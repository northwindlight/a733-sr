#!/usr/bin/env python3
"""
A733 视频超分服务 —— 上传/哔哩哔哩链接 -> NPU 超分 -> 下载

流水线（全部在本机 A733 上，不依赖外部机器）：
    [源视频]
      └ vdecpipe ── VE 硬解 H.264（流式，一个进程全程活着喂管道）
          └ srpipe  ── NPU 分块 x4 超分（块 352x224，带 margin 防接缝）
              └ ffmpeg 缩放到目标分辨率，输出 NV12（★必须在这一步缩：
                 VE 编码器不会缩，给它小一号的 -d 它只会裁左上角）
                  └ vencoderdemo_v2 ── VE 硬编 H.264（-s == -d，纯编码）
                      └ ffmpeg 混流原音轨 -> mp4（★不能用 -shortest，见 README）

（没有 VE 硬解时走软解：ffmpeg 解码 + 缩放到源分辨率 + 转 gbrp 平面喂 srpipe。）

分块的原因：NBG 是定形状的。要么把源降采样（丢信息），要么重编形状
（要重测内存池；640x360 x4 实测会溢出 84MB 池 -> 输出错乱但不报错）。
分块两者都不需要，且代价仍 ∝ 源总像素（实测 0.547 µs/px）。

只依赖 python3 标准库 + numpy + 系统里的 ffmpeg/yt-dlp/srpipe。
"""
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

import sysmon
from fractions import Fraction

# ---------------------------------------------------------------- 配置
BASE     = os.path.dirname(os.path.abspath(__file__))
DATA     = os.path.join(BASE, "data")          # 上传 + 产物
WORK     = os.path.join(BASE, "work")          # 每任务的临时目录
JOBS     = os.path.join(DATA, "jobs.json")
STATIC   = os.path.join(BASE, "static")

def _env(k, d):
    return os.environ.get(k) or d

# 系统状态采集：一个后台线程按 1 Hz 刷新快照，网页用 SSE 订阅。
# ★必须单点采样：CPU 利用率是两次 /proc/stat 的差值，每个连接各采各的会互相
#   吃掉差值（第二个标签页一开，第一个的读数就跳 0）。
SYS = sysmon.Sampler()
SYS_MAX_CLIENTS = int(_env("SR_SYS_CLIENTS", "4"))   # 1GB 内存的板子，别让线程炸开

VENV_DIR = _env("SR_VENC_DIR",  os.path.join(BASE, "vendor/venc"))
NPU_BIN  = _env("SR_SRPIPE",    os.path.join(BASE, "srpipe"))
VIP_LIB  = _env("SR_VIP_LIB",   os.path.join(BASE, "vendor/viplite"))
NBG      = _env("SR_NBG",       os.path.join(BASE, "models", "anv3_352x224.nb"))
VENC_BIN = _env("SR_VENC_BIN",  os.path.join(VENV_DIR, "vencoderdemo_v2"))
COOKIES  = _env("SR_COOKIES",   os.path.join(BASE, "bili_cookies.txt"))  # 可选，有则能下更高清
# GPU 色彩转换开关。需要 img-bxm-dkms（/dev/dri/card1）+ libPVROCL。
GPU_CONV = _env("SR_GPU", "1") not in ("0", "", "no", "false")

# ★VE 硬解开关。把整片用专用硅解成 NV12 裸流落到 work 目录，之后的分段只做
#   「seek + crop + 搬运」，软解和 swscale 都从 SR 阶段消失。
#   尺子是【挪热点】：实测 SR 阶段那步 ffmpeg 的 CPU 从 15.9 ms/帧 降到 5.1 ms/帧。
#   代价是多一个串行预pass（VE 解码本身只 2.95 ms/帧 CPU，但 NV12 落盘受 eMMC 限速，
#   实测 ~22 ms/帧 wall），以及中间文件要占盘 —— 所以下面有容量闸，超了就退回软解。
#   ★解码器用【系统自带】的 cedarc 库：我们自己编的那套 libawh264 会在
#     H264DecoderInit 里段错误（见 deploy/ve-decode/README.md）。
VE_DECODE = _env("SR_VE_DECODE", "1") not in ("0", "", "no", "false")
VDEC_BIN  = _env("SR_VDEC_BIN",  os.path.join(BASE, "vdecpipe"))
VE_MAX_INTERMEDIATE = int(_env("SR_VE_MAX_GB", "20")) << 30   # 流式改造后已不用（见 sweep_old）

# ★块形状必须等于 NBG 的输入形状，两者是【一起定】的。
# NBG 定形状 ⇒ 每块 NPU 代价固定（实测只与块数有关，与 core 面积无关），
# 所以整帧代价 = 块数 x 常数，能优化的只有块数。旧 320x180 的 core 是 288x148，
# 960x540 要 4x4=16 块；352x224 的 core 是 320x192，960x540 正好 3x3=9 块。
# 实测 960x540：NPU 492.5 -> 395.7 ms/帧，整帧 666 -> 600 ms。
#   1280x720  25 -> 16 块 (-12%)      1920x1080  56 -> 36 块 (-12%)
#   640x360    9 ->  4 块 (-39%)      320x180    1 ->  1 块 (反而略差，可忽略)
TILE_W, TILE_H = 352, 224      # 必须等于 NBG 的输入形状
SR_SCALE       = 4             # 该 NBG 是 x4
MARGIN         = 16            # 分块防接缝的余量（实测 16 足够：边界差 2.07 -> 0.00）
CHUNK_FRAMES   = 300           # 每个编码分段的帧数，限制临时盘占用

PORT = int(os.environ.get("SR_PORT", "8080"))

os.makedirs(DATA, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

# ---------------------------------------------------------------- 任务表
_lock = threading.Lock()
_jobs = {}

# ---------------------------------------------------------------- 取消
# 取消要能杀掉正在跑的那一段：流水线里跑的是
#   ffmpeg | srpipe | ffmpeg   （bash 管道）和 vencoderdemo
# 所以起进程时要用 start_new_session 开新进程组，取消时 killpg 整组，
# 否则只杀 bash、下游的 srpipe/ffmpeg 会变成孤儿继续吃 NPU。
_cancel = set()
_procs = {}


class Cancelled(Exception):
    pass


# ★串行闸：板子只有一个 NPU、一个编码器实例。两个任务同时跑会互相抢，
#   各自半速 —— 不如排队跑完一个再跑下一个。README 一直是这么写的，
#   这里把实现补上（原来是无脑起线程，并发跑）。
_run_lock = threading.Lock()


def request_cancel(jid):
    with _lock:
        _cancel.add(jid)
        p = _procs.get(jid)
    if p is not None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def cancel_requested(jid):
    with _lock:
        return jid in _cancel


def _register(jid, p):
    with _lock:
        _procs[jid] = p


# 流式 VE 解码器：一个进程全程活着给管道喂数据，跨段存在，所以不能走 _procs
# （那里一条任务只放得下一个进程，段里还要放 srpipe）。单独一张表，谁都能收。
_ve_procs = {}


def _ve_kill(jid):
    """收掉解码器。正常结束、出错、取消三条路都要走它 —— 漏一条它就挂在管道上
    不走，而且一直占着 VE 硬件，下一个任务会莫名其妙地慢。"""
    with _lock:
        it = _ve_procs.pop(jid, None)
    if not it:
        return
    dec, raws = it
    try:
        if dec.stdout:
            dec.stdout.close()
    except Exception:
        pass
    try:
        dec.terminate()
        dec.wait(timeout=5)
    except Exception:
        try: dec.kill()
        except Exception: pass
    try: os.remove(raws)
    except OSError: pass


def _unregister(jid, p):
    with _lock:
        if _procs.get(jid) is p:
            _procs.pop(jid, None)


def _load():
    try:
        with open(JOBS) as f:
            _jobs.update(json.load(f))
    except Exception:
        pass


def _save():
    tmp = JOBS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_jobs, f, ensure_ascii=False, indent=1)
    os.replace(tmp, JOBS)


def set_job(jid, **kw):
    with _lock:
        j = _jobs.setdefault(jid, {})
        j.update(kw)
        j["updated"] = time.time()
        _save()


def get_job(jid):
    with _lock:
        j = _jobs.get(jid)
        return dict(j) if j else None


def all_jobs():
    with _lock:
        return sorted(_jobs.items(), key=lambda kv: kv[1].get("created", 0), reverse=True)


# ---------------------------------------------------------------- 工具
def run_stream(jid, cmd, on_line):
    """流式跑命令：逐行回调，用于 yt-dlp 这种要报进度的。
    （原来用 run() 的 communicate() 会把输出全缓冲住 —— 下载期间界面
      停在 2% 一动不动，看着像卡死。）"""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, start_new_session=True)
    _register(jid, p)
    tail = []
    try:
        for line in p.stdout:
            tail.append(line)
            if len(tail) > 80:
                tail.pop(0)
            try:
                on_line(line)
            except Exception:
                pass          # 进度解析失败绝不能影响下载
        p.wait()
    finally:
        _unregister(jid, p)
    if cancel_requested(jid):
        raise Cancelled()
    return p.returncode, "".join(tail)


def _monitor_srpipe(jid, path, base, total, t_start, stop):
    """跟着 srpipe 的 stderr 走 —— 它每处理完一帧就写一行 `FRAME n`。

    没有这个的话，界面在一整段（大源要几分钟到十几分钟）里只有一次进度更新，
    看着就是卡死。开了它进度条才是真的在动。

    ★节流：set_job 会落盘 jobs.json，逐帧写会变成每秒几十次小文件写。
      所以最快 1 秒更新一次。
    """
    f = None
    last = -1
    last_t = 0.0
    while not stop.is_set():
        if f is None:
            try:
                f = open(path, errors="ignore")
            except OSError:
                time.sleep(0.3)
                continue
        line = f.readline()
        if not line:
            time.sleep(0.25)
            continue
        m = re.match(r"FRAME (\d+)", line)
        if not m:
            continue
        n = int(m.group(1))
        if n == last:
            continue
        last = n
        now = time.time()
        if now - last_t < 1.0 and not stop.is_set():
            continue                       # 节流，但收尾那次仍然写
        last_t = now
        done = min(base + n + 1, total or base + n + 1)
        el = max(0.01, now - t_start)
        rate = done / el
        eta = int((total - done) / rate) if (total and rate > 0) else 0
        set_job(jid, progress=min(89, 10 + (done / total * 80 if total else 0)),
                frames_done=done, fps_now=round(rate, 2), eta_s=eta)
    if f:
        try:
            f.close()
        except Exception:
            pass


def run(cmd, jid=None, **kw):
    """跑一条命令，失败抛异常并带上 stderr 尾巴。可取消。"""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, start_new_session=True, **kw)
    if jid:
        _register(jid, p)
    try:
        out, err = p.communicate()
    finally:
        if jid:
            _unregister(jid, p)
    if jid and cancel_requested(jid):
        raise Cancelled()
    if p.returncode != 0:
        tail = "\n".join((err or out or "").strip().split("\n")[-12:])
        raise RuntimeError(f"命令失败({p.returncode}): {' '.join(cmd[:3])}...\n{tail}")
    return subprocess.CompletedProcess(cmd, 0, out, err)


def streams(path):
    p = run(["ffprobe", "-v", "error", "-print_format", "json",
             "-show_streams", path])
    return json.loads(p.stdout)["streams"]


def probe(path):
    p = run(["ffprobe", "-v", "error", "-print_format", "json",
             "-show_streams", "-show_format", path])
    info = json.loads(p.stdout)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if not v:
        raise RuntimeError("这个文件里没有视频流")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    fps = (float(num) / float(den)) if float(den) else 0.0
    return {
        "w": int(v["width"]), "h": int(v["height"]),
        # ★保留原始有理数帧率。29.97 这类帧率取整成 30 会让音画在长片上错开
        #   （535 秒的片子差 0.5 秒），所以打时间戳一律用 num/den，不用那个 float。
        "fps_num": int(float(num)), "fps_den": int(float(den)),
        "fps": fps, "has_audio": a is not None,
        "duration": float(info["format"].get("duration") or 0),
        "nb_frames": int(v.get("nb_frames") or 0),
        "codec": v.get("codec_name") or "",
    }


# ---------------------------------------------------------------- 流水线
def check_output(path, want_w, want_h, want_audio, n_done, want_dur, fr):
    """★产物自证 —— 拿产物的客观量跟【请求】比，不是看 rc、不是看界面标签。

    三个 bug 全都活过了"端到端跑通"的验收，因为验收只看界面上的「完成」：
      ① `-shortest` 让 aac 编出 0 KiB，产物静音，ffmpeg rc=0
      ② VE 编码器把 `-d` 当裁剪框，3840x2160 要 1920x1080 就交左上 1/4 画面
      ③ 裸 H.264 没时间戳，`-r` 没给对就快放（152 帧播 2.0 秒）
    这里逐条对着请求查，对不上就抛 —— **绝不打印成功**。

    单独抽成函数是为了能拿构造出来的坏文件直接测（tools/gatetest.sh），
    不用每次都跑一遍五分钟的完整流水线。
    """
    st = streams(path)
    v = next((s for s in st if s["codec_type"] == "video"), None)
    if v is None:
        raise RuntimeError("产物里没有视频流")
    if (int(v["width"]), int(v["height"])) != (want_w, want_h):
        raise RuntimeError(
            f"产物分辨率是 {v['width']}x{v['height']}，不是要求的 {want_w}x{want_h}。"
            f"VE 编码器只会裁不会缩（-d 小于 -s 时它交的是左上角那块），"
            f"缩放必须由 srpipe 后面那一级 ffmpeg 做。")
    if want_audio and not any(s["codec_type"] == "audio" for s in st):
        raise RuntimeError("源有音轨，产物里没有 —— 混音那一步静默丢了音频。")
    # 容差 ±1 帧：容器里的 nb_frames 是头部元数据，实测连"一段纯 libx264 裸流 concat
    # 起来"都会报得比实际解码出来的多 1 帧，卡死相等会误报。差一大截（比如少一半）
    # 才是真故障，那种照样拦得住。
    nv = int(v.get("nb_frames") or 0)
    if nv and abs(nv - n_done) > 1:
        raise RuntimeError(f"产物 {nv} 帧，但编码段加起来是 {n_done} 帧 —— 拼接丢了帧。")
    # 时长也要对得上：帧数 / 精确帧率。差半帧以上就说明 -r 没吃到真实帧率
    #   （用 30 顶替 29.97 时，9 分钟的片子会差 0.5 秒，音画后半段就不同步了）。
    #   容差 2 帧：容器报出来的 duration 实测是 (帧数-1)/帧率（最后一包的时长没算进去），
    #   卡 1 帧会贴着边界。快放那个 bug 差的是几十帧，2 帧的容差照样拦得住。
    vd = float(v.get("duration") or 0)
    if vd and abs(vd - want_dur) > max(2.0 / float(fr), 0.1):
        raise RuntimeError(
            f"产物时长 {vd:.3f}s，应该是 {want_dur:.3f}s（{n_done} 帧 / "
            f"{fr.numerator}/{fr.denominator}）—— 时间戳没按源帧率打。")


def npu_env():
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = VIP_LIB + ":" + e.get("LD_LIBRARY_PATH", "")
    return e


def pipeline(jid, src, scale, target_h, t_start, sharp=0.4):
    """
    scale    : 用户选的"超分倍率" 2 或 4 —— 只影响最终输出尺寸，不影响 NPU 代价
    target_h : 用户选的最终高度（0 = 自动 = 源高 * scale）
    """
    wd = os.path.join(WORK, jid)
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(wd, exist_ok=True)

    set_job(jid, stage="探测源视频", progress=8)
    meta = probe(src)
    W, H = meta["w"], meta["h"]
    fps = meta["fps"] or 30.0
    # 精确帧率（有理数）。ffprobe 给不出有效值时才退回按毫秒近似的那个。
    # ★这是"音画对准"的基准：产物时长必须是 帧数 / 这个值，音频也按它切。
    fr = (Fraction(meta["fps_num"], meta["fps_den"])
          if meta["fps_num"] > 0 and meta["fps_den"] > 0 else Fraction(round(fps * 1000), 1000))

    out_w = W * SR_SCALE
    out_h = H * SR_SCALE

    if target_h:
        final_h = target_h
        final_w = int(round(target_h * W / H / 2)) * 2      # 保持比例，且宽为偶数
    else:
        final_h = H * scale
        final_w = int(round(W * scale / 2)) * 2

    # 最终尺寸不能超过 SR 出来的尺寸（再大也没有信息）
    if final_h > out_h:
        final_h, final_w = out_h, out_w

    # ★VE 编码器【只会裁、不会缩】。给它 -s 3840x2160 -d 1920x1080，它交出来的不是
    #   缩小后的整帧，而是**左上角 1920x1080 那一块**（实测：单独把一张正确的 4K NV12
    #   喂给它，出来就是左上 1/4，和整条流水线的产物一模一样）。vencoder_platform_v2.h
    #   里那个 Adscaler 是锐化参数，不是缩放器。
    #   ⇒ 缩放不能交给它：改成在 srpipe 后面挂一级 ffmpeg swscale，并让编码器永远
    #     -s == -d。走管道 ⇒ 4K 的中间帧不进磁盘，落盘的只有目标尺寸（12.4 -> 3.1 MB/帧）。
    #   flags=area 是真正的面积平均，实测 24.5 ms/帧（占 850 ms/帧的 3%）。
    ds = (final_w != out_w or final_h != out_h)
    mid_w, mid_h = (final_w, final_h) if ds else (out_w, out_h)

    tiles_x = -(-W // (TILE_W - 2 * MARGIN))
    tiles_y = -(-H // (TILE_H - 2 * MARGIN))
    npu_ms = W * H * 0.547 / 1000.0            # 实测 0.547 µs/源像素

    # 落盘的中间文件是**目标**分辨率的 NV12（缩放那一级 ffmpeg 在管道里，4K 不进盘）。
    # 分段大小按它定，把临时盘控在 ~2 GB 以内。
    bytes_per_frame = mid_w * mid_h * 3 // 2
    chunk = max(10, min(CHUNK_FRAMES, int(2e9 / max(1, bytes_per_frame))))

    total_frames = meta["nb_frames"] or int(meta["duration"] * fps)
    est = npu_ms * total_frames / 1000.0 if total_frames else 0
    set_job(jid, stage="准备中", meta={
        "src": f"{W}x{H}", "sr": f"{out_w}x{out_h}",
        "final": f"{final_w}x{final_h}", "fps": round(fps, 2),
        "tiles": f"{tiles_x}x{tiles_y}={tiles_x*tiles_y}", "npu_ms": round(npu_ms, 1),
        "frames": total_frames, "est_npu_s": round(est, 1),
    })

    # ---------- 可选：VE 硬解（流式，全程与超分并行）----------
    # 解码器【一个进程全程活着】往管道里吐 NV12，每一段让 srpipe 读够 chunk 帧就退出，
    # 剩下的字节留在管道里给下一段。
    # 以前不是这样的：解码器先把整片写成 NV12 文件再审，那有三个毛病 ——
    #   ① 有容量闸（超 20G/剩余空间 40% 就退回软解）⇒ 1080p 长片根本用不上硬解
    #   ② 开局要空等几分钟，界面完全没进度
    #   ③ 半截 dump 能一路混过去，静默交半个视频
    # 解码（122fps）远快于超分（~1.5fps），所以管道永远不会饿着 NPU；
    # 流式省不掉 NPU 的时间，省掉的是上面那三件。
    ah = (H + 15) // 16 * 16              # ★解码器输出的高按 16 对齐（960x540 -> 960x544）
    ve = None                             # (解码进程, 裸流路径)；None = 走软解
    if VE_DECODE and meta.get("codec") == "h264" and os.path.exists(VDEC_BIN):
        raws = os.path.join(wd, "src.h264")
        declog = os.path.join(wd, "vdec.log")
        set_job(jid, stage="抽裸流", progress=5)
        # 解码器只吃 elementary stream，不吃 mp4 容器
        run(["bash", "-c", f'ffmpeg -v error -nostdin -y -i {shq(src)} '
                           f'-c:v copy -bsf:v h264_mp4toannexb -f h264 {shq(raws)}'], jid=jid)
        # ★数据走 fd3、日志走 fd1。这个 demo 的日志是裸 printf，全在 stdout 上，
        #   直接 `-o /dev/stdout` 会把日志插进 NV12 流里（实测 21 处），
        #   于是流被垫歪、后面的帧全部错位 —— 而且大小看着还挺像整数帧。
        # ★-ss 0 -sn N 也必须给：demo 默认「一帧都不存且不报错」。
        n = max(1, total_frames) + 8
        dec = subprocess.Popen(
            ["bash", "-c",
             f'exec 3>&1; exec env LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu '
             f'{shq(VDEC_BIN)} -i {shq(raws)} -codFmat 1 -o /dev/fd/3 -outFmat 6 '
             f'-n {n} -ss 0 -sn {n} 1> {shq(declog)}'],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        ve = (dec, raws)
        with _lock:
            _ve_procs[jid] = ve
        set_job(jid, stage="VE 硬解已接上（边解边超分）", progress=6)

    # ---------- 分段处理 ----------
    t_sr_start = time.time()
    n_done = 0
    parts = []
    idx = 0
    while True:
        # 本段从 n_done 帧开始
        seg = os.path.join(wd, f"seg{idx}.h264")
        nv12 = os.path.join(wd, f"seg{idx}.nv12")

        # srpipe 直接吐 NV12（源×4 分辨率）。★注意：缩放【不是】VE 编码器做的 ——
        # 它只会裁。曾经这里写着"缩放交给 VE，白送"，那句话是错的，代价是用户拿到的
        # 每一个 1080p 产物都只有左上 1/4 画面。真正做缩放的是下面管道里那级 ffmpeg。
        # 数据量砍半：gbrp 3 字节/像素 -> NV12 1.5。
        # 锐化由 srpipe 自己做，不让开关静默失效。
        sh_arg = f" --sharpen {sharp}" if sharp and sharp > 0 else ""
        # ★GPU 做色彩转换，和 NPU 并行。选它不是因为它快（实测比 CPU 慢），
        #   是因为 CPU 是那个 60°C 就降频的热区，把活挪走能少降频。
        gpu_arg = " --gpu" if GPU_CONV else ""
        if ve:
            # 流式硬解：直接从解码器的管道读，连中间那级 ffmpeg 都省了
            # （它原来只干一件事：把解码器补到 16 对齐的 544 行裁成 540，
            #   实测每帧 5.1ms CPU）。现在 srpipe 自己带 --in-h 裁。
            # ★顺序读，不 seek —— 管道里是连续的帧，seek 不了也不需要。
            head = (
                f'LD_LIBRARY_PATH={VIP_LIB} {PIN}{shq(NPU_BIN)} {shq(NBG)} {W} {H} {SR_SCALE} '
                f'{TILE_W} {TILE_H} --margin {MARGIN} --nv12 --in-nv12 --in-h {ah} '
                f'--frames {chunk}{sh_arg}{gpu_arg} '
                f'2> {shq(wd + "/sr.err")}'
            )
        else:
            # 软解路径：ffmpeg 解码 + 缩放到源分辨率 + 转 gbrp 平面，喂给 srpipe
            # ★这里刻意【不】钉解码那级 ffmpeg：它是多线程的，钉到 2 个大核反而更慢。
            #   只钉 srpipe（单线程）和后面那级缩放。
            head = (
                f'ffmpeg -v error -nostdin '
                f'-ss {n_done / fps:.6f} -i {shq(src)} '
                f'-frames:v {chunk} -an -vf scale={W}:{H} -pix_fmt gbrp -f rawvideo - '
                f'| LD_LIBRARY_PATH={VIP_LIB} {PIN}{shq(NPU_BIN)} {shq(NBG)} '
                f'{W} {H} {SR_SCALE} '
                f'{TILE_W} {TILE_H} --margin {MARGIN} --nv12{sh_arg}{gpu_arg} '
                f'2> {shq(wd + "/sr.err")}'
            )
        # 两条路都在这里收口：SR 出来是源×4，如果不是目标尺寸，就在管道里缩一次。
        if ds:
            cmd = (f'set -o pipefail; {head} | {PIN}ffmpeg -v error -nostdin -f rawvideo '
                   f'-pix_fmt nv12 -s {out_w}x{out_h} -i - '
                   f'-vf scale={final_w}:{final_h}:flags=area -f rawvideo -pix_fmt nv12 '
                   f'-y {shq(nv12)}')
        else:
            cmd = f'set -o pipefail; {head} > {shq(nv12)}'
        if cancel_requested(jid):
            raise Cancelled()
        # ★这一段是整条流水线里最长的一步（大源要几分钟到十几分钟），
        #   必须在**开跑之前**就把 stage 设对，否则界面会一直停在上一句
        #   （上一句是"准备中"，于是看起来像卡死 —— 用户实测踩到）。
        set_job(jid, stage=f"超分第 {idx+1} 段"
                           f"（{n_done}~{min(n_done + chunk, total_frames or n_done + chunk)} 帧）",
                progress=max(10, 10 + (n_done / total_frames * 80 if total_frames else 0)))
        sr_err = os.path.join(wd, "sr.err")
        stop_evt = threading.Event()
        mon = threading.Thread(target=_monitor_srpipe,
                               args=(jid, sr_err, n_done, total_frames, t_sr_start, stop_evt),
                               daemon=True)
        mon.start()
        try:
            # VE 流式路：把解码器的管道接到 srpipe 的 stdin。★Python 一直持有这个
            # 读端 ⇒ 段与段之间解码器不会被 SIGPIPE 打死，它只是写满管道缓冲区后
            # 阻塞，等下一段来读。少了这一手，长片会在第二段开头就断。
            p = run(["bash", "-c", cmd], jid=jid,
                    stdin=(ve[0].stdout if ve else None))
        finally:
            stop_evt.set()
        if p.returncode != 0:
            raise RuntimeError("超分段失败:\n" + (p.stderr or "")[-800:])

        if not os.path.exists(nv12) or os.path.getsize(nv12) < 1024:
            break                                    # 没有更多帧了

        got = os.path.getsize(nv12) // (mid_w * mid_h * 3 // 2)   # 中间已经是目标尺寸
        if got == 0:
            break

        # VE 硬编这一段。★-s 和 -d 必须【相等】：编码器不会缩，-d 小了它只会裁。
        #   尺寸的活已经在上面那级 ffmpeg 干完了，这里就是纯编码。
        set_job(jid, stage=f"硬件编码 第{idx+1}段")
        # ★-r 必须给源帧率：不给的话编码器用自己的默认帧率打时间戳，
        #   152 帧会被塞进 2 秒（实测源 10.2 秒的片子出来播 2.0 秒，快放 5 倍），
        #   而且 dts 非单调、后面 concat 也会跟着错。
        run(["bash", "-c",
             f'LD_LIBRARY_PATH={VENV_DIR} {shq(VENC_BIN)} -i {shq(nv12)} '
             f'-n {got} -f 0 -o {shq(seg)} -s {final_w}x{final_h} -d {final_w}x{final_h} '
             f'-r {max(1, int(round(fps)))}'],
            jid=jid)

        os.remove(nv12)
        parts.append(seg)

        n_done += got
        pct = 10 + (n_done / total_frames * 80 if total_frames else 0)
        # ★ETA 按**实测速率**外推，不要用"NPU 理论耗时" —— 那个只算 NPU，
        #   实测端到端是它的 3~4 倍（解码/组帧/NV12 转换/编码都算上），
        #   报出来会严重低估，用户以为卡死了。
        el = max(0.01, time.time() - t_sr_start)
        rate = n_done / el
        eta = int((total_frames - n_done) / rate) if (total_frames and rate > 0) else 0
        set_job(jid, progress=min(90, pct), frames_done=n_done,
                fps_now=round(rate, 2), eta_s=eta,
                stage=f"超分+编码 {n_done}/{total_frames or '?'} 帧"
                      + (f"（约剩 {eta//60} 分）" if eta > 60 else ""))
        idx += 1

        if got < chunk:
            break                                    # 源已经放完

    # 解码器用完就收 —— 它还挂在管道上，不收就一直占着 VE 硬件
    _ve_kill(jid)

    if not parts:
        raise RuntimeError("没有产出任何编码段 —— 源视频可能读不出来")

    # ★产出帧数必须和源对得上，少一大截就是出了故障，【不能报「完成」】。
    #   这个闸是拿血换的：srpipe 因为不认 --in-nv12，每段都少读一半帧、
    #   每段都"正常"结束，于是任务显示完成、给下载，交出去半个视频 ——
    #   界面上一切正常，只有文件大小不对（用户就是这样发现的）。
    if total_frames and n_done < total_frames * 98 // 100:
        raise RuntimeError(
            f"只产出了 {n_done}/{total_frames} 帧就断了，不当作成功。"
            f"上游提前结束（不是源真的放完了）。常见原因："
            f"srpipe / 编码器与当前调用参数不匹配（例如二进制没跟着 app.py 一起更新）。")

    # ---------- 拼接 + 混音 ----------
    set_job(jid, stage="拼接", progress=92)
    listf = os.path.join(wd, "parts.txt")
    with open(listf, "w") as f:
        for pth in parts:
            f.write(f"file '{pth}'\n")

    out = os.path.join(DATA, jid + ".mp4")
    # ★-r 必须在 -i【之前】：那几段是裸 H.264（elementary stream 里根本没有时间戳），
    #   ffmpeg 解析时一律按默认 25fps 打点，跟源的真实帧率毫无关系 ——
    #   结果是产物时长整个是错的（实测源 10.2 秒的片子出来播 2.0 秒，快放 5 倍；
    #   而且不管 80 帧还是 152 帧都恰好 2.0 秒）。把输入帧率钉成源帧率才对。
    #   这个 bug 一直都在，只是没人去核过产物时长。
    # ★★混音这一步有两个坑，都是「rc=0、界面显示完成、东西是错的」：
    #   ① 不能用 -shortest。concat 进来的裸 H.264 没有时间戳（下面这行 -r 只能补
    #      输入帧率，补不出每包的 pts），ffmpeg 因此算不出视频流的结束时间，
    #      于是 -shortest 判定「视频已经结束」⇒ aac 一帧都没编，输出 audio:0KiB。
    #      实测：加 -shortest 产物无音轨，去掉就有；两次 rc 都是 0。
    #   ② -map 后面的 ? 会把「映射不到音轨」变成静默通过。既然上面已经确认源有音轨，
    #      就不要 ? —— 映射失败就该报错。
    #   下面还有一道产物自证闸兜底（分辨率 / 音轨 / 帧数 / 时长）。
    # ★帧率用【有理数】原文，不要取整。30 和 30000/1001 差 0.1%，9 分钟的片子
    #   就是 0.5 秒的累积偏移 —— 画面比声音快半秒，越到后面越明显。
    #   ffmpeg 的 -r 吃 "30000/1001" 这种写法。
    fps_r = f"{fr.numerator}/{fr.denominator}"
    dur = float(Fraction(n_done * fr.denominator, fr.numerator))   # 产物应有的时长
    if meta["has_audio"]:
        head = (f'ffmpeg -v error -nostdin -f concat -safe 0 -r {fps_r} -i {shq(listf)} '
                f'-i {shq(src)} -map 0:v:0 -map 1:a:0 -c:v copy ')
        # ★音轨【原样复制】，不重编码：源多半已经是 aac，重编一代就多一代损失，
        #   而且这一步本来就没有需要重编的理由（视频也是 copy）。
        #   -t 精确切到产物时长。实测 150 帧的段切完仍是 150 帧，不会啃掉最后一帧。
        try:
            run(["bash", "-c", head + f'-c:a copy -t {dur:.6f} -y {shq(out)}'], jid=jid)
        except RuntimeError:
            # 复制不进 mp4 的音轨（opus/vorbis 之类）会在这里失败 —— 失败是响的，
            # 不是静默的。退回重编码，并用 atrim 精确切音频（-t 那版会连视频一起截）。
            run(["bash", "-c",
                 head + f'-c:a aac -b:a 192k -af atrim=end={dur:.6f},asetpts=N/SR/TB '
                        f'-y {shq(out)}'], jid=jid)
    else:
        cmd = (f'ffmpeg -v error -nostdin -f concat -safe 0 -r {fps_r} -i {shq(listf)} '
               f'-c:v copy -y {shq(out)}')
        run(["bash", "-c", cmd], jid=jid)

    # ★产物必须自证 —— 见 check_output 的注释（单独抽成函数就为了能拿坏文件直接测）
    check_output(out, final_w, final_h, meta["has_audio"], n_done, dur, fr)

    shutil.rmtree(wd, ignore_errors=True)
    size = os.path.getsize(out)
    set_job(jid, stage="完成", progress=100, status="done",
            out=os.path.basename(out), size=size,
            took=round(time.time() - t_start, 1),
            fps_done=round(n_done / max(0.01, time.time() - t_start), 2))


def _pick_cpus():
    """默认把重活钉在 capacity 最大的那组核上。

    A733 是 6×A55(cap 385) + 2×A76(cap 1024) —— 单核容量差 **2.66 倍**。
    小核不是"省电"：同一份活在小核上要跑 2.66 倍的时间，做完更晚、回 idle 更晚，
    对"跑完就等 NPU"这种突发型负载是**又慢又费**。
    实测（2026-09-28）不钉的时候 srpipe 在 7/4/3/3/3/3/4/6/4 之间乱跳 ——
    单线程的活被调度器摊到空闲的小核上去了，大核空着。

    返回 None 表示不限制。可用 SR_CPUSET 覆盖（"6,7"，或 0/off 关掉）。
    """
    env = os.environ.get("SR_CPUSET")
    if env is not None:
        env = env.strip()
        if env in ("", "0", "off", "no", "false"):
            return None
        try:
            return sorted({int(x) for x in env.replace(" ", ",").split(",") if x})
        except ValueError:
            return None
    caps = {}
    for d in os.listdir("/sys/devices/system/cpu"):
        if d.startswith("cpu") and d[3:].isdigit():
            try:
                with open(f"/sys/devices/system/cpu/{d}/cpu_capacity") as f:
                    caps[int(d[3:])] = int(f.read().strip())
            except (OSError, ValueError):
                pass
    if not caps:
        return None
    big = sorted(c for c, v in caps.items() if v == max(caps.values()))
    # 全都一样 = 同构机器，钉了没意义（钉到全部核 = 空操作，但别写进命令行里）
    return big if len(big) < len(caps) else None


CPUS = _pick_cpus()
if CPUS and not shutil.which("taskset"):
    print("警告: 没有 taskset（util-linux），不做钉核", flush=True)
    CPUS = None
# 命令行前缀：'taskset -c 6,7 '。不限制时是空串。
# 用 taskset 而不是 subprocess 的 preexec_fn —— 本服务是多线程的，
# preexec_fn 在 fork 和线程之间不安全。
PIN = f'taskset -c {",".join(map(str, CPUS))} ' if CPUS else ""


def shq(s):
    return "'" + str(s).replace("'", "'\\''") + "'"


def worker(jid, src, scale, target_h, url, sharp=0.4):
    t0 = time.time()
    try:
        if url:
            set_job(jid, stage="下载中（bilibili）", progress=2, status="running")
            dst = os.path.join(DATA, jid + ".src.mp4")
            # --newline：让下载进度一条一行，不然会挤在一行里解析不到
            cmd = ["yt-dlp", "--no-playlist", "--no-update", "--newline",
                   "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
                   "--merge-output-format", "mp4", "-o", dst]
            if os.path.exists(COOKIES):
                cmd += ["--cookies", COOKIES]
            cmd.append(url)

            def _on_dl(line):
                m = re.search(r"\[download\]\s+([\d.]+)%", line)
                if m:
                    set_job(jid, progress=2 + float(m.group(1)) * 0.06,
                            stage="下载中（bilibili）%s%%" % m.group(1))
                elif line.startswith("[Merger]") or "[ffmpeg]" in line:
                    set_job(jid, progress=8, stage="下载完成，正在合并音视频")

            rc, out = run_stream(jid, cmd, _on_dl)
            if rc != 0 or not os.path.exists(dst):
                raise RuntimeError("下载失败:\n" + "\n".join(out.strip().split("\n")[-8:]))
            src = dst
        set_job(jid, status="running", _src=src)   # 记下来，收尾时删它
        if _run_lock.locked():
            set_job(jid, stage="排队等前面的任务跑完（板子只有一个 NPU）")
        with _run_lock:
            if cancel_requested(jid):
                raise Cancelled()
            pipeline(jid, src, scale, target_h, t0, sharp)
    except Cancelled:
        set_job(jid, status="cancelled", stage="已取消", progress=100,
                error="用户取消", took=round(time.time() - t0, 1))
        shutil.rmtree(os.path.join(WORK, jid), ignore_errors=True)
    except Exception as e:
        set_job(jid, status="error", stage="失败", error=str(e)[:1500],
                progress=100, detail=traceback.format_exc()[-1500:])
    finally:
        # 三条退出路径（成功/取消/失败）都要收：解码器是跨段活着的独立进程，
        # 漏一条它就挂在管道上不走、还一直占着 VE 硬件；源文件也要删。
        _ve_kill(jid)
        sweep_old(jid)


# ---------------------------------------------------------------- HTTP
_sse_lock = threading.Lock()
_sse_n = [0]                     # 当前 SSE 连接数（用列表当可变盒子）


class H(BaseHTTPRequestHandler):
    server_version = "a733-sr"

    def log_message(self, fmt, *a):
        print("[%s] %s" % (self.log_date_time_string(), fmt % a), flush=True)

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _file(self, path, ctype="application/octet-stream", download=None):
        if not os.path.isfile(path):
            return self._json({"error": "文件不存在"}, 404)
        sz = os.path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(sz))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, 1 << 16)

    def _sse_sys(self):
        """SSE 推系统状态。每个连接一个线程，最多 SYS_MAX_CLIENTS 个。

        内容是「变化才推」：温度每秒都在动，所以实际上就是 1 Hz。
        用 HTTP/1.1 + Connection: close —— 响应体靠关连接定界，
        这是 EventSource 认的合法形态（类默认是 HTTP/1.0，这里显式抬一下）。
        """
        with _sse_lock:
            if _sse_n[0] >= SYS_MAX_CLIENTS:
                return self._json({"error": "监视连接数已满(%d)" % SYS_MAX_CLIENTS}, 503)
            _sse_n[0] += 1
        try:
            self.protocol_version = "HTTP/1.1"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")   # 万一前面挂了反代，别缓冲
            self.end_headers()
            # 先推一次：CPU 利用率要等第二个采样点才有值，先告诉前端「在等」
            self.wfile.write(b"retry: 3000\n\n")
            last = None
            while True:
                b = json.dumps(SYS.snapshot(), ensure_ascii=False)
                if b != last:
                    self.wfile.write(("data: " + b + "\n\n").encode())
                    last = b
                time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                                   # 关页面就是关连接，正常
        finally:
            with _sse_lock:
                _sse_n[0] -= 1

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._file(os.path.join(STATIC, "index.html"),
                              "text/html; charset=utf-8")
        if u.path == "/api/sys":
            return self._sse_sys()
        if u.path == "/api/sys.json":              # 调试/巡检用的一次性快照
            return self._json(SYS.snapshot())
        if u.path == "/api/jobs":
            return self._json({"jobs": [dict(v, id=k) for k, v in all_jobs()][:40]})
        if u.path == "/api/status":
            jid = (q.get("id") or [""])[0]
            j = get_job(jid)
            if not j:
                return self._json({"error": "无此任务"}, 404)
            # ★必须把 id 也放进去：id 本来是字典的 key，不在值里。
            #   前端轮询拿到 j 之后要拿 j.id 去拼"取消"和"下载"的链接，
            #   漏了它就变成 undefined —— 按钮看着在、点了没反应，而且不报错。
            j["id"] = jid
            return self._json(j)
        if u.path == "/api/download":
            jid = (q.get("id") or [""])[0]
            j = get_job(jid)
            if not j or not j.get("out"):
                return self._json({"error": "还没好"}, 404)
            base = re.sub(r"[^0-9A-Za-z_.\-]", "_", unquote((q.get("name") or [""])[0]))
            name = (base or "sr") + "_sr.mp4"
            return self._file(os.path.join(DATA, j["out"]), "video/mp4", name)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/api/upload":
            name = unquote((q.get("name") or ["input.mp4"])[0])
            scale = int((q.get("scale") or ["2"])[0])
            th = int((q.get("target") or ["0"])[0])
            sh = float((q.get("sharp") or ["0.4"])[0])
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return self._json({"error": "空文件"}, 400)
            jid = uuid.uuid4().hex[:12]
            dst = os.path.join(DATA, jid + ".src" + (os.path.splitext(name)[1] or ".mp4"))
            got = 0
            with open(dst, "wb") as f:
                while got < n:
                    chunk = self.rfile.read(min(1 << 20, n - got))
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
            if got < n:
                os.remove(dst)
                return self._json({"error": "上传中断"}, 400)
            set_job(jid, status="running", stage="排队", progress=1,
                    created=time.time(), name=name, scale=scale, target=th,
                    src_mb=round(n / 1048576, 1))
            threading.Thread(target=worker, args=(jid, dst, scale, th, None, sh),
                             daemon=True).start()
            return self._json({"id": jid})

        if u.path == "/api/cancel":
            jid = (q.get("id") or [""])[0]
            j = get_job(jid)
            if not j:
                return self._json({"error": "无此任务"}, 404)
            if j.get("status") in ("done", "error", "cancelled"):
                return self._json({"ok": True, "note": "已经结束了"})
            request_cancel(jid)
            set_job(jid, stage="正在取消…")
            return self._json({"ok": True})

        if u.path == "/api/url":
            ln = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(ln) or b"{}")
            url = (body.get("url") or "").strip()
            if not re.match(r"^https?://(www\.)?(bilibili\.com|b23\.tv)/", url):
                return self._json({"error": "只接受 bilibili.com / b23.tv 链接"}, 400)
            jid = uuid.uuid4().hex[:12]
            set_job(jid, status="running", stage="排队", progress=1, created=time.time(),
                    name=url, scale=int(body.get("scale") or 2),
                    target=int(body.get("target") or 0))
            threading.Thread(target=worker,
                             args=(jid, None, int(body.get("scale") or 2),
                                   int(body.get("target") or 0), url,
                                   float(body.get("sharp", 0.4))), daemon=True).start()
            return self._json({"id": jid})

        return self._json({"error": "not found"}, 404)


def sweep_stale():
    """重启后，进程内那些 running 任务的线程已经没了 —— 标成「已中断」，
    否则界面上会永远挂着一个假的进度条。"""
    n = 0
    with _lock:
        for jid, j in _jobs.items():
            if j.get("status") == "running":
                j.update(status="error", stage="已中断", progress=100,
                         error="服务重启了，这个任务没跑完（子进程已随服务一起收掉）。重传一次即可。")
                n += 1
        if n:
            _save()
    if n:
        print(f"清理了 {n} 个僵死任务", flush=True)


KEEP_JOBS = int(_env("SR_KEEP_JOBS", "8"))


def sweep_old(jid=None):
    """删掉没用的东西。两件：

    ① **上传的源文件用完就删。** 它在 DATA 里躺着一份完整副本（实测 20 个任务
       攒了 604 MB，全是 *.src.*，单个最大 406 MB），成品出来后再没有任何用处 ——
       失败信息里写的也一直是「重传一次即可」。跑着的任务不删。
    ② 任务记录只留最近 KEEP_JOBS 个，连同成品文件一起删，免得 jobs.json
       和磁盘无限涨（用户反馈「任务一直累积，现在有 20 个」）。
    """
    rm, rmdir = [], []
    with _lock:
        for k, j in _jobs.items():
            if j.get("status") == "running":
                continue                      # 还在跑（含排队）的源文件不能动
            j.pop("_src", None)
            rm += _files(k, ".src")           # ① 源文件：成品出来就没用了
        order = sorted(_jobs.items(), key=lambda kv: kv[1].get("created", 0))
        for k, j in (order[:-KEEP_JOBS] if len(order) > KEEP_JOBS else []):
            rm += _files(k, "")               # ② 砍掉的旧任务连成品一起删
            _jobs.pop(k, None)
        # ③ 孤儿 work 目录：崩溃/被 kill/重启留下的（实测有一个挂了 12 小时）。
        #    跑着的任务目录不能动，其余全清。
        try:
            for d in os.listdir(WORK):
                if _jobs.get(d, {}).get("status") != "running":
                    rmdir.append(os.path.join(WORK, d))
        except OSError:
            pass
        _save()
    for f in rm:
        try:
            os.remove(f)
        except OSError:
            pass
    for d in rmdir:
        shutil.rmtree(d, ignore_errors=True)


def _files(jid, kind):
    """DATA 下属于这个任务的文件。kind=".src" 只要源文件，"" = 全要。
    ★按【文件名】找而不是只认 _src 字段：早期任务记录里没那个字段，
      只认字段的话它们的源文件永远删不掉（实测残留 7 个）。"""
    out = []
    try:
        for f in os.listdir(DATA):
            if f.startswith(jid + ".") and (not kind or kind in f):
                out.append(os.path.join(DATA, f))
    except OSError:
        pass
    return out


def main():
    _load()
    sweep_stale()
    sweep_old()
    SYS.start()
    print(f"A733 视频超分服务  监听 0.0.0.0:{PORT}", flush=True)
    print(f"  NPU: {NBG}  块 {TILE_W}x{TILE_H} x{SR_SCALE} margin={MARGIN}", flush=True)
    print(f"  重活钉核: {('cpu ' + ','.join(map(str, CPUS))) if CPUS else '不限制（SR_CPUSET=0 可关）'}",
          flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
