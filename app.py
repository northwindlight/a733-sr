#!/usr/bin/env python3
"""
A733 视频超分服务 —— 上传/哔哩哔哩链接 -> NPU 超分 -> 下载

流水线（全部在本机 A733 上，不依赖外部机器）：
    [源视频]
      └ ffmpeg 解码 + 缩放到源分辨率，输出 gbrp 平面帧
          └ srpipe  ── NPU 分块 x4 超分（块 320x180，带 margin 防接缝）
              └ ffmpeg 缩放到目标分辨率，输出 NV12
                  └ vencoderdemo_v2 ── VE 硬编 H.264
                      └ ffmpeg 混流原音轨 -> mp4

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
VE_MAX_INTERMEDIATE = int(_env("SR_VE_MAX_GB", "20")) << 30

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
        "fps": fps, "has_audio": a is not None,
        "duration": float(info["format"].get("duration") or 0),
        "nb_frames": int(v.get("nb_frames") or 0),
        "codec": v.get("codec_name") or "",
    }


# ---------------------------------------------------------------- 流水线
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

    out_w = W * SR_SCALE
    out_h = H * SR_SCALE

    if target_h:
        final_h = target_h
        final_w = int(round(target_h * W / H / 2)) * 2      # 保持比例，且宽为偶数
    else:
        final_h = H * scale
        final_w = int(round(W * scale / 2)) * 2

    # 最终尺寸不能超过 SR 出来的尺寸（VE 编码器只会缩、不会放）
    if final_h > out_h:
        final_h, final_w = out_h, out_w

    tiles_x = -(-W // (TILE_W - 2 * MARGIN))
    tiles_y = -(-H // (TILE_H - 2 * MARGIN))
    npu_ms = W * H * 0.547 / 1000.0            # 实测 0.547 µs/源像素

    # 中间文件是 SR 分辨率（源×4）的 NV12 —— 720p 源就是 22 MB/帧。
    # 分段大小按它定，把临时盘控在 ~2 GB 以内（原来是按目标尺寸算的，会超）。
    bytes_per_frame = out_w * out_h * 3 // 2
    chunk = max(10, min(CHUNK_FRAMES, int(2e9 / max(1, bytes_per_frame))))

    total_frames = meta["nb_frames"] or int(meta["duration"] * fps)
    est = npu_ms * total_frames / 1000.0 if total_frames else 0
    set_job(jid, stage="准备中", meta={
        "src": f"{W}x{H}", "sr": f"{out_w}x{out_h}",
        "final": f"{final_w}x{final_h}", "fps": round(fps, 2),
        "tiles": f"{tiles_x}x{tiles_y}={tiles_x*tiles_y}", "npu_ms": round(npu_ms, 1),
        "frames": total_frames, "est_npu_s": round(est, 1),
    })

    # ---------- 可选：VE 硬解预处理（整片 → NV12 裸流）----------
    nv12_src = None                       # (文件, 对齐后的高)；None = 走软解
    if VE_DECODE and meta.get("codec") == "h264" and os.path.exists(VDEC_BIN):
        ah = (H + 15) // 16 * 16          # ★解码器输出的高按 16 对齐（960x540 -> 960x544）
        need = W * ah * 3 // 2 * max(1, total_frames)
        free = shutil.disk_usage(WORK).free
        if need >= min(VE_MAX_INTERMEDIATE, free * 4 // 10):
            set_job(jid, stage=f"源太大，NV12 中间文件要 {need/2**30:.1f}G —— 退回软解")
        else:
            raws = os.path.join(wd, "src.h264")
            dump = os.path.join(wd, "src.nv12")
            set_job(jid, stage="VE 硬解中（专用硅解码，这段 CPU 不参与）", progress=6)
            # ① 抽裸流：解码器只吃 elementary stream，不吃 mp4 容器
            run(["bash", "-c", f'ffmpeg -v error -nostdin -y -i {shq(src)} '
                               f'-c:v copy -bsf:v h264_mp4toannexb -f h264 {shq(raws)}'], jid=jid)
            # ② 硬解整片 → NV12。★-ss 0 -sn N 必须给：demo 默认「一帧都不存且不报错」
            n = max(1, total_frames) + 8
            # ★这一段对长片要跑几分钟到十几分钟（受 eMMC 写入限速），而且它是
            #   一个不可中断的子进程 —— 不给进度的话界面就冻在 6%，看着像卡死
            #   （用户实测反馈"卡了半天"）。demo 自己不报进度，就盯输出文件大小。
            ve_exp = W * ah * 3 // 2 * max(1, total_frames)
            ve_stop = threading.Event()

            def _ve_tick():
                while not ve_stop.wait(3.0):
                    try:
                        g = os.path.getsize(dump) // (W * ah * 3 // 2)
                    except OSError:
                        continue
                    set_job(jid, stage=f"VE 硬解中 {g}/{total_frames} 帧"
                                       f"（专用硅解码，这段 CPU 不参与）",
                            progress=min(9, 5 + 4 * os.path.getsize(dump) / max(1, ve_exp)))

            ve_th = threading.Thread(target=_ve_tick, daemon=True)
            ve_th.start()
            run(["bash", "-c", f'LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu {shq(VDEC_BIN)} '
                               f'-i {shq(raws)} -codFmat 1 -o {shq(dump)} -outFmat 6 '
                               f'-n {n} -ss 0 -sn {n}'], jid=jid)
            ve_stop.set()
            # ★必须按【整片】验，不能只验「≥1 帧」。硬解半路停掉（磁盘满/驱动抽风）
            #   会留下一个能用的短文件，后面每段都"正常"跑完，最后静默交半个视频。
            fsz = W * ah * 3 // 2
            got_f = os.path.getsize(dump) // fsz
            if os.path.getsize(dump) >= fsz * max(1, total_frames) * 98 // 100:
                nv12_src = (dump, ah)
            elif got_f >= 1:
                set_job(jid, stage=f"硬解只出了 {got_f}/{total_frames} 帧，退回软解")
            else:
                set_job(jid, stage="硬解没出东西，退回软解")
            for f in (raws, dump) if not nv12_src else (raws,):
                try: os.remove(f)
                except OSError: pass

    # ---------- 分段处理 ----------
    t_sr_start = time.time()
    n_done = 0
    parts = []
    idx = 0
    while True:
        # 本段从 n_done 帧开始
        seg = os.path.join(wd, f"seg{idx}.h264")
        nv12 = os.path.join(wd, f"seg{idx}.nv12")

        # srpipe 直接吐 NV12（源×4 分辨率），缩放交给 VE 编码器 —— VE 自带缩放，白送。
        # 这样省掉原来那一步 ffmpeg swscale：实测它是整条流水线的瓶颈
        # （5120x2880 时 ffmpeg 占 368% CPU，srpipe 才 45%）。
        # 数据量也砍半：gbrp 3 字节/像素 -> NV12 1.5。
        # 锐化改由 srpipe 自己做（新路径里 ffmpeg 不参与后期了，不能让开关静默失效）。
        sh_arg = f" --sharpen {sharp}" if sharp and sharp > 0 else ""
        # ★GPU 做色彩转换，和 NPU 并行。选它不是因为它快（实测比 CPU 慢），
        #   是因为 CPU 是那个 60°C 就降频的热区，把活挪走能少降频。
        gpu_arg = " --gpu" if GPU_CONV else ""
        if nv12_src:
            # VE 硬解路径：输入已是 NV12 裸流 ⇒ 只 seek + crop，不解码、不 swscale。
            # 管道数据量也从 gbrp 的 3 字节/像素降到 NV12 的 1.5。
            dump, ah = nv12_src
            inarg = (f'-f rawvideo -pix_fmt nv12 -s {W}x{ah} '
                     f'-ss {n_done / fps:.6f} -i {shq(dump)} -frames:v {chunk} '
                     f'-vf crop={W}:{H}:0:0 -f rawvideo -')   # ★必须显式 :0:0 —— crop 默认是【居中】裁
            nv12flag = " --in-nv12"
        else:
            # 软解路径：解码 + 缩放到源分辨率 + 转 gbrp 平面
            inarg = (f'-ss {n_done / fps:.6f} -i {shq(src)} '
                     f'-frames:v {chunk} -an -vf scale={W}:{H} -pix_fmt gbrp -f rawvideo -')
            nv12flag = ""
        cmd = (
            f'set -o pipefail; ffmpeg -v error -nostdin {inarg} '
            f'| LD_LIBRARY_PATH={VIP_LIB} {shq(NPU_BIN)} {shq(NBG)} {W} {H} {SR_SCALE} '
            f'{TILE_W} {TILE_H} --margin {MARGIN} --nv12{nv12flag}{sh_arg}{gpu_arg} '
            f'2> {shq(wd + "/sr.err")} > {shq(nv12)}'
        )
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
            p = run(["bash", "-c", cmd], jid=jid)
        finally:
            stop_evt.set()
        if p.returncode != 0:
            raise RuntimeError("超分段失败:\n" + (p.stderr or "")[-800:])

        if not os.path.exists(nv12) or os.path.getsize(nv12) < 1024:
            break                                    # 没有更多帧了

        got = os.path.getsize(nv12) // (out_w * out_h * 3 // 2)   # 中间是 SR 分辨率
        if got == 0:
            break

        # VE 硬编这一段：-s 给 SR 尺寸、-d 给目标尺寸 —— 缩放由 VE 做
        set_job(jid, stage=f"硬件编码 第{idx+1}段")
        # ★-r 必须给源帧率：不给的话编码器用自己的默认帧率打时间戳，
        #   152 帧会被塞进 2 秒（实测源 10.2 秒的片子出来播 2.0 秒，快放 5 倍），
        #   而且 dts 非单调、后面 concat 也会跟着错。
        run(["bash", "-c",
             f'LD_LIBRARY_PATH={VENV_DIR} {shq(VENC_BIN)} -i {shq(nv12)} '
             f'-n {got} -f 0 -o {shq(seg)} -s {out_w}x{out_h} -d {final_w}x{final_h} '
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

    # VE 硬解的中间产物用完就删 —— 它是整片源分辨率的 NV12，几百 MB 到几 GB
    if nv12_src:
        try: os.remove(nv12_src[0])
        except OSError: pass

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
    fps_i = max(1, int(round(fps)))
    if meta["has_audio"]:
        cmd = (f'ffmpeg -v error -nostdin -f concat -safe 0 -r {fps_i} -i {shq(listf)} '
               f'-i {shq(src)} -map 0:v:0 -map 1:a:0? -c:v copy -c:a aac -b:a 128k '
               f'-shortest -y {shq(out)}')
    else:
        cmd = (f'ffmpeg -v error -nostdin -f concat -safe 0 -r {fps_i} -i {shq(listf)} '
               f'-c:v copy -y {shq(out)}')
    run(["bash", "-c", cmd], jid=jid)

    shutil.rmtree(wd, ignore_errors=True)
    size = os.path.getsize(out)
    set_job(jid, stage="完成", progress=100, status="done",
            out=os.path.basename(out), size=size,
            took=round(time.time() - t_start, 1),
            fps_done=round(n_done / max(0.01, time.time() - t_start), 2))


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
        set_job(jid, status="running")
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


def main():
    _load()
    sweep_stale()
    SYS.start()
    print(f"A733 视频超分服务  监听 0.0.0.0:{PORT}", flush=True)
    print(f"  NPU: {NBG}  块 {TILE_W}x{TILE_H} x{SR_SCALE} margin={MARGIN}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
