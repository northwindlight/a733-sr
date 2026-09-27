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

# ---------------------------------------------------------------- 配置
BASE     = os.path.dirname(os.path.abspath(__file__))
DATA     = os.path.join(BASE, "data")          # 上传 + 产物
WORK     = os.path.join(BASE, "work")          # 每任务的临时目录
JOBS     = os.path.join(DATA, "jobs.json")
STATIC   = os.path.join(BASE, "static")

def _env(k, d):
    return os.environ.get(k) or d

VENV_DIR = _env("SR_VENC_DIR",  os.path.join(BASE, "vendor/venc"))
NPU_BIN  = _env("SR_SRPIPE",    os.path.join(BASE, "srpipe"))
VIP_LIB  = _env("SR_VIP_LIB",   os.path.join(BASE, "vendor/viplite"))
NBG      = _env("SR_NBG",       os.path.join(BASE, "models", "anv3.nb"))
VENC_BIN = _env("SR_VENC_BIN",  os.path.join(VENV_DIR, "vencoderdemo_v2"))
COOKIES  = _env("SR_COOKIES",   os.path.join(BASE, "bili_cookies.txt"))  # 可选，有则能下更高清

TILE_W, TILE_H = 320, 180      # 必须等于 NBG 的输入形状
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

    total_frames = meta["nb_frames"] or int(meta["duration"] * fps)
    est = npu_ms * total_frames / 1000.0 if total_frames else 0
    set_job(jid, stage="准备中", meta={
        "src": f"{W}x{H}", "sr": f"{out_w}x{out_h}",
        "final": f"{final_w}x{final_h}", "fps": round(fps, 2),
        "tiles": f"{tiles_x}x{tiles_y}={tiles_x*tiles_y}", "npu_ms": round(npu_ms, 1),
        "frames": total_frames, "est_npu_s": round(est, 1),
    })

    # ---------- 分段处理 ----------
    n_done = 0
    parts = []
    idx = 0
    while True:
        # 本段从 n_done 帧开始
        seg = os.path.join(wd, f"seg{idx}.h264")
        nv12 = os.path.join(wd, f"seg{idx}.nv12")

        # ★CAS（对比度自适应锐化）而不是 unsharp：unsharp 在平坦区过度锐化会出晕，
        #   CAS 按局部对比度决定强度，正是"模型把边缘糊了"这个场景要的。
        #   实测拉普拉斯均值 2.42 -> 3.37（strength 0.6），且不动平坦区。
        fin = f",cas=strength={sharp}" if sharp and sharp > 0 else ""
        cmd = (
            f'ffmpeg -v error -nostdin -ss {n_done / fps:.6f} -i {shq(src)} '
            f'-frames:v {CHUNK_FRAMES} -an -vf scale={W}:{H} -pix_fmt gbrp -f rawvideo - '
            f'| LD_LIBRARY_PATH={VIP_LIB} {shq(NPU_BIN)} {shq(NBG)} {W} {H} {SR_SCALE} '
            f'{TILE_W} {TILE_H} --margin {MARGIN} 2> {shq(wd + "/sr.err")} '
            f'| ffmpeg -v error -nostdin -f rawvideo -pix_fmt gbrp -s {out_w}x{out_h} -i - '
            f'-vf "scale={final_w}:{final_h}{fin}" -pix_fmt nv12 -f rawvideo -y {shq(nv12)}'
        )
        if cancel_requested(jid):
            raise Cancelled()
        p = run(["bash", "-c", cmd], jid=jid)
        if p.returncode != 0:
            raise RuntimeError("超分/缩放段失败:\n" + (p.stderr or "")[-800:])

        if not os.path.exists(nv12) or os.path.getsize(nv12) < 1024:
            break                                    # 没有更多帧了

        got = os.path.getsize(nv12) // (final_w * final_h * 3 // 2)
        if got == 0:
            break

        # 用 VE 硬编这一段
        set_job(jid, stage=f"硬件编码 第{idx+1}段")
        run(["bash", "-c",
             f'LD_LIBRARY_PATH={VENV_DIR} {shq(VENC_BIN)} -i {shq(nv12)} '
             f'-n {got} -f 0 -o {shq(seg)} -s {final_w}x{final_h} -d {final_w}x{final_h}'],
            jid=jid)

        os.remove(nv12)
        parts.append(seg)

        n_done += got
        pct = 10 + (n_done / total_frames * 80 if total_frames else 0)
        set_job(jid, progress=min(90, pct), frames_done=n_done,
                stage=f"超分+编码 {n_done}/{total_frames or '?'} 帧")
        idx += 1

        if got < CHUNK_FRAMES:
            break                                    # 源已经放完

    if not parts:
        raise RuntimeError("没有产出任何编码段 —— 源视频可能读不出来")

    # ---------- 拼接 + 混音 ----------
    set_job(jid, stage="拼接", progress=92)
    listf = os.path.join(wd, "parts.txt")
    with open(listf, "w") as f:
        for pth in parts:
            f.write(f"file '{pth}'\n")

    out = os.path.join(DATA, jid + ".mp4")
    if meta["has_audio"]:
        cmd = (f'ffmpeg -v error -nostdin -f concat -safe 0 -i {shq(listf)} '
               f'-i {shq(src)} -map 0:v:0 -map 1:a:0? -c:v copy -c:a aac -b:a 128k '
               f'-shortest -y {shq(out)}')
    else:
        cmd = (f'ffmpeg -v error -nostdin -f concat -safe 0 -i {shq(listf)} '
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
            cmd = ["yt-dlp", "--no-playlist", "--no-update",
                   "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
                   "--merge-output-format", "mp4", "-o", dst]
            if os.path.exists(COOKIES):
                cmd += ["--cookies", COOKIES]
            cmd.append(url)
            p = run(cmd, jid=jid)
            if p.returncode != 0 or not os.path.exists(dst):
                tail = "\n".join((p.stderr or p.stdout or "").strip().split("\n")[-8:])
                raise RuntimeError("下载失败:\n" + tail)
            src = dst
        set_job(jid, status="running")
        pipeline(jid, src, scale, target_h, t0, sharp)
    except Cancelled:
        set_job(jid, status="cancelled", stage="已取消", progress=100,
                error="用户取消", took=round(time.time() - t0, 1))
        shutil.rmtree(os.path.join(WORK, jid), ignore_errors=True)
    except Exception as e:
        set_job(jid, status="error", stage="失败", error=str(e)[:1500],
                progress=100, detail=traceback.format_exc()[-1500:])


# ---------------------------------------------------------------- HTTP
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

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._file(os.path.join(STATIC, "index.html"),
                              "text/html; charset=utf-8")
        if u.path == "/api/jobs":
            return self._json({"jobs": [dict(v, id=k) for k, v in all_jobs()][:40]})
        if u.path == "/api/status":
            jid = (q.get("id") or [""])[0]
            j = get_job(jid)
            return self._json(j or {"error": "无此任务"}, 200 if j else 404)
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
    print(f"A733 视频超分服务  监听 0.0.0.0:{PORT}", flush=True)
    print(f"  NPU: {NBG}  块 {TILE_W}x{TILE_H} x{SR_SCALE} margin={MARGIN}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
