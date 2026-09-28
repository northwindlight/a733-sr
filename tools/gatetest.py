#!/usr/bin/env python3
"""★每个自动闸门都要故意破坏一次，确认它真的会响。

`app.check_output` 单独抽成函数就是为了这个：用 ffmpeg 现造几个**坏产物**直接喂进去，
不用每次跑一遍五分钟的完整流水线。

跑法：  python3 tools/gatetest.py

不需要 A733 板子，任何装了 ffmpeg 的机器都能跑（纯查容器元数据）。
"""
import os
import subprocess
import sys
import tempfile
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app                                     # noqa: E402

FPS = 15
N = 150
DUR = N / FPS                                  # 10.0 s

# 参考请求：1920x1080、150 帧、10 秒、要有音轨
WANT = dict(want_w=1920, want_h=1080, want_audio=True,
            n_done=N, want_dur=DUR, fr=Fraction(FPS, 1))


def build(path, w, h, frames, fps=FPS, audio=True, seconds=None):
    """造一个 mp4。seconds 不给就按 frames/fps；给了就照它（用来造"快放"那种时长错的）。"""
    if seconds is None:
        src = f"testsrc2=size={w}x{h}:rate={fps}:duration={frames / fps:.6f}"
    else:
        src = f"testsrc2=size={w}x{h}:rate={frames / seconds:.6f}:duration={seconds:.6f}"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", src]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=30",
                "-c:a", "aac", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, check=True)


def main():
    d = tempfile.mkdtemp(prefix="gatetest.")
    cases = []

    # ① 好产物 —— 闸门必须【放行】。只会响的闸门等于没有闸门。
    p = os.path.join(d, "good.mp4")
    build(p, 1920, 1080, N)
    cases.append(("好产物（应该放行）", p, WANT, False))

    # ② 分辨率错 —— VE 编码器把 -d 当裁剪框那次的形状
    p = os.path.join(d, "crop.mp4")
    build(p, 960, 540, N)
    cases.append(("编码器裁成左上 1/4（960x540）", p, WANT, True))

    # ③ 没音轨 —— -shortest 让 aac 编出 0 KiB 那次的形状
    p = os.path.join(d, "mute.mp4")
    build(p, 1920, 1080, N, audio=False)
    cases.append(("源有音轨但产物静音", p, WANT, True))

    # ④ 时长错 —— 帧数对但快放（-r 没吃到真实帧率）
    p = os.path.join(d, "fast.mp4")
    build(p, 1920, 1080, N, seconds=6.0)
    cases.append(("快放（150 帧塞进 6 秒）", p, WANT, True))

    # ⑤ 丢帧 —— 拼接掉了一大半
    p = os.path.join(d, "short.mp4")
    build(p, 1920, 1080, 75)
    cases.append(("只拼上一半帧（75/150）", p, WANT, True))

    bad = 0
    for name, path, want, should_fire in cases:
        try:
            app.check_output(path, **want)
            fired, msg = False, ""
        except RuntimeError as e:
            fired, msg = True, str(e).split("\n")[0][:90]
        ok = (fired == should_fire)
        bad += not ok
        print(f"{'✓' if ok else '✗'} {name:34s} 闸门{'响了' if fired else '没响'}"
              f"{'' if ok else '  ← 不对！'}")
        if fired:
            print(f"    {msg}")
    print()
    print("全部符合预期" if not bad else f"有 {bad} 项不符合预期")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
