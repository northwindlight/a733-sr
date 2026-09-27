#!/usr/bin/env python3
"""
量化分块超分的接缝。

原理：同一帧用**两套不同的分块网格**各跑一遍。远离块边界的地方两次结果应该
几乎一样（只差量化噪声）；如果分块引入了接缝，差异会在边界那一列冒出来。

用法：
  # 1) 源图 A 全幅跑一遍
  srpipe anv3.nb 640 360 4 320 180 --margin M < frame.gbrp > A.raw
  # 2) 把源图裁掉左边 160 像素，再跑一遍（块边界于是落在别处）
  srpipe anv3.nb 480 360 4 320 180 --margin M < crop.gbrp > B.raw
  # 3) 比
  seamtest.py A.raw B.raw --a-shape 2560x1440 --b-shape 1920x1440 --crop-x 160
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--a-shape", required=True, help="A 的输出 WxH")
    ap.add_argument("--b-shape", required=True, help="B 的输出 WxH")
    ap.add_argument("--crop-x", type=int, required=True,
                    help="B 是从源图左边裁掉多少像素得到的（源像素）")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--tile", type=int, default=320, help="块宽（源像素）")
    ap.add_argument("--margin", type=int, default=0)
    args = ap.parse_args()

    aw, ah = (int(x) for x in args.a_shape.lower().split("x"))
    bw, bh = (int(x) for x in args.b_shape.lower().split("x"))
    A = np.fromfile(args.a, dtype=np.uint8).reshape(3, ah, aw).astype(np.int16)
    B = np.fromfile(args.b, dtype=np.uint8).reshape(3, bh, bw).astype(np.int16)

    # B 的第 0 列 == A 的第 crop_x*scale 列
    off = args.crop_x * args.scale
    n = min(aw - off, bw)
    a = A[:, :, off:off + n]
    b = B[:, :, :n]
    d = np.abs(a - b).mean(axis=(0, 1))

    # A 的块边界（输出列），换算成这里的相对列号
    step = (args.tile - 2 * args.margin) * args.scale
    boundaries = [k * step - off for k in range(1, aw // step + 2)
                  if 0 <= k * step - off < n]

    far = np.ones(n, dtype=bool)
    for bx in boundaries:
        far[max(0, bx - 32):bx + 32] = False

    print(f"重叠区 {n} 列，A 的块边界在相对列 {boundaries}")
    if far.any():
        print(f"  远离边界  平均绝对差 = {d[far].mean():.3f}")
    for bx in boundaries:
        print(f"  边界列 {bx:5d} 平均绝对差 = {d[bx]:.3f}")
    if boundaries and far.any():
        ratio = d[[b for b in boundaries]].mean() / max(1e-9, d[far].mean())
        verdict = "★有接缝" if ratio > 3 else "看不太出来"
        print(f"  => 边界/非边界 = {ratio:.2f}×  {verdict}")


if __name__ == "__main__":
    sys.exit(main())
