#!/bin/bash
# CPU 抠块 vs GPU 抠块 —— 两者必须产出逐字节相同的瓦片组
# 用 srpipe --dump-tiles 直接吐出瓦片缓冲区，比逐字节
set -eu
SRC=${1:?用法: gputiletest.sh <源视频>}
NBG=${SR_NBG:-/opt/sr/models/anv3.nb}
BIN=${SR_SRPIPE:-/opt/sr/srpipe}
export LD_LIBRARY_PATH=${SR_VIP_LIB:-/home/northwind/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
fail=0
# 覆盖：内块、左右边缘（窗口越界要靠钳制）、纯钳制（源比块还小）
for geo in "960 540" "1280 720" "640 360" "320 180"; do
    set -- $geo; W=$1; H=$2
    IN="ffmpeg -v error -nostdin -ss 30 -i $SRC -frames:v 1 -an -vf scale=$W:$H -pix_fmt gbrp -f rawvideo -"
    $IN | $BIN "$NBG" $W $H 4 320 180 --margin 16 --dump-tiles "$TMP/c.bin" 2>/dev/null
    $IN | $BIN "$NBG" $W $H 4 320 180 --margin 16 --gpu-in --dump-tiles "$TMP/g.bin" 2>/dev/null
    if cmp -s "$TMP/c.bin" "$TMP/g.bin"; then r="✓"; else r="✗"; fail=1; fi
    printf "  %-12s %s\n" "${W}x${H}" "$r"
done
[ $fail = 0 ] && echo "✓ GPU 抠块与 CPU 逐字节相同" || { echo "✗ 有不一致"; exit 1; }
