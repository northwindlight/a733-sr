#!/bin/bash
# srpipe 的字节 LUT 快路 vs awnn 浮点路 —— 必须逐字节相同，快才有意义
#
# 用法：tools/abtest.sh <源视频> [宽 高]
# 依赖：板上的 ffmpeg / 一个 NBG / 编好的 srpipe
set -eu
SRC=${1:?用法: abtest.sh <源视频> [宽 高]}
W=${2:-960}; H=${3:-540}
NBG=${SR_NBG:-/opt/sr/models/anv3.nb}
BIN=${SR_SRPIPE:-/opt/sr/srpipe}
export LD_LIBRARY_PATH=${SR_VIP_LIB:-/home/northwind/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

IN="ffmpeg -v error -nostdin -ss 30 -i $SRC -frames:v 3 -an -vf scale=$W:$H -pix_fmt gbrp -f rawvideo -"
$IN | $BIN "$NBG" $W $H 4 320 180 --margin 16 --nv12 --sharpen 0.4 --float-out > "$TMP/float.nv12"
$IN | $BIN "$NBG" $W $H 4 320 180 --margin 16 --nv12 --sharpen 0.4            > "$TMP/raw.nv12"

ls -l "$TMP/float.nv12" "$TMP/raw.nv12" | awk '{print "  "$5" "$9}'
if cmp "$TMP/float.nv12" "$TMP/raw.nv12"; then
    echo "✓ 两条路输出逐字节相同"
else
    echo "✗ 不同 —— 字节 LUT 的钳制/取整和浮点路不一致"; exit 1
fi
