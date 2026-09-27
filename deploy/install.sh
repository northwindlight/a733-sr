#!/bin/bash
# A733 视频超分 —— 一键安装（在板子上以 root 跑：sudo bash deploy/install.sh）
#
# 做四件事：
#   1. 编 srpipe（链 awnn / VIP Lite）
#   2. 拉预编译的 NBG（从 northwindlight/a733-npu 的 Release）
#   3. 装 udev 规则（让普通用户免 sudo 用 VE 与 dma_heap）
#   4. 装 systemd 单元 + Caddy 反代，并启动
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
SR_DIR=${SR_DIR:-/opt/sr}
RUN_USER=${SR_RUN_USER:-${SUDO_USER:-$(id -un)}}
AI_SDK=${AI_SDK:-$HOME/ai-sdk}
NBG_URL=${NBG_URL:-https://github.com/northwindlight/a733-npu/releases/download/nbg-animevideov3-v3/network_binary.nb}

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m!!\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "请用 sudo 跑"

# ---------------------------------------------------------------- 0. 依赖
say "检查依赖"
for c in ffmpeg ffprobe yt-dlp caddy g++ python3; do
    command -v "$c" >/dev/null || die "缺 $c —— sudo apt-get install -y ffmpeg yt-dlp caddy build-essential"
done
python3 -c 'import numpy' 2>/dev/null || die "缺 numpy —— sudo apt-get install -y python3-numpy"

# ★Debian 源里的 yt-dlp 太老，下 B 站必然 HTTP 412。只警告不阻断（不上 B 站就用不着）
YTDLP_V=$(yt-dlp --version 2>/dev/null || echo 0)
if [ "${YTDLP_V%%.*}" -lt 2026 ] 2>/dev/null; then
    printf '\033[1;33m注意\033[0m: yt-dlp 是 %s，太老 —— 下 B 站会 412。\n' "$YTDLP_V"
    echo "      换成官方 standalone 二进制："
    echo "        curl -L -o /usr/local/bin/yt-dlp \\"
    echo "          https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux_aarch64"
    echo "        chmod 755 /usr/local/bin/yt-dlp"
fi

# ---------------------------------------------------------------- 1. 编 srpipe
# ai-sdk 里的 awnn 源码 + VIP Lite 库
if [ ! -d "$AI_SDK/examples/libawnn_viplite" ]; then
    for cand in /home/*/ai-sdk "$HOME/ai-sdk" /root/ai-sdk; do
        [ -d "$cand/examples/libawnn_viplite" ] && { AI_SDK=$cand; break; }
    done
fi
[ -d "$AI_SDK/examples/libawnn_viplite" ] || die "找不到 ai-sdk（Allwinner 官方 NPU SDK）。
   设 AI_SDK=<路径> 再跑，或按 docs/RUNTIME.md 去下载。"

VIP_SRC=""
for v in "$AI_SDK"/viplite-tina/lib/aarch64-none-linux-gnu/v2.0 \
         "$AI_SDK"/viplite-tina/lib/glibc-gcc*/v*/; do
    [ -f "$v/libNBGlinker.so" ] && { VIP_SRC=$v; break; }
done
[ -n "$VIP_SRC" ] || die "ai-sdk 里找不到 v2.0 的 VIP Lite 库"

say "编译 srpipe（VIP Lite: $VIP_SRC）"
g++ -O2 -o "$REPO/srpipe" "$REPO/srpipe.c" \
    "$AI_SDK/examples/libawnn_viplite/awnn_lib.c" \
    "$AI_SDK/examples/libawnn_viplite/awnn_quantize.c" \
    -I"$VIP_SRC/inc" -I"$AI_SDK/examples/libawnn_viplite" -I"$AI_SDK" \
    -L"$VIP_SRC" -Wl,-rpath-link,"$VIP_SRC" -lNBGlinker -lVIPhal -lm
say "srpipe 编好了"

# ---------------------------------------------------------------- 2. 目录 + NBG
say "建 $SR_DIR"
mkdir -p "$SR_DIR"/{models,static,data,work,vendor}
install -m755 "$REPO/srpipe"        "$SR_DIR/srpipe"
install -m644 "$REPO/app.py"        "$SR_DIR/app.py"
install -m644 "$REPO/static/index.html" "$SR_DIR/static/index.html"
install -m644 "$REPO/README.md"     "$SR_DIR/README.md"

# VIP 运行库：拷进来，免得到处配 LD_LIBRARY_PATH
cp -a "$VIP_SRC"/libNBGlinker.so "$VIP_SRC"/libVIPhal.so "$SR_DIR/vendor/" 2>/dev/null || true

if [ -s "$SR_DIR/models/anv3.nb" ]; then
    say "已有 NBG，跳过下载"
else
    say "下载 NBG（$NBG_URL）"
    curl -fL --retry 3 -o "$SR_DIR/models/anv3.nb" "$NBG_URL" \
        || die "下载失败。也可以自己用 a733-npu 编一个，放到 $SR_DIR/models/anv3.nb"
fi

# VE 编解码那套（libvencoder 等）不在本仓库里，见 docs/RUNTIME.md
VENC_DIR=${SR_VENC_DIR:-/home/$RUN_USER/venc2}
if [ ! -x "$VENC_DIR/vencoderdemo_v2" ]; then
    printf '\033[1;33m注意\033[0m: %s/vencoderdemo_v2 不存在。\n' "$VENC_DIR"
    echo "      VE 硬编那套要在 Tina SDK 上自己编，见 docs/RUNTIME.md。"
    echo "      编好后放进 $VENC_DIR/，或设 SR_VENC_DIR 指向它。"
fi

chown -R "$RUN_USER:$RUN_USER" "$SR_DIR"

# ---------------------------------------------------------------- 3. udev
say "装 udev 规则（免 sudo 用 VE / dma_heap）"
cat > /etc/udev/rules.d/60-a733-media.rules <<'EOF'
# A733 媒体设备：让普通用户免 sudo 使用 VE(编解码) 与 dma_heap(ion 替代)
# NPU 的 /dev/vipcore 出厂已是 0666，这里补齐 VE 侧
KERNEL=="cedar_dev",     MODE="0666"
KERNEL=="cedar_dev_ve2", MODE="0666"
SUBSYSTEM=="dma_heap",   MODE="0666"
EOF
udevadm control --reload-rules && udevadm trigger || true

# ---------------------------------------------------------------- 4. 服务
say "装 systemd 单元与 Caddy 配置"
sed -e "s|/opt/sr|$SR_DIR|g" -e "s|^User=.*|User=$RUN_USER|" \
    -e "s|^Group=.*|Group=$RUN_USER|" \
    -e "s|SR_VENC_DIR=.*||" \
    "$REPO/deploy/sr-web.service" > /etc/systemd/system/sr-web.service

# 把编码器路径写进服务（本仓库不含那套库，只能指到用户自己编的地方）
if [ -x "$VENC_DIR/vencoderdemo_v2" ]; then
    sed -i "s|^Environment=SR_PORT=8080|Environment=SR_PORT=${SR_PORT:-8080}\nEnvironment=SR_VENC_DIR=$VENC_DIR\nEnvironment=SR_VIP_LIB=$SR_DIR/vendor|" \
        /etc/systemd/system/sr-web.service
fi

install -m644 "$REPO/deploy/Caddyfile" /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile >/dev/null 2>&1 || \
    printf '\033[1;33m注意\033[0m: Caddyfile 校验没过，检查 /etc/caddy/Caddyfile\n'

systemctl daemon-reload
systemctl enable --now sr-web caddy
sleep 2

# ---------------------------------------------------------------- 5. 自检
IP=$(hostname -I | awk '{print $1}')
echo
say "装完了"
systemctl is-active sr-web >/dev/null && echo "  sr-web    : active" || echo "  sr-web    : 起不来，看 journalctl -u sr-web"
systemctl is-active caddy  >/dev/null && echo "  caddy     : active" || echo "  caddy     : 起不来，看 journalctl -u caddy"
echo
echo "  浏览器打开  http://$IP/"
echo
echo "  日志:  journalctl -u sr-web -f"
echo "  哔哩哔哩下载要 cookie 的话，放到 $SR_DIR/bili_cookies.txt（chmod 600）"
