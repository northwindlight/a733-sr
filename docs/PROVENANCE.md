# 来源清单 —— 每个件是从哪拿的

本仓库**只含自己写的代码**（`app.py` / `srpipe.c` / 前端 / 部署脚本）。
其余全部是外部件，这里逐个写清出处、版本、许可。

> ⚠️ 有两条要特别注意：
> 1. **`ai-sdk` 是第三方镜像，不是全志官方仓库**（见 §2）
> 2. **VE 编解码库是 Allwinner 的预编译二进制，许可不明，本仓库不转分发**（见 §3）

---

## 1. NPU 模型（NBG）

| | |
|---|---|
| 拿到的东西 | `anv3.nb`（animevideov3 x4，输入 320×180） |
| 从哪拿 | `github.com/northwindlight/a733-npu` 的 Release `nbg-animevideov3-v3` |
| 直链 | `https://github.com/northwindlight/a733-npu/releases/download/nbg-animevideov3-v3/network_binary.nb` |
| 怎么来的 | 该仓库用 Pegasus/ACUITY 工具链从 ONNX 转换（CI 产出，见其 `build_nbg.yml`） |
| 再上游 | 权重是 Real-ESRGAN 的 `realesr-animevideov3.pth`，来自 `github.com/xinntao/Real-ESRGAN` 的 release `v0.2.5.0` |
| 许可 | 权重 BSD-3-Clause（Real-ESRGAN） |

**为什么模型不放进本仓库**：Pegasus/ACUITY 工具链装在 2.7 GB 的私有容器里，
塞进这里不合适；而且 NBG 是二进制大件，放 Release 更合适。

`install.sh` 会自动从这个直链拉取。你也可以自己用 `a733-npu` 编一个别的形状放进去。

### 1.1 ★转换工具链的 Docker 镜像（链条上唯一的单点故障）

| | |
|---|---|
| 是什么 | Pegasus / ACUITY 转换容器（`ghcr.io/northwindlight/ubuntu-npu`），2.7 GB |
| 从哪来 | **使用者自己找 Radxa 要的**（不走公开分发渠道） |
| 可见性 | ⚠️ **私有包，不对外开放** |
| 引用方式 | `a733-npu` 的 `container.txt` 按 **digest** 钉死 |

**★这是整条链上唯一不可恢复的依赖**：那个 ghcr 包一旦被删，
上游 `a733-npu` 就永久失去构建（重新转换 NBG 的）能力，且既有产物都无法解释。
而且**没法用 Release 兜底** —— GitHub 单文件 Release 附件上限 2 GiB，镜像是 2.7 GB。

**恢复路径**：再去找 Radxa 要一份（联系人/渠道由使用者自己保留）。
拿到后按 `a733-npu/container.txt` 里的两步换 digest。

**对本仓库的影响：没有。** `a733-sr` 只从 Release 拉**编好的 NBG**，
不需要这个镜像 —— 只有"想自己重编一个别的形状的模型"时才会撞上它。

---

## 2. NPU 用户态：awnn API + VIP Lite 运行库

| | |
|---|---|
| 拿到的东西 | `awnn_lib.c` / `awnn_quantize.c` / `awnn_lib.h`、`log/log.h`、`vip_lite.h`、`libNBGlinker.so` / `libVIPhal.so` |
| 从哪拿 | **`github.com/ZIFENG278/ai-sdk`** |
| 版本 | commit `fc90006d0f6569da2f6726c2d8395877686f5aca`（2025-10-20，"fix: missing -I flag in Makefile"），无 tag |
| 路径 | `examples/libawnn_viplite/`、`log/`、`viplite-tina/lib/aarch64-none-linux-gnu/v2.0/` |
| 许可 | 仓库未标许可，内容是全志的 SDK 源码 |

> ⚠️ **这是第三方镜像，不是全志官方仓库。** 全志的 AI SDK 官方只随 SDK 包/镜像分发，
> 不走公开 git。这个镜像内容与板子镜像里的一致（`vpm_run`、VIP Lite v2.0.3.2-AW-2024-08-30
> 能对上），但**长期可用性没有保证**，请自行留档。

本仓库**不转分发**这些文件，`install.sh` 让你指向本机已有的 `ai-sdk`（默认 `~/ai-sdk`），
或设 `AI_SDK=<路径>`。板子镜像通常自带（Radxa 的 Debian 镜像里有）。

---

## 3. VE 编解码库（Allwinner cedarc）

| | |
|---|---|
| 拿到的东西 | `libvencoder.so` / `libvenc_base.so` / `libcdc_base.so` / `libMemAdapter.so`（源码编译）<br>`libvenc_codec.so` / `libVE.so`（预编译二进制） |
| 源码从哪拿 | **`gitlab.com/tina5.0_aiot/media/cedarc-release/libcedarc_v2.0`**，分支 `product-aiot-stable` |
| 预编译件在哪 | 同仓库 `library/toolchain-sunxi-aarch64-glibc-gcc-v1320/v2/libvenc_codec.so`<br>和 `library/toolchain-sunxi-aarch64-glibc-gcc-v1320/libVE.so` |
| 许可 | 仓库未标许可（Allwinner 专有） |

> ⚠️ **这是专有二进制，许可不明，本仓库不转分发。** `install.sh` 只做检测与提示。
> 自己编的完整配方见本文件 §3.1。

**★关键**：必须用 `libcedarc_v2.0` 这份（V-line 新 API），
用老的 `libcedarc` 仓库那份**永远不认这颗芯片**（`ic_version 0x21320` 被拒）。
判据见 `a733-npu` 与 §3.2。

### 3.1 完整构建配方

```bash
git clone -b product-aiot-stable \
  https://gitlab.com/tina5.0_aiot/media/cedarc-release/libcedarc_v2.0.git
cd libcedarc_v2.0 && ./bootstrap
LIBDIR=$PWD/library/toolchain-sunxi-aarch64-glibc-gcc-v1320
./configure --prefix=$PWD/install --host=aarch64-linux-gnu \
  --disable-aftertreatment --disable-vdecoderdemo \
  CFLAGS="-DCONF_KERNEL_VERSION_6_6 -DCONF_USE_IOMMU -D__LINUX__ -O2 -g \
          -Wno-error -Wno-implicit-function-declaration" \
  CXXFLAGS="<同上>" \
  LDFLAGS="-L$LIBDIR -L$LIBDIR/v2"
make -j4
make -j4 LDFLAGS="-L$LIBDIR -L$LIBDIR/v2 -Wl,-rpath-link,$LIBDIR/v2 -Wl,-rpath-link,$LIBDIR"
```

四个必踩的开关：

1. `-DCONF_KERNEL_VERSION_6_6` ⇒ 才会走 `/dev/dma_heap/system` 而不是 `/dev/ion`
   （板上没有 ion，漏了就报 `open ion failed`）
2. `-DCONF_USE_IOMMU`（A733 的 `machinfo/a733/config.mk` 就是这么写的）
3. `LDFLAGS` 必须含 **`$LIBDIR/v2`**（`libvenc_codec.so` 在那），
   且 demo 链接要 `-Wl,-rpath-link,$LIBDIR/v2`
4. `--disable-aftertreatment`（解码侧，`-lscaledown` 在 v1320 里不存在）

链接 demo 时还有两个坑：`-lm` 必须放在**库之后**并用 `-Wl,--no-as-needed`
（**`libvenc_codec.so` 没声明 `NEEDED libm.so.6`**，是 Allwinner 的缺陷）；
改了 CFLAGS 后要 `make clean`，否则 `.o` 不比源码旧会**静默不重编**。

### 3.2 怎么判断拿对了库

- `libvenc_codec.so` 的 `NEEDED` 只有 `libcdc_base` / `libVE` / `libMemAdapter` / `libvenc_base`
- 它导出 `H264EncOpenVer2` / `video_encoder_h264_ver2` / `_h265` / `_jpeg`（**没有 ver1**）
- 老的 `libvenc_h264.so` 那套会打印
  `error: the driver do not support the ic 21320` —— 看到它就是拿错库了

---

## 4. 板子固件与系统包

| 件 | 出处 |
|---|---|
| 系统镜像 | Radxa 的 Debian 13 (trixie) A733 CLI 镜像 |
| `libcedarc-dev-{2.0.0,3.0.0}-arm64` v1.0.7 | Radxa 源 `https://radxa-repo.github.io/a733-trixie-test`（2022 年的构建，**编解码器不认 A733**，只用来做对照） |
| `gstreamer1.0-omx-allwinner` | 同上（OMX 那条路是坏的：`libOmxVdec.so` 缺 `log_unset_level`，见 RUNTIME.md） |
| apt 镜像 | 清华 `mirrors.tuna.tsinghua.edu.cn` |

## 5. 通用工具（apt 装）

| 件 | 出处 | 备注 |
|---|---|---|
| ffmpeg / ffprobe 7.1.5 | Debian 13 | 解码 + 缩放 + 混流 |
| caddy 2.6.2 | Debian 13 | 反向代理 |
| yt-dlp | **Debian 源里那份是 2025.04.30，太老，B 站必然 412** | 换成官方 standalone 二进制（`github.com/yt-dlp/yt-dlp` releases 的 `yt-dlp_linux_aarch64`，实测 2026.06.09 可用） |
| build-essential / g++ 14.2 | Debian 13 | 编 srpipe |
| python3 3.13 + python3-numpy | Debian 13 | 跑 app.py |

---

## 6. 校准数据（只在上游 a733-npu 用到）

DIV2K：`https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip`
—— 量化校准用，**仅限非商业研究**许可，裁切图不能提交进公开仓库。
本仓库不涉及（NBG 是编好的）。

---

## 汇总：本仓库**不含**什么

| 不含 | 原因 | 怎么补 |
|---|---|---|
| NBG 模型 | 大二进制 + 工具链太重 | `install.sh` 从 a733-npu Release 拉 |
| awnn / VIP Lite | 第三方镜像，非官方 | 用镜像里自带的 `~/ai-sdk` |
| VE 编解码库 | **专有二进制，许可不明** | 按 §3.1 自己编 |
| 任何凭据 | 安全 | cookie 自己放，已进 `.gitignore` |
