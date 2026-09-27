# a733-sr — A733 NPU 视频超分（端到端方案）

把一块 **129 元的 Radxa CUBIE A7Z（Allwinner A733）** 变成一台独立的视频超分盒子：
浏览器上传视频或贴哔哩哔哩链接，NPU 跑超分，VE 硬编输出，下载成品。全程不依赖任何外部机器。

```
浏览器 ──► Caddy ──► app.py ──┬─ 上传 / b23 链接（yt-dlp）
                              └─ ffmpeg 解码
                                   └ srpipe   ← NPU 分块 x4 超分（本仓库的核心）
                                       └ ffmpeg 缩放 + CAS 锐化
                                           └ vencoderdemo  ← VE 硬编 H.264
                                               └ ffmpeg 混音轨 → mp4
```

---

## 这块板子的实际水平（全部实测，非规格推算）

| | 实测 |
|---|---|
| NPU 超分 | **每块 ~41.5 ms × 块数**（animevideov3 x4，块 352×224）<br>见下面「分块的代价模型」——**不是** ∝ 源总像素 |
| VE 编码 | 1080p **96 fps** / 4K **24.5 fps**（像素率恒定 ~200 Mpx/s） |
| VE 解码 | 1080p **122 fps** / 4K **34 fps** |
| 编解码并发 | **完全并行**（两个独立硬件实例），1080p 合计 84 fps |

对比 Intel UHD 630（Gen9.5，规格推算）：**解码约一半**（282 vs ~498 Mpx/s），
**编码吞吐基本打平**（203–257 vs ~249 Mpx/s）。但 Intel 编码器的画质工具链
（B 帧、lookahead、自适应量化）成熟得多——**吞吐打平不等于质量打平**。

**VE 侧完全不是瓶颈**，唯一瓶颈是 NPU。

---

## 为什么是「分块」

NBG（NPU 模型）是**编译期定形状**的。想处理任意分辨率的源，只有两条笨路：

1. **把源降采样到 NBG 的输入尺寸** —— 先丢信息再让模型猜回来，得不偿失
2. **重新编译 NBG** —— 要重测内存池。实测：320×180（5.76 万像素）正确；
   **640×360（23 万像素）x4 会溢出 84 MB 内存池**，表现为「局部平滑、全局错乱」的
   横条纹——**不报错、不崩溃**，极易误判为能跑

**分块（tiling）两条都不需要**：源原样切成小块，逐块过 NPU，拼回 N 倍大的整帧。

### ★分块的代价模型（这条曾经记反过，2026-09-28 更正）

NBG 定形状 ⇒ **每块 NPU 代价是固定的**（实测每块恒定 ~41.5 ms），
**与每块实际用到多少面积无关**。所以：

```
整帧代价 = 块数 × 每块常数
块数     = ceil(宽 / core_w) × ceil(高 / core_h)
core     = 块尺寸 − 2×margin      （margin 是防接缝的重叠圈）
```

**能优化的只有块数**，而块数由「core 能不能整除常见视频宽高」决定。
margin 只要 ≥1 就可能把整数倍顶上去：960/320 = 3 是整的，但 960/288 = 3.33
⇒ 4 列，**多出 7 块 = +61% 的 NPU 时间**。实测 960×540：

| margin | 块数 | NPU/帧 |
|---|---|---|
| 0 | 3×3 = **9** | 373 ms |
| 16 | 4×4 = 16 | 687 ms |

margin 0 便宜得多但有静态竖缝，不能要。**正确做法是把 NBG 的输入形状
选成 core 能整除常见宽高**——这就是块尺寸取 352×224（core 320×192）而不是
320×180（core 288×148）的原因：

| 源 | 320×180 | 352×224 |
|---|---|---|
| 960×540 | 4×4 = 16 | **3×3 = 9**（−23%） |
| 1280×720 | 5×5 = 25 | **4×4 = 16**（−12%） |
| 1920×1080 | 7×8 = 56 | **6×6 = 36**（−12%） |
| 640×360 | 3×3 = 9 | **2×2 = 4**（−39%） |

### 接缝是真的，而且量得出来

3×3 卷积 ×18 层，感受野半径约 18 像素。块边界处网络看不到真实邻居
（只能看到钳制复制出来的像素），那圈输出是「少信息」的。

做法：同一帧用**两套不同的分块网格**各跑一遍，在重叠区逐列比。

| | 无 margin | `--margin 16` |
|---|---|---|
| 块边界那一列的平均绝对差 | **2.073** | **0.000** |
| 远离边界 | 0.141 | 0.040 |

**关键在于它位置固定、每帧一样**——静态竖线正是人眼最敏感的东西，
比同量级的随机噪声恶劣得多。所以默认开 `--margin 16`。

**而且代价不额外增加**：640×360 源无 margin 是 2×2=4 块，带 margin 是 3×3=9 块，
但总处理像素都是 518K（640/320=2 整除，加任何 margin 都要多围一圈）。
换个更大的块（480×270 无 margin 也是 4 块）总像素一模一样。
**一样贵，前者无接缝。**

复现：

```bash
python3 tools/seamtest.py A.raw B.raw --a-shape 2560x1440 --b-shape 1920x1440 \
        --crop-x 160 --margin 16
```

---

## 性能：源分辨率越高越慢

NPU 代价 **∝ 源总像素**（与超分倍率几乎无关——x4 的那些卷积全在输入分辨率上跑，
x2 只是最后一层输出通道少 4 倍）。所以决定速度的是**喂多大**，不是放大几倍。

| 源分辨率 | NPU 每帧 | 端到端 |
|---|---|---|
| 320×180 | 31 ms | ~4.5 帧/秒 |
| 360p (640×360) | 126 ms | ~3.6 帧/秒（含 margin 与流水线开销） |
| 480p | 216 ms | ~2 帧/秒 |
| 720p | 504 ms | ~0.8 帧/秒 |
| 1080p | 1.13 s | ~0.4 帧/秒 |

**这是批处理速度，不是实时。** 硬件决定，不可调优。

---

## 快速开始（在 A733 板子上）

前提：板子跑 Debian 13（trixie），有 `~/ai-sdk`（Allwinner 官方 NPU SDK）。

```bash
sudo apt-get install -y ffmpeg yt-dlp caddy build-essential
git clone https://github.com/northwindlight/a733-sr && cd a733-sr
sudo bash deploy/install.sh          # 装 udev 规则、编译 srpipe、拉 NBG、装服务
```

然后浏览器打开 `http://<板子IP>/`。

`install.sh` 会从 [a733-npu](https://github.com/northwindlight/a733-npu) 的
Release 拉取预编译的 NBG，也会在 `~/ai-sdk/examples/` 找 awnn 源码来编 `srpipe`。

### 可配置项（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `SR_PORT` | `8080` | Web 服务端口 |
| `SR_VIP_LIB` | `vendor/viplite` | VIP Lite 运行库目录 |
| `SR_VENC_DIR` | `vendor/venc` | `libvencoder` 那一套（含 `vencoderdemo_v2`） |
| `SR_VENC_BIN` | `$SR_VENC_DIR/vencoderdemo_v2` | VE 编码器 demo 可执行文件 |
| `SR_NBG` | `models/anv3.nb` | NPU 模型 |
| `SR_COOKIES` | `bili_cookies.txt` | 哔哩哔哩 cookie（可选，**别提交**） |
| `SR_SYS_CLIENTS` | `4` | 状态栏最多允许几条 SSE 长连接 |

---

## 状态栏（温度 / CPU / 内存 / NPU / GPU）

页面顶部那条是实时状态，走 **SSE**（`/api/sys`）而不是轮询。
`/api/sys.json` 是一次性快照，调试用（`curl` 一下就知道数）。

采集在 **`sysmon.py` 里单独一个线程**按 1 Hz 刷新快照，HTTP 那边只读快照：

- **必须单点采样**：CPU 利用率是两次 `/proc/stat` 求差。每个连接各采各的会
  互相吃掉差值 —— 第二个标签页一开，第一个的读数就跳 0。
- 前端**不重建 DOM**：每秒重建一次会让 hover 和文字选择一直被打断。
  只在「顺序真的变了」时才重建行（温度会互相超车），其余原地改数值。

### 温度条的量程，不是 0~100°C

是**该传感器自己的首个降频阈值**。A733 的 8 个 zone 阈值差很多：

| 传感器 | 首个阈值 |
|---|---|
| CPU 小核 / 大核 / GPU | 60 °C |
| 板面 `skin_zone` | **50 °C** |
| 小核 / 大核·空闲参考 | 70 / 90 °C |
| DDR / NPU | 110 °C |

所以条的长短**不能跨行直接比**：同样 50 °C，对 CPU 才 83%，对板面已经到线。
行尾的 `/60` `/50` 就是各自的量程。数字变黄（≥85%）变红（≥95%）才是真警告。

带「空闲」的是热管理用来做基准的辅助区，不是主传感器，但读数是真的。

### ★GPU / NPU 硬件计数器读不到（需要 root）

两个计数器都在 debugfs 里，**而 `/sys/kernel/debug` 整个目录是 `drwx------ root`**：

| | 路径 |
|---|---|
| NPU 负载 | `/sys/kernel/debug/viplite/core_loading` → `NPU Loading ----> Core0: 0%` |
| GPU 负载 | `/sys/kernel/debug/pvr/status` → `GPU Utilisation: 42%` + 2D/GEOM/3D/CDM 分项 |

服务以普通用户跑，进不去这个目录，所以状态栏这两格显示 `—` 和原因。
**注意不是文件本身 0600 —— 是父目录 0700，光给文件 chmod 没用。**
NPU 的**频率**（devfreq）不需要权限，照常显示。

代码把「驱动没加载」和「读不到」分开报：判据不能是 `os.path.exists(path)`
（debugfs 进不去时它对底下任何路径都返回 False），得拿 debugfs 之外的节点当代理。

---

## 来源 —— 每样东西是从哪个仓库拿的

本仓库**只含自己写的代码**。其余外部件逐个列明出处、版本、许可，
完整清单见 **[docs/PROVENANCE.md](docs/PROVENANCE.md)**，摘要：

| 件 | 从哪拿 | 本仓库是否转分发 |
|---|---|---|
| **NBG 模型** | [northwindlight/a733-npu](https://github.com/northwindlight/a733-npu) 的 Release `nbg-animevideov3-v3`（该仓库用 Pegasus/ACUITY 从 ONNX 转）<br>再上游：Real-ESRGAN `realesr-animevideov3.pth` @ [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) release `v0.2.5.0`（BSD-3-Clause） | ❌ `install.sh` 自动拉 |
| **awnn API + VIP Lite**（NPU 用户态） | **`github.com/ZIFENG278/ai-sdk`** @ `fc90006d`（2025-10-20）<br>⚠️ **第三方镜像，不是全志官方仓库** | ❌ 用镜像里自带的 `~/ai-sdk` |
| **VE 编解码库** | **`gitlab.com/tina5.0_aiot/media/cedarc-release/libcedarc_v2.0`** 分支 `product-aiot-stable`<br>（`libvenc_codec.so` / `libVE.so` 是该仓库 `library/toolchain-sunxi-aarch64-glibc-gcc-v1320/v2/` 里的预编译件）<br>⚠️ **Allwinner 专有二进制，许可不明** | ❌ 按 PROVENANCE §3.1 自己编 |
| 板子固件 | Radxa A733 Debian 13 (trixie) 镜像 | — |
| `libcedarc-dev-*-arm64` v1.0.7 | Radxa 源 `radxa-repo.github.io/a733-trixie-test`（2022 年构建，**编解码器不认 A733**，只作对照） | — |
| ffmpeg 7.1.5 / caddy 2.6.2 / g++ 14.2 / python3.13+numpy | Debian 13 | — |
| **yt-dlp** | ⚠️ Debian 源里那份 2025.04.30 **太老，B 站必然 412**，要换 [yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp) 的 standalone 二进制 | — |
| 校准数据 DIV2K | `data.vision.ee.ethz.ch`（仅非商业研究）—— **只在 a733-npu 用到**，本仓库不涉及 | — |

**两条最要紧的**：
1. `ai-sdk` 是**第三方镜像**，官方不走公开 git，长期可用性没保证，请自行留档
2. VE 那套是**专有二进制**，所以本仓库不转分发，只给构建配方

---

## 已知问题

### ★「油画感」+ 边缘不锐 —— 模型问题，不是滤镜问题

`animevideov3` 是给**动画**调的轻量模型（16 层卷积 / 64 特征），训练目标是
平面色块 + 硬线条。喂**实拍**素材会把纹理抹成塑料块，锐边又还原不出来。

- **前置去噪没用，甚至更糟**：去噪先削掉细节，模型没有依据只能猜得更多
- **后置锐化只补边缘，补不回纹理**。默认开轻 CAS（`strength=0.4`）：
  实测拉普拉斯均值 2.42 → 3.37；`unsharp` 更猛（3.62）但在平坦区出晕，所以选 CAS
- **根治要换模型**：`realesr-general-x4v3` 是**同一套 SRVGGNetCompact 架构**
  （训练集换成通用/实拍），**代码路径完全一样**，只是 `num_conv` 16→32（约 2× 慢）。
  加一个 `models/*/model.json` 走 a733-npu 的 CI 就能编出来

### 编码器只会缩、不会放

VE 编码器自带缩放，1080p→720p 免费。但**放大直接失败**
（`h264 encoder wait interrupt overtime`，只出 29 字节）。所以最终分辨率不能
超过超分输出的尺寸，`app.py` 里已经做了钳制。

### 其它

- 单 worker 串行——同时传两个任务会排队（板子 1 GB 内存，并行跑不动）
- 任务表只存在内存 + `data/jobs.json`，重启不丢记录但会丢运行中的任务

---

## 文件

| 路径 | 说明 |
|---|---|
| `app.py` | Web 服务 + 流水线（单文件，只用 Python 标准库 + numpy） |
| `sysmon.py` | 系统状态采集（温度/CPU/内存/NPU/GPU），1 Hz 快照，SSE 推给前端 |
| `srpipe.c` | NPU 分块超分（C，链 awnn / VIP Lite） |
| `static/index.html` | 前端（无框架、无 CDN 依赖） |
| `deploy/install.sh` | 一键安装 |
| `deploy/Caddyfile` | 反向代理 |
| `deploy/sr-web.service` | systemd 单元 |
| `tools/seamtest.py` | 接缝量化工具 |
| `docs/RUNTIME.md` | 运维手册 + 完整踩坑记录 |
| `docs/PROVENANCE.md` | **来源清单：每个外部件的出处/版本/许可** |

---

## 许可

**MPL-2.0**（Mozilla Public License 2.0）—— 文件级 copyleft：
改动本仓库的源文件要把改动也以 MPL-2.0 开放，但把它和别的代码链接、
或用它做服务不触发传染。

```
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
```

**外部件各自的许可另算**，不受本仓库影响：

| 件 | 许可 |
|---|---|
| 超分权重 `realesr-animevideov3` | BSD-3-Clause（Real-ESRGAN） |
| `ai-sdk`（awnn / VIP Lite） | 未标许可（第三方镜像，见 PROVENANCE §2） |
| Allwinner cedarc（VE 编解码） | 专有，未标许可（见 PROVENANCE §3） |
| ffmpeg / caddy / numpy 等 | 各自的（Debian 包） |
