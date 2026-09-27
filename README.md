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
| NPU 超分 | 0.547 µs / **源像素**（animevideov3 x4，跨分辨率线性） |
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
代价仍 ∝ **源总像素**，与块大小无关——所以分块是免费的。

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

---

## 依赖的两个外部件

这个仓库**不含**下面两样，`install.sh` 会去取：

1. **NBG 模型** —— 由 [northwindlight/a733-npu](https://github.com/northwindlight/a733-npu)
   从 ONNX 转换而来（Pegasus/ACUITY 工具链太重，不适合塞进本仓库）。
   模型是 Real-ESRGAN `realesr-animevideov3`（BSD-3-Clause）。
2. **VE 编解码库** —— Allwinner Tina SDK（`gitlab.com/tina5.0_aiot`）编出来的
   `libvencoder` / `libvenc_codec` 那一套。见 [docs/RUNTIME.md](docs/RUNTIME.md)。

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
| `srpipe.c` | NPU 分块超分（C，链 awnn / VIP Lite） |
| `static/index.html` | 前端（无框架、无 CDN 依赖） |
| `deploy/install.sh` | 一键安装 |
| `deploy/Caddyfile` | 反向代理 |
| `deploy/sr-web.service` | systemd 单元 |
| `tools/seamtest.py` | 接缝量化工具 |
| `docs/RUNTIME.md` | 运维手册 + 完整踩坑记录 |

---

## 许可

Apache-2.0。超分权重 `realesr-animevideov3` 为 BSD-3-Clause（Real-ESRGAN）。
