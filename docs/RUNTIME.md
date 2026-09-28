# A733 视频超分服务

上传视频或哔哩哔哩链接 → NPU 超分 → 下载。全部在 A733 板子上跑，不依赖外部机器。

浏览器打开 **`http://<板子IP>/`**（板子的局域网地址，端口 80）。

---

## 架构

```
浏览器 ──► Caddy(:80) ──► app.py(:8080, ThreadingHTTPServer)
                            │
                            ├─ 上传    → /opt/sr/data/<id>.src.mp4
                            ├─ b23 链接 → yt-dlp（带 cookie）
                            │
                            └─ 流水线（每个任务一个线程）
                                 vdecpipe ── VE 硬解 H.264（流式喂管道）
                                   └ srpipe ── NPU 分块 x4 超分（352x224 块）
                                       └ ffmpeg 缩放到目标分辨率 → NV12（★必须在这缩，
                                         编码器不会缩，-d 小了它只裁左上角）
                                           └ vencoderdemo_v2 ── VE 硬编 H.264（-s == -d）
                                               └ ffmpeg 混入原音轨 → mp4（★不能用 -shortest）
```

**关键点：只有压缩视频过网络，帧从不落地传输。** NPU 和 VE 都在板子上，所以不存在"帧来回搬"的带宽问题。

---

## 为什么是"分块"而不是"换个形状的模型"

NBG（NPU 模型）是**定形状**的——编译时输入尺寸就固定了。要处理任意分辨率的源，两条路：

1. **把源降采样到 NBG 的输入尺寸** —— 直接丢信息，超分再猜回来，得不偿失
2. **重新编译 NBG** —— 要重测内存池。实测：320×180（5.76 万像素）输出正确；640×360（23 万像素）x4 会**溢出 84 MB 内存池**，表现为"局部平滑、全局错乱"的横条纹，**不报错、不崩溃**，极易误判为能跑

**分块（tiling）两条都不需要**：源原样切成 320×180 的小块，逐块过 NPU，再拼回 4 倍大的整帧。代价仍 ∝ **源总像素**，与块大小无关。

### 接缝与 margin

3×3 卷积 ×18 层的感受野半径约 18 像素。块边界处网络看不到真实邻居（只能看到钳制复制出来的像素），那圈输出是"少信息"的。

**实测**（同一帧用两套不同分块网格跑，在重叠区逐列比）：

| | 无 margin | margin=16 |
|---|---|---|
| 块边界那一列的平均绝对差 | **2.073** | **0.000** |
| 远离边界 | 0.141 | 0.040 |

差异是真实存在的，而且**位置固定、每帧一样**——静态竖线正是人眼最敏感的东西。所以默认开 margin=16。

**代价不额外增加**：640×360 源无 margin 是 2×2=4 块；带 margin 是 3×3=9 块，总处理像素都是 518K。

---

## 性能（实测，非估算）

NPU 代价 **∝ 源总像素**，约 **0.547 µs/源像素**（animevideov3 x4，跨分辨率实测线性）：

| 源分辨率 | NPU 每帧 | 端到端 |
|---|---|---|
| 320×180 | 31 ms | ~4.5 帧/秒 |
| 360p (640×360) | 126 ms | ~3.6 帧/秒（含 margin 与流水线开销） |
| 480p | 216 ms | ~2 帧/秒 |
| 720p | 504 ms | ~0.8 帧/秒 |
| 1080p | 1.13 s | ~0.4 帧/秒 |

**这是批处理速度，不是实时。** 源分辨率越高越慢，这是硬件决定的，不是可以调优的地方。

VE 侧（解码 + 编码）完全不是瓶颈——1080p 编解码并发时合计 84 帧/秒，比 NPU 快两个数量级。

---

## 已知问题

### 1. "油画感" + 边缘不锐 —— 模型问题，不是滤镜问题

`animevideov3` 是给**动画**调的轻量模型（16 层卷积，64 特征），训练目标是平面色块 + 硬线条。喂**实拍**素材会把纹理抹成塑料块，然后锐边又还原不出来。

- **前置去噪没用**（甚至更糟）：去噪会先削掉细节，模型没有依据只能猜得更多，油画感更重。
- **后置锐化只能补边缘**，补不回纹理。默认开了轻微 CAS（`strength=0.4`），实测拉普拉斯均值 2.42 → 3.37；`unsharp` 更猛（3.62）但在平坦区会出晕，所以选了 CAS（对比度自适应，按局部对比度决定强度）。
- **根治要换模型**：`realesr-general-x4v3` 是**同一套架构**（SRVGGNetCompact）但训练集是通用/实拍内容，**代码路径完全一样**，只是 `num_conv` 从 16 变成 32（约 2× 慢）。加一个 `models/*/model.json` 走 CI 编出来即可，见 `a733-npu` 仓库。

### 2. ★编码器**既不会缩也不会放** —— 它只会裁（2026-09-28 更正）

这里原本写着「VE 编码器自带缩放，1080p→720p 免费」。**那句话是错的。**

实测：一张**正确的** 3840×2160 NV12 喂给 `vencoderdemo_v2 -s 3840x2160 -d 1920x1080`，
出来的**不是**缩小后的整帧，而是**左上角 1920×1080 那一块**，**rc=0、无任何报错**。
（放大也不行：`wait interrupt overtime`，只出 29 字节。）

`-d`/`--dstsize` 这个名字在暗示"目标尺寸、它会帮你缩"，实际是**裁剪框**。
所以：**`-s` 必须等于 `-d`**，要改尺寸就在喂给编码器之前自己缩 ——
`app.py` 是在 srpipe 和编码器之间挂一级管道里的 `ffmpeg -vf scale=…:flags=area`
（4K→1080p 实测 24.5 ms/帧），最终分辨率不能超过超分输出（钳制还在）。

细节与复现步骤见 README 的「已知问题」。

### 3. ffmpeg 的 `hqdn3d` 在第一帧上是 no-op

它会静默返回原帧。用逐帧抽帧（`-ss ... -frames:v 1`）测去噪效果时，**测出来的"没区别"是假的**。要测就得喂连续帧流，或换 `nlmeans`。

---

## 运维

```bash
systemctl status sr-web        # Web 服务
journalctl -u sr-web -f        # 日志
systemctl restart sr-web       # 改完 app.py 后重启

systemctl status caddy         # 反向代理
```

**文件位置**

| 路径 | 是什么 |
|---|---|
| `/opt/sr/app.py` | Web 服务 + 流水线（单文件，只用标准库 + numpy） |
| `/opt/sr/srpipe` | NPU 分块超分（C，链 awnn/VIP Lite） |
| `/opt/sr/models/anv3.nb` | animevideov3 x4，输入 320×180 |
| `/opt/sr/static/index.html` | 前端 |
| `/opt/sr/data/` | 上传的源 + 产物（可定期清） |
| `/opt/sr/bili_cookies.txt` | 哔哩哔哩 cookie（600，含凭据，别外传） |
| `/etc/udev/rules.d/60-a733-media.rules` | 让普通用户免 sudo 用 VE / dma_heap |

**改 app.py 后**：`scp` 到 `/tmp` → `sudo mv` 到 `/opt/sr/` → `sudo systemctl restart sr-web`

**改 srpipe.c 后**：在板子上重编（见下），再 `sudo cp` 到 `/opt/sr/srpipe`

```bash
V=$HOME/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0
g++ -O2 -o srpipe srpipe.c \
  $HOME/ai-sdk/examples/libawnn_viplite/awnn_lib.c \
  $HOME/ai-sdk/examples/libawnn_viplite/awnn_quantize.c \
  -I$V/inc -I$HOME/ai-sdk/examples/libawnn_viplite -I$HOME/ai-sdk \
  -L$V -Wl,-rpath-link,$V -lNBGlinker -lVIPhal -lm
```

---

## 踩过的坑（照着别再踩）

1. **`awnn` 的 ALOGD 就是裸 `printf`** —— 会写进 stdout。srpipe 启动时先把 fd 1 存下来再 `freopen(/dev/null)`，数据走保存的那个 fd。之前直接输出到 stdout，每帧被日志污染 ~1KB。
2. **`vpm_run` 不能用来做视频**：每帧重新 load NBG，而且默认把输出写成 56 MB ASCII 文本（单帧多 ~10 秒）。必须走 awnn API 一次 load。
3. **NPU 输出布局是 planar（NCHW）**，恒等映射就对了 —— 喂 ffmpeg 用 `-pix_fmt gbrp`。⚠ 我一度以为 `--nhwc` 分支能验证布局，其实两个分支都是 `out[i]=g(f[i])`，**那个测试什么也没证明**。
4. **`fs.protected_regular=2`**（Debian 默认）+ sticky `/tmp`：**连 root 都不能覆写别人属主的已存在文件**（EACCES）。调试脚本请用独立工作目录，别在 `/tmp` 里复用文件名。
5. **`strace -p` 挂到正在用 `/dev/cedar_dev` 的进程上会把板子拖死到需要重启**；`kill -9` 正在编码的进程也会把 VE 驱动留在坏状态（症状：每帧打印一条 `wait ve mem sync idle bit too long`，性能掉 20 倍）。要观测就改代码埋点。
6. **别用 `| head` 给管道收尾测构建**：`$?` 变成 `head` 的，构建失败也看着像成功。
