# VE 硬解接入

解码从 CPU 软解换成 VE 专用硅。`srpipe --in-nv12`（NV12 → G,B,R 平面瓦片）
和 app.py 里的预处理都已完成并上线。

## ★先纠正一个我自己搞错的结论

排查过程中我一度判定"解码器需要 root"——**错了**。真相是：

`LD_LIBRARY_PATH` 让加载器选中了**我们自己编的 cedarc_v2 那套库**，而它会在
`H264DecoderInit` 里段错误（`si_addr=0x40`，NULL 解引用）。

| 库 | 自编（cedarc_v2） | 系统自带 |
|---|---|---|
| `libawh264.so` | `5a4e3036…` ✗ 崩 | `a02910d4…` ✓ |
| `libvdecoder.so` | `45b86d8a…` | `e223d459…` |

**换成系统库，普通用户直接跑通。** 编码器那条线用自编库没问题（我们编了
`libvenc_codec.so` 的 V-line 新 API），所以之前一直没暴露。

### 但 `sunxi_soc_info` 那条 udev 规则是真的需要

两套库都会 `open("/dev/sunxi_soc_info")`。它出厂是 `0600 root:root`，读不到
就 NULL 解引用段错误（**连日志都来不及打**，看起来像"要 root"）。
已加进 `60-a733-media.rules`：

```
KERNEL=="sunxi_soc_info", MODE="0644"
```

★定位手法：root 与普通用户各跑一次 `strace -f -e trace=openat,ioctl`，
`diff` 两条 trace —— 分岔点是**插件扫描目录**不同（root 扫 `/usr/lib`、
用户扫 `LD_LIBRARY_PATH`），这就是"加载了不同库"的指纹。

## 已确认的事实

| 事 | 结论 |
|---|---|
| ffmpeg 的 `h264_v4l2m2m` | **用不了** —— `sunxi-ve.ko` 不是 V4L2 驱动（零个 v4l2 符号），`/sys/class/video4linux` 空 |
| 解码器输入 | 必须**裸 H.264 流**（`ffmpeg -c:v copy -bsf:v h264_mp4toannexb -f h264`），不吃 mp4 |
| demo 的存帧开关 | 默认 `nSavePictureStartNumber=0xffffff` / `nSavePictureNumber=0` ⇒ **不传 `-ss 0 -sn N` 一帧都不存，且不报错** |
| 输出格式 | `-outFmat 6` = NV12，**高按 16 对齐**（960×540 → 960×544） |
| 硬度缩放 | `libawh264.so` 有 `H264DecoderSetExtraScaleInfo`，但 cedarC 那层 `SetExtraScaleInfo` **从没被赋值** ⇒ `ConfigExtraScaleInfo()` 是**空操作**。想要"解出来就是目标尺寸"得自己补这层 |
| **解码正确性** | 与 ffmpeg 软解**逐字节相同**（Y 平面 平均 0.000 / 最大 0） |

## ★原始 demo 会卡死（本目录补丁修的就是这个）

`vdecoderdemo` 是个**播放器** demo。它的显示线程一进来就检查 `state & DEMO_EXIT`
并自杀，而 `DEMO_EXIT` **包含 parser 的错误位** —— parser 稍有问题，写文件和
还帧的线程就没了，解码线程随后空转到帧池耗尽：

> 进程活着、CPU 10%、输出停在 **14 帧**（= 帧缓冲数）不再增长

**看起来完全像"解码慢"。** 补丁：显示线程只认解码器自己的退出条件，且拿到帧
立刻 `ReturnPicture`。

## 实测收益（尺子是"挪热点"）

| 阶段 | CPU/帧 |
|---|---|
| 现状 ffmpeg（软解 + scale + gbrp） | 15.9 ms |
| VE 硬解 → NV12（预pass） | 2.95 ms（13% CPU，纯 I/O 受限） |
| 新 ffmpeg（只 seek + crop + 搬运） | 5.1 ms |
| **SR 阶段的 ffmpeg CPU** | **15.9 → 5.1 ms（−68%）** |

管道数据量也减半（gbrp 3 字节/像素 → NV12 1.5）。

**代价**：多一个串行预pass。VE 解码本身只 2.95 ms/帧 CPU，但 NV12 落盘受
eMMC 限速（实测 35 MB/s）拖到 ~22 ms/帧 wall；中间文件要占盘，所以 app.py 里
有容量闸（默认 20 GB 或剩余空间 40%），超了自动退回软解并在界面上说明。

## ★踩到的坑：`crop` 默认是【居中】裁

`-vf crop=W:H` 在 960×544 的输入上裁的是**第 2~541 行**（`x=(iw-ow)/2`），
不是第 0 行。必须写 `crop=W:H:0:0`。这个 bug 让 Y 平面平均差 2.134 ——
恰好等于"偏移 2 行"的数值，所以我一开始还以为是解码器不对。
