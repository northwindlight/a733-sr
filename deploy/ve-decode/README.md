# VE 硬解接入（进行中）

目标：把解码从 CPU 软解换成 VE 专用硅（`srpipe --in-nv12` 那一半已经做完并验证）。

## 已确认的事实

| 事 | 结论 |
|---|---|
| ffmpeg 的 `h264_v4l2m2m` | **用不了** —— `sunxi-ve.ko` 不是 V4L2 驱动（二进制里没有任何 v4l2 符号），`/sys/class/video4linux` 是空的 |
| 解码器后端 | `libawh264.so` 里有 `H264DecoderSetExtraScaleInfo` ⇒ **硬件支持缩放**，但 cedarC 那层没接线（`DecoderInterface->SetExtraScaleInfo` 从没被赋值），`ConfigExtraScaleInfo()` 是**空操作** |
| demo 的输入 | 必须**裸 H.264 流**（`ffmpeg -c:v copy -bsf:v h264_mp4toannexb -f h264`），不吃 mp4 容器 |
| demo 的存帧开关 | 默认 `nSavePictureStartNumber=0xffffff` / `nSavePictureNumber=0` ⇒ **不传 `-ss 0 -sn N` 一帧都不存**，而且不报错，看着像卡死 |
| 输出格式 | `-outFmat 6` = NV12，**高度按 16 对齐**（960×540 → 960×544） |

## ★原始 demo 会卡死（本补丁修的就是这个）

`vdecoderdemo` 是个**播放器** demo：解码线程把帧交给「显示线程」，显示线程再还回帧池。
它的显示线程一进来就检查 `state & DEMO_EXIT` 并自杀 —— 而 `DEMO_EXIT` **包含 parser 的错误位**。
于是 parser 稍有问题，写文件和还帧的线程就没了：解码线程随后空转到帧池耗尽
（**正好停在帧缓冲数那么多帧上**），进程活着、CPU 10%、输出文件不再增长。

本目录的补丁把它改成：显示线程只认解码器自己的退出条件，且**拿到帧就立刻
`ReturnPicture`**，不进显示列表、不假装在显示。

## ⚠️ 卡点：解码器要 root

- root 跑：正常（960×544 NV12，199 帧，干净退出）
- 非 root 跑：**段错误**（0 字节输出，无日志）
- **原始 demo 非 root 也一样段错误** ⇒ 不是补丁的问题

`strings libcdc_base.so` 显示它要 `/dev/dma_heap/reserved` 和 `/dev/dma_heap/system`，
而板上 **`/dev/dma_heap/` 是空的**（`60-a733-media.rules` 里那条 dma_heap 规则在空转）。
大概率就是这里，但**没验**。

**没有擅自加 sudoers / setcap / 改服务用户** —— 这属于权限变更，要用户拍板。
