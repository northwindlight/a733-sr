/*
 * srpipe —— A733 NPU 超分滤波器（分块 / tiling）
 *
 * 为什么分块：NBG 是定形状的。想处理任意分辨率的源，要么把源降采样到
 * NBG 的输入尺寸（丢信息），要么换形状重编（要重测内存池，且 640x360 x4
 * 实测会溢出 84MB 内存池 -> 输出错乱但不报错）。
 * 分块两者都不需要：源原样切块，逐块过 NPU，再拼回 4 倍大的整帧。
 * 代价仍 ∝ 源总像素（实测 0.547 µs/px），与块大小无关。
 *
 * ★数据走 stdin/stdout 管道。awnn 库里 ALOGD 就是裸 printf 会污染 stdout，
 *   所以启动时先把 fd 1 存下来，再把 stdout 重定向到 /dev/null。
 *   进度一律走 stderr。
 *
 * ★接缝问题：3x3 卷积 x18 层的感受野半径约 18 像素。块边界处网络看不到真实
 *   邻居（会看到钳制复制出来的像素），那圈输出是"少信息"的，且位置固定、
 *   每帧一样 —— 静态竖线正是人眼最敏感的。实测边界列的平均绝对差是远离边界
 *   处的 14.7 倍（2.07 vs 0.141）。
 *   修法：每块多读 margin 像素、输出只取中间有效区。代价不额外增加：
 *   640x360 源无 margin 是 2x2=4 块，margin=16 是 3x3=9 块，总处理像素都是 518K。
 *
 * 用法：
 *   srpipe <nbg> <in_w> <in_h> <scale> <tile_w> <tile_h>
 *          [--margin M] [--nv12] [--sharpen A] [--profile]
 *          [--gpu] [--gpu-in] [--float-out] [--dump-tiles <f>]
 *
 *   --gpu       输出侧：GPU 做 RGB->NV12 + 锐化（与 NPU 并行）—— 默认开，省 CPU 热点
 *   --gpu-in    输入侧：GPU 抠块。★实测【没用】，默认关，见 gpu_in_thread 上面的注释
 *   --float-out 强制走 awnn 的浮点输出路（只用于 A/B 验证字节等价）
 *   --dump-tiles 把这帧的瓦片组原样吐出，用于比 CPU/GPU 两条抠块路是否逐字节相同
 *
 * 默认输出平面 RGB（gbrp），配 ffmpeg 用。
 * ★--nv12 直接吐 NV12（Y 平面 + 交织 CbCr，2x2 下采样），为了喂 VE 硬编：
 *   ① 数据量从 3 字节/像素降到 1.5，管道/磁盘都省一半
 *   ② 省掉 ffmpeg 那一步 swscale —— 实测它在大尺寸上是整条流水线的瓶颈
 *      （5120x2880 时 ffmpeg 占 368% CPU，srpipe 才 45%）
 *   ③ 编码器原生就吃 NV12，可以再用 FIFO 把中间文件也省掉
 *
 *   输入 stdin ：每帧 in_w*in_h*3 字节，平面排列 R 面|G 面|B 面（= ffmpeg gbrp）
 *   输出 stdout：每帧 (in_w*scale)*(in_h*scale)*3 字节，同样平面排列
 *
 * 例：640x360 的源，x4，用 320x180 的块 -> 2x2=4 块/帧，出 2560x1440
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <awnn_lib.h>
/* ★awnn_internal.h 只为拿 Awnn_Context_t 的结构定义（要绕开 awnn_run 的浮点反量化）。
 *   它没有 include guard，而且里面是【定义】了两个全局数组 time_begin/time_end
 *   —— awnn_lib.c 里也定义了同样两个，直接 include 会 multiple definition。
 *   借宏给它们改个名绕过去。 */
#define time_begin srpipe_unused_time_begin
#define time_end   srpipe_unused_time_end
#include "awnn_internal.h"
#undef time_begin
#undef time_end
#include <pthread.h>
#include <semaphore.h>
#include <CL/cl.h>

#define CL_CHECK_OR_DIE(e, what) do { if ((e) != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL 错误 %d @ %s\n", (int)(e), what); exit(1); } } while (0)

static void die(const char *m) { fprintf(stderr, "srpipe: %s\n", m); exit(1); }
static void write_full(int fd, const void *buf, size_t n);   /* 定义在后面 */
static double _ms(void);                                    /* 定义在后面 */

/* ============================ GPU 转换（可选） ============================
 * 为什么单独开线程：转换和 NPU 是不同硬件，串行做等于白白浪费 NPU 那 486ms。
 * 双缓冲 + 信号量，NPU 线程在算第 N+1 块时 GPU 线程在转第 N 帧。
 *
 * 实测（3840x2160，见 gputest.c）：GPU 73ms vs CPU 57ms —— GPU 更慢，
 * 但在流水线里这一级只需跑赢 NPU 的 486ms，两者都远远够。
 * 选 GPU 的理由不是速度，是**把热点从 CPU 挪开**（CPU 60°C 就降频）。
 */
static const char *CL_SRC =
"__kernel void gbrp2nv12_sharp(__global const uchar *G, __global const uchar *B,\n"
"                              __global const uchar *R, __global uchar *Y,\n"
"                              __global uchar *UV, __global uchar *YT,\n"
"                              const int w, const int h, const int sh)\n"
"{\n"
"    int bx = get_global_id(0) * 2, by = get_global_id(1) * 2;\n"
"    if (bx >= w || by >= h) return;\n"
"    int sr = 0, sg = 0, sb = 0;\n"
"    for (int dy = 0; dy < 2; dy++) {\n"
"        int yy = by + dy; if (yy >= h) yy = h - 1;\n"
"        for (int dx = 0; dx < 2; dx++) {\n"
"            int xx = bx + dx; if (xx >= w) xx = w - 1;\n"
"            int i = yy * w + xx;\n"
"            int r = R[i], g = G[i], b = B[i];\n"
"            sr += r; sg += g; sb += b;\n"
"            int v = ((16829 * r + 33039 * g + 6416 * b) >> 16) + 16;\n"
"            Y[i] = (uchar)(v < 16 ? 16 : (v > 235 ? 235 : v));\n"
"        }\n"
"    }\n"
"    int r = sr / 4, g = sg / 4, b = sb / 4;\n"
"    int cb = ((-9711 * r - 19098 * g + 28784 * b) >> 16) + 128;\n"
"    int cr = (( 28784 * r - 24103 * g - 4681 * b) >> 16) + 128;\n"
"    cb = cb < 16 ? 16 : (cb > 240 ? 240 : cb);\n"
"    cr = cr < 16 ? 16 : (cr > 240 ? 240 : cr);\n"
"    int o = (by / 2) * w + bx;\n"
"    UV[o] = (uchar)cb; UV[o + 1] = (uchar)cr;\n"
"}\n"
"/* 亮度 3x3 unsharp。★必须单独一个内核：\n"
" *   锐化要读「未锐化的当前帧 Y」，而在同一个内核里写 Y 的邻居是没同步的\n"
" *   （work-item 之间无顺序保证）。拆开 + 中间拷一份 YT 才是对的。\n"
" *   我第一版把拷贝放在内核之前 —— 拷到的是上一帧的 Y，输出全错（实测平均差 45）。\n"
" */\n"
"__kernel void unsharp_y(__global uchar *Y, __global const uchar *YT,\n"
"                        const int w, const int h, const int sh)\n"
"{\n"
"    int x = get_global_id(0), y = get_global_id(1);\n"
"    if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) return;\n"
"    int c = y * w + x, s9 = 0;\n"
"    for (int j = -1; j <= 1; j++)\n"
"        for (int k = -1; k <= 1; k++) s9 += YT[c + j * w + k];\n"
"    int v = (int)YT[c] + ((((int)YT[c] - s9 / 9) * sh) >> 8);\n"
"    Y[c] = (uchar)(v < 16 ? 16 : (v > 235 ? 235 : v));\n"
"}\n";

/* ---- 输入侧：把整帧 gbrp「抠」成 NPU 要的瓦片布局 ----
 * 纯数据搬运，不做任何数值变换 —— 所以能和 CPU 路径【逐字节】比对，
 * 这是这条流水线里唯一一处可以做到完全等价验证的地方。
 *
 * 为什么要搬走：CPU 上这一趟是 8640 次 320 字节的跨行 memcpy
 * （16 块 x 3 平面 x 180 行），实测比它搬运的数据量该有的时间贵几十倍，
 * 而 CPU 正是 60°C 就降到 416MHz 的那个热区。
 *
 * 输出布局：tiles[(ty*nx+tx)][plane][y][x]，plane 顺序 G,B,R（与 gbrp 一致）。
 */
static const char *CL_IN_SRC =
"__kernel void frame_to_tiles(__global const uchar *G, __global const uchar *B,\n"
"                             __global const uchar *R, __global uchar *T,\n"
"                             const int iw, const int ih, const int tw, const int th,\n"
"                             const int nx, const int cw, const int ch, const int mg)\n"
"{\n"
"    int gx = get_global_id(0);          /* 0 .. nx*tw-1 */\n"
"    int gy = get_global_id(1);          /* 0 .. NY*th-1 */\n"
"    int tx = gx / tw, lx = gx - tx * tw;\n"
"    int ty = gy / th, ly = gy - ty * th;\n"
"    int sx = tx * cw - mg + lx;         /* 窗口左上角可以越界，靠钳制补边 */\n"
"    int sy = ty * ch - mg + ly;\n"
"    if (sx < 0) sx = 0; else if (sx >= iw) sx = iw - 1;\n"
"    if (sy < 0) sy = 0; else if (sy >= ih) sy = ih - 1;\n"
"    int i = sy * iw + sx, o = ly * tw + lx, plane = tw * th;\n"
"    __global uchar *base = T + (size_t)(ty * nx + tx) * 3 * plane;\n"
"    base[o]            = G[i];\n"
"    base[plane + o]    = B[i];\n"
"    base[2 * plane + o]= R[i];\n"
"}\n";

/* --- GPU 线程的共享状态（双缓冲）--- */
typedef struct {
    int w, h, sharp;
    unsigned char *rgb[2];        /* 生产者填这里 */
    unsigned char *yuv[2];        /* 消费者写这里 */
    sem_t full, empty;            /* full: 有帧可转; empty: 有槽位可填 */
    int idx_in, idx_out;
    volatile int stop;
    cl_context ctx; cl_command_queue q; cl_kernel k, k_sh;
    cl_mem mG, mB, mR, mY, mUV, mYT;
    double t_conv;                /* 累计转换耗时(ms)，仅做报告 */
    volatile long frames;
    volatile long posted;         /* 生产者交出去的帧数，用来排空后再停 */
    int data_fd;                  /* GPU 线程负责写输出 */
} gpu_ctx_t;

static void *gpu_thread(void *arg)
{
    gpu_ctx_t *g = (gpu_ctx_t *)arg;
    while (1) {
        sem_wait(&g->full);
        int i = g->idx_out;
        if (g->stop) break;
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        double t0 = ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;

        cl_int e;
        size_t px = (size_t)g->w * g->h, rgb = px, yuv = px * 3 / 2;
        e = clEnqueueWriteBuffer(g->q, g->mG, CL_TRUE, 0, rgb, g->rgb[i], 0, NULL, NULL);
        e |= clEnqueueWriteBuffer(g->q, g->mB, CL_TRUE, 0, rgb, g->rgb[i] + px, 0, NULL, NULL);
        e |= clEnqueueWriteBuffer(g->q, g->mR, CL_TRUE, 0, rgb, g->rgb[i] + 2 * px, 0, NULL, NULL);
        (void)e;
        size_t gsz[2] = { ((size_t)g->w + 1) / 2, ((size_t)g->h + 1) / 2 };
        size_t lsz[2] = { 16, 4 };
        if (lsz[0] > gsz[0]) lsz[0] = gsz[0];
        if (lsz[1] > gsz[1]) lsz[1] = gsz[1];
        clEnqueueNDRangeKernel(g->q, g->k, 2, NULL, gsz, lsz, 0, NULL, NULL);
        /* ★锐化必须在转换内核【之后】：先拷当前帧的 Y 到 YT，再让第二个内核写回 Y */
        if (g->sharp > 0) {
            clEnqueueCopyBuffer(g->q, g->mY, g->mYT, 0, 0, px, 0, NULL, NULL);
            size_t gs1[2] = { (size_t)g->w, (size_t)g->h };
            size_t ls1[2] = { 64, 4 };
            if (ls1[0] > gs1[0]) ls1[0] = gs1[0];
            if (ls1[1] > gs1[1]) ls1[1] = gs1[1];
            clEnqueueNDRangeKernel(g->q, g->k_sh, 2, NULL, gs1, ls1, 0, NULL, NULL);
        }
        /* ★Y 和 UV 是两个 buffer，必须分别读回来、拼进 NV12 的两段。
         *   我第一版只读了 mY 却按整个 yuv 尺寸读 —— UV 那半截是未初始化内存，
         *   表现成"Y 平面逐字节相同、UV 全错"（实测 UV 平均差 128）。 */
        clEnqueueReadBuffer(g->q, g->mY,  CL_TRUE, 0, px,      g->yuv[i],        0, NULL, NULL);
        clEnqueueReadBuffer(g->q, g->mUV, CL_TRUE, 0, px / 2,  g->yuv[i] + px,   0, NULL, NULL);
        clFinish(g->q);

        clock_gettime(CLOCK_MONOTONIC, &ts);
        g->t_conv += (ts.tv_sec * 1e3 + ts.tv_nsec / 1e6) - t0;
        g->frames++;
        if (!g->stop) {
            write_full(g->data_fd, g->yuv[i], yuv);
            fprintf(stderr, "FRAME %ld\n", g->frames - 1);
        }
        g->idx_out ^= 1;
        sem_post(&g->empty);
    }
    return NULL;
}

/* ========================= 输入侧：GPU 抠块（可选） =========================
 * 与输出侧各自一个线程 + 各自的 cl_command_queue。
 * 刻意【不】把两件事塞进同一个线程：那样 gather(f+1) 会被 convert(f-1) 挡住，
 * 排出来的周期是 NPU+GPU 串联（推算 1126ms），而不是 NPU 单独（686ms）。
 * 两个线程各自的循环都只有单一职责，不用去证明一个交叉调度的时序是对的。
 *
 * 输入侧只保留【一帧在飞】：主线程拿空槽(read) -> 填 -> 丢给 GPU -> 等瓦片。
 * 这一级 GPU 只需要跑赢 NPU（实测 gather 远小于 NPU），不需要更深。
 */
typedef struct {
    /* ★in_h = 输入缓冲区的【实际高】，ih = 我们要的【裁剪高】。两者分开是因为
     *   VE 解码器的输出高按 16 对齐（960x540 的源它吐 960x544）：Y 平面是
     *   iw*in_h，UV 平面从 iw*in_h 开始，但我们只要前 ih 行。
     *   不给 --in-h 时 in_h = ih，行为与以前完全一致。 */
    int iw, ih, in_h, tw, th, cw, ch, nx, ny, margin, in_nv12;
    size_t frame_in, tile_in, all_tiles;
    unsigned char *src[3];            /* 主线程从管道读进来的整帧 gbrp */
    unsigned char *tiles[2];          /* GPU 收集出来的瓦片组 */
    sem_t free_slots;                 /* 计数 3：主线程等一个已被消费的 src 槽 */
    sem_t in_full;                    /* 主线程 -> GPU：有一帧可抠 */
    sem_t tiles_ready;                /* GPU -> 主线程：瓦片好了 */
    volatile int slot_in, slot_out, stop;
    volatile long posted, done;
    double t_gpu;                     /* 抠块线程累计耗时(ms) */
    cl_context ctx; cl_command_queue q; cl_kernel k, k_nv12;
    cl_mem mG, mB, mR, mT, mY, mUV;
} gin_t;

static void *gpu_in_thread(void *arg)
{
    gin_t *g = (gin_t *)arg;
    int out_idx = 0;
    for (;;) {
        sem_wait(&g->in_full);
        if (g->stop) break;
        int i = g->slot_in, o = out_idx;
        out_idx ^= 1;

        double _t = _ms();
        cl_int e;
        size_t px = (size_t)g->iw * g->ih;
        const unsigned char *s = g->src[i];
        if (g->in_nv12) {
            e  = clEnqueueWriteBuffer(g->q, g->mY,  CL_TRUE, 0, px,     s,        0, NULL, NULL);
            e |= clEnqueueWriteBuffer(g->q, g->mUV, CL_TRUE, 0, px / 2, s + px,   0, NULL, NULL);
            if (e != CL_SUCCESS) { fprintf(stderr, "srpipe: GPU 上传失败 %d\n", (int)e); exit(1); }
            size_t gsz[2] = { (size_t)g->nx * g->tw, (size_t)g->ny * g->th };
            clEnqueueNDRangeKernel(g->q, g->k_nv12, 2, NULL, gsz, NULL, 0, NULL, NULL);
        } else {
            e  = clEnqueueWriteBuffer(g->q, g->mG, CL_TRUE, 0, px, s,          0, NULL, NULL);
            e |= clEnqueueWriteBuffer(g->q, g->mB, CL_TRUE, 0, px, s + px,     0, NULL, NULL);
            e |= clEnqueueWriteBuffer(g->q, g->mR, CL_TRUE, 0, px, s + 2 * px, 0, NULL, NULL);
            if (e != CL_SUCCESS) { fprintf(stderr, "srpipe: GPU 上传失败 %d\n", (int)e); exit(1); }
            size_t gsz[2] = { (size_t)g->nx * g->tw, (size_t)g->ny * g->th };
            clEnqueueNDRangeKernel(g->q, g->k, 2, NULL, gsz, NULL, 0, NULL, NULL);
        }
        clEnqueueReadBuffer(g->q, g->mT, CL_TRUE, 0, g->all_tiles, g->tiles[o], 0, NULL, NULL);
        clFinish(g->q);

        g->t_gpu += _ms() - _t;
        sem_post(&g->free_slots);
        g->done++;
        g->slot_out = o;          /* ★必须在 post 之前写，主线程 post 之后才读它 */
        sem_post(&g->tiles_ready);
    }
    return NULL;
}

/* NV12 版同一件事：一趟做完 NV12 -> RGB + 排成瓦片。
 *
 * 为什么需要它：VE 硬解吐出来的就是 NV12（用 vdecoderdemo 整片 dump，见 app.py），
 * 而 NPU 要的是 G,B,R 平面瓦片。中间那步 ffmpeg 的 yuv->gbrp 是纯 CPU 开销，
 * 直接在这里做掉。
 *
 * ★色度是【最近邻】上采样（每个像素取它 2x2 块的那对 UV）。ffmpeg 的 swscale
 *   用的是更精细的插值，所以换路之后颜色会有极小差异 —— 但 Y（细节全在这）
 *   是原样的，且输出还要过 VE 的 4:2:0 编码，色度精度不是瓶颈。已记在文档里。
 * ★系数是 BT.601 有限范围的标准整数反变换，与 srpipe 输出侧用的正向系数
 *   （从 ffmpeg 回归出来的那套）互为逆。
 */
static const char *CL_IN_NV12_SRC =
"__kernel void nv12_to_tiles(__global const uchar *Yp, __global const uchar *UV,\n"
"                            __global uchar *T,\n"
"                            const int iw, const int ih, const int tw, const int th,\n"
"                            const int nx, const int cw, const int ch, const int mg)\n"
"{\n"
"    int gx = get_global_id(0), gy = get_global_id(1);\n"
"    int tx = gx / tw, lx = gx - tx * tw;\n"
"    int ty = gy / th, ly = gy - ty * th;\n"
"    int sx = tx * cw - mg + lx;\n"
"    int sy = ty * ch - mg + ly;\n"
"    if (sx < 0) sx = 0; else if (sx >= iw) sx = iw - 1;\n"
"    if (sy < 0) sy = 0; else if (sy >= ih) sy = ih - 1;\n"
"    int c = (int)Yp[sy * iw + sx] - 16; if (c < 0) c = 0;\n"
"    int ui = (sy >> 1) * iw + ((sx >> 1) << 1);\n"
"    int d = (int)UV[ui] - 128, e = (int)UV[ui + 1] - 128;\n"
"    int r = (298 * c + 409 * e + 128) >> 8;\n"
"    int g = (298 * c - 100 * d - 208 * e + 128) >> 8;\n"
"    int b = (298 * c + 516 * d + 128) >> 8;\n"
"    r = r < 0 ? 0 : (r > 255 ? 255 : r);\n"
"    g = g < 0 ? 0 : (g > 255 ? 255 : g);\n"
"    b = b < 0 ? 0 : (b > 255 ? 255 : b);\n"
"    int o = ly * tw + lx, plane = tw * th;\n"
"    __global uchar *base = T + (size_t)(ty * nx + tx) * 3 * plane;\n"
"    base[o]             = (uchar)g;\n"
"    base[plane + o]     = (uchar)b;\n"
"    base[2 * plane + o] = (uchar)r;\n"
"}\n";

/* CPU 版同一件事 —— 保留它是为了能逐字节验证 GPU 路径（见 tools/gputiletest.sh） */
static void gather_cpu(const unsigned char *fin, unsigned char *tiles, const gin_t *g)
{
    const size_t plane = (size_t)g->tw * g->th;
    for (int ty = 0; ty < g->ny; ty++) {
        for (int tx = 0; tx < g->nx; tx++) {
            const int ox = tx * g->cw, oy = ty * g->ch;
            const int x0 = ox - g->margin, y0 = oy - g->margin;
            const int fast_x = (x0 >= 0 && x0 + g->tw <= g->iw);
            unsigned char *blk = tiles + (size_t)(ty * g->nx + tx) * 3 * plane;
            for (int c = 0; c < 3; c++) {
                const unsigned char *src = fin + (size_t)c * g->iw * g->ih;
                unsigned char *dst = blk + (size_t)c * plane;
                for (int y = 0; y < g->th; y++) {
                    int sy = y0 + y;
                    if (sy < 0) sy = 0; else if (sy >= g->ih) sy = g->ih - 1;
                    if (fast_x) {
                        memcpy(dst + (size_t)y * g->tw, src + (size_t)sy * g->iw + x0, g->tw);
                    } else {
                        const unsigned char *srow = src + (size_t)sy * g->iw;
                        unsigned char *drow = dst + (size_t)y * g->tw;
                        for (int x = 0; x < g->tw; x++) {
                            int sx = x0 + x;
                            if (sx < 0) sx = 0; else if (sx >= g->iw) sx = g->iw - 1;
                            drow[x] = srow[sx];
                        }
                    }
                }
            }
        }
    }
}

/* CPU 版 NV12 -> 瓦片。与上面的 GPU 内核【逐字节】对齐（同一套整数系数与钳制），
 * 否则 tools/gputiletest.sh 会报不一致。 */
static void gather_nv12_cpu(const unsigned char *fin, unsigned char *tiles, const gin_t *g)
{
    const unsigned char *Yp = fin, *UV = fin + (size_t)g->iw * g->in_h;
    const size_t plane = (size_t)g->tw * g->th;
    for (int ty = 0; ty < g->ny; ty++) {
        for (int tx = 0; tx < g->nx; tx++) {
            const int x0 = tx * g->cw - g->margin, y0 = ty * g->ch - g->margin;
            unsigned char *blk = tiles + (size_t)(ty * g->nx + tx) * 3 * plane;
            unsigned char *G = blk, *B = blk + plane, *R = blk + 2 * plane;
            for (int y = 0; y < g->th; y++) {
                int sy = y0 + y;
                if (sy < 0) sy = 0; else if (sy >= g->ih) sy = g->ih - 1;
                const unsigned char *yrow = Yp + (size_t)sy * g->iw;
                const unsigned char *urow = UV + (size_t)(sy >> 1) * g->iw;
                for (int x = 0; x < g->tw; x++) {
                    int sx = x0 + x;
                    if (sx < 0) sx = 0; else if (sx >= g->iw) sx = g->iw - 1;
                    int c = (int)yrow[sx] - 16; if (c < 0) c = 0;
                    int d = (int)urow[(sx >> 1) << 1] - 128;
                    int e = (int)urow[((sx >> 1) << 1) + 1] - 128;
                    int r = (298 * c + 409 * e + 128) >> 8;
                    int gg = (298 * c - 100 * d - 208 * e + 128) >> 8;
                    int b = (298 * c + 516 * d + 128) >> 8;
                    G[y * g->tw + x] = (unsigned char)(gg < 0 ? 0 : (gg > 255 ? 255 : gg));
                    B[y * g->tw + x] = (unsigned char)(b  < 0 ? 0 : (b  > 255 ? 255 : b));
                    R[y * g->tw + x] = (unsigned char)(r  < 0 ? 0 : (r  > 255 ? 255 : r));
                }
            }
        }
    }
}

static ssize_t read_full(int fd, void *buf, size_t n)
{
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, (char *)buf + got, n - got);
        if (r == 0) return got == 0 ? 0 : -1;
        if (r < 0)  return -1;
        got += (size_t)r;
    }
    return (ssize_t)got;
}

static void write_full(int fd, const void *buf, size_t n)
{
    const char *p = (const char *)buf;
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) die("写输出失败（下游断了吗）");
        p += w; n -= (size_t)w;
    }
}

#include <time.h>
static double _ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e3+t.tv_nsec/1e6;}
static double t_read,t_tile,t_npu,t_asm,t_yuv,t_sh,t_wr;   /* 各阶段累计(ms) */

int main(int argc, char **argv)
{
    if (argc < 7) {
        fprintf(stderr,
            "usage: %s <nbg> <in_w> <in_h> <scale> <tile_w> <tile_h> [选项]\n"
            "  读 stdin 的帧（默认 gbrp 平面，--in-nv12 则读 NV12），\n"
            "  输出 scale 倍大的帧（默认 gbrp，--nv12 则输出 NV12）\n"
            "  --in-nv12          输入是 NV12（Y 平面 + 交织 CbCr）\n"
            "  --in-h H           输入缓冲区实际高（默认 = in_h 参数）。\n"
            "                     VE 解码器输出高按 16 对齐，靠它把补的行裁掉\n"
            "  --frames N         只处理 N 帧就正常退出（流式分段用）\n"
            "  --margin N         块间重叠圈\n"
            "  --nv12             输出 NV12\n"
            "  --sharpen F        锐化强度\n"
            "  --gpu / --gpu-in   GPU 做输出转换 / GPU 抠块\n"
            "  --profile          打印各阶段耗时\n", argv[0]);
        return 2;
    }
    const char *nbg = argv[1];
    int iw = atoi(argv[2]), ih = atoi(argv[3]), scale = atoi(argv[4]);
    int tw = atoi(argv[5]), th = atoi(argv[6]);
    int margin = 0, out_nv12 = 0;
    float sharpen = 0.0f;
    int profile = 0, use_gpu = 0, use_gpu_in = 0, in_nv12 = 0;
    const char *dump_tiles = NULL;
    int force_float = 0;
    int in_h = 0;                 /* 输入缓冲区高，0 = 跟 ih 一样 */
    long max_frames = 0;          /* >0 = 只处理这么多帧就正常收工 */
    for (int i = 7; i < argc; i++) {
        if (!strcmp(argv[i], "--margin") && i + 1 < argc) margin = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nv12")) out_nv12 = 1;
        else if (!strcmp(argv[i], "--in-h") && i + 1 < argc) in_h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--frames") && i + 1 < argc) max_frames = atol(argv[++i]);
        else if (!strcmp(argv[i], "--sharpen") && i + 1 < argc) sharpen = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--profile")) profile = 1;
        else if (!strcmp(argv[i], "--gpu")) use_gpu = 1;
        else if (!strcmp(argv[i], "--gpu-in")) use_gpu_in = 1;
        else if (!strcmp(argv[i], "--in-nv12")) in_nv12 = 1;
        else if (!strcmp(argv[i], "--float-out")) force_float = 1;   /* A/B 用：强制走 awnn 浮点路 */
        else if (!strcmp(argv[i], "--dump-tiles") && i + 1 < argc) dump_tiles = argv[++i];
    }
    if (sharpen < 0.0f) sharpen = 0.0f;
    if (iw <= 0 || ih <= 0 || scale <= 0 || tw <= 0 || th <= 0) die("参数不合法");
    if (margin < 0) margin = 0;
    if (2 * margin >= tw || 2 * margin >= th) die("margin 太大了");
    const int cw = tw - 2 * margin, ch = th - 2 * margin;   /* 每块的有效（core）尺寸 */

    const int ow = iw * scale, oh = ih * scale;      /* 整帧输出 */
    const int stw = tw * scale, sth = th * scale;    /* 每块的输出 */
    const int scw = cw * scale, sch = ch * scale;    /* 有效区在输出里的大小 */
    const int nx = (iw + cw - 1) / cw, ny = (ih + ch - 1) / ch;   /* 按 core 步进 */

    /* ★关键：先把 stdout 存下来再重定向到 /dev/null，挡掉 awnn 的 printf */
    int data_fd = dup(STDOUT_FILENO);
    if (data_fd < 0) die("dup(stdout) 失败");
    if (!freopen("/dev/null", "w", stdout)) die("重定向 stdout 失败");

    fprintf(stderr, "srpipe: %dx%d -> %dx%d  块 %dx%d (margin %d, core %dx%d)  %dx%d=%d 块/帧\n",
            iw, ih, ow, oh, tw, th, margin, cw, ch, nx, ny, nx * ny);

    if (in_h <= 0) in_h = ih;
    if (in_h < ih) die("--in-h 不能小于裁剪高 ih");
    const size_t frame_in  = in_nv12 ? (size_t)iw * in_h * 3 / 2   /* NV12 = Y + 交织 CbCr */
                                     : (size_t)iw * ih * 3;
    const size_t frame_out = out_nv12 ? (size_t)ow * oh * 3 / 2     /* NV12 = Y + 交织 CbCr(1/2) */
                                      : (size_t)ow * oh * 3;
    const size_t rgb_bytes = (size_t)ow * oh * 3;   /* 组帧始终是平面 RGB */
    const size_t tile_in   = (size_t)tw * th * 3;

    unsigned char *fin   = (unsigned char *)malloc(frame_in);
    unsigned char *fout  = (unsigned char *)malloc(rgb_bytes);   /* 平面 RGB 整帧 */
    unsigned char *yuv   = out_nv12 ? (unsigned char *)malloc(frame_out) : NULL;
    const size_t all_tiles = (size_t)nx * ny * tile_in;          /* 一帧的全部瓦片 */
    const size_t tile_out_bytes = (size_t)stw * sth * 3;         /* NPU 单块输出字节数 */
    unsigned char *tiles = (unsigned char *)malloc(all_tiles);
    unsigned char *qbuf  = (unsigned char *)malloc(tile_out_bytes);   /* 量化输出中转 */
    if (!fin || !fout || !tiles || !qbuf) die("malloc 失败");
    (void)qbuf;

    /* ---- 抠块（输入侧）：几何对 CPU/GPU 两条路完全一样 ---- */
    gin_t gin; memset(&gin, 0, sizeof(gin));
    gin.iw = iw; gin.ih = ih; gin.in_h = in_h; gin.tw = tw; gin.th = th;
    gin.cw = cw; gin.ch = ch; gin.nx = nx; gin.ny = ny; gin.margin = margin;
    gin.in_nv12 = in_nv12;
    /* --gpu-in 的内核把 UV 平面写死在 Y 之后（偏移 iw*ih），缓冲区也按 iw*ih 分配，
     * 装不下 in_h > ih 的输入。这条路本来就默认关、实测也没用（弱 GPU），
     * 与其去改内核，不如直接拒绝这个组合。 */
    if (use_gpu_in && in_h != ih)
        die("--gpu-in 不支持 --in-h（输入缓冲区高 != 裁剪高）；用 CPU 抠块那条路");
    gin.frame_in = frame_in; gin.tile_in = tile_in; gin.all_tiles = all_tiles;
    gin.slot_in = 0; gin.slot_out = 0; gin.stop = 0; gin.posted = 0; gin.done = 0;
    pthread_t gin_tid = 0;

    if (use_gpu_in) {
        cl_int e; cl_platform_id plat; cl_device_id dev; cl_uint nd;
        if (clGetPlatformIDs(1, &plat, &nd) != CL_SUCCESS ||
            clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, &nd) != CL_SUCCESS)
            die("找不到 OpenCL GPU（装了 img-bxm-dkms 吗？modprobe pvrsrvkm）");
        gin.ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e);
        gin.q   = clCreateCommandQueue(gin.ctx, dev, 0, &e);
        cl_program pr = clCreateProgramWithSource(gin.ctx, 1, &CL_IN_SRC, NULL, &e);
        if (clBuildProgram(pr, 1, &dev, "-cl-fast-relaxed-math", NULL, NULL) != CL_SUCCESS) {
            char log[8192] = {0};
            clGetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
            fprintf(stderr, "GPU 抠块内核编译失败:\n%s\n", log);
            die("GPU 抠块内核编译失败");
        }
        cl_program pr2 = clCreateProgramWithSource(gin.ctx, 1, &CL_IN_NV12_SRC, NULL, &e);
        if (clBuildProgram(pr2, 1, &dev, "-cl-fast-relaxed-math", NULL, NULL) != CL_SUCCESS) {
            char log[8192] = {0};
            clGetProgramBuildInfo(pr2, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
            fprintf(stderr, "GPU NV12 抠块内核编译失败:\n%s\n", log);
            die("GPU NV12 抠块内核编译失败");
        }
        gin.k_nv12 = clCreateKernel(pr2, "nv12_to_tiles", &e);
        gin.k = clCreateKernel(pr, "frame_to_tiles", &e);   /* NV12 走 k_nv12，这个用不上 */
        cl_mem mG = clCreateBuffer(gin.ctx, CL_MEM_READ_ONLY, (size_t)iw * ih, NULL, &e);
        cl_mem mB = clCreateBuffer(gin.ctx, CL_MEM_READ_ONLY, (size_t)iw * ih, NULL, &e);
        cl_mem mR = clCreateBuffer(gin.ctx, CL_MEM_READ_ONLY, (size_t)iw * ih, NULL, &e);
        cl_mem mY = clCreateBuffer(gin.ctx, CL_MEM_READ_ONLY, (size_t)iw * ih, NULL, &e);
        cl_mem mUV = clCreateBuffer(gin.ctx, CL_MEM_READ_ONLY, (size_t)iw * ih / 2, NULL, &e);
        gin.mT = clCreateBuffer(gin.ctx, CL_MEM_WRITE_ONLY, all_tiles, NULL, &e);
        gin.mY = mY; gin.mUV = mUV;
        CL_CHECK_OR_DIE(clSetKernelArg(gin.k, 0, sizeof(cl_mem), &mG), "inG");
        CL_CHECK_OR_DIE(clSetKernelArg(gin.k, 1, sizeof(cl_mem), &mB), "inB");
        CL_CHECK_OR_DIE(clSetKernelArg(gin.k, 2, sizeof(cl_mem), &mR), "inR");
        CL_CHECK_OR_DIE(clSetKernelArg(gin.k, 3, sizeof(cl_mem), &gin.mT), "inT");
        CL_CHECK_OR_DIE(clSetKernelArg(gin.k, 4, sizeof(cl_int), &gin.iw), "inIW");
        CL_CHECK_OR_DIE(clSetKernelArg(gin.k, 5, sizeof(cl_int), &gin.ih), "inIH");
        CL_CHECK_OR_DIE(clSetKernelArg(gin.k, 6, sizeof(cl_int), &gin.tw), "inTW");
        CL_CHECK_OR_DIE(clSetKernelArg(gin.k, 7, sizeof(cl_int), &gin.th), "inTH");
        CL_CHECK_OR_DIE(clSetKernelArg(gin.k, 8, sizeof(cl_int), &gin.nx), "inNX");
        CL_CHECK_OR_DIE(clSetKernelArg(gin.k, 9, sizeof(cl_int), &gin.cw), "inCW");
        CL_CHECK_OR_DIE(clSetKernelArg(gin.k,10, sizeof(cl_int), &gin.ch), "inCH");
        CL_CHECK_OR_DIE(clSetKernelArg(gin.k,11, sizeof(cl_int), &gin.margin), "inMG");
        gin.mG = mG; gin.mB = mB; gin.mR = mR;
        {
            cl_int e2 = CL_SUCCESS;
            cl_mem a0 = in_nv12 ? mY : mG, a1 = in_nv12 ? mUV : mB, a2 = in_nv12 ? gin.mT : mR;
            e2 |= clSetKernelArg(gin.k_nv12, 0, sizeof(cl_mem), &a0);
            e2 |= clSetKernelArg(gin.k_nv12, 1, sizeof(cl_mem), &a1);
            e2 |= clSetKernelArg(gin.k_nv12, 2, sizeof(cl_mem), &gin.mT);
            /* ★NV12 内核签名是 (Yp, UV, T, iw, ih, tw, th, nx, cw, ch, mg) —— 11 个参数。
             *   别照抄 gbrp 那 12 个的下标，会整体错位一格，而且“看着能跑”。 */
            e2 |= clSetKernelArg(gin.k_nv12,  3, sizeof(cl_int), &gin.iw);
            e2 |= clSetKernelArg(gin.k_nv12,  4, sizeof(cl_int), &gin.ih);
            e2 |= clSetKernelArg(gin.k_nv12,  5, sizeof(cl_int), &gin.tw);
            e2 |= clSetKernelArg(gin.k_nv12,  6, sizeof(cl_int), &gin.th);
            e2 |= clSetKernelArg(gin.k_nv12,  7, sizeof(cl_int), &gin.nx);
            e2 |= clSetKernelArg(gin.k_nv12,  8, sizeof(cl_int), &gin.cw);
            e2 |= clSetKernelArg(gin.k_nv12,  9, sizeof(cl_int), &gin.ch);
            e2 |= clSetKernelArg(gin.k_nv12, 10, sizeof(cl_int), &gin.margin);
            CL_CHECK_OR_DIE(e2, "nv12 args");
        }
        for (int i = 0; i < 3; i++) {
            gin.src[i] = (unsigned char *)malloc(frame_in);
            if (!gin.src[i]) die("GPU 抠块 src malloc 失败");
        }
        for (int i = 0; i < 2; i++) {
            gin.tiles[i] = (unsigned char *)malloc(all_tiles);
            if (!gin.tiles[i]) die("GPU 抠块 tiles malloc 失败");
        }
        sem_init(&gin.free_slots, 0, 3);
        sem_init(&gin.in_full, 0, 0);
        sem_init(&gin.tiles_ready, 0, 0);
        pthread_create(&gin_tid, NULL, gpu_in_thread, &gin);
        fprintf(stderr, "srpipe: GPU 抠块已启用（与 NPU 并行，瓦片 %zu KB/帧）\n",
                all_tiles / 1024);
    }

    awnn_init();
    Awnn_Context_t *ctx = awnn_create(nbg);
    if (!ctx) die("awnn_create 失败");

    /* ★★ 绕开 awnn_run 的浮点反量化。
     *
     * awnn_run 对 uint8 输出干的是：
     *     for (j...) fp_data[j] = quantize_maps[i][*(data + j)];
     * 每块 276 万个 float【写出去】（11MB），16 块就是每帧 177MB 的浮点写 ——
     * 全在 CPU 上、全落在我原先标的"NPU 690ms"里。然后拼帧又把这 177MB 读回来
     * 转字节。我们最终只要 0..255 的字节，这趟浮点往返纯属白烧。
     *
     * 改成：自己调 vip_run_network，直接拿量化后的 uint8，
     * 用一张 256 项【字节】LUT 一步到位。数值与浮点路径逐字节等价
     * （同一张 quantize_maps，同样的 (int)(v+0.5f) 再钳制）。
     */
    int raw_out = 0;
    unsigned char lut[256];
    {
        int df = ctx->output_params[0].vip_param.data_format;
        if (!force_float && ctx->output_count == 1 && df == VIP_BUFFER_FORMAT_UINT8 && ctx->quantize_maps[0]) {
            for (int j = 0; j < 256; j++) {
                int iv = (int)(ctx->quantize_maps[0][j] * 255.0f + 0.5f);
                lut[j] = (unsigned char)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
            }
            raw_out = 1;
            fprintf(stderr, "srpipe: NPU 输出是量化 uint8 -> 走字节 LUT，"
                            "省掉每帧 177MB 的浮点反量化 + 回读\n");
        } else {
            fprintf(stderr, "srpipe: %s(data_format=%d, outputs=%u) -> 走 awnn 的浮点路径\n",
                    force_float ? "被 --float-out 强制" : "NPU 输出不是量化 uint8",
                    df, ctx->output_count);
        }
    }

    void *in_buffers[1] = { tiles };

    /* ---- 可选：GPU 转换线程（双缓冲）---- */
    gpu_ctx_t gpu; gpu.stop = 0; gpu.t_conv = 0; gpu.frames = 0;
    pthread_t gpu_tid = 0;
    if (use_gpu) {
        memset(&gpu, 0, sizeof(gpu));
        gpu.data_fd = data_fd;
        gpu.w = ow; gpu.h = oh; gpu.sharp = (int)(sharpen * 256.0f + 0.5f);
        cl_int e; cl_platform_id plat; cl_device_id dev; cl_uint nd;
        if (clGetPlatformIDs(1, &plat, &nd) != CL_SUCCESS ||
            clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, &nd) != CL_SUCCESS)
            die("找不到 OpenCL GPU（装了 img-bxm-dkms 吗？modprobe pvrsrvkm）");
        gpu.ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e);
        gpu.q   = clCreateCommandQueue(gpu.ctx, dev, 0, &e);
        cl_program pr = clCreateProgramWithSource(gpu.ctx, 1, &CL_SRC, NULL, &e);
        if (clBuildProgram(pr, 1, &dev, "-cl-fast-relaxed-math", NULL, NULL) != CL_SUCCESS) {
            char log[8192] = {0};
            clGetProgramBuildInfo(pr, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
            fprintf(stderr, "GPU 内核编译失败:\n%s\n", log);
            die("GPU 内核编译失败");
        }
        gpu.k    = clCreateKernel(pr, "gbrp2nv12_sharp", &e);
        gpu.k_sh = clCreateKernel(pr, "unsharp_y", &e);
        for (int i = 0; i < 2; i++) {
            gpu.rgb[i] = (unsigned char *)malloc(rgb_bytes);
            gpu.yuv[i] = (unsigned char *)malloc(frame_out);
            if (!gpu.rgb[i] || !gpu.yuv[i]) die("GPU 双缓冲 malloc 失败");
        }
        gpu.mG  = clCreateBuffer(gpu.ctx, CL_MEM_READ_ONLY,  rgb_bytes, NULL, &e);
        gpu.mB  = clCreateBuffer(gpu.ctx, CL_MEM_READ_ONLY,  rgb_bytes, NULL, &e);
        gpu.mR  = clCreateBuffer(gpu.ctx, CL_MEM_READ_ONLY,  rgb_bytes, NULL, &e);
        gpu.mY  = clCreateBuffer(gpu.ctx, CL_MEM_READ_WRITE, frame_out, NULL, &e);
        gpu.mYT = clCreateBuffer(gpu.ctx, CL_MEM_READ_WRITE, (size_t)ow * oh, NULL, &e);
        gpu.mUV = clCreateBuffer(gpu.ctx, CL_MEM_WRITE_ONLY, frame_out, NULL, &e);
        CL_CHECK_OR_DIE(clSetKernelArg(gpu.k, 0, sizeof(cl_mem), &gpu.mG), "argG");
        CL_CHECK_OR_DIE(clSetKernelArg(gpu.k, 1, sizeof(cl_mem), &gpu.mB), "argB");
        CL_CHECK_OR_DIE(clSetKernelArg(gpu.k, 2, sizeof(cl_mem), &gpu.mR), "argR");
        CL_CHECK_OR_DIE(clSetKernelArg(gpu.k, 3, sizeof(cl_mem), &gpu.mY), "argY");
        CL_CHECK_OR_DIE(clSetKernelArg(gpu.k, 4, sizeof(cl_mem), &gpu.mUV), "argUV");
        CL_CHECK_OR_DIE(clSetKernelArg(gpu.k, 5, sizeof(cl_mem), &gpu.mYT), "argYT");
        CL_CHECK_OR_DIE(clSetKernelArg(gpu.k, 6, sizeof(cl_int), &gpu.w), "argW");
        CL_CHECK_OR_DIE(clSetKernelArg(gpu.k, 7, sizeof(cl_int), &gpu.h), "argH");
        CL_CHECK_OR_DIE(clSetKernelArg(gpu.k, 8, sizeof(cl_int), &gpu.sharp), "argSH");
        CL_CHECK_OR_DIE(clSetKernelArg(gpu.k_sh, 0, sizeof(cl_mem), &gpu.mY),  "shY");
        CL_CHECK_OR_DIE(clSetKernelArg(gpu.k_sh, 1, sizeof(cl_mem), &gpu.mYT), "shYT");
        CL_CHECK_OR_DIE(clSetKernelArg(gpu.k_sh, 2, sizeof(cl_int), &gpu.w),   "shW");
        CL_CHECK_OR_DIE(clSetKernelArg(gpu.k_sh, 3, sizeof(cl_int), &gpu.h),   "shH");
        CL_CHECK_OR_DIE(clSetKernelArg(gpu.k_sh, 4, sizeof(cl_int), &gpu.sharp), "shS");
        sem_init(&gpu.full, 0, 0);
        sem_init(&gpu.empty, 0, 2);
        gpu.idx_in = 0; gpu.idx_out = 0;
        pthread_create(&gpu_tid, NULL, gpu_thread, &gpu);
        fprintf(stderr, "srpipe: GPU 转换已启用（双缓冲，与 NPU 并行）\n");
    }

    long frame = 0;
    for (;;) {
        /* ★只处理 N 帧就正常收工（--frames）。流式流水线靠它把一条长管道
         *   切成编码器能吃的段：解码器一直活着往管道里吐，每个段让 srpipe
         *   读够 N 帧就退出，剩下的字节留在管道里给下一段。
         *   【不能】在这里多读一个字节 —— 管道里是连续帧，多读就错位了。 */
        if (max_frames > 0 && frame >= max_frames) break;
        double _t0=_ms();
        /* --- 读一帧 + 抠成瓦片 ---
         * GPU 路径：管道【直接读进】抠块线程的输入槽，主线程一个字节都不搬，
         *           然后等它的瓦片（只留一帧在飞）。
         * CPU 路径：读进 fin 再自己 gather。
         * 两条路产出完全相同布局的 tiles，所以能逐字节比对（tools/gputiletest.sh）。 */
        unsigned char *tb = tiles;
        if (use_gpu_in) {
            sem_wait(&gin.free_slots);
            gin.slot_in = (int)(frame % 3);
            if (read_full(STDIN_FILENO, gin.src[gin.slot_in], frame_in) <= 0) {
                gin.stop = 1;                 /* 叫停抠块线程，别让它挂在 in_full 上 */
                sem_post(&gin.in_full);
                break;
            }
            t_read += _ms()-_t0; _t0=_ms();
            gin.posted++;
            sem_post(&gin.in_full);
            sem_wait(&gin.tiles_ready);
            tb = gin.tiles[gin.slot_out];
        } else {
            if (read_full(STDIN_FILENO, fin, frame_in) <= 0) break;
            t_read += _ms()-_t0; _t0=_ms();
            if (in_nv12) gather_nv12_cpu(fin, tiles, &gin);
            else         gather_cpu(fin, tiles, &gin);
        }
        t_tile += _ms()-_t0; _t0=_ms();

        /* 调试/验证用：把这一帧的瓦片组原样吐出来，两条路（CPU / GPU）比对 */
        if (dump_tiles) {
            FILE *df = fopen(dump_tiles, "wb");
            if (!df) die("打不开 --dump-tiles 的文件");
            fwrite(tb, 1, all_tiles, df);
            fclose(df);
            fprintf(stderr, "DUMPED_TILES %s %zu bytes\n", dump_tiles, all_tiles);
            _exit(0);
        }

        for (int ty = 0; ty < ny; ty++) {
            for (int tx = 0; tx < nx; tx++) {
                /* 瓦片已经在 tb 里排好了，这里只是换个指针 —— 一次拷贝都没有 */
                in_buffers[0] = tb + (size_t)(ty * nx + tx) * tile_in;
                double _t1 = _ms();
                awnn_set_input_buffers(ctx, in_buffers);
                int lim = scw;
                if (tx * scw + lim > ow) lim = ow - tx * scw;
                if (lim < 0) lim = 0;

                if (raw_out) {
                    /* ---- 快路：直接拿量化 uint8 + 字节 LUT ---- */
                    if (vip_run_network(ctx->network) != VIP_SUCCESS) die("vip_run_network 失败");
                    t_npu += _ms()-_t1; _t1=_ms();
                    vip_buffer ob = ctx->output_buffers[0];
                    if (vip_flush_buffer(ob, VIP_BUFFER_OPER_TYPE_INVALIDATE) != VIP_SUCCESS)
                        die("输出 buffer 失效化失败");
                    const unsigned char *qm = (const unsigned char *)vip_map_buffer(ob);
                    if (!qm) die("vip_map_buffer 失败");
                    /* ★必须先整块 memcpy 出来：vip_map_buffer 给的通常是 uncached 映射，
                     *   逐字节查表读它会退化成每字节一次设备内存访问 —— 实测反而慢 5 倍
                     *   （84s vs 16s / 20 帧）。awnn 自己也是先 memcpy 再处理的。 */
                    memcpy(qbuf, qm, tile_out_bytes);
                    vip_unmap_buffer(ob);
                    const unsigned char *q = qbuf;
                    for (int c = 0; c < 3; c++) {
                        const unsigned char *sp = q + (size_t)c * stw * sth;
                        unsigned char *dstf = fout + (size_t)c * ow * oh;
                        for (int y = 0; y < sch; y++) {
                            int oy = ty * sch + y;
                            if (oy >= oh) break;
                            unsigned char *row = dstf + (size_t)oy * ow + tx * scw;
                            const unsigned char *srow = sp + (size_t)(y + margin * scale) * stw
                                                           + margin * scale;
                            for (int x = 0; x < lim; x++) row[x] = lut[srow[x]];
                        }
                    }
                } else {
                    /* ---- 慢路：awnn 的浮点输出 ---- */
                    awnn_run(ctx);
                    float **out = awnn_get_output_buffers(ctx);
                    t_npu += _ms()-_t1; _t1=_ms();
                    if (!out || !out[0]) die("拿不到输出 buffer");
                    const float *f = out[0];
                    for (int c = 0; c < 3; c++) {
                        const float *src = f + (size_t)c * stw * sth;
                        unsigned char *dstf = fout + (size_t)c * ow * oh;
                        for (int y = 0; y < sch; y++) {
                            int oy = ty * sch + y;
                            if (oy >= oh) break;
                            unsigned char *row = dstf + (size_t)oy * ow + tx * scw;
                            const float *srow = src + (size_t)(y + margin * scale) * stw
                                                    + margin * scale;
                            for (int x = 0; x < lim; x++) {
                                float v = srow[x] * 255.0f;
                                int iv = (int)(v + 0.5f);
                                if (iv < 0) iv = 0; else if (iv > 255) iv = 255;
                                row[x] = (unsigned char)iv;
                            }
                        }
                    }
                }
                /* ★按块计时（原来 _t0 在拼帧前才重置，t_tile 会把上一块的拼帧也算进去）*/
                t_asm += _ms()-_t1;
            }
        }

        _t0 = _ms();
        if (use_gpu) {
            /* 把刚拼好的整帧交给 GPU 线程，自己不转 —— NPU 立刻去啃下一帧 */
            sem_wait(&gpu.empty);
            memcpy(gpu.rgb[gpu.idx_in], fout, rgb_bytes);
            gpu.idx_in ^= 1;
            gpu.posted++;
            sem_post(&gpu.full);
            t_yuv += _ms()-_t0;
        } else if (out_nv12) {
            /* 整帧平面 -> NV12（BT.601 有限范围，16..235 / 128 中心）。
             *
             * ★平面顺序是 G, B, R —— 不是 R, G, B！
             * 这是从数据里回归出来的：拿 ffmpeg 的 gbrp->nv12 当参考做最小二乘，
             * 拟合系数正好是 BT.601 的系数按 (R,G,B)->(G,B,R) 置换。
             * 也正因为 NPU 的输出顺序恰好就是 gbrp 的定义，整条流水线才对得上。
             * 之前按 R,G,B 读，往返误差 7.18；改对之后应降到 <1。
             *
             * 色度 2x2 平均必须在整帧而不是每块里做 —— 2x2 块会跨块边界。 */
            const unsigned char *G = fout, *B = fout + (size_t)ow * oh,
                                *R = fout + (size_t)2 * ow * oh;
            unsigned char *Yp = yuv, *UV = yuv + (size_t)ow * oh;
            for (int y = 0; y < oh; y++) {
                unsigned char *yp = Yp + (size_t)y * ow;
                const unsigned char *rp = R + (size_t)y * ow,
                                    *gp = G + (size_t)y * ow,
                                    *bp = B + (size_t)y * ow;
                for (int x = 0; x < ow; x++) {
                    int r = rp[x], g = gp[x], b = bp[x];
                    /* BT.601 limited: 0.2568/0.5041/0.0979, 16..235 */
                    int v = ((16829 * r + 33039 * g + 6416 * b) >> 16) + 16;
                    if (v < 16) v = 16; else if (v > 235) v = 235;
                    yp[x] = (unsigned char)v;
                }
            }
            for (int y = 0; y < oh; y += 2) {
                unsigned char *uvp = UV + (size_t)(y / 2) * ow;
                for (int x = 0; x < ow; x += 2) {
                    int sr = 0, sg = 0, sb = 0;
                    for (int dy = 0; dy < 2; dy++) {
                        int yy = y + dy; if (yy >= oh) yy = oh - 1;
                        const unsigned char *rp = R + (size_t)yy * ow,
                                            *gp = G + (size_t)yy * ow,
                                            *bp = B + (size_t)yy * ow;
                        for (int dx = 0; dx < 2; dx++) {
                            int xx = x + dx; if (xx >= ow) xx = ow - 1;
                            sr += rp[xx]; sg += gp[xx]; sb += bp[xx];
                        }
                    }
                    int r = sr >> 2, g = sg >> 2, b = sb >> 2;
                    /* BT.601 limited 色差: -0.1482/-0.2914/+0.4392 与 +0.4392/-0.3678/-0.0714 */
                    int cb = ((-9711 * r - 19098 * g + 28784 * b) >> 16) + 128;
                    int cr = (( 28784 * r - 24103 * g - 4681 * b) >> 16) + 128;
                    if (cb < 16) cb = 16; else if (cb > 240) cb = 240;
                    if (cr < 16) cr = 16; else if (cr > 240) cr = 240;
                    uvp[x]     = (unsigned char)cb;
                    uvp[x + 1] = (unsigned char)cr;
                }
            }
            /* 亮度锐化（3x3 unsharp）。放在这里而不是用 ffmpeg 的 cas：
             * 新流水线里 ffmpeg 已经不参与后期了（缩放交给 VE），
             * 而把一整帧再交给 ffmpeg 只为锐化不划算。
             * 边缘一圈不处理 —— 省掉边界钳制，且边界本来也不该被锐化。 */
            t_yuv += _ms()-_t0; _t0=_ms();
            if (sharpen > 0.0f) {
                unsigned char *tmp = (unsigned char *)malloc((size_t)ow * oh);
                if (tmp) {
                    memcpy(tmp, Yp, (size_t)ow * oh);
                    int a = (int)(sharpen * 256.0f + 0.5f);
                    for (int y = 1; y < oh - 1; y++) {
                        unsigned char *yp = Yp + (size_t)y * ow;
                        const unsigned char *tp = tmp + (size_t)y * ow;
                        for (int x = 1; x < ow - 1; x++) {
                            int s9 = tp[x-ow-1] + tp[x-ow] + tp[x-ow+1]
                                   + tp[x-1]    + tp[x]    + tp[x+1]
                                   + tp[x+ow-1] + tp[x+ow] + tp[x+ow+1];
                            int v = tp[x] + (((tp[x] - s9 / 9) * a) >> 8);
                            if (v < 16) v = 16; else if (v > 235) v = 235;
                            yp[x] = (unsigned char)v;
                        }
                    }
                    free(tmp);
                }
            }
            t_sh += _ms()-_t0; _t0=_ms();
            write_full(data_fd, yuv, frame_out);
            t_wr += _ms()-_t0;
        } else {
            write_full(data_fd, fout, rgb_bytes);
        }
        fprintf(stderr, "FRAME %ld\n", frame);
        frame++;
    }

    if (use_gpu_in) {
        while (gin.done < gin.posted) usleep(2000);
        gin.stop = 1;
        sem_post(&gin.in_full);
        pthread_join(gin_tid, NULL);
    }
    if (use_gpu) {
        /* ★必须先排空再停：直接设 stop 的话，线程拿到令牌就 break，
         *   正在排队的那一帧（最后一帧）会被丢掉。
         *   frames 在写完一帧之后才 ++，所以 frames==posted 表示全写完了。 */
        while (gpu.frames < gpu.posted) usleep(2000);
        gpu.stop = 1;
        sem_post(&gpu.full);            /* 唤醒它去看到 stop */
        pthread_join(gpu_tid, NULL);
    }
    awnn_destroy(ctx);
    awnn_uninit();
    close(data_fd);
    if (use_gpu_in)
        fprintf(stderr, "GPU 抠块: %ld 帧，合计 %.0f ms => %.2f ms/帧（与 NPU 并行）\n",
                gin.done, gin.t_gpu, gin.done ? gin.t_gpu / gin.done : 0);
    if (use_gpu)
        fprintf(stderr, "GPU: %ld 帧，转换合计 %.0f ms => %.2f ms/帧\n",
                gpu.frames, gpu.t_conv, gpu.frames ? gpu.t_conv / gpu.frames : 0);
    if (profile && frame > 0) {
        double tot=t_read+t_tile+t_npu+t_asm+t_yuv+t_sh+t_wr;
        fprintf(stderr, "PROFILE %ld 帧  每帧 ms: 读入%.1f %s%.1f NPU(含去量化)%.1f "
                        "拼帧%.1f RGB->NV12 %.1f 锐化%.1f 写出%.1f | 合计%.1f\n",
                frame, t_read/frame, use_gpu_in ? "等GPU抠块" : "抠块", t_tile/frame,
                t_npu/frame, t_asm/frame, t_yuv/frame, t_sh/frame, t_wr/frame, tot/frame);
    }
    fprintf(stderr, "DONE %ld\n", frame);
    return 0;
}
