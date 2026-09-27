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
 *          [--margin M] [--nv12] [--sharpen A]
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
#include <pthread.h>
#include <semaphore.h>
#include <CL/cl.h>

#define CL_CHECK_OR_DIE(e, what) do { if ((e) != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL 错误 %d @ %s\n", (int)(e), what); exit(1); } } while (0)

static void die(const char *m) { fprintf(stderr, "srpipe: %s\n", m); exit(1); }
static void write_full(int fd, const void *buf, size_t n);   /* 定义在后面 */

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
            "usage: %s <nbg> <in_w> <in_h> <scale> <tile_w> <tile_h>\n"
            "  读 stdin 的 gbrp 平面帧，输出 scale 倍大的 gbrp 平面帧\n", argv[0]);
        return 2;
    }
    const char *nbg = argv[1];
    int iw = atoi(argv[2]), ih = atoi(argv[3]), scale = atoi(argv[4]);
    int tw = atoi(argv[5]), th = atoi(argv[6]);
    int margin = 0, out_nv12 = 0;
    float sharpen = 0.0f;
    int profile = 0, use_gpu = 0;
    for (int i = 7; i < argc; i++) {
        if (!strcmp(argv[i], "--margin") && i + 1 < argc) margin = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nv12")) out_nv12 = 1;
        else if (!strcmp(argv[i], "--sharpen") && i + 1 < argc) sharpen = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--profile")) profile = 1;
        else if (!strcmp(argv[i], "--gpu")) use_gpu = 1;
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

    const size_t frame_in  = (size_t)iw * ih * 3;
    const size_t frame_out = out_nv12 ? (size_t)ow * oh * 3 / 2     /* NV12 = Y + 交织 CbCr(1/2) */
                                      : (size_t)ow * oh * 3;
    const size_t rgb_bytes = (size_t)ow * oh * 3;   /* 组帧始终是平面 RGB */
    const size_t tile_in   = (size_t)tw * th * 3;
    const size_t tile_out  = (size_t)stw * sth * 3;

    unsigned char *fin   = (unsigned char *)malloc(frame_in);
    unsigned char *fout  = (unsigned char *)malloc(rgb_bytes);   /* 平面 RGB 整帧 */
    unsigned char *yuv   = out_nv12 ? (unsigned char *)malloc(frame_out) : NULL;
    unsigned char *tin   = (unsigned char *)malloc(tile_in);
    unsigned char *tout  = (unsigned char *)malloc(tile_out);
    if (!fin || !fout || !tin || !tout) die("malloc 失败");

    awnn_init();
    Awnn_Context_t *ctx = awnn_create(nbg);
    if (!ctx) die("awnn_create 失败");

    void *in_buffers[] = { tin };

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
        double _t0=_ms();
        if (read_full(STDIN_FILENO, fin, frame_in) <= 0) break;
        t_read += _ms()-_t0; _t0=_ms();

        for (int ty = 0; ty < ny; ty++) {
            for (int tx = 0; tx < nx; tx++) {
                /* --- 抠块（含边界钳制，边缘块复制最后一行/列，不出黑边）--- */
                const int ox = tx * cw, oy = ty * ch;        /* core 在源里的原点 */
                const int x0 = ox - margin;                  /* 窗口左上角 */
                /* ★快慢两条路：内块横向完全不越界，每行就是一段连续内存，直接 memcpy。
                 *   原来逐字节 + 每像素两次比较 —— 实测 4.3MB 拷了 243ms（18MB/s），
                 *   占了整帧的 15%，比 NPU 之外任何一步都贵。 */
                const int fast_x = (x0 >= 0 && x0 + tw <= iw);
                for (int c = 0; c < 3; c++) {
                    const unsigned char *src = fin + (size_t)c * iw * ih;
                    unsigned char *dst = tin + (size_t)c * tw * th;
                    if (fast_x) {
                        for (int y = 0; y < th; y++) {
                            int sy = oy - margin + y;
                            if (sy < 0) sy = 0; else if (sy >= ih) sy = ih - 1;
                            memcpy(dst + (size_t)y * tw, src + (size_t)sy * iw + x0, tw);
                        }
                    } else {
                        for (int y = 0; y < th; y++) {
                            int sy = oy - margin + y;
                            if (sy < 0) sy = 0; else if (sy >= ih) sy = ih - 1;
                            const unsigned char *srow = src + (size_t)sy * iw;
                            unsigned char *drow = dst + (size_t)y * tw;
                            for (int x = 0; x < tw; x++) {
                                int sx = x0 + x;
                                if (sx < 0) sx = 0; else if (sx >= iw) sx = iw - 1;
                                drow[x] = srow[sx];
                            }
                        }
                    }
                }

                t_tile += _ms()-_t0; _t0=_ms();
                awnn_set_input_buffers(ctx, in_buffers);
                awnn_run(ctx);
                float **out = awnn_get_output_buffers(ctx);
                t_npu += _ms()-_t0; _t0=_ms();
                if (!out || !out[0]) die("拿不到输出 buffer");
                const float *f = out[0];

                /* --- 转 uint8 并放回整帧对应位置 --- */
                for (int c = 0; c < 3; c++) {
                    const float *src = f + (size_t)c * stw * sth;
                    unsigned char *dstf = fout + (size_t)c * ow * oh;
                    for (int y = 0; y < sch; y++) {
                        int oy = ty * sch + y;                 /* ★输出坐标也按 core 步进 */
                        if (oy >= oh) break;
                        unsigned char *row = dstf + (size_t)oy * ow;
                        /* 取块输出的中央有效区（去掉 margin 圈，输出是输入 scale 倍）*/
                        const float *srow = src + (size_t)(y + margin * scale) * stw
                                                + margin * scale;
                        for (int x = 0; x < scw; x++) {
                            int ox = tx * scw + x;
                            if (ox >= ow) break;
                            float v = srow[x] * 255.0f;
                            int iv = (int)(v + 0.5f);
                            if (iv < 0) iv = 0; else if (iv > 255) iv = 255;
                            row[ox] = (unsigned char)iv;
                        }
                    }
                }
            }
        }

        t_asm += _ms()-_t0; _t0=_ms();
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
    if (use_gpu)
        fprintf(stderr, "GPU: %ld 帧，转换合计 %.0f ms => %.2f ms/帧\n",
                gpu.frames, gpu.t_conv, gpu.frames ? gpu.t_conv / gpu.frames : 0);
    if (profile && frame > 0) {
        double tot=t_read+t_tile+t_npu+t_asm+t_yuv+t_sh+t_wr;
        fprintf(stderr, "PROFILE %ld 帧  每帧 ms: 读入%.1f 抠块%.1f NPU(含去量化)%.1f "
                        "拼帧%.1f RGB->NV12 %.1f 锐化%.1f 写出%.1f | 合计%.1f\n",
                frame, t_read/frame, t_tile/frame, t_npu/frame, t_asm/frame,
                t_yuv/frame, t_sh/frame, t_wr/frame, tot/frame);
    }
    fprintf(stderr, "DONE %ld\n", frame);
    return 0;
}
