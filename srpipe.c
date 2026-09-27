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
 *   srpipe <nbg> <in_w> <in_h> <scale> <tile_w> <tile_h> [--margin M]
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

static void die(const char *m) { fprintf(stderr, "srpipe: %s\n", m); exit(1); }

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
    int margin = 0;
    for (int i = 7; i < argc; i++)
        if (!strcmp(argv[i], "--margin") && i + 1 < argc) margin = atoi(argv[++i]);
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
    const size_t frame_out = (size_t)ow * oh * 3;
    const size_t tile_in   = (size_t)tw * th * 3;
    const size_t tile_out  = (size_t)stw * sth * 3;

    unsigned char *fin   = (unsigned char *)malloc(frame_in);
    unsigned char *fout  = (unsigned char *)malloc(frame_out);
    unsigned char *tin   = (unsigned char *)malloc(tile_in);
    unsigned char *tout  = (unsigned char *)malloc(tile_out);
    if (!fin || !fout || !tin || !tout) die("malloc 失败");

    awnn_init();
    Awnn_Context_t *ctx = awnn_create(nbg);
    if (!ctx) die("awnn_create 失败");

    void *in_buffers[] = { tin };

    long frame = 0;
    for (;;) {
        if (read_full(STDIN_FILENO, fin, frame_in) <= 0) break;

        for (int ty = 0; ty < ny; ty++) {
            for (int tx = 0; tx < nx; tx++) {
                /* --- 抠块（含边界钳制，边缘块复制最后一行/列，不出黑边）--- */
                const int ox = tx * cw, oy = ty * ch;        /* core 在源里的原点 */
                for (int c = 0; c < 3; c++) {
                    const unsigned char *src = fin + (size_t)c * iw * ih;
                    unsigned char *dst = tin + (size_t)c * tw * th;
                    for (int y = 0; y < th; y++) {
                        int sy = oy - margin + y;
                        if (sy < 0) sy = 0; else if (sy >= ih) sy = ih - 1;
                        for (int x = 0; x < tw; x++) {
                            int sx = ox - margin + x;
                            if (sx < 0) sx = 0; else if (sx >= iw) sx = iw - 1;
                            dst[y * tw + x] = src[(size_t)sy * iw + sx];
                        }
                    }
                }

                awnn_set_input_buffers(ctx, in_buffers);
                awnn_run(ctx);
                float **out = awnn_get_output_buffers(ctx);
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

        write_full(data_fd, fout, frame_out);
        fprintf(stderr, "FRAME %ld\n", frame);
        frame++;
    }

    awnn_destroy(ctx);
    awnn_uninit();
    close(data_fd);
    fprintf(stderr, "DONE %ld\n", frame);
    return 0;
}
