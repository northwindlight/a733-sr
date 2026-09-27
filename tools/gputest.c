/*
 * gputest —— 量 PowerVR 做 GBRP(平面) -> NV12 转换到底多快
 *
 * 目的：决定"把色彩转换交给 GPU"这条路值不值得走。
 * 反面教材：树莓派 V3D 上同类活比 CPU 慢 64 倍。
 *
 * 内核干的事和 srpipe 里那段 C 完全一样：
 *   每个 work-item 处理一个 2x2 块 —— 读 12 字节(G,B,R 各 4)，
 *   写 4 个 Y 和 1 对 UV。
 *
 * 编译：gcc -O2 -o gputest gputest.c -lPVROCL -lm
 * 跑：  ./gputest [宽 高 次数]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>

#define CL_CHECK(e, what) do { if ((e) != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL 错误 %d @ %s\n", (int)(e), what); exit(1); } } while (0)

static const char *SRC =
"__kernel void gbrp2nv12(__global const uchar *G, __global const uchar *B,\n"
"                        __global const uchar *R, __global uchar *Y,\n"
"                        __global uchar *UV, const int w, const int h)\n"
"{\n"
"    int bx = get_global_id(0) * 2;\n"          /* 每个 item 一个 2x2 块 */
"    int by = get_global_id(1) * 2;\n"
"    if (bx >= w || by >= h) return;\n"
"    int sr = 0, sg = 0, sb = 0, n = 0;\n"
"    for (int dy = 0; dy < 2; dy++) {\n"
"        int yy = by + dy; if (yy >= h) yy = h - 1;\n"
"        for (int dx = 0; dx < 2; dx++) {\n"
"            int xx = bx + dx; if (xx >= w) xx = w - 1;\n"
"            int i = yy * w + xx;\n"
"            int r = R[i], g = G[i], b = B[i];\n"
"            sr += r; sg += g; sb += b; n++;\n"
"            int v = ((16829 * r + 33039 * g + 6416 * b) >> 16) + 16;\n"
"            Y[i] = (uchar)(v < 16 ? 16 : (v > 235 ? 235 : v));\n"
"        }\n"
"    }\n"
"    int r = sr / n, g = sg / n, b = sb / n;\n"
"    int cb = ((-9711 * r - 19098 * g + 28784 * b) >> 16) + 128;\n"
"    int cr = (( 28784 * r - 24103 * g - 4681 * b) >> 16) + 128;\n"
"    cb = cb < 16 ? 16 : (cb > 240 ? 240 : cb);\n"
"    cr = cr < 16 ? 16 : (cr > 240 ? 240 : cr);\n"
"    int o = (by / 2) * w + bx;\n"
"    UV[o] = (uchar)cb; UV[o + 1] = (uchar)cr;\n"
"}\n";

static double now_ms(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(int argc, char **argv)
{
    int w = argc > 1 ? atoi(argv[1]) : 3840;
    int h = argc > 2 ? atoi(argv[2]) : 2160;
    int iters = argc > 3 ? atoi(argv[3]) : 20;

    cl_platform_id plat; cl_device_id dev; cl_uint n = 0;
    CL_CHECK(clGetPlatformIDs(1, &plat, &n), "getPlatformIDs");
    CL_CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, &n), "getDeviceIDs");
    char nm[128] = {0}; cl_uint cu = 0; cl_ulong gmem = 0;
    clGetDeviceInfo(dev, CL_DEVICE_NAME, 128, nm, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, NULL);
    printf("设备: %s  CU=%u  mem=%.0f MB\n", nm, cu, gmem / 1048576.0);

    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    CL_CHECK(err, "createContext");
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    CL_CHECK(err, "createCommandQueue");
    cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &err);
    CL_CHECK(err, "createProgram");
    err = clBuildProgram(prog, 1, &dev, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192] = {0};
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "编译内核失败:\n%s\n", log);
        return 1;
    }
    cl_kernel k = clCreateKernel(prog, "gbrp2nv12", &err);
    CL_CHECK(err, "createKernel");

    size_t px = (size_t)w * h;
    size_t rgb = px, yuv = px * 3 / 2;
    unsigned char *G = malloc(rgb), *B = malloc(rgb), *R = malloc(rgb);
    unsigned char *Y = malloc(yuv), *UV = malloc(yuv);
    for (size_t i = 0; i < px; i++) { G[i] = (i * 7) & 0xff; B[i] = (i * 3) & 0xff; R[i] = (i * 5) & 0xff; }
    printf("帧: %dx%d  RGB=%.1f MB  NV12=%.1f MB\n", w, h, rgb / 1048576.0, yuv / 1048576.0);

    cl_mem mG = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rgb, G, &err); CL_CHECK(err, "bufG");
    cl_mem mB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rgb, B, &err); CL_CHECK(err, "bufB");
    cl_mem mR = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rgb, R, &err); CL_CHECK(err, "bufR");
    cl_mem mY = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, yuv, NULL, &err); CL_CHECK(err, "bufY");
    cl_mem mUV = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, yuv, NULL, &err); CL_CHECK(err, "bufUV");

    CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem), &mG), "arg0");
    CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_mem), &mB), "arg1");
    CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem), &mR), "arg2");
    CL_CHECK(clSetKernelArg(k, 3, sizeof(cl_mem), &mY), "arg3");
    CL_CHECK(clSetKernelArg(k, 4, sizeof(cl_mem), &mUV), "arg4");
    CL_CHECK(clSetKernelArg(k, 5, sizeof(cl_int), &w), "arg5");
    CL_CHECK(clSetKernelArg(k, 6, sizeof(cl_int), &h), "arg6");

    size_t gsz[2] = { (size_t)((w + 1) / 2), (size_t)((h + 1) / 2) };
    size_t lsz[2] = { 16, 4 };
    lsz[0] = lsz[0] > gsz[0] ? gsz[0] : lsz[0];
    lsz[1] = lsz[1] > gsz[1] ? gsz[1] : lsz[1];

    /* 预热一次（首次含内核编译/加载） */
    CL_CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, gsz, lsz, 0, NULL, NULL), "kernel warm");
    CL_CHECK(clFinish(q), "finish warm");

    double t0 = now_ms();
    for (int i = 0; i < iters; i++)
        CL_CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, gsz, lsz, 0, NULL, NULL), "kernel");
    CL_CHECK(clFinish(q), "finish");
    double dt = now_ms() - t0;

    printf("GPU: %d 次 %.1f ms  => %.2f ms/帧  (%.1f 帧/秒)\n",
           iters, dt, dt / iters, iters * 1000.0 / dt);
    double bw = (rgb * 3 + yuv) * iters / (dt / 1000.0) / 1e9;   /* GB/s */
    printf("有效带宽: %.2f GB/s（读 3 平面 + 写 NV12）\n", bw);

    /* ---- CPU 做同一件事（同一份数据、同一时刻，直接对等比较）---- */
    {
        unsigned char *yc = malloc(yuv), *uvc = malloc(yuv);
        double c0 = now_ms();
        for (int it = 0; it < iters; it++) {
            for (int y = 0; y < h; y++) {
                unsigned char *yp = yc + (size_t)y * w;
                for (int x = 0; x < w; x++) {
                    size_t i = (size_t)y * w + x;
                    int v = ((16829 * R[i] + 33039 * G[i] + 6416 * B[i]) >> 16) + 16;
                    yp[x] = (unsigned char)(v < 16 ? 16 : (v > 235 ? 235 : v));
                }
            }
            for (int y = 0; y < h; y += 2)
                for (int x = 0; x < w; x += 2) {
                    int sr=0,sg=0,sb=0;
                    for (int dy=0;dy<2;dy++) for (int dx=0;dx<2;dx++){
                        size_t i=(size_t)(y+dy)*w+(x+dx); sr+=R[i]; sg+=G[i]; sb+=B[i]; }
                    int r=sr/4,g=sg/4,b=sb/4;
                    int cb=((-9711*r-19098*g+28784*b)>>16)+128;
                    int cr=((28784*r-24103*g-4681*b)>>16)+128;
                    size_t o=(size_t)(y/2)*w+x;
                    uvc[o]=(unsigned char)(cb<16?16:(cb>240?240:cb));
                    uvc[o+1]=(unsigned char)(cr<16?16:(cr>240?240:cr));
                }
        }
        double cdt = now_ms() - c0;
        printf("CPU: %d 次 %.1f ms  => %.2f ms/帧  (%.1f 帧/秒)\n",
               iters, cdt, cdt/iters, iters*1000.0/cdt);
        printf("  => GPU/CPU = %.2fx  %s\n", (dt/iters)/(cdt/iters),
               (dt < cdt) ? "（GPU 更快）" : "（★CPU 更快）");
        free(yc); free(uvc);
    }

    /* 正确性抽查：取几个点跟 CPU 比 */
    unsigned char *y2 = malloc(yuv);
    CL_CHECK(clEnqueueReadBuffer(q, mY, CL_TRUE, 0, yuv, y2, 0, NULL, NULL), "readback");
    int bad = 0;
    for (size_t i = 0; i < px; i += 9973) {
        int v = ((16829 * (int)R[i] + 33039 * (int)G[i] + 6416 * (int)B[i]) >> 16) + 16;
        if (v < 16) v = 16; if (v > 235) v = 235;
        if (abs(v - (int)y2[i]) > 1) bad++;
    }
    printf("正确性抽查: %s\n", bad == 0 ? "✓ 与 CPU 一致" : "✗ 有不一致");
    return 0;
}
