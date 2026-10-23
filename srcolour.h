/* Source-faithful RGB interpolation, shared integer arithmetic for CPU/OpenCL.
 * Catmull-Rom bicubic, exact 1/8 phases for the supported native x2/x4 models. */
#ifndef SR_COLOUR_H
#define SR_COLOUR_H
static int sr_clip(int x,int lo,int hi) { return x<lo?lo:(x>hi?hi:x); }
static const int sr_cubic[32]={0,1024,0,0,-49,987,93,-7,-72,888,232,-24,
    -75,745,399,-45,-64,576,576,-64,-45,399,745,-75,-24,232,888,-72,-7,93,987,-49};
static void sr_source_rgb(const unsigned char *s,int x,int y,int w,int h,int in_h,
                          int nv,int in709,int rgb[3])
{
    int i=y*w+x,p=w*h;
    if(!nv) { rgb[0]=s[2*p+i];rgb[1]=s[i];rgb[2]=s[p+i];return; }
    int c=sr_clip((int)s[i]-16,0,255),uv=w*in_h+(y/2)*w+(x/2)*2;
    int d=(int)s[uv]-128,e=(int)s[uv+1]-128;
    rgb[0]=sr_clip((298*c+(in709?459:409)*e+128)>>8,0,255);
    rgb[1]=sr_clip((298*c-(in709?55:100)*d-(in709?136:208)*e+128)>>8,0,255);
    rgb[2]=sr_clip((298*c+(in709?541:516)*d+128)>>8,0,255);
}
static void sr_original_rgb(const unsigned char *s,int x,int y,int w,int h,int in_h,
                            int nv,int in709,int scale,int rgb[3])
{
    int xx=sr_clip((2*x+1)*128/scale-128,0,(w-1)*256);
    int yy=sr_clip((2*y+1)*128/scale-128,0,(h-1)*256);
    const int *wx=sr_cubic+((xx&255)>>5)*4,*wy=sr_cubic+((yy&255)>>5)*4;
    int sum[3]={0,0,0};
    for(int j=0;j<4;j++) {
        int horizontal[3]={0,0,0};
        for(int i=0;i<4;i++) {
            int v[3];sr_source_rgb(s,sr_clip((xx>>8)+i-1,0,w-1),
                sr_clip((yy>>8)+j-1,0,h-1),w,h,in_h,nv,in709,v);
            for(int c=0;c<3;c++) horizontal[c]+=v[c]*wx[i];
        }
        for(int c=0;c<3;c++) sum[c]+=((horizontal[c]+8)>>4)*wy[j];
    }
    for(int c=0;c<3;c++) rgb[c]=sr_clip((sum[c]+32768)>>16,0,255);
}
static void sr_mix_cpu(unsigned char *out,const unsigned char *s,int w,int h,int in_h,
                       int nv,int in709,int scale,int strength)
{
    int ow=w*scale,oh=h*scale;size_t p=(size_t)ow*oh;
    for(int y=0;y<oh;y++) for(int x=0;x<ow;x++) {
        int rgb[3];sr_original_rgb(s,x,y,w,h,in_h,nv,in709,scale,rgb);
        for(int c=0;c<3;c++) {
            unsigned char *v=out+(size_t)((c+2)%3)*p+(size_t)y*ow+x;
            *v=(unsigned char)((*v*strength+rgb[c]*(256-strength)+128)>>8);
        }
    }
}
static const char *CL_COLOUR_HELPERS=
"__constant int sr_cubic[32]={0,1024,0,0,-49,987,93,-7,-72,888,232,-24,\n"
"-75,745,399,-45,-64,576,576,-64,-45,399,745,-75,-24,232,888,-72,-7,93,987,-49};\n"
"int3 sr_source_rgb(__global const uchar*s,int x,int y,int w,int h,int in_h){\n"
" int i=y*w+x;\n"
"#if INPUT_NV12\n"
" int c=max((int)s[i]-16,0),uv=w*in_h+(y/2)*w+(x/2)*2;\n"
" int d=(int)s[uv]-128,e=(int)s[uv+1]-128;\n"
"#if IN_BT709\n"
" return clamp((int3)((298*c+459*e+128)>>8,(298*c-55*d-136*e+128)>>8,(298*c+541*d+128)>>8),0,255);\n"
"#else\n"
" return clamp((int3)((298*c+409*e+128)>>8,(298*c-100*d-208*e+128)>>8,(298*c+516*d+128)>>8),0,255);\n"
"#endif\n"
"#else\n"
" int p=w*h;return (int3)(s[2*p+i],s[i],s[p+i]);\n"
"#endif\n"
"}\n"
"__kernel void source_horizontal(__global const uchar*s,__global short4*H,int iw,int ih,int in_h,int scale){\n"
" int x=get_global_id(0),y=get_global_id(1),w=iw*scale; if(x>=w||y>=ih)return;\n"
" int xx=clamp((2*x+1)*128/scale-128,0,(iw-1)*256),px=((xx&255)>>5)*4;int3 sum=(int3)(0);\n"
" for(int i=0;i<4;i++)sum+=sr_source_rgb(s,clamp((xx>>8)+i-1,0,iw-1),y,iw,ih,in_h)*sr_cubic[px+i];\n"
" sum=(sum+(int3)(8))>>4;H[y*w+x]=(short4)(sum.x,sum.y,sum.z,0);\n"
"}\n"
"__kernel void source_mix(__global uchar*G,__global uchar*B,__global uchar*R,\n"
" __global const short4*H,int iw,int ih,int in_h,int scale,int strength){\n"
" int x=get_global_id(0),y=get_global_id(1),w=iw*scale;\n"
" if(x>=w||y>=ih*scale)return;int i=y*w+x;\n"
" int yy=clamp((2*y+1)*128/scale-128,0,(ih-1)*256),py=((yy&255)>>5)*4;int3 sum=(int3)(0);\n"
" for(int j=0;j<4;j++)sum+=convert_int3(H[clamp((yy>>8)+j-1,0,ih-1)*w+x].xyz)*sr_cubic[py+j];\n"
" int3 base=clamp((sum+(int3)(32768))>>16,0,255);\n"
" R[i]=(uchar)(((int)R[i]*strength+base.x*(256-strength)+128)>>8);\n"
" G[i]=(uchar)(((int)G[i]*strength+base.y*(256-strength)+128)>>8);\n"
" B[i]=(uchar)(((int)B[i]*strength+base.z*(256-strength)+128)>>8);\n"
"}\n";
#endif
