#!/usr/bin/env python3
"""Hardware regression for source fidelity and BT.601/709 (native x2 model).
SR_SRPIPE=/path/srpipe SR_NBG=/path/model.nb SR_MARGIN=64 SR_INPUT_INT16=1 python3 tools/qualitycheck.py
"""
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np

binary=os.environ.get('SR_SRPIPE','/opt/sr/srpipe')
model=os.environ.get('SR_NBG','/opt/sr/models/animejanai_v3_x2.nb')
margin=os.environ.get('SR_MARGIN','16')
w,h,ah=64,40,48
rng=np.random.default_rng(20261023)
rgb=rng.integers(0,256,(h,w,3),dtype=np.uint8)
gbr=rgb.transpose(2,0,1)[[1,2,0]].tobytes()
nv=np.concatenate([rng.integers(16,236,w*ah,dtype=np.uint8),rng.integers(16,241,w*ah//2,dtype=np.uint8)]).tobytes()
base=[binary,model,str(w),str(h),'2','352','224','--margin',margin]
if os.environ.get('SR_INPUT_INT16')=='1': base.append('--input-int16')
def run(data,*args):
 p=subprocess.run(base+list(args),input=data,capture_output=True)
 if p.returncode: raise RuntimeError(p.stderr.decode(errors='replace'))
 return p.stdout

# Independent floating-point Catmull-Rom reference, no integer coefficient LUT.
def bicubic_reference(a):
 yy=np.clip((np.arange(h*2)+.5)/2-.5,0,h-1)
 xx=np.clip((np.arange(w*2)+.5)/2-.5,0,w-1)
 def coeff(f): return [-.5*f+f*f-.5*f**3,1-2.5*f*f+1.5*f**3,.5*f+2*f*f-1.5*f**3,-.5*f*f+.5*f**3]
 wx,wy=coeff(xx%1),coeff(yy%1);out=np.zeros((h*2,w*2,3))
 for j in range(4):
  for i in range(4):
   v=a[np.clip(yy.astype(int)+j-1,0,h-1)[:,None],np.clip(xx.astype(int)+i-1,0,w-1)[None,:]]
   out+=v*wy[j][:,None,None]*wx[i][None,:,None]
 return np.clip(np.floor(out+.5),0,255).astype(np.uint8)
zero=np.frombuffer(run(gbr,'--strength','0'),np.uint8).reshape(3,h*2,w*2)[[2,0,1]].transpose(1,2,0)
error=np.abs(zero.astype(int)-bicubic_reference(rgb).astype(int))
assert error.max()<=1, ('source interpolation',error.max())
print('PASS zero AI matches independent bicubic reference, max error',error.max(),flush=True)
def reference_yuv(planar,width,height,matrix):
 cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','gbrp','-s',f'{width}x{height}',
      '-i','-','-vf',f'scale=out_color_matrix={matrix}','-frames:v','1','-pix_fmt','yuv444p','-f','rawvideo','-']
 return np.frombuffer(subprocess.run(cmd,input=planar,check=True,capture_output=True).stdout,np.uint8).reshape(3,height,width)
zero_gbr=zero.transpose(2,0,1)[[1,2,0]].copy()
for matrix in ['bt601','bt709']:
 actual=np.frombuffer(run(gbr,'--strength','0','--nv12','--output-matrix',matrix),np.uint8)
 ref=reference_yuv(zero_gbr.tobytes(),w*2,h*2,matrix)
 yerr=np.abs(actual[:w*h*4].reshape(h*2,w*2).astype(int)-ref[0].astype(int))
 # Match the hardware path's RGB 2x2 average before chroma conversion.
 averaged=(zero_gbr.astype(np.uint16).reshape(3,h,2,w,2).sum(axis=(2,4))//4).astype(np.uint8)
 uvref=reference_yuv(averaged.tobytes(),w,h,matrix)
 uv=actual[w*h*4:].reshape(h,w,2).transpose(2,0,1)
 uverr=np.abs(uv.astype(int)-uvref[1:].astype(int))
 assert yerr.max()<=2 and uverr.max()<=2,(matrix,yerr.max(),uverr.max())
 print('PASS FFmpeg matrix reference',matrix,'Y/UV max error',yerr.max(),uverr.max(),flush=True)
for label,data,opts in [('GBR',gbr,[]),('NV12 padded',nv,['--in-nv12','--in-h',str(ah)])]:
 for im,om in [('bt601','bt601'),('bt709','bt709'),('bt601','bt709')]:
  for strength in ['0','0.35','1']:
   args=opts+['--nv12','--strength',strength,'--input-matrix',im,'--output-matrix',om,'--preserve-bottom','0.2']
   cpu=run(data,*args);gpu=run(data,*args,'--gpu')
   assert len(cpu)==w*h*4*3//2 and cpu==gpu,(label,im,om,strength)
  print('PASS CPU/GPU',label,im,om,'strength 0/35/100%',flush=True)
 with tempfile.TemporaryDirectory() as d:
  a,b=Path(d)/'cpu',Path(d)/'gpu'
  run(data,*opts,'--input-matrix','bt709','--dump-tiles',str(a))
  run(data,*opts,'--input-matrix','bt709','--gpu-in','--dump-tiles',str(b))
  assert a.read_bytes()==b.read_bytes()
for args in [('--strength','nan'),('--strength','-1'),('--strength','abc'),('--input-matrix','bad'),('--output-matrix','bad')]:
 p=subprocess.run(base+list(args),input=b'',capture_output=True)
 assert p.returncode!=0,args
print('PASS BT709 input tiling and invalid parameter rejection',flush=True)
