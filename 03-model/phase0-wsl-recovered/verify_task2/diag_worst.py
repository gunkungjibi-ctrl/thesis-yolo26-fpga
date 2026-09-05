import sys, glob
import numpy as np, torch
sys.path.insert(0, "/workspace/thesis/phase0")
from quantize_yolo_pytorch import build_float_model, TARGET, IMGSZ
import cv2
def sig(z): return 1.0/(1.0+np.exp(-z))
img=sorted(glob.glob("/workspace/thesis/phase0/calib_images/*"))[0]
im=cv2.imread(img); im=cv2.resize(im,(IMGSZ,IMGSZ))
im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
x=torch.from_numpy(np.ascontiguousarray(np.transpose(im,(2,0,1)))).unsqueeze(0)
m=build_float_model("/workspace/thesis/phase0/yolov8n.pt").eval()
with torch.no_grad(): g=m(x)
from pytorch_nndct.apis import torch_quantizer
ex=torch.randn(1,3,IMGSZ,IMGSZ)
q=torch_quantizer(quant_mode="test",module=m,input_args=(ex,),device=torch.device("cpu"),target=TARGET).quant_model
with torch.no_grad(): qo=q(x)
gg=g[0].cpu().numpy()[0]; qq=qo[0].cpu().numpy()[0]   # OUT[0] (144,80,80)
cls_g=gg[64:]; cls_q=qq[64:]                          # (80,80,80)
d=np.abs(sig(cls_g)-sig(cls_q))
idx=np.unravel_index(np.argmax(d), d.shape)
print(f"worst class pixel idx={idx}")
print(f"  float logit={cls_g[idx]:.3f} -> sigmoid={sig(cls_g[idx]):.4f}")
print(f"  quant logit={cls_q[idx]:.3f} -> sigmoid={sig(cls_q[idx]):.4f}")
print(f"global stats OUT[0]: float[min={gg.min():.1f} max={gg.max():.1f}] quant[min={qq.min():.1f} max={qq.max():.1f}]")
print(f"  #class logits>5 (prob~1): float={np.sum(cls_g>5)} quant={np.sum(cls_q>5)}")
