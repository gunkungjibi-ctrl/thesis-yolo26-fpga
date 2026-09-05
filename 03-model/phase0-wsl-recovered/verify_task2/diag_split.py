import sys, glob, os
import numpy as np, torch
sys.path.insert(0, "/workspace/thesis/phase0")
from quantize_yolo_pytorch import build_float_model, TARGET, IMGSZ
import cv2
def cs(a,b):
    a=a.flatten().astype(np.float64); b=b.flatten().astype(np.float64)
    d=np.linalg.norm(a)*np.linalg.norm(b); return float(np.dot(a,b)/d) if d>0 else float("nan")
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
for i,(gg,qq) in enumerate(zip(g,qo)):
    gg=gg.cpu().numpy(); qq=qq.cpu().numpy()
    box_g,box_q=gg[:,:64],qq[:,:64]; cls_g,cls_q=gg[:,64:],qq[:,64:]
    print(f"OUT[{i}] {gg.shape[2]}x{gg.shape[3]}")
    print(f"   BOX(0:64)   cos={cs(box_g,box_q):.6f}  max_err={np.max(np.abs(box_g-box_q)):.2f}  g_range=[{box_g.min():.1f},{box_g.max():.1f}]")
    print(f"   CLASS(64:)  cos={cs(cls_g,cls_q):.6f}  max_err={np.max(np.abs(cls_g-cls_q)):.2f}  g_range=[{cls_g.min():.1f},{cls_g.max():.1f}]")
