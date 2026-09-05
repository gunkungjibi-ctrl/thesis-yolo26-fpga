import sys, glob
import numpy as np, torch
sys.path.insert(0, "/workspace/thesis/phase0")
from quantize_yolo_pytorch import build_float_model, TARGET, IMGSZ
import cv2
def sig(z): return 1.0/(1.0+np.exp(-np.clip(z,-30,30)))
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
print("=== detection-level: #cell ที่ max class prob เกิน threshold ===")
for th in [0.25, 0.5]:
    print(f"-- thr={th} --")
    for i,(gg,qq) in enumerate(zip(g,qo)):
        gg=gg.cpu().numpy()[0]; qq=qq.cpu().numpy()[0]
        pg=sig(gg[64:]).max(0); pq=sig(qq[64:]).max(0)   # max over classes -> (H,W)
        ng=int((pg>th).sum()); nq=int((pq>th).sum())
        # นับ cell ที่ทั้งคู่ตรงกัน (agreement)
        agree=int(((pg>th)&(pq>th)).sum())
        print(f"  OUT[{i}] {gg.shape[1]}x{gg.shape[2]}  float={ng} quant={nq} agree={agree}")
