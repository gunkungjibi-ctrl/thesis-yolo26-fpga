import sys, glob
import numpy as np, torch
sys.path.insert(0, "/workspace/thesis/phase0")
from quantize_yolo_pytorch import build_float_model, TARGET, IMGSZ
import cv2
def cs(a,b):
    a=a.flatten().astype(np.float64); b=b.flatten().astype(np.float64)
    d=np.linalg.norm(a)*np.linalg.norm(b); return float(np.dot(a,b)/d) if d>0 else float("nan")
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
print("=== CLASS หลัง sigmoid (space จริงที่ counting ใช้) ===")
for i,(gg,qq) in enumerate(zip(g,qo)):
    gg=gg.cpu().numpy(); qq=qq.cpu().numpy()
    cg,cq=sig(gg[:,64:]),sig(qq[:,64:])
    md=np.max(np.abs(cg-cq))
    print(f"OUT[{i}] {gg.shape[2]}x{gg.shape[3]}  cos={cs(cg,cq):.6f}  max_abs_err={md:.5f}  (0..1 prob)")
