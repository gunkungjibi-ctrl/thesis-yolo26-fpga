#!/usr/bin/env python3
# M2-B2 re-verify — เทียบ float vs quant(test) ต่อ tensor + พิมพ์ logit range
# ใช้พิสูจน์ว่า fine-tune (LeakyReLU) หด logit range -> cos_sim กลับขึ้น > 0.99
# ต้องรัน AFTER calib (มี quantize_result/quant_info.json ที่ตรงกับ --weights)
#   python quantize_yolo_pytorch.py --weights <ft.pt> --quant_mode calib --subset_len 128 --batch_size 16
#   python verify_task2/verify_ft_compare.py --weights <ft.pt>
import os, sys, glob, argparse
import numpy as np, torch, cv2
sys.path.insert(0, "/workspace/thesis/phase0")
from quantize_yolo_pytorch import build_float_model, TARGET, IMGSZ


def cos_sim(a, b):
    a = a.flatten().astype(np.float64); b = b.flatten().astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--calib_dir", default="/workspace/thesis/phase0/calib_images")
    ap.add_argument("--img", default="")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--thresh", type=float, default=0.99)
    args = ap.parse_args()

    img = args.img or sorted(glob.glob(os.path.join(args.calib_dir, "*")))[0]
    im = cv2.imread(img); im = cv2.resize(im, (args.imgsz, args.imgsz))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(np.ascontiguousarray(np.transpose(im, (2, 0, 1)))).unsqueeze(0)

    model = build_float_model(args.weights).eval()
    with torch.no_grad():
        golden = model(x)

    from pytorch_nndct.apis import torch_quantizer
    ex = torch.randn(1, 3, args.imgsz, args.imgsz)
    q = torch_quantizer(quant_mode="test", module=model, input_args=(ex,),
                        device=torch.device("cpu"), target=TARGET).quant_model
    with torch.no_grad():
        quant = q(x)

    print(f"\n=== M2-B2 re-verify: weights={os.path.basename(args.weights)}  img={os.path.basename(img)} ===")
    all_ok = True
    for i, (g, qq) in enumerate(zip(golden, quant)):
        g = g.cpu().numpy(); qq = qq.cpu().numpy()
        hw = f"{g.shape[2]}x{g.shape[3]}"
        box_g, cls_g = g[:, :64], g[:, 64:]
        mae = float(np.max(np.abs(g - qq)))
        cs = cos_sim(g, qq)
        ok = cs > args.thresh; all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] OUT[{i}] {hw}  cos_sim={cs:.6f}  max_abs_err={mae:.3f}"
              f"  | logit range BOX=[{box_g.min():.1f},{box_g.max():.1f}] "
              f"CLASS=[{cls_g.min():.1f},{cls_g.max():.1f}]")
    print(f"=== {'PASS' if all_ok else 'FAIL'} (เกณฑ์ cos_sim > {args.thresh} ทุก tensor) ===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
