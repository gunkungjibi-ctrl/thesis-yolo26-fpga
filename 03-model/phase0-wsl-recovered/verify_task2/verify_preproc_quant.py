#!/usr/bin/env python3
# งานที่ 2 — verify (run INSIDE Vitis AI 3.0 container, env: vitis-ai-pytorch)
#   A: layout round-trip (CHW golden vs HWC .bin ที่ dpu_infer รับ)
#   B: float vs quant(test mode) fidelity per tensor
# reuse ของจริงจาก Phase 0 script (preprocessing/wrapper/quantizer เดิม)
import os, sys, glob, argparse
import numpy as np
import cv2
import torch

# import โครงเดิมจาก Phase 0 (อย่า re-implement preprocessing)
sys.path.insert(0, "/workspace/thesis/phase0")
from quantize_yolo_pytorch import (
    build_float_model, load_calib_batch, TARGET, IMGSZ,
)

def cos_sim(a, b):
    a = a.flatten().astype(np.float64); b = b.flatten().astype(np.float64)
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 0 else float("nan")

def preprocess_one(img_path, imgsz):
    """ตรงกับ load_calib_batch เป๊ะ: resize -> BGR2RGB -> /255 -> CHW"""
    im = cv2.imread(img_path)
    if im is None:
        raise FileNotFoundError(img_path)
    im = cv2.resize(im, (imgsz, imgsz))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(im, (2, 0, 1))           # CHW (torch/NCHW)
    return np.ascontiguousarray(chw)

def part_A_layout(img_path, imgsz, out_bin):
    print("\n=== ส่วน A: layout round-trip (จับ HWC/CHW silent bug) ===")
    chw = preprocess_one(img_path, imgsz)        # (3,640,640)
    hwc = np.ascontiguousarray(np.transpose(chw, (1, 2, 0)))  # (640,640,3) HWC
    hwc.astype(np.float32).tofile(out_bin)       # นี่คือไฟล์ที่ dpu_infer รับ
    print(f"[A] เขียน {out_bin}  shape(HWC)={hwc.shape}  bytes={hwc.nbytes}")

    # อ่านกลับเหมือน dpu_infer แล้ว permute กลับเป็น CHW -> ต้อง bit-identical
    back = np.fromfile(out_bin, dtype=np.float32).reshape(imgsz, imgsz, 3)
    back_chw = np.transpose(back, (2, 0, 1))
    max_abs = float(np.max(np.abs(back_chw - chw)))
    identical = bool(np.array_equal(back_chw, chw))
    print(f"[A] round-trip max_abs_diff={max_abs:.3e}  bit_identical={identical}")
    if not identical:
        print("[A] FAIL: .bin ที่ dpu_infer รับ ไม่ตรงภาพต้นทาง -> layout ผิด")
        return False, chw
    print("[A] PASS: input.bin (HWC) = ภาพเดียวกับ golden (CHW) แค่ permute")
    return True, chw

def part_B_fidelity(weights, chw, imgsz):
    print("\n=== ส่วน B: float vs quant(test) fidelity (proxy DPU accuracy) ===")
    from pytorch_nndct.apis import torch_quantizer
    device = torch.device("cpu")
    x = torch.from_numpy(chw).unsqueeze(0).to(device)   # (1,3,640,640) NCHW

    model = build_float_model(weights).to(device).eval()
    with torch.no_grad():
        golden = model(x)                                # tuple of 3 CHW tensors
    print(f"[B] golden tensors = {[tuple(t.shape) for t in golden]}")

    example = torch.randn(1, 3, imgsz, imgsz, device=device)
    quantizer = torch_quantizer(
        quant_mode="test", module=model,
        input_args=(example,), device=device, target=TARGET,
    )
    qmodel = quantizer.quant_model
    with torch.no_grad():
        quant = qmodel(x)

    all_ok = True
    for i, (g, q) in enumerate(zip(golden, quant)):
        g = g.cpu().numpy(); q = q.cpu().numpy()
        mae = float(np.max(np.abs(g - q)))
        cs = cos_sim(g, q)
        flag = "OK" if cs > 0.99 else "LOW"
        if cs <= 0.99: all_ok = False
        print(f"[B] OUT[{i}] shape={g.shape}  max_abs_err={mae:.4f}  cos_sim={cs:.6f}  [{flag}]")
    print(f"[B] {'PASS' if all_ok else 'WARN'}: "
          f"{'quant ใกล้ float (proxy ผ่าน)' if all_ok else 'cos_sim ต่ำ — ตรวจ quant'}")
    print("[B] หมายเหตุ: quant(torch) == DPU จริง ยังพิสูจน์ไม่ได้ (Phase 1 gate)")
    return all_ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/workspace/thesis/phase0/yolov8n.pt")
    ap.add_argument("--img", default="")
    ap.add_argument("--calib_dir", default="/workspace/thesis/phase0/calib_images")
    ap.add_argument("--out_bin", default="/workspace/thesis/phase0/verify_task2/input_float_hwc.bin")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    args = ap.parse_args()
    img = args.img or sorted(glob.glob(os.path.join(args.calib_dir, "*")))[0]
    print(f"[info] target={TARGET}  img={img}")
    a_ok, chw = part_A_layout(img, args.imgsz, args.out_bin)
    b_ok = part_B_fidelity(args.weights, chw, args.imgsz)
    print(f"\n=== สรุป: A(layout)={'PASS' if a_ok else 'FAIL'}  "
          f"B(fidelity)={'PASS' if b_ok else 'WARN'} ===")
    sys.exit(0 if a_ok else 1)

if __name__ == "__main__":
    main()
