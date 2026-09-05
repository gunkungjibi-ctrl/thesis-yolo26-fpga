#!/usr/bin/env python3
# M5 — เปรียบเทียบ output จริงจาก VART CPU runner (yolo_dpu_infer, M4)
# กับ golden ที่ได้จาก PyTorch float model
#   ฝั่ง M4  : out_<H>x<W>.bin  float32  NHWC  (dequantized แล้ว)
#   ฝั่ง golden: model(x) -> tuple of (1,C,H,W)  float32  NCHW (PyTorch conv output)
# จับคู่ tensor ด้วยขนาด spatial (H,W) ไม่ใช้ index/order เพราะ VART ไม่การันตี order
# เกณฑ์ผ่าน: cos_sim > 0.99 ทุก tensor
import os, re, sys, glob, argparse
import numpy as np
import torch

sys.path.insert(0, "/workspace/thesis/phase0")
from quantize_yolo_pytorch import build_float_model, IMGSZ

BIN_NAME_RE = re.compile(r"_(\d+)x(\d+)\.bin$")


def cos_sim(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else float("nan")


def preprocess_one(img_path, imgsz):
    """ต้องตรงกับ preprocess_into ใน yolo_dpu_infer.cpp เป๊ะ: plain resize -> BGR2RGB -> /255 -> CHW"""
    import cv2
    im = cv2.imread(img_path)
    if im is None:
        raise FileNotFoundError(img_path)
    im = cv2.resize(im, (imgsz, imgsz))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(im, (2, 0, 1))
    return np.ascontiguousarray(chw)


def load_dpu_bins(bin_dir, prefix):
    """โหลด out_<H>x<W>.bin ทั้งหมดที่มี prefix ตรงกัน -> dict{(H,W): array NHWC}"""
    pattern = os.path.join(bin_dir, f"{prefix}_*x*.bin")
    found = {}
    for path in sorted(glob.glob(pattern)):
        m = BIN_NAME_RE.search(os.path.basename(path))
        if not m:
            continue
        h, w = int(m.group(1)), int(m.group(2))
        arr = np.fromfile(path, dtype=np.float32)
        c = arr.size // (h * w)
        if c * h * w != arr.size:
            print(f"[warn] {path}: size {arr.size} ไม่หาร h*w ลงตัว, ข้าม")
            continue
        found[(h, w)] = arr.reshape(h, w, c)  # NHWC (batch=1 ตัดทิ้งแล้ว)
        print(f"[load] {path}  shape(HWC)={found[(h, w)].shape}")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/workspace/thesis/phase0/yolov8n.pt")
    ap.add_argument("--img", required=True, help="ภาพเดียวกับที่ป้อน yolo_dpu_infer")
    ap.add_argument("--bin_dir", default=".", help="โฟลเดอร์ที่มี out_<H>x<W>.bin จาก M4")
    ap.add_argument("--prefix", default="out", help="out_prefix ที่ใช้ตอนรัน yolo_dpu_infer")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--thresh", type=float, default=0.99)
    args = ap.parse_args()

    # 1) golden จาก PyTorch float model (input เดียวกับที่ yolo_dpu_infer ใช้)
    chw = preprocess_one(args.img, args.imgsz)
    x = torch.from_numpy(chw).unsqueeze(0)  # (1,3,H,W) NCHW
    model = build_float_model(args.weights).eval()
    with torch.no_grad():
        golden = model(x)  # tuple of (1,C,H,W)
    golden_by_hw = {}
    for i, g in enumerate(golden):
        g = g.cpu().numpy()
        _, c, h, w = g.shape
        golden_by_hw[(h, w)] = g[0]  # CHW
        print(f"[golden] OUT[{i}] shape(CHW)={g[0].shape}")

    # 2) dpu outputs (.bin จาก M4)
    dpu_by_hw = load_dpu_bins(args.bin_dir, args.prefix)

    if not dpu_by_hw:
        print(f"[FAIL] ไม่เจอไฟล์ {args.prefix}_*x*.bin ใน {args.bin_dir}")
        sys.exit(1)

    # 3) จับคู่ด้วย (H,W) แล้วเทียบ
    print("\n=== M5: DPU(VART CPU runner) vs PyTorch float golden ===")
    all_ok = True
    for hw in sorted(golden_by_hw, key=lambda t: -t[0]):
        g_chw = golden_by_hw[hw]  # (C,H,W)
        if hw not in dpu_by_hw:
            print(f"[MISS] golden {hw} ไม่มี .bin คู่ (ขนาด spatial ไม่ตรงกับ M4 output ใด ๆ)")
            all_ok = False
            continue
        d_hwc = dpu_by_hw[hw]              # (H,W,C)
        d_chw = np.transpose(d_hwc, (2, 0, 1))  # -> CHW เทียบกับ golden

        if d_chw.shape != g_chw.shape:
            print(f"[FAIL] {hw}: channel ไม่ตรง golden={g_chw.shape} dpu={d_chw.shape}")
            all_ok = False
            continue

        mae = float(np.max(np.abs(g_chw - d_chw)))
        cs = cos_sim(g_chw, d_chw)
        ok = cs > args.thresh
        all_ok = all_ok and ok
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {hw[0]}x{hw[1]}  shape={g_chw.shape}  "
              f"max_abs_err={mae:.4f}  cos_sim={cs:.6f}")

    print(f"\n=== สรุป M5: {'PASS' if all_ok else 'FAIL'} "
          f"(เกณฑ์ cos_sim > {args.thresh}) ===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
