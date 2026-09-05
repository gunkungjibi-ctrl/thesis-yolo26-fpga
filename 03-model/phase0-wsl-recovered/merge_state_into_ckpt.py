#!/usr/bin/env python3
# รวม fine-tuned weights (plain state_dict จาก Colab) เข้ากับ ckpt ของ yolov8n
# ให้ได้ .pt ที่ ultralytics 8.0.x ใน Vitis AI container โหลดได้ตรง ๆ
# รัน INSIDE container: pip install --no-deps 'ultralytics<8.1' ก่อน
#
#   base yolov8n.pt (SiLU model object, เวอร์ชัน container) + state_dict (fine-tuned params)
#     -> yolov8n_leaky_ft.pt  (params = fine-tuned, activation ยังเป็น SiLU ใน ckpt)
#   ตอน build_float_model โหลด .pt นี้แล้ว swap SiLU->LeakyReLU ซ้ำ -> ตรงกับตอน train
import argparse, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="yolov8n.pt", help="ckpt ต้นฉบับ (container-native)")
    ap.add_argument("--state", default="yolov8n_leaky_state.pt", help="plain state_dict จาก Colab")
    ap.add_argument("--out", default="yolov8n_leaky_ft.pt")
    args = ap.parse_args()

    base = torch.load(args.base, map_location="cpu")   # ultralytics ckpt dict
    sd = torch.load(args.state, map_location="cpu")     # OrderedDict ของ tensor
    if isinstance(sd, dict) and "model" in sd and not torch.is_tensor(next(iter(sd.values()), None)):
        # เผลอเซฟทั้ง ckpt มา — ดึง state_dict ออก
        sd = sd["model"].float().state_dict()

    model_obj = base["model"]
    missing, unexpected = model_obj.load_state_dict(sd, strict=False)
    print(f"[merge] loaded state_dict: {len(sd)} tensors")
    print(f"[merge] missing keys   = {len(missing)}")
    print(f"[merge] unexpected keys = {len(unexpected)}")
    if missing:
        print("  MISSING (คีย์ที่โมเดล container มีแต่ state_dict ไม่มี — ควรเป็น 0):")
        for k in missing[:10]:
            print("   -", k)
    if unexpected:
        print("  UNEXPECTED (คีย์ใน state_dict ที่โมเดลไม่รับ — ควรเป็น 0):")
        for k in unexpected[:10]:
            print("   -", k)
    if missing or unexpected:
        print("[merge] WARN: key ไม่ตรง 100% — architecture ระหว่าง Colab/container อาจต่างกันเล็กน้อย, ตรวจก่อนใช้")
    else:
        print("[merge] OK: key ตรงครบ 100%")

    # ล้าง ema/optimizer เก่า (params ใน model_obj อัปเดตแล้ว)
    base["ema"] = None
    base["optimizer"] = None
    base["updates"] = None
    torch.save(base, args.out)
    print(f"[merge] wrote {args.out} -> ใช้ --weights {args.out} กับ quantize/verify ได้เลย")


if __name__ == "__main__":
    main()
