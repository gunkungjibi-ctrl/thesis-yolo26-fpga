#!/usr/bin/env python3
# M2-B2 — fine-tune yolov8n หลังสลับ SiLU->LeakyReLU (แก้ต้นเหตุ logit range ระเบิด)
# รัน INSIDE Vitis AI 3.0 container (torch 1.12.1 CPU) — ต้อง pip install 'ultralytics<8.1' ก่อน
#
# เป้าหมาย: ให้ conv/bn weight ปรับตัวเข้ากับ LeakyReLU ผ่าน forward จริงตอน train
#   -> logit distribution หด (จาก ±2000) -> int8 quantize ไม่ saturate -> cos_sim กลับขึ้น
#
# กันบั๊กสำคัญ: substitute SiLU->LeakyReLU ทำผ่าน CALLBACK บน trainer.model (+ ema)
#   ไม่พึ่งว่า ultralytics จะเก็บ module substitution ให้ตอนสร้างโมเดล
#   (สิ่งที่ต้องได้จริง = forward ตอน train ใช้ LeakyReLU; activation ตอน reload
#    ไม่สำคัญเพราะ build_float_model swap เป็น LeakyReLU ซ้ำอยู่แล้ว — ดู quantize_yolo_pytorch.py)
import argparse, os, shutil
import torch.nn as nn
from ultralytics import YOLO

# slope ต้องตรงกับ replace_silu_with_leakyrelu ใน quantize_yolo_pytorch.py เป๊ะ
SLOPE = 0.1015625


def swap_silu_to_leaky(module):
    """แทน SiLU ทุกตัวด้วย LeakyReLU (inplace=False เพื่อเลี่ยง autograd inplace error ตอน train;
    numerically เท่ากับ inplace=True ที่ quantize ใช้)"""
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.LeakyReLU(SLOPE, inplace=False))
            n += 1
        else:
            n += swap_silu_to_leaky(child)
    return n


def make_callback():
    def _cb(trainer):
        n = swap_silu_to_leaky(trainer.model)
        # swap EMA ด้วย เพื่อให้ val/บันทึกใช้ LeakyReLU สอดคล้องกัน (ema.ema = deepcopy)
        m = 0
        if getattr(trainer, "ema", None) is not None and getattr(trainer.ema, "ema", None) is not None:
            m = swap_silu_to_leaky(trainer.ema.ema)
        print(f"[cb] SiLU->LeakyReLU({SLOPE}) swapped: model={n} ema={m}")
        if n == 0:
            raise SystemExit("[cb] FATAL: ไม่พบ SiLU ในโมเดล — activation อาจไม่ใช่ SiLU, ตรวจ yaml")
    return _cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--data", default="coco128.yaml")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default="runs_ft")
    ap.add_argument("--name", default="yolov8n_leaky")
    ap.add_argument("--out", default="yolov8n_leaky_ft.pt",
                    help="path ปลายทางที่จะ copy best.pt ไป (บน mounted volume)")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.add_callback("on_pretrain_routine_end", make_callback())

    results = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, project=args.project, name=args.name,
        cache=False,            # ดิสก์จำกัด — ไม่ cache
        optimizer="SGD", lr0=0.001, warmup_epochs=1.0,  # fine-tune: lr ต่ำ
        pretrained=False,       # โหลด weight เองผ่าน YOLO(weights) แล้ว
        verbose=True, plots=False, val=True,
    )

    save_dir = results.save_dir if hasattr(results, "save_dir") else os.path.join(args.project, args.name)
    best = os.path.join(str(save_dir), "weights", "best.pt")
    if not os.path.exists(best):
        best = os.path.join(str(save_dir), "weights", "last.pt")
    shutil.copy(best, args.out)
    print(f"[ok] fine-tuned weights -> {args.out} (from {best})")


if __name__ == "__main__":
    main()
