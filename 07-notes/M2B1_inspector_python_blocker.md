# M2-B1 blocker — nndct Inspector ↔ YOLO26 Python version clash

วันที่ 2026-08-15 · ระหว่างพยายามรัน hardware-aware Inspector บน YOLO26n

> ## ✅ ปิดแล้ว 2026-09-05 — ดู `03-model/track-b-yolo26n-target/RUNBOOK_M2B3.md`
>
> **ทางที่ใช้จริง = ทางเลือกที่ 4 ซึ่งตอนเขียนโน้ตนี้ยังไม่ได้คิดถึง: ตัด ultralytics ออกจากคอนเทนเนอร์ทั้งหมด**
>
> เขียนกราฟ YOLO26n ใหม่เป็น PyTorch ล้วนที่ `quantize/yolo26n_dpu.py` — syntax Python 3.7,
> ใช้เฉพาะ op ที่มีใน torch 1.12, ไม่ import ultralytics เลย → import ได้ใน env `vitis-ai-pytorch` ตรงๆ
> น้ำหนักส่งเข้าเป็น **plain state_dict** (ไม่ใช่ pickle ที่มีคลาส ultralytics ฝังอยู่ ซึ่ง torch 1.12 อ่านไม่ได้)
>
> **หลักประกันว่ากราฟไม่เพี้ยน:** `export_yolo26n_state_dict.py` โหลด state_dict แบบ `strict`
> แล้วเทียบ output กับ ultralytics ทุกครั้งที่รัน ถ้าไม่ตรงจะ abort
> ผลตรวจ: **708 tensors ตรงครบ · `max|diff| = 3.8e-06`** (= noise ของ float32)
>
> **ทำไมไม่เลือกทางที่ 1–3 ที่จดไว้:**
> - ทางที่ 1 (ONNX op histogram) — ทำไปแล้วใน M2-B1 และตอบคำถามได้ระดับหนึ่ง **แต่ไม่ใช่ gate** เพราะ gate คือ `vai_c_xir` ซึ่งต้องมี `_int.xmodel` จาก nndct
> - ทางที่ 2 (Vitis AI 3.5) — ยังไม่แนะนำ เหตุผลเดิม (ไม่มี KV260 prebuilt)
> - ทางที่ 3 (ลง nndct ใน py3.9) — เป็นไปได้ต่ำเหมือนที่ประเมินไว้
>
> **สิ่งที่พบเพิ่มระหว่างทาง:** ต่อให้ patch walrus 5 บรรทัดให้ ultralytics import ได้บน py3.7
> (สแกนแล้วโค้ดใน `ultralytics/nn/` สะอาด py3.7 100% ตัวที่ติดอยู่ที่ `utils/` 5 ไฟล์)
> ก็ยังเสี่ยงพังกับ **torch 1.12** อยู่ดี (เช่น `torch.load(weights_only=)` ที่เพิ่งมีใน torch 1.13)
> การเขียนกราฟเองจึงคุมความเสี่ยงได้ดีกว่า และได้ผลพลอยได้คือตัดปัญหา `chunk`/`split` ไปในตัว

## ปัญหา
รัน `quantize_yolo26n_pytorch.py --inspect` ไม่ได้ เพราะ **ต้องมี 2 แพ็กเกจที่ต้องการ Python คนละเวอร์ชัน**:

| ต้องการ | อยู่ที่ | Python |
|---|---|---|
| `pytorch_nndct` (Inspector + `torch_quantizer`) | conda env `vitis-ai-pytorch` เท่านั้น | **3.7.12** |
| `ultralytics==8.4.71` (โหลด/สร้างกราฟ YOLO26n) | ต้องติดตั้งเอง | pip บังคับ **Python ≥ 3.8** |

`pip install ultralytics==8.4.71` ใน py3.7 → `No matching distribution` (เวอร์ชันสูงสุดที่ลงได้บน 3.7 คือ 8.0.151 ซึ่ง **ยังไม่รู้จัก YOLO26**)

## สำรวจ env ในคอนเทนเนอร์ `xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106`
```
base                -> Python 3.9.9   (ว่างเปล่า: ไม่มี torch/nndct/cv2/numpy)
vitis-ai-pytorch    -> Python 3.7.12  (มี torch 1.12.1 + pytorch_nndct)
vitis-ai-wego-torch -> Python 3.7.8
```
→ ไม่มี env ไหนมีทั้ง nndct และ py3.8+ พร้อมกัน. nndct ถูก build ผูกกับ py3.7+torch1.12.1 ย้ายข้าม py ตรง ๆ ไม่ได้

**นี่คือรากของ "Track B เสี่ยง" ที่ระบุในแผน** — toolchain Vitis AI 3.0 (2022, py3.7) เก่ากว่า YOLO26 (ต้อง ultralytics ใหม่ที่ต้อง py3.8+)

## ทางเลือกที่ยังเดินได้ (ยังไม่ได้ตัดสิน — ให้ผู้สานต่อเลือก)

1. **Export ONNX แยก แล้ววิเคราะห์ op histogram เอง** (แนะนำ, ตรงกับ README ขั้น 2)
   - สร้าง env py3.9+ (host หรือ venv ใหม่) ลง `ultralytics 8.4.71` + torch → รัน `export_yolo26n_for_dpu.py` ได้ `.onnx`
   - เทียบ op ใน onnx กับ watchlist DPUCZDX8G: `TopK, GatherElements, GatherND, ReduceMax, Mod, ScatterND, NonMaxSuppression, Range`
   - ตอบคำถามเดียวกับ Inspector (op ไหนตก CPU) โดยไม่ต้องใช้ nndct — เพียงพอสำหรับ feasibility finding
   - **ข้อจำกัด:** ไม่ได้ผ่าน hardware-aware quantizer จริง แต่ gate จริงคือ `vai_c_xir` อยู่แล้ว (Inspector เป็นแค่ advisory)

2. **หา Vitis AI ใหม่กว่า** (3.5+) ที่ env เป็น py3.8+ — แต่ 3.5 ไม่มี KV260 prebuilt (ดู README เตือน) → เสี่ยงพังงานอื่น ไม่แนะนำ

3. **ลง torch+nndct ใน base py3.9 เอง** — nndct wheel ผูก py3.7 ABI, ความเป็นไปได้ต่ำ

## บทเรียนเข้าเล่ม
การ deploy โมเดล YOLO รุ่นใหม่บน DPU stack เดิม ติดที่ **toolchain/Python compatibility ก่อนจะถึงเรื่อง op mapping ด้วยซ้ำ** — เป็นข้อจำกัดเชิงปฏิบัติที่ควรรายงาน
