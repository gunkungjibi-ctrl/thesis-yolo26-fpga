# M5 — ทำไม cos_sim (float vs quant) ตกที่ output 40x40 / 20x20

วันที่สรุป: 2026-08-11
ขอบเขต: ขุดหาสาเหตุที่ part B ของ verify_task2 ให้ผล **WARN** — cos_sim ต่ำกว่าเกณฑ์ 0.99 ที่ 2 ใน 3 output tensor
รันจริงใน Vitis AI 3.0 container (`vitis-ai-pytorch`, ต้อง `pip install --no-deps 'ultralytics<8.1'` ก่อน — ไม่ persist เพราะ `--rm`)

---

## ข้อสรุป (TL;DR)

1. **ไม่ใช่บั๊กของ pipeline / preprocessing / compare script** — เป็นผลลูกโซ่จากการที่ **SiLU→LeakyReLU ยังไม่ได้ fine-tune (M2-B2 ยังไม่ทำ)**
2. logit distribution ของ golden float model เอง **เพี้ยน** (range ระเบิดถึง ±2000 ที่ scale หยาบ) → int8 per-tensor fixpos เดียวรับ range ไม่ไหว → saturate → cos_sim ตก
3. **root fix = ทำ M2-B2 (fine-tune หลัง LeakyReLU) ก่อน** แล้ว re-verify M5
4. metric ที่ใช้ (cosine บน raw pre-decode logit) มองโลกแง่ร้ายเกินไป — สิ่งที่ควร verify จริงคือ detection agreement / mAP หลัง decode+NMS

---

## หลักฐาน (รันจาก `verify_task2/diag_*.py`)

### 1. แยก BOX(0:64 DFL) vs CLASS(64: class) — `diag_split.py`

| Output | BOX cos | CLASS cos | CLASS float logit range |
|---|---|---|---|
| 80x80 | 0.963 | 0.992 | [-128.7, 11.1] |
| 40x40 | 0.944 | 0.941 | [-340.5, 29.8] |
| **20x20** | 0.925 | **0.893** | **[-2007.4, 169.4]** |

→ สมมติฐานแรก (BOX/DFL เป็นตัวฉุด) **ถูกหักล้าง** — CLASS แย่กว่าหรือพอกัน และทั้งคู่เสื่อมตาม scale ที่หยาบลง
→ raw head logit range ระเบิด ±65 (80x80) → ±340 (40x40) → **±2000 (20x20)**: feature map ยิ่งลึก/หยาบ activation ยิ่งใหญ่

### 2. saturation/clamping พิสูจน์ตรง ๆ — `diag_worst.py`

- global stats OUT[0]: float `[-128.7, 56.1]` แต่ quant ถูก clamp เหลือ `[-64.0, 38.5]`
  → int8 × 2^(-fixpos) แทน range เต็มไม่ได้ (int8 มีแค่ 256 ระดับ, range ±2000 → step ≈ 16/ระดับ)
- worst class pixel: float logit `6.004` (sigmoid=0.9975 = **เจอ object**) → quant logit `-11.000` (sigmoid=0.0000 = **พลาด**)
  → sign flip จาก quant grid หยาบ + saturation

### 3. เบาะแสชี้ต้นตอ — `diag_detcount.py`

- cell ที่ max class prob > 0.5: **556 cells** ที่ 40x40, 204 ที่ 20x20
- ผิดปกติมหาศาล (ภาพ COCO ปกติมี object ไม่กี่ตัว) → logit ที่พุ่ง ±2000 ไม่ใช่พฤติกรรมโมเดลที่ train ดี
- ชี้ว่า golden float model เองก็เพี้ยน เพราะ `build_float_model` ทำ `SiLU→LeakyReLU` **โดยไม่ retrain**

### 4. post-sigmoid (counting space) — `diag_sigmoid.py`

- cos ยิ่งต่ำ (0.65 / 0.75 / 0.71) แต่ **หลอกตา** — sigmoid ของ logit ±2000 อิ่มตัวเป็น 0/1 หมด, cosine บน field เกือบ binary ที่ sparse จึง noisy → metric นี้ไม่ควรใช้ตัดสิน

---

## กลไก (chain of causation)

```
SiLU→LeakyReLU ไม่ fine-tune (M2-B2 ยังไม่ทำ)
   └─► logit distribution เพี้ยน, range ระเบิด (±2000 ที่ scale หยาบ)
          └─► int8 per-tensor fixpos เดียว รับ range ไม่ไหว → saturate/clamp
                 └─► cos_sim ตก + detection พลิก (เจอ↔พลาด)
```

## สิ่งที่ต้องทำต่อ

1. **M2-B2 (fine-tune หลัง LeakyReLU) ก่อน** — น่าจะลด logit magnitude → แก้ทั้ง 2 ชั้นพร้อมกัน แล้ว cos_sim ควรกลับขึ้น = dependency ของ M5
2. **verify ด้วย metric ที่ถูกระดับ** — detection agreement / mAP หลัง decode+NMS ไม่ใช่ raw tensor cosine
3. **M5 execute จริงผ่าน VART** ยังทำในคอนเทนเนอร์ x86 ไม่ได้ (`no DpuController found for DPUCZDX8G`) — ผูกกับบอร์ด (M9)

## หมายเหตุ reproduce

- ผลนี้ทำบน **YOLOv8n (Track A)** ผ่าน PyTorch quantizer test-mode (proxy DPU) ไม่ใช่ VART runner จริง
- ภาพ: `calib_images/000000000009.jpg` (ตัวแรก sorted)
- ยืนยันตัวเลข part B ตรงกับ `verify_task2/verify_run.log` เดิม (0.991 / 0.942 / 0.894)
