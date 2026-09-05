# รายงานปัญหา: Quantization Fidelity ต่ำที่ output หยาบ (M5 WARN)

วันที่: 2026-08-11
สถานะ: ระบุสาเหตุแล้ว ยังไม่แก้ (block M5)
เอกสารเทคนิคประกอบ: [`M5_cossim_dropoff_diagnosis.md`](M5_cossim_dropoff_diagnosis.md)

---

## 1. ปัญหาคืออะไร (สรุปสำหรับคนอ่านเร็ว)

ตอน verify ว่าโมเดลหลัง quantize (int8) ยังให้ผลใกล้เคียงโมเดล float เดิมหรือไม่
เราวัดด้วย **cosine similarity** ต่อ output tensor (เกณฑ์ผ่าน > 0.99) ผลออกมา:

| Output tensor | cos_sim | ผ่านเกณฑ์ 0.99? |
|---|---|---|
| 80x80 (object เล็ก) | 0.991 | ✅ ผ่าน |
| 40x40 (object กลาง) | 0.942 | ❌ ตก |
| 20x20 (object ใหญ่) | 0.894 | ❌ ตก |

แปลว่า **โมเดล int8 เพี้ยนจาก float มากขึ้นเรื่อย ๆ ที่ output หยาบ** (ซึ่งรับผิดชอบ object ขนาดกลาง-ใหญ่)

**ต้นตอที่แท้จริง (ไม่ใช่บั๊กโค้ด):**
โมเดลถูกดัดแปลง `SiLU → LeakyReLU` (เพราะ DPU ไม่รองรับ SiLU native) **แต่ยังไม่ได้ fine-tune หลังดัดแปลง**
→ ทำให้ค่า logit ภายในโมเดลเพี้ยน range ระเบิดถึง **±2000** ที่ scale หยาบ
→ int8 มีแค่ 256 ระดับ แทน range ±2000 ไม่ไหว → ค่าถูกบด/อิ่มตัว (saturate)
→ ผลลัพธ์บาง pixel พลิกจาก "เจอ object" เป็น "ไม่เจอ" (float logit 6.0 → quant −11.0)

```
SiLU→LeakyReLU ไม่ fine-tune  →  logit range ระเบิด  →  int8 saturate  →  detection พลิก
```

---

## 2. สำคัญต่อโปรเจกต์นี้ยังไง

โปรเจกต์นี้ปลายทางคือ **นับ object (package) บนวิดีโอ** แล้ววัด accuracy (counting MAE) เทียบ GPU baseline
ถ้า quantization เพี้ยนที่ output หยาบ ผลกระทบเป็นลูกโซ่:

1. **object ขนาดกลาง-ใหญ่จะตรวจพลาด/เพี้ยน** — 40x40 กับ 20x20 คือ detection head ที่จับของใหญ่ พอ cos_sim ตกแปลว่า box/class ที่ scale นี้เชื่อถือไม่ได้
2. **counting MAE จะสูง** — นับพลาดเพราะ detection หาย → ตัวเลข benchmark หลัก (จุดขายของ thesis) จะดูแย่โดยไม่ใช่ความผิดของ FPGA แต่เป็นความผิดของ quantize
3. **debug บนบอร์ดจะสับสน** — นี่คือเหตุผลที่ M5 มีอยู่: วันบอร์ดมา ถ้าไม่ verify ก่อน เราจะแยกไม่ออกว่า detection ห่วยเพราะ (ก) preprocessing bug (ข) quantize เพี้ยน หรือ (ค) DPU มีปัญหา — การรู้ตั้งแต่ตอนนี้ว่า "quantize เพี้ยนเพราะยังไม่ fine-tune" ตัดตัวแปรออกไปหนึ่งตัวล่วงหน้า
4. **gate ของ M5 ผ่านไม่ได้จริง** จนกว่าจะแก้ → block การประกาศว่า "pipeline ถูกต้องเชิงตัวเลข ก่อนขึ้นบอร์ด"

> หมายเหตุระดับความรุนแรง: นี่ **ไม่ใช่ blocker ที่ทำให้โปรเจกต์ล้ม** — Track A (YOLOv8n) compile ผ่าน gate แล้ว และปัญหานี้เป็นเรื่อง accuracy ที่มีทางแก้ชัดเจน แต่ **ต้องแก้ก่อนเก็บตัวเลข benchmark จริง** ไม่งั้นผลที่ได้จะไม่สะท้อนความสามารถจริงของ FPGA

---

## 3. แก้ยังไง (วิธีหลัก)

### ✅ วิธีที่แนะนำ: ทำ M2-B2 — Fine-tune หลัง SiLU→LeakyReLU

หลังสลับ activation ให้ train ต่ออีกไม่กี่ epoch เพื่อให้ weight ปรับตัวเข้ากับ LeakyReLU
→ logit distribution กลับมาปกติ (range หดจาก ±2000 ลงมาระดับ ±10..±50 ตามที่โมเดลที่ train ดีควรเป็น)
→ พอ range แคบลง int8 ก็แทนได้ครบ ไม่ saturate → cos_sim ควรกลับขึ้นผ่าน 0.99

**ทำไมวิธีนี้ตรงจุด:** มันแก้ที่ **ต้นเหตุ** (logit range ระเบิด) ไม่ใช่ปลายเหตุ และเป็นงานที่วางแผนไว้อยู่แล้วใน timeline (M2-B2) แค่ยังไม่ได้ทำ — ไม่ใช่งานเพิ่มนอกแผน

**ขั้นตอน:**
1. โหลด float model, สลับ SiLU→LeakyReLU (มีโค้ดใน `quantize_yolo_pytorch.py` แล้ว)
2. fine-tune บน dataset (COCO subset หรือ conveyor frames) จน accuracy กลับมาใกล้เดิม
3. re-quantize จาก weight ใหม่
4. rerun `verify_task2` (part B) → เช็ก cos_sim ผ่าน 0.99 ทั้ง 3 tensor
5. เมื่อบอร์ดมา rerun ผ่าน VART จริง + `compare_dpu_vs_golden.py`

**ต้นทุน:** ต้องมี GPU + เวลา train (แต่มี GPU baseline อยู่แล้ว) — งานระดับชั่วโมง ไม่ใช่วัน

---

## 4. วิธีอื่น (ถ้าไม่แก้ด้วย fine-tune หรือใช้เสริม)

| วิธี | แนวคิด | ข้อดี | ข้อเสีย / ข้อจำกัด |
|---|---|---|---|
| **B. QAT (Quantization-Aware Training)** | จำลอง quant ตอน train เลย ไม่ใช่ quantize ทีหลัง (PTQ) | fidelity สูงสุด, โมเดลเรียนรู้ที่จะทน quant | งานหนักกว่า fine-tune ธรรมดา, ต้องตั้ง pipeline QAT ของ vai_q_pytorch |
| **C. ไม่สลับ SiLU (คง activation เดิม)** | เก็บ SiLU ไว้ ไม่ทำ LeakyReLU | ไม่มี logit ระเบิดจากการสลับ | **DPU ไม่รองรับ SiLU** → op ตกไป CPU/USER → ไม่ผ่าน gate 1-DPU-subgraph = ขัดเป้าหมาย Phase 0 (นี่คือเหตุผลที่สลับตั้งแต่แรก) |
| **D. เปลี่ยน metric การ verify** | วัด mAP / detection-agreement หลัง decode+NMS แทน raw-tensor cosine | สะท้อน accuracy จริงที่ผู้ใช้สนใจ, cosine บน raw logit มองแง่ร้ายเกินจริง | **ไม่ได้แก้โมเดล** แค่เปลี่ยนไม้บรรทัด — ถ้า detection พลิกจริง mAP ก็จะตกอยู่ดี ใช้เป็น "ตัววัดที่ถูกต้องกว่า" คู่กับ A ไม่ใช่แทน |
| **E. Per-channel quantization** | ให้แต่ละ channel มี scale ของตัวเอง (แทน per-tensor เดียว) | รับ dynamic range กว้างได้ดีกว่ามาก | **DPUCZDX8G รองรับแค่ per-tensor power-of-2 (fixpos)** — ฮาร์ดแวร์ทำ per-channel ไม่ได้ → ใช้ไม่ได้กับ target นี้ |
| **F. Fallback เป็นโมเดล Model-Zoo ที่พิสูจน์แล้ว** | ใช้ YOLOv5/YOLOv8 quantized version ทางการที่ Xilinx การันตี | เสี่ยงต่ำ, มี reference accuracy | เสีย novelty ของ thesis (แต่ fallback plan เดิมมีไว้อยู่แล้วเผื่อกรณีเลวร้าย) |

**คำแนะนำ:** ทำ **A (fine-tune) เป็นหลัก + D (เพิ่ม metric mAP) เป็นเกณฑ์ยืนยัน**
ถ้า A ยังไม่พอค่อยยก **B (QAT)**; C/E ใช้ไม่ได้เพราะติดข้อจำกัดฮาร์ดแวร์; F เก็บเป็น safety net สุดท้าย

---

## 5. ถ้าไม่แก้ จะเป็นยังไง

- **on-board detection ที่ scale กลาง-ใหญ่จะเชื่อถือไม่ได้** — object ใหญ่ (ใกล้กล้อง) หรือกลางจะถูกนับพลาดบ่อย
- **counting MAE สูงผิดจริง** → ตัวเลข benchmark หลักของ thesis (FPGA vs GPU) จะดูแย่โดยไม่ยุติธรรมกับ FPGA — สรุปผลผิดว่า "FPGA แม่นน้อย" ทั้งที่จริงเป็นเพราะ quantize ไม่ fine-tune
- **เสียโอกาส debug ล่วงหน้า** — วันบอร์ดมาจะแยกไม่ออกว่า detection ห่วยเพราะอะไร (preprocessing / quantize / DPU) เพราะไม่ได้ตัดตัวแปร quantize ทิ้งไปก่อน → debug บนบอร์ดช้าและแพงกว่า
- **M5 gate ค้าง** → ไม่สามารถประกาศ "pipeline ถูกต้องเชิงตัวเลขก่อนขึ้นบอร์ด" ได้ ซึ่งเป็นทั้งจุดประสงค์ของ M5

> ข้อควรระวังในการเขียนเล่ม: ถ้าจำเป็นต้องส่งโดยยังไม่ได้ fine-tune จริง ๆ **ต้องรายงานตรง ๆ** ว่าตัวเลขที่ได้เป็น lower-bound เพราะ activation substitution ยังไม่ retrain — อย่านำเสนอเป็น accuracy สูงสุดของ FPGA

---

## 6. สรุปตัดสินใจ

1. **แก้ด้วย A (fine-tune / M2-B2)** = root fix, อยู่ในแผนอยู่แล้ว, ต้นทุนต่ำ → **ทำอันนี้**
2. เพิ่ม **D (metric mAP/detection agreement)** เป็นเกณฑ์ยืนยันคู่กับ cosine
3. เก็บ **B (QAT)** ไว้ถ้า A ไม่พอ, **F (fallback model)** เป็น safety net สุดท้าย
4. **C, E ใช้ไม่ได้** — ติดข้อจำกัดฮาร์ดแวร์ DPUCZDX8G (ไม่รองรับ SiLU / ไม่รองรับ per-channel)
5. **M5 ปิดไม่ได้จนกว่าจะทำ M2-B2** → ปรับ dependency ใน timeline แล้ว

**ลำดับงานที่ควรเป็น:** M2-B1 (inspector) → **M2-B2 (fine-tune)** → re-quantize → M5 re-verify (proxy) → [รอบอร์ด] → M5 execute จริงผ่าน VART (M9)
