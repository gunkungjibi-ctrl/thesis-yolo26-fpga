# แผนงาน Pre-Staging ระหว่างรอบอร์ด KV260

**โครงงาน:** Real-Time Video-Based Object Counting on FPGA (KV260 / Vitis AI 3.0)
**บริบท:** บอร์ด KV260 ยังไม่มาถึง — Phase 1 (board bring-up) ยังเริ่มไม่ได้ เอกสารนี้คืองาน pre-staging ที่ทำบน host ได้โดยไม่ต้องใช้บอร์ด เพื่อให้วันที่บอร์ดมาถึงเสียบแล้วเดินได้ทันที
**สถานะอ้างอิง:** Phase 0 ✅ | GPU baseline ✅ | Phase 1 ⏳ รอบอร์ด

---

## หลักการจัดลำดับ (de-risk critical-path-first)

เรียงตาม **น้ำหนักต่อ Phase 1 + ความเสี่ยงถ้าไม่ทำล่วงหน้า** งานที่อยู่บน critical path ของ Phase 1 และทำล่วงหน้าได้ → ทำก่อน

| # | งาน | น้ำหนัก | ทำบน host ได้? | gate/ผลลัพธ์ |
|---|------|---------|----------------|--------------|
| 1 | VART C++ host code | สูงสุด | ✅ เขียน+compile-check | โค้ดพร้อม deploy ทันทีที่บอร์ดมา |
| 2 | Input preprocessing + CPU emulation verify | สูง | ✅ verify เชิงตัวเลข | xmodel ให้ผลตรง PyTorch float |
| 3 | Ubuntu image + DPU firmware checklist | กลาง | ✅ ค้น+เตรียม | ยืนยัน image ตรง B4096/fingerprint |
| 4 | Counting ground-truth dataset | กลาง | ✅ annotate | dataset พร้อมวัด MAE (Phase 2-3) |

---

## งานที่ 1 — VART C++ Host Code (น้ำหนักสูงสุด)

**ทำไมสำคัญ:** เป็นงานก้อนใหญ่ที่สุดของ Phase 1 ที่ **ไม่ต้องรอบอร์ด** — เขียนและ compile-check บน host ได้ทั้งหมด แม้รันจริงไม่ได้ ถ้าเขียนเสร็จล่วงหน้า วันบอร์ดมาเหลือแค่ deploy + debug runtime

**ขอบเขต:**
- โหลด `yolo26n_kv260.xmodel` (ไฟล์สุดท้ายจาก Phase 0)
- สร้าง VART runner จาก DPU subgraph
- ป้อน input tensor → รัน DPU → ดึง raw output 3 tensors

**Spec ที่ต้องตรง (จาก Phase 0 `xdputil xmodel -l`):**
- input: `[1, 640, 640, 3]` **NHWC** (ไม่ใช่ NCHW), fixpos 6 → input int8 = `round(float × 2^6)` clamp [-128,127]
- output: 3 tensors `[1,40,40,144]` (fixpos 0), `[1,20,20,144]` (fixpos -2), `[1,80,80,144]` (fixpos 1)
- DPU Arch: DPUCZDX8G_ISA1_B4096, fingerprint `0x101000056010407`

**จุดเสี่ยง (เขียนกันไว้):**
- **NHWC vs NCHW** — VART รับ NHWC, ต้อง transpose จาก preprocessing ที่เป็น CHW (จุดเดียวกับที่ CPU emulation ต้องระวัง)
- **fixpos/scale** — input ต้อง quantize ตาม fixpos 6, output ต้อง dequantize ตาม fixpos แต่ละ scale ก่อนส่งไป decode บน PS
- **dequantize เป็นงาน PS** — output 3 tensors เป็น int8 ต้องแปลงกลับ float (× 2^-fixpos) บน PS ก่อน decode (DFL + sigmoid)

**สถานะ:** ยังไม่เริ่ม — เป็นงานต่อไปที่แนะนำ (critical path สูงสุด)

**หมายเหตุ runtime:** Phase 0 ระบุใช้ **VART-based native C++** (ไม่ใช่ PYNQ) ตามที่ตั้งไว้

---

## งานที่ 2 — Input Preprocessing + CPU Emulation Verify

**ทำไมสำคัญ:** ถ้า preprocessing ผิด inference บนบอร์ดจะให้ผลขยะ **โดยไม่มี error ฟ้อง** (เงียบแบบเดียวกับ wrapper bug ใน Phase 0) — verify เชิงตัวเลขบน host ก่อนได้ จับ bug ก่อนบอร์ดมา. **งานนี้พักไว้ตอนต้น session** (ตอนหันไปทำ GPU baseline ก่อน)

**Preprocessing spec (lock แล้วจาก Phase 0 calib — `load_calib_batch` บรรทัด 211-225):**
- `cv2.resize(im, (640, 640))` — **plain resize ไม่ letterbox** (ต้องตรงกับ calib! แม้ letterbox จะ "ถูก" กว่าตามหลัก YOLO)
- `cv2.cvtColor(BGR2RGB)` — BGR→RGB
- `.astype(float32) / 255.0` — normalize /255 **มี**
- transpose CHW (สำหรับ PyTorch) → ตอน VART ต้อง NHWC

**สถาปัตยกรรมการ verify:**
```
รูป test 1 รูป
   ├─► [PyTorch path] BackboneHead float → raw 3 tensors (golden reference)
   └─► [VART CPU path] preprocess → xmodel (CPU runner) → raw 3 tensors
            diff: max abs error + cosine similarity ต่อ tensor
```
- ตรงกัน → preprocessing + xmodel ถูก พร้อมขึ้นบอร์ด
- ต่าง → มี bug (NHWC/NCHW สลับ, fixpos ผิด, resize ผิด) จับก่อนบอร์ดมา

**ทำได้ใน:** Vitis AI container เดิม (`xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106`) — มีทั้ง torch + xir/vart CPU runner

**ข้อจำกัด:** CPU emulation verify ได้แค่ **numerical correctness** ไม่ใช่ latency/power (นั่นต้องรอ DPU จริง)

**สถานะ:** ยังไม่เริ่ม — preprocessing spec lock แล้ว พร้อมเขียน verify script

---

## งานที่ 3 — Ubuntu Image + DPU Firmware Checklist

**ทำไมสำคัญ:** fingerprint บน Ubuntu image อาจไม่ตรง `0x101000056010407` ที่ฝังใน xmodel เพราะ **DPU overlay ผูกกับ image** ถ้าไม่ตรงต้อง re-compile — รู้ล่วงหน้าดีกว่าเจอตอนบอร์ดมา (นี่คือความเสี่ยงอันดับต้นของ Phase 1)

**สิ่งที่ต้องค้น/เตรียม (ค้นเว็บก่อน — ข้อมูลเปลี่ยนได้):**
- เวอร์ชัน **Ubuntu for Kria** ที่ตรงกับ Vitis AI 3.0 / DPUCZDX8G B4096
- แพ็กเกจ DPU firmware (`xlnx-firmware-kv260-*`) ที่ให้ B4096 overlay
- ขั้นตอน flash SD card + boot
- **เช็คว่า DPU overlay ที่มากับ image เป็น B4096 และ fingerprint ตรง** (ไม่ใช่ B3136 หรือ config อื่น)

**Gate (เมื่อบอร์ดมา):** `xmutil loadapp` → `xdputil query` อ่าน fingerprint จริง → เทียบ `0x101000056010407`
- ตรง → ใช้ xmodel เดิมได้เลย
- ไม่ตรง → re-compile xmodel ด้วย arch.json ของบอร์ดจริง

**ผู้ใช้ระบุ boot image:** **Ubuntu** (ไม่ใช่ Petalinux/pre-built Vitis AI image)

**สถานะ:** ยังไม่เริ่ม — ต้องค้นข้อมูล Ubuntu-on-Kria + firmware package ปัจจุบัน

---

## งานที่ 4 — Counting Ground-Truth Dataset

**ทำไมสำคัญ:** Phase 0/baseline ใช้ COCO128 ซึ่ง**ใช้พิสูจน์ feasibility ได้แต่วัด accuracy จริงไม่ได้** ก่อน Phase 3 (วัดผล) ต้องมี dataset สายพานจริงสำหรับวัด MAE ของ line-crossing counting

**สถาปัตยกรรม dataset (locked, three-role):**
1. **Detection dataset** — Roboflow box/parcel/polybag, remap ทุก class → id 0 ("package"), สำหรับ train/quantize/mAP
2. **Counting ground-truth** — Pexels videos (กล้อง top-down นิ่ง, วัตถุแยกชิ้น), label จำนวนต่อเฟรมด้วยมือ
3. **Robustness demo** — Figure AI F.03 livestream clip, ตัด 2-3 นาที, label ด้วยมือ; **on-screen counter เป็น cumulative total อย่าใช้เป็น ground truth**

**สิ่งที่ต้องทำ:**
- หา/annotate conveyor frames (ทางเลือก: Edge Impulse "Cans on conveyor belt", iron ore conveyor dataset)
- per-frame bbox (สำหรับ mAP) หรือ verified count (สำหรับ MAE)
- เปลี่ยน calibration data จาก COCO128 → conveyor frames (ก่อน re-quantize ใน Phase 3)

**Counting scope (locked):** single-class ("package") — รวมทุกวัตถุบนสายพานเป็น class เดียว

**หลักการ:** per-frame detection ≠ counting — ต้องใช้ tracking-based line-crossing (temporal) ไม่งั้นนับซ้ำทุกเฟรม

**สถานะ:** ยังไม่เริ่ม — dataset architecture lock แล้ว เหลือจัดหา + annotate

---

## ลำดับแนะนำ

1. **งานที่ 1 (VART C++)** — critical path สูงสุด งานใหญ่สุด ทำล่วงหน้าได้เต็ม
2. **งานที่ 2 (preprocessing verify)** — ต่อยอดจากงาน 1 (ใช้ preprocessing เดียวกัน) + จับ bug เงียบก่อนบอร์ดมา
3. **งานที่ 3 (Ubuntu image)** — ค้นข้อมูล + เตรียม checklist (de-risk fingerprint)
4. **งานที่ 4 (dataset)** — ทำคู่ขนานได้ ไม่ block งานอื่น

**หมายเหตุ:** งาน 1+2 ใช้ preprocessing spec เดียวกัน (plain resize /255 BGR→RGB) — ทำงาน 1 เสร็จได้ artifact ต่อยอดงาน 2 ทันที ไม่ทับซ้อน

---

## Gate ของ Phase 1 (เมื่อบอร์ดมาถึง)

Phase 1 จะเริ่มและปิดได้เมื่อบอร์ด KV260 มาถึง แล้วทำตามลำดับ:
1. Board bring-up — boot Ubuntu, `xmutil loadapp` โหลด DPU overlay
2. **Fingerprint verify (critical)** — `xdputil query` เทียบ `0x101000056010407` (ไม่ตรง → re-compile)
3. **First live DPU inference** — รัน `yolo26n_kv260.xmodel` ด้วย VART → raw 3 tensors `[1,40,40,144]/[1,20,20,144]/[1,80,80,144]` ถูก shape

**Phase 1 gate = first live DPU inference ให้ shape ถูก** (ไม่ใช่แค่บอร์ดบูตได้) — เหมือน Phase 0 ที่ยึด `DPU subgraph number 1` เป็น gate เดียวที่นับ
