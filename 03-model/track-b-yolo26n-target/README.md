# Track B — YOLO26n (TARGET, ยังไม่ผ่าน gate)

**สถานะ: ยังไม่ได้รัน quantize/compile ให้จบ** — มีแค่สคริปต์ export + quantize ที่เขียนไว้
Gate เดียวที่นับ: `vai_c_xir` รายงาน **1 DPU subgraph** และ `xdputil xmodel -l` ไม่มี op ไหนขึ้น device `USER`/`CPU`

## ลำดับงาน (ทำตามนี้ ห้ามข้าม)

### ขั้น 0 — เคลียร์ DPU target ก่อน
อย่าเพิ่งรันอะไรจนกว่าจะรู้แน่ว่า B3136 หรือ B4096 (ดู `../README.md` ประเด็นที่ 1)
ค่านี้ไปอยู่ใน `quantize_yolo26n_pytorch.py::TARGET` และ `compile/compile_yolo26n.sh::ARCH` ต้องตรงกัน

### ขั้น 1 — Inspector ก่อน (ถูกที่สุด รู้ผลเร็วสุด)
```bash
python quantize/quantize_yolo26n_pytorch.py --weights ../common/weights/yolo26n.pt --inspect
```
Inspector บอกล่วงหน้าว่า op ไหนจะตกไป CPU/PS — **advisory เท่านั้น ไม่ใช่ gate** แต่ถ้าตรงนี้เละ ไม่ต้องเสียเวลา compile
บันทึกผลลง `artifacts/inspect_report/`

### ขั้น 2 — export + ตรวจ op histogram
```bash
python export/export_yolo26n_for_dpu.py --weights ../common/weights/yolo26n.pt
```
สคริปต์ตั้ง `end2end=False` (ultralytics 8.4.71 ใช้ชื่อนี้ **ไม่มี underscore** — โน้ตเก่าที่เขียน `_end2end` ผิดและจะ no-op เงียบ)
→ ทำให้ `postprocess()` ถูกข้าม ซึ่งเป็นที่มาของ **TopK ×2 + GatherElements ×2** ที่ DPUCZDX8G map ไม่ได้
watchlist: `TopK, GatherElements, GatherND, ReduceMax, Mod, ScatterND, NonMaxSuppression, Range`

### ขั้น 3 — ⚠️ fine-tune หลังสลับ SiLU → LeakyReLU
DPUCZDX8G ไม่รองรับ SiLU → สคริปต์สลับเป็น `LeakyReLU(0.1015625 = 26/256)` ซึ่งเป็นค่า DPU-friendly
**การสลับนี้ทำให้ accuracy เพี้ยน** ต้อง fine-tune ก่อน ไม่งั้น calibrate แล้วได้ INT8 ขยะ
> สำหรับพิสูจน์ *compile feasibility* อย่างเดียว ข้ามขั้นนี้ได้ แต่ห้ามเอาตัวเลข accuracy ที่ได้ไปเขียนรายงาน

### ขั้น 4 — quantize (สองรอบตาม API)
```bash
python quantize/quantize_yolo26n_pytorch.py --weights yolo26n_finetuned.pt \
  --calib_dir calib_images --quant_mode calib --subset_len 200
python quantize/quantize_yolo26n_pytorch.py --weights yolo26n_finetuned.pt \
  --calib_dir calib_images --quant_mode test --subset_len 1 --batch_size 1 --deploy
```
calib ต้องเป็น **ภาพสายพานจริง 100–1000 รูป** — ใช้ random data = Phase 0 เป็นโมฆะ

### ขั้น 5 — compile = gate
```bash
bash compile/compile_yolo26n.sh
xdputil xmodel yolo26n_kv260.xmodel -l
```

## เงื่อนไขตัดสิน

| ผล | ทำอะไรต่อ |
|---|---|
| 1 DPU subgraph, ไม่มี op ตก CPU | ✅ Track B ขึ้นเป็นโมเดลหลัก, Track A เป็น comparison baseline ในเล่ม |
| มี op ตก CPU แต่น้อย | วัด latency penalty จริง → ถ้ารับได้ก็ใช้ต่อ + รายงานเป็น finding |
| compile ไม่ผ่าน / op ตกเยอะ | ❌ ปิด Track B → เดินหน้าด้วย Track A **แต่ผลลัพธ์นี้ยังเป็น contribution** (องค์ความรู้ว่า YOLO26 ติดตรงไหนบน DPUCZDX8G) เขียนเข้าเล่มได้เต็ม ๆ |

**ไม่ว่าผลออกทางไหน ให้จดไว้ว่า op ไหนไม่รองรับและเพราะอะไร** — นั่นคือส่วนที่เป็น contribution จริงของ track นี้ ไม่ใช่แค่ "ผ่าน/ไม่ผ่าน"
