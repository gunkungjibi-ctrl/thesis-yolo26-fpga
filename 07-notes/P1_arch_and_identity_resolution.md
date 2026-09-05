# P1 — เคลียร์ประเด็น DPU arch (B3136 vs B4096) และ identity ของ xmodel

วันที่สรุป: 2026-08-11
ขอบเขต: ตรวจ artifact ที่กู้มาใน `03-model/phase0-wsl-recovered/` เพื่อปิดปม P1 ก่อนไป compile รอบต่อไป

---

## ข้อสรุป (TL;DR)

1. **DPU arch จริงที่ compile = `DPUCZDX8G_ISA1_B4096`** — ไม่ใช่ B3136 ตามที่ comment/RUNBOOK เก่าเขียนไว้
2. **artifact ที่ผ่าน Phase 0 gate (`yolo26n_kv260.xmodel`) แท้จริงคือ YOLOv8n (Track A) — ชื่อไฟล์ตั้งผิด**
   YOLO26n (Track B, NMS-free head) **ยังไม่เคยผ่าน gate**
3. **ไม่ต้อง re-quantize / re-compile** เพื่อแก้เรื่อง arch — โมเดลผ่าน DPU 100% ทั้งบน B3136 และ B4096

---

## ประเด็นที่ 1: arch = B4096 (หลักฐาน 3 แหล่ง ตรงกัน)

| แหล่ง | ระบุ |
|---|---|
| `vai_c_xir_yolo26n_kv260.log:3` (log compile จริง) | `Target architecture: DPUCZDX8G_ISA1_B4096` |
| `meta.json` (runner metadata ฝังใน xmodel) | `"target": "DPUCZDX8G_ISA1_B4096"` |
| `compile_yolo26n.sh:21` ใช้ `/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json` | arch.json default ของ KV260 ใน Vitis AI 3.0 = B4096 |

- xmodel md5 = `69adf8bcd1276feeff6de220bcb39b3e` ตรงกับ `md5sum.txt` → log/meta/xmodel เป็นชุดเดียวกัน
- compile log: `Total device subgraph number 5, DPU subgraph number 1` → ผ่าน gate (DPU subgraph = 1)

### B3136 มาจากไหน (ทั้งหมดเป็นข้อความ/ชื่อไฟล์เก่า ไม่ใช่การ compile จริง)
- comment ใน `compile_yolo26n.sh:5` และ `:55`
- title ใน `RUNBOOK_phase0.md:1`
- ไฟล์ `quantize_result/inspect_DPUCZDX8G_ISA1_B3136.txt` = **Inspector dry-run** ลองยิงกับ target B3136 เพื่อเช็คความเข้ากันได้ ไม่ใช่ผล compile

### ทำไมไม่ต้อง re-compile
`inspect_DPUCZDX8G_ISA1_B3136.txt` และ `inspect_DPUCZDX8G_ISA1_B4096.txt` เขียนตรงกันว่า
`All the operators are assigned to the DPU.` → โมเดลรันบน DPU ได้ครบทั้งสอง arch การเลือก B4096 (default ของบอร์ด) ไม่มีผลเสีย

---

## ประเด็นที่ 2: xmodel ที่ผ่าน gate = YOLOv8n ไม่ใช่ YOLO26n

ไฟล์ชื่อ `yolo26n_kv260.xmodel` แต่เนื้อในเป็น **YOLOv8n** (ยืนยันหลาย layer):

| หลักฐาน | ค่า | แปลว่า |
|---|---|---|
| output channel ของหัว (`inspect_*B4096.txt`) | `[1,20,20,144]`, `[1,40,40,144]` = **144 ch** | 144 = 4×16(DFL) + 80(COCO) = **หัว v8** |
| `quantize_yolo_pytorch.py:105-106` (ตัวแยกในสคริปต์เอง) | v8n → 144 ch (reg_max=16); yolo26n → **84 ch** (reg_max=1, ตัด DFL) | ถ้าเป็น yolo26n ต้องเป็น 84 ไม่ใช่ 144 |
| `source range` ของ output node ใน inspect | ชี้ `quantize_yolo_pytorch.py(162) forward` | traced มาจากสคริปต์ v8 ไม่ใช่ `quantize_yolo26n.py` |
| โครงหัวใน inspect | `Detect[22] / cv2 / cv3` | โครง Detect head แบบ v8 คลาสสิก |
| `quantize_yolo_pytorch.py:247` | `--weights default="yolov8n.pt"` | น้ำหนักต้นทาง = yolov8n |

**หมายเหตุ กับดักที่เกือบพลาด:** มี pipeline yolo26n แยกอยู่จริง
(`export_yolo26n_for_dpu.py` โหลด `yolo26n.pt` → `yolo26n_o2m_leakyrelu.onnx` → `quantize_yolo26n.py`)
แต่ pipeline นี้ **ยังไม่ได้ผลิต xmodel ที่ผ่าน gate** — อย่าตัดสิน identity จากชื่อไฟล์/ชื่อ export script ให้ดู channel count + source range เสมอ

`yolo26n.pt` (md5 `cf3cca69f04cf639bafdeb2644bd0843`) และ `yolov8n.pt` (md5 `0305608151dd1725c9d7da6882ae52d5`) เป็นไฟล์คนละตัวจริง — yolo26n.pt มีอยู่ พร้อมสำหรับ Track B แต่ยังไม่ถูกนำไป quantize/compile สำเร็จ

ชื่อ log ระดับบน `inspect_v8n_b4096.log`, `inspect_v8n_robust.log` = ชื่อตรงกับตัวจริง (เป็น v8) แค่ output xmodel ถูกเซฟชื่อ yolo26n ผิด

---

## สิ่งที่ยังค้าง (จะโผล่ Phase 1 ตอนบอร์ดมา)

- **Fingerprint check**: arch.json ใน docker มี default fingerprint; DPU จริงบนบอร์ดขึ้นกับ overlay ที่โหลดด้วย `xmutil loadapp`
  handover เดิมระบุ fingerprint `0x101000056010407` — พอบอร์ดมา ให้รัน `xdputil query` เทียบก่อนเชื่อผลรันบนบอร์ด (อ้าง `compile_yolo26n.sh:47-56`)
- ยังไม่ได้รัน `xdputil xmodel yolo26n_kv260.xmodel -l` เพราะต้องใช้ใน docker/บอร์ด (ทำไม่ได้บน desktop) — แต่ inspect + compile log เพียงพอยืนยัน partition แล้ว

## แนะนำขั้นต่อไป (ไม่ต้องรอบอร์ด)
1. **เปลี่ยนชื่อ artifact ให้ตรงความจริง** เพื่อกันสับสนในเล่มวิทยานิพนธ์:
   `yolo26n_kv260.xmodel` → ควรสื่อว่าเป็น **Track A / YOLOv8n / B4096** (เช่น `yolov8n_kv260_B4096.xmodel`)
   *ทำกับสำเนา ไม่แตะ backup ต้นฉบับ เพื่อรักษาหลักฐาน*
2. เดินหน้า Track B: quantize `yolo26n_o2m_leakyrelu.onnx` (หรือ `yolo26n.pt` ผ่าน pytorch flow) → คาดว่าหัวจะเป็น 84 ch → compile B4096
3. งานอิสระจากบอร์ด (M4–M8): VART C++ host, CPU emulation, ground-truth dataset ทำล่วงหน้าได้
