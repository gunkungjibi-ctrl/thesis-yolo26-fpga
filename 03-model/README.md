# 03-model — Two-track model strategy

Pipeline (ทั้งสอง track เหมือนกัน): **train/export → quantize → compile (.xmodel) → deploy**

โปรเจคเดินสองเคสขนานกัน ไม่ใช่เคสเดียวแล้วค่อยถอย:

| | Track A — YOLOv8n | Track B — YOLO26n |
|---|---|---|
| บทบาท | **baseline ที่พิสูจน์แล้ว** / safety net ของทั้งโครงงาน | **โมเดลเป้าหมายตามชื่อโครงงาน** — ยังพิสูจน์ไม่ผ่าน |
| Phase 0 gate (`DPU subgraph number 1`) | ✅ ผ่าน (มิ.ย. 2026) | ⏳ ยังไม่ผ่าน / ยังไม่ได้รัน |
| Flow ที่ใช้ | vai_q_pytorch → vai_c_xir | vai_q_pytorch → vai_c_xir (เหมือนกัน) |
| ความเสี่ยง | ต่ำ — มี prior work เยอะ | สูง — NMS-free head, ไม่อยู่ใน Model Zoo |
| ถ้าล้ม | โครงงานยังจบได้ | ตกกลับไปใช้ Track A |

**กฎ:** ห้ามลบ/ทับ Track A เพื่อทำ Track B — Track A คือสิ่งที่การันตีว่าโครงงานมีระบบใช้งานได้ Track B คือส่วนที่เพิ่มคุณค่าเชิงวิชาการ (องค์ความรู้เรื่อง deploy โมเดล YOLO ใหม่บน DPU)

---

## โครงสร้าง

```
03-model/
├── common/weights/          weights ที่ใช้ร่วม (yolo26n.pt)
├── track-a-yolov8n-baseline/
│   ├── quantize/  quantize_yolov8n_pytorch.py   ← ตัวที่ทำให้ผ่าน Phase 0
│   ├── compile/   compile_yolov8n.sh
│   └── artifacts/ **ต้องดึง .xmodel + vai_c_xir log จาก WSL มาไว้ที่นี่**
├── track-b-yolo26n-target/
│   ├── export/    export_yolo26n_for_dpu.py
│   ├── quantize/  quantize_yolo26n_pytorch.py
│   ├── compile/   compile_yolo26n.sh  (เขียนใหม่ ชี้ YOLO26nBackboneHead_int.xmodel)
│   └── artifacts/
├── _deprecated/             flow vai_q_onnx เดิม เก็บไว้อ้างอิง ห้ามใช้ต่อ
└── RUNBOOK_phase0_ORIGINAL.md
```

---

## ⚠️ ประเด็นค้างที่ต้องเคลียร์ (ทั้งสอง track)

### 1. DPU target ขัดกันเอง — B3136 vs B4096
| แหล่ง | ระบุว่า |
|---|---|
| `quantize_yolov8n_pytorch.py` → `TARGET` | `DPUCZDX8G_ISA1_B3136` |
| `RUNBOOK_phase0_ORIGINAL.md` | B3136 |
| `GPU_Baseline_Handover.md` / `PreStaging_Plan` | **B4096**, fingerprint `0x101000056010407` |
| `01-docs/platform_selection.docx` | KV260 ใส่ **B4096** ได้ (ตรงกับ Model Zoo) |

ถ้า quantize ด้วย target B3136 แต่บอร์ดโหลด overlay B4096 → **fingerprint check ตอน Phase 1 จะ fail** ต้อง re-quantize + re-compile ใหม่ทั้งชุด
**วิธีเคลียร์:** เปิด `vai_c_xir_*.log` ตัวจริง อ่านว่าใช้ arch.json ตัวไหน + `xdputil xmodel *.xmodel -l` ดู DPU arch ที่ฝังในไฟล์ → แล้วแก้ให้ตรงกันทุกที่ **ทำก่อนลง Track B** ไม่งั้นทำงานซ้ำสองรอบ

### 2. Track A ที่ผ่าน Phase 0 เป็นโมเดล 80 class ไม่ใช่ single-class "package"
output tensors `[1,40,40,144]` → 144 = 64 (DFL) + 80 (COCO classes)
แต่ counting scope ที่ล็อกไว้คือ single-class `package` → **ต้อง re-train/re-quantize/re-compile อีกรอบก่อน Phase 3 อยู่ดี**
ของที่มีตอนนี้พิสูจน์ได้แค่ *compile feasibility* ไม่ใช่โมเดล deployment สุดท้าย

### 3. Track B ยังไม่ผ่านเงื่อนไข accuracy
`export_yolo26n_for_dpu.py` สลับ SiLU → LeakyReLU(0.1015625) ซึ่ง **ทำให้ accuracy เพี้ยน** ต้อง fine-tune หลังสลับก่อนใช้ calibrate ไม่งั้น INT8 ได้ผลขยะ (ตอนนี้ยังไม่ได้ fine-tune)

### 4. artifact จริงยังอยู่ใน WSL `~/thesis/` เท่านั้น
`.xmodel`, `vai_c_xir_*.log`, `quant_info.json`, `results_fp32_final.json`, `calib_images/` — ยังไม่มีสำเนาบนเดสก์ท็อป
WSL `.vhdx` เคยพังมาแล้วหนึ่งรอบ (ดู `07-notes/Session_Summary_GPU_Baseline.md` บั๊กที่ 3) → **copy ออกมาก่อน**
