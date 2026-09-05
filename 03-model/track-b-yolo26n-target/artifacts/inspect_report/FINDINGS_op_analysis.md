# M2-B1 — YOLO26n op analysis on DPUCZDX8G (B4096)

วันที่ 2026-08-15 · export ผ่าน env py3.9 แยก (`D:\yolo26-export-env`, ultralytics 8.4.71 / torch 2.8.0+cpu / onnx 1.19.1)
ไฟล์: `yolo26n_o2m_leakyrelu.onnx` (457 nodes) · histogram: `op_histogram.json`

## วิธีทำ (แทน nndct Inspector ที่รันไม่ได้)
nndct Inspector รันไม่ได้เพราะ python clash (ดู `07-notes/M2B1_inspector_python_blocker.md`)
→ ใช้ทางเลือก 1: export ONNX (`end2end=False`, SiLU→LeakyReLU) แล้ววิเคราะห์ op graph เอง

## ✅ ข่าวดี: NMS-free head ไม่เป็นปัญหา
ตั้ง `end2end=False` → **watchlist ops หายหมด**: ไม่มี TopK, GatherElements, GatherND, ReduceMax, Mod, ScatterND, NonMaxSuppression, Range
(postprocess ที่สร้าง TopK×2 + GatherElements×2 ถูกข้าม; decode/NMS ไปทำบน ARM PS ตอน deploy)
SiLU→LeakyReLU: แทนครบ (0 SiLU / 105+ LeakyReLU)

## 🔴 ตัวบล็อกจริง: attention block 2 จุด "กลาง" เน็ตเวิร์ก
| ตำแหน่ง | โมดูล | ops |
|---|---|---|
| node ~148–153 (**~32%**) | `/model.10/m/m.0/attn` | MatMul → Softmax → MatMul |
| node ~315–320 (**~69%**) | `/model.22/m.0/m.0.1/attn` | MatMul → Softmax → MatMul |

โครงสร้าง Q·Kᵀ → Softmax → ·V (self-attention). บน DPUCZDX8G (Zynq US+):
- **MatMul แบบ data×data** (ไม่ใช่ conv/weight-constant) → DPU map ไม่ได้
- **Softmax** → ไม่ใช่ DPU op → ตกไป ARM PS

เพราะอยู่ที่ ~32% และ ~69% (**กลางกราฟ ไม่ใช่ปลาย**) → จะ**ตัด DPU subgraph เป็น ~3 ชิ้น**
→ **gate "1 DPU subgraph" ผ่านไม่ได้ตามสภาพ** (นี่คือ mechanism ที่แท้จริง)

## op อื่นที่ไม่ใช่ปัญหา
- MaxPool×3 (~28%, SPPF) — DPU รองรับ
- Resize×2 (~37%/46%, FPN upsample nearest) — DPU รองรับ
- Sigmoid×1 (~99.6%, ปลายสุด decode) — ปลายกราฟ ไม่ตัด subgraph

## สรุป/ทางเลือก Track B (ให้ผู้สานต่อตัดสิน)
ตัวบล็อก YOLO26n บน DPUCZDX8G **ไม่ใช่ NMS-free head** (แก้ได้ด้วย end2end=False) **แต่คือ attention (A2C2f/PSA) 2 จุดกลางเน็ต**

1. **ยอมรับ multi-subgraph** — ปล่อย attention รันบน PS, วัด latency penalty จริง → รายงานเป็น finding
2. **ถอด/แทน attention ด้วยบล็อก conv-only แล้ว retrain** — ให้เหลือ 1 subgraph (งานหนัก + กระทบ accuracy)
3. **เขียนเป็น contribution** — "YOLO26 ติด DPUCZDX8G ตรง attention MatMul+Softmax กลางกราฟ" คือองค์ความรู้ที่จับต้องได้

**กติกาสำคัญ:** histogram นี้เป็น advisory. gate จริงยังคือ `vai_c_xir` log — ควรลอง compile จริงเพื่อยืนยันว่าตัดเป็นกี่ subgraph (ยืนยัน mechanism ข้างบน)

---

## ภาคผนวก (2026-09-05) — ยืนยันจาก source ของโมเดล ไม่ใช่แค่ ONNX node name

ตอนเตรียม M2-B3 ได้ introspect โมเดลที่ ultralytics build จริง (`yolo26n.yaml`, scale n) → ยืนยันข้อสรุปข้างบนครบ และได้รายละเอียดเพิ่ม

### attention 2 จุด — ยืนยันตรงกับที่อ่านจาก ONNX

| เลเยอร์ | โมดูลจริง | ที่มาของ attention |
|---|---|---|
| `model.10` | `C2PSA(256, 256, n=1)` | `.m[0]` = `PSABlock` → `.attn` = `Attention(dim=128, heads=2, key_dim=32, head_dim=64)` |
| `model.22` | `C3k2(384, 256, n=1, c3k=True, e=0.5, **attn=True**)` | `.m[0]` = `Sequential(Bottleneck, PSABlock)` → `.m[0][1].attn` = `Attention(dim=128, heads=2, key_dim=32, head_dim=64)` |

`model.22` มี attention เพราะ arg ตัวที่ 4 ใน yaml (`[-1, 1, C3k2, [1024, True, 0.5, True]]`) map เข้า
`C3k2.__init__(..., c3k=True, e=0.5, attn=True)` — **ไม่ใช่ C3k2 ธรรมดา** ตรงกับ path `/model.22/m.0/m.0.1/attn` ที่เจอใน ONNX เป๊ะ

source ของ `Attention.forward` ยืนยัน mechanism: `q.transpose(-2,-1) @ k` → `* scale` → `.softmax(dim=-1)` → `v @ attn.transpose(-2,-1)`
= MatMul(data×data) → Softmax → MatMul(data×data) ตามที่วิเคราะห์ไว้

### 🔴 ข้อค้นพบใหม่ที่ ONNX histogram ไม่ได้ชี้: **YOLO26 ไม่มี DFL**

`yolo26.yaml` ตั้ง `reg_max: 1` → `Detect.dfl = nn.Identity()`

| | Track A (YOLOv8n) | Track B (YOLO26n) |
|---|---|---|
| `reg_max` | 16 | **1** |
| channel ต่อหัว (single class) | 4×16 + 1 = **65** | 4×1 + 1 = **5** |
| decode ฝั่ง PS | softmax 16 bin ต่อด้าน × arange | **ไม่มี DFL** — อ่าน l,t,r,b ตรงๆ |

**ผลกระทบ:** `04-deploy/board/yolo_dpu_detect.py` (เขียนไว้สำหรับ 65 ch + DFL) **ใช้กับ YOLO26n ไม่ได้ทันที** ต้องแก้ decoder
แต่เป็นข่าวดีเชิง performance — ตัด softmax+matmul ต่อ anchor ออกจาก PS ซึ่งเป็นฝั่งที่เป็นคอขวดอยู่แล้ว (preproc+decode = 84% ของ pipeline)

### หมายเหตุ: YOLO26 มี 2 หัว

`end2end: True` ใน yaml ทำให้ `Detect` สร้าง `one2one_cv2/one2one_cv3` เป็น deepcopy ของ `cv2/cv3`
- **`cv2/cv3`** = one-to-many ใช้คู่กับ NMS → **flow ปัจจุบันเลือกอันนี้** (ให้เทียบกับ Track A ได้ตรงๆ, NMS อยู่บน PS เหมือนกัน)
- **`one2one_cv2/cv3`** = NMS-free branch → เลือกได้ด้วย `--use_one2one`

การตั้ง `end2end=False` ตอน export (ที่ทำใน M2-B1) = เลือกเดินหัว o2m ซึ่งเป็นเหตุผลที่ TopK/GatherElements หายไป — ตรงกับที่สรุปไว้

---

## M2-B3 (2026-09-05) — รัน `vai_c_xir` จริง: พังเร็วกว่าที่คาด แต่ตรงจุดเดิม

**สภาพแวดล้อม:** Docker Desktop เดสก์ท็อป, `xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106`, conda env `vitis-ai-pytorch` (py3.7.12, torch 1.12.1) — ตรง spec ทุกข้อ

**ผล:**
- **Pass 1 (calibrate, 32 รูปจาก `02-dataset/calib/images/`) ผ่านสมบูรณ์** — forward ครบ 32/32, เขียน `quant_info.json` สำเร็จ
- **Pass 2 (`--quant_mode test --deploy`, export เป็น XIR xmodel) ล้มเหลว** — **ไม่ถึงขั้นรัน `vai_c_xir` ด้วยซ้ำ** พังตั้งแต่ขั้น nndct แปลง traced graph → XIR graph (`export_xmodel` / `dump_xmodel`)

**Error เต็ม:**
```
[VAIQ_ERROR][QUANTIZER_TORCH_EXPORT_XMODEL]: Failed convert graph 'YOLO26nBackboneHead' to xmodel.
...
nndct_shared.utils.exception.AddXopError: Failed to add op(
  name:YOLO26nBackboneHead::YOLO26nBackboneHead/C2PSA[model]/ModuleList[10]/Sequential[m]/PSABlock[0]/Attention[attn]/22573,
  type:nndct_elemwise_mul) in xGraph.: 'Caught an unknown exception!'
```
(ก่อนหน้านั้นมี `[VAIQ_WARN]: YOLO26nBackboneHead::<id> is not tensor.` อีก 10 บรรทัด สำหรับ id แถว `model.10` และ `model.22` — สอดคล้องกับ node ในบล็อก attention ทั้งสองจุด)

log เต็ม: `03-model/track-b-yolo26n-target/quantize/export_xmodel_FAIL_2026-09-05.log` (pass 1 log: `quantize_calib_pass1_2026-09-05.log`)

**อ่าน error นี้ยังไง — ทำไมยัง "ตรงจุดเดิม" ที่ M2-B1 ทำนายไว้:**

ดู `Attention.forward` (`yolo26n_dpu.py:159`):
```python
attn = (q.transpose(-2, -1) @ k) * self.scale   # <- op ที่พัง: elemwise_mul
attn = attn.softmax(dim=-1)
```
`self.scale` เป็น **python float คงที่** (`key_dim ** -0.5`) ไม่ใช่ tensor/weight ที่ผ่าน quantizer — พอ nndct พยายามสร้าง `nndct_elemwise_mul` (fixed-point binary op) จาก tensor (ผลจาก matmul, ผ่าน quant) คูณกับ scalar constant ตัวนี้ ตัว `xgraph.create_fixed_normal_op` ใน C++ layer โยน exception ที่ไม่มีข้อความอธิบาย (`'Caught an unknown exception!'`) — เป็น edge case ที่ nndct's XIR converter ไม่รองรับ ไม่ใช่บั๊กใน `yolo26n_dpu.py`

**นี่คือ node แรกสุดของ path attention ที่ M2-B1 ชี้ไว้ (`model.10` = `C2PSA[model]`) — ยืนยันตรงเป๊ะว่าจุดที่ตกคือ attention block ตามที่วิเคราะห์จาก ONNX/source ไว้ก่อนหน้า เพียงแต่ตัวที่ตกจริงคือ scale-multiply ก่อนถึง softmax ไม่ใช่ matmul หรือ softmax เอง**

**สรุปตามตารางตัดสินในบรีฟ:** ผลออกมาเป็นแถว **"compile error — มี op ที่ compiler ปฏิเสธตรงๆ"** — เป็น contribution ของ track นี้ตามที่ตกลงไว้ ไม่ใช่ความล้มเหลวของงาน

**ยังไม่ได้ลอง:** เปลี่ยน `* self.scale` เป็นคูณก่อน matmul (`(q * (self.scale ** 0.5)).transpose(-2,-1) @ (k * (self.scale ** 0.5))`) เผื่อเลี่ยง binary-op-กับ-scalar-constant นี้ได้ — ยังไม่ verify ว่าจะเปลี่ยนผลลัพธ์เชิงตัวเลขหรือไม่ (ต้องเช็ค bit-exact ใหม่ถ้าจะลอง) → ทิ้งไว้เป็นข้อเสนอสำหรับคนที่ทำต่อ ไม่ใช่ทำต่อในรอบนี้เพราะบรีฟระบุว่า compile error ก็เป็น finding ที่ปิดคำถามได้แล้ว
