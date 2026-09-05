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
