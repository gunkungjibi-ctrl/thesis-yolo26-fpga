# 04-deploy/board — VART C++ host code (M4)

โค้ดฝั่ง PS (Arm Cortex-A53) ของ KV260: โหลด `.xmodel` → รัน DPU → ดึง raw output 3 tensors
งานนี้คือ **งานที่ 1 (น้ำหนักสูงสุด)** ของแผน pre-staging — เขียน + compile-check บน host ได้ทั้งหมด **ไม่ต้องรอบอร์ด**

| ไฟล์ | หน้าที่ |
|---|---|
| `yolo_dpu_infer.cpp` | โค้ดหลัก: deserialize xmodel → DPU subgraph → preprocess → run → dequantize → dump `.bin` |
| `CMakeLists.txt` | build config (OpenCV + VART/XIR/unilog/glog) |
| `build.sh` | compile-check (CMake หรือ g++ fallback) |

## Gate ของ M4

> **เขียน + compile-check ผ่านบน host** — ยังไม่ต้องรันจริง (ไม่มี DPU บน host)

รันจริง = Phase 1 (M9) เมื่อบอร์ดมาถึง

## วิธี compile-check (ใน Vitis AI 3.0 container)

โค้ดนี้ต้องการ header/lib ของ VART+XIR ที่มีอยู่ **ในคอนเทนเนอร์เท่านั้น** (ไม่มีบน Windows host)
รันจาก WSL:

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106 \
  bash -lc "cd /workspace/04-deploy/board && ./build.sh"
```

ผ่าน = ได้ไฟล์ `build/yolo_dpu_infer` โดยไม่มี compile/link error
> ⚠️ ชื่อ lib (`vart-runner-ext`, `unilog`) อาจต่างเล็กน้อยตาม build ของคอนเทนเนอร์ —
> ถ้า link ไม่ผ่าน เช็กด้วย `ls /usr/lib | grep -E 'vart|xir|unilog'` แล้วปรับใน `CMakeLists.txt`

## รันบนบอร์ด (Phase 1 — เมื่อบอร์ดมา)

```bash
./yolo_dpu_infer  yolo26n_kv260.xmodel  test.jpg  out/raw
# -> out/raw_80x80.bin  out/raw_40x40.bin  out/raw_20x20.bin  (float32 NHWC)
```

## I/O contract (ยึด `xdputil xmodel <f>.xmodel -l` เป็นความจริง)

- **input** `[1,640,640,3]` **NHWC** int8 — fixpos อ่านจาก tensor ตอน runtime (ไม่ hardcode)
  - preprocessing = **plain resize 640** (ไม่ letterbox) → BGR2RGB → `/255` → quantize `round(x·2^fixpos)` clamp `[-128,127]`
  - ต้องตรงกับ calibration ของ Phase 0 เป๊ะ (`load_calib_batch`) — ไม่งั้น inference เพี้ยนเงียบ ๆ
- **output** 3 tensors int8 NHWC — channel ขึ้นกับหัวโมเดล:
  - YOLOv8n (Track A, artifact ที่ผ่าน gate ตอนนี้): `144 = 64 DFL + 80 class`
  - YOLO26n (Track B, โมเดลเป้าหมาย): `84 = 4 box + 80 class` (reg_max=1, ไม่มี DFL)
  - แยกแต่ละ tensor ด้วย **ขนาด spatial** (`80×80 / 40×40 / 20×20`) ไม่ยึด index — order ของ runtime ไม่การันตี
  - dequantize บน PS: `float = int8 · 2^(-fixpos)` **ก่อน** ส่งไป decode
- **DPU** `DPUCZDX8G_ISA1_B4096`
  - ⚠️ fingerprint ของ xmodel ต้องตรงกับ DPU ที่ `xmutil loadapp` โหลดบนบอร์ดจริง — เช็กตอน Phase 1 ด้วย `xdputil query` (ดู M6 checklist)

## จุดเสี่ยงที่เขียนกันไว้ในโค้ด

1. **NHWC vs NCHW** — VART รับ NHWC; cv2 ให้ HWC อยู่แล้ว จึง **ไม่ transpose** (CHW ใช้เฉพาะ PyTorch path)
2. **fixpos/scale** — อ่านจาก `tensor->get_attr<int>("fix_point")` ทุกครั้ง; re-compile arch อื่น fixpos เปลี่ยนได้
3. **dequantize = งาน PS** — DPU คืน int8; decode (DFL softmax+conv, sigmoid) ทำต่อบน PS ในขั้นถัดไป (ยังไม่รวมในไฟล์นี้)

## ต่อยอด → M5 (CPU emulation verify)

`.bin` ที่ dump ออกมา (float32 NHWC) เอาไปเทียบกับ PyTorch golden ของ `verify_task2/` ได้ตรง ๆ
(max abs err + cosine similarity ต่อ tensor) — bridge M4 → M5 โดยไม่ต้องเขียน preprocessing ซ้ำ
