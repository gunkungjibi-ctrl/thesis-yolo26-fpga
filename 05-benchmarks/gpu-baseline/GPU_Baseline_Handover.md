# เอกสารสรุปงาน — GPU Baseline Pre-Staging

**โครงงาน:** Real-Time Video-Based Object Counting on FPGA (KV260 / Vitis AI 3.0)
**งานใน session นี้:** สร้าง GPU baseline (high-power reference) เทียบกับ KV260 — งาน Phase 3 ที่ดึงมาทำล่วงหน้าระหว่างรอบอร์ด
**วันที่:** 26 มิถุนายน 2026
**สถานะ:** GPU baseline pre-staging — **เสร็จสมบูรณ์** ✅

> **หมายเหตุสำคัญเรื่อง Phase:** งานนี้ **ไม่ใช่ Phase 1** Phase 1 = board bring-up (ต้องมีบอร์ด KV260 จริง) ซึ่ง**ยังไม่เริ่ม เพราะบอร์ดยังไม่มาถึง** GPU baseline เป็นงานวัดผล (Phase 3) ที่ทำบน host ได้โดยไม่ต้องใช้บอร์ด จึงดึงมาทำล่วงหน้าเพื่อ de-risk

---

## 1. เป้าหมายและที่มา

Thesis ต้องเทียบ KV260 (edge DPU, INT8, ~5W) กับ baseline ฝั่ง high-power 2 แกน: **คุณภาพผลลัพธ์ (accuracy)** และ **พลังงาน (FPS/Watt)** GPU baseline ตัวนี้ให้ตัวเลข FPS/Watt ฝั่ง GPU ที่วัดได้จริง (ไม่ใช่อ้างสเปค)

**หลักการ apple-to-apple:**
- โมเดลเดียวกับ Phase 0: `yolov8n.pt`
- preprocessing ตรงกับ Phase 0 calib เป๊ะ: plain resize 640×640 → BGR→RGB → /255 → CHW (อ้าง `quantize_yolo_pytorch.py::load_calib_batch`)
- input: COCO128 (128 รูปจริง, ตัวเดียวกับ calib)

---

## 2. Hardware & Environment

| | |
|---|---|
| **GPU** | NVIDIA GeForce GTX 1660 Ti **with Max-Q Design** (mobile, 6GB, 60W cap, idle ~3.8W) |
| **สถาปัตยกรรม** | Turing **TU116** — **ไม่มี Tensor Core** (สำคัญต่อการตีความผล) |
| **Host** | LAPTOP-1OPT5Q1T, WSL2 (Ubuntu 25.10 "resolute") |
| **Env** | conda `gpubench` (miniforge), Python 3.12.13 |
| **torch** | 2.11.0**+cu128** (CUDA 12.8 wheel, driver รองรับสูงสุด CUDA 13.0) |
| **ultralytics** | 8.4.75 |
| **power measurement** | NVML ผ่าน `nvidia-ml-py` 13.x (`import pynvml`) — แม่นกว่า parse nvidia-smi |

**หมายเหตุ env ที่เจ็บมาก่อนถึงจุดนี้ (lessons):**
- Python 3.14 (default ของ resolute) **ใช้ไม่ได้** — ไม่มี CUDA wheel → ต้องใช้ Python 3.12 ผ่าน miniforge
- deadsnakes PPA **ใช้ไม่ได้บน resolute** (ไม่ publish repo) → miniforge เป็นทางออก
- `/tmp` เป็น tmpfs 1.4GB → pip OOM ต้องตั้ง `TMPDIR` ไป disk จริง + `--no-cache-dir`
- **Windows C: เคยเหลือ 1.22GB** → WSL `.vhdx` โตไม่ได้ → filesystem พัง (read-only → I/O error → segfault) ต้องเคลียร์ C: ให้เหลือ >12GB ก่อนลง torch (~6GB)
- ลง torch ก่อน แล้ว `pip install ultralytics --no-deps` กันมันลาก torch CPU build มาทับ

---

## 3. ตัวเลข Baseline ทางการ (FP32)

> รายงาน **คู่** เสมอ — real-time (latency-bound) + peak (throughput-bound) เพื่อ reviewer-proof ทั้งสองมุม

| โหมด | throughput | power | ประสิทธิภาพ | uncertainty | ใช้เทียบ |
|---|---|---|---|---|---|
| **Real-time (batch=1)** | 31.5 FPS | 22.8 W | **1.38 FPS/W** | spread ±6 FPS (Max-Q ramp) | KV260 รัน real-time ต่อเฟรม |
| **Peak (batch=32)** | 264 img/s | 58.7 W | **4.50 img/s/W** | นิ่ง (util 96%+) | ขอบบน GPU เต็มศักยภาพ |

**Batch sweep (หาจุดอิ่มตัว):**

| batch | img/s | util | img/s/W |
|---|---|---|---|
| 1 | 25.7 | 33% | 1.21 |
| 4 | 108.5 | 33% | 2.47 |
| 8 | 239.1 | 61% | 4.16 |
| 16 | 259.1 | 83% | 4.50 |
| **32** | **263.9** | **96%** | **4.51** ← peak |
| 64 | 263.9 | 98% | 4.50 (latency เบิ้ล ไม่ได้ throughput เพิ่ม) |

**raw data:** `~/thesis/gpu_baseline/results_fp32_final.json`

---

## 4. ข้อจำกัดที่ต้องเขียนใน methodology (สำคัญ)

1. **Max-Q throttling** — batch=1 มี clock ramp (360–1335 MHz) ทำให้ latency variance สูง (spread ±6 FPS). ใช้ **multi-run median-of-medians** (5 รอบ) ลด outlier แต่ยังมี uncertainty — รายงานเป็นช่วง ไม่ใช่จุดเดียว. `nvmlDeviceSetGpuLockedClocks` **ทำไม่ได้บน WSL** (`Insufficient Permissions`) จึงตรึง clock ไม่ได้

2. **GPU util ต่ำที่ batch=1 (33%)** — YOLOv8n เบาเกินกว่าจะเติม GPU ที่ batch เดียว. นี่เป็น **ลักษณะของ latency-bound serving** ไม่ใช่ข้อบกพร่อง — เป็นเหตุผลว่าทำไมต้องรายงาน peak คู่ด้วย

3. **FP16 ใช้ไม่ได้บน TU116 (finding ที่ verify แล้ว)** — GTX 1660 Ti ตัด Tensor Core ออก, cuBLAS/cuDNN ไม่มี optimized FP16 kernel สำหรับสถาปัตยกรรมนี้:
   - microbenchmark matmul: **FP16 227ms vs FP32 33ms** (FP16 ช้ากว่า ~7 เท่า บน pure matmul ไม่เกี่ยว YOLO)
   - YOLOv8n peak: FP16 ~75 img/s vs FP32 264 img/s
   - ทดสอบทั้ง `model.half()` และ `torch.autocast` (AMP) — ช้าเหมือนกันทั้งคู่
   - เป็น **known issue** ของ TU116 (PyTorch #121957, ultralytics/yolov5 #1866, NVIDIA forum 165310) — ไม่ใช่ข้อจำกัดของ WSL หรือ environment
   - **สรุป: baseline ใช้ FP32 เท่านั้น** ซึ่งสมเหตุผล เพราะ KV260 เทียบเป็น INT8 อยู่แล้ว (edge-DPU-INT8 vs mobile-GPU-FP32)

---

## 5. ไฟล์ที่ได้

| ไฟล์ | คืออะไร |
|---|---|
| `~/thesis/gpu_baseline/gpu_baseline.py` | measurement script (multi-run, sweep, NVML power/util/clock, autocast, OOM guard, JSON save) |
| `~/thesis/gpu_baseline/results_fp32_final.json` | raw data ทางการ (แนบ thesis เป็นภาคผนวก) |

**วิธีรันซ้ำ:**
```bash
source ~/miniforge3/bin/activate && conda activate gpubench
cd ~/thesis/gpu_baseline
python gpu_baseline.py --weights ~/thesis/phase0/yolov8n.pt \
  --images ~/thesis/phase0/calib_images --runs 5 --sweep --save results_fp32_final.json
```

---

## 6. สถานะโครงงานและงานต่อไป

| Phase | สถานะ |
|---|---|
| **Phase 0** (host quantize+compile) | ✅ เสร็จ — `DPU subgraph number 1` |
| **GPU baseline** (Phase 3 ทำล่วงหน้า) | ✅ เสร็จ — FP32 baseline คู่ วัดจริง นิ่ง |
| **Phase 1** (board bring-up) | ⏳ **ยังไม่เริ่ม — รอบอร์ด KV260** |

**Phase 1 gate (เมื่อบอร์ดมา):** `xmutil loadapp` → `xdputil query` เทียบ fingerprint `0x101000056010407` → first live DPU inference ให้ raw 3 tensors `[1,40,40,144]/[1,20,20,144]/[1,80,80,144]` ถูก shape

**Pre-staging ที่ยังทำได้ระหว่างรอบอร์ด (ไม่ต้องใช้บอร์ด):**
- เขียน VART C++ host code ล่วงหน้า (งานก้อนใหญ่สุดของ Phase 1)
- เตรียม Ubuntu boot image (Kria) + DPU firmware checklist — verify ว่าตรงกับ B4096/fingerprint
- Input preprocessing + CPU emulation verify (เทียบ xmodel vs PyTorch float)
- หา/annotate counting ground-truth dataset (conveyor frames) — แทน COCO128 ก่อน Phase 3

**ความเสี่ยง Phase 1 ที่จดไว้:** fingerprint บน Ubuntu image อาจไม่ตรง `0x101000056010407` (DPU overlay ผูกกับ image) → ถ้าไม่ตรงต้อง re-compile xmodel ด้วย arch.json ของบอร์ดจริง
