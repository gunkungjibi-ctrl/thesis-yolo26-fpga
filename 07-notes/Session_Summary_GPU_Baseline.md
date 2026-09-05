# สรุป Session — GPU Baseline Pre-Staging

**โครงงาน:** Real-Time Video-Based Object Counting on FPGA (KV260 / Vitis AI 3.0)
**วันที่:** 26 มิถุนายน 2026
**เป้าหมาย session:** สร้าง GPU baseline (high-power reference) เทียบ KV260 ระหว่างรอบอร์ด
**ผลลัพธ์:** ✅ เสร็จสมบูรณ์ — ได้ FP32 baseline คู่ + raw data + script ที่รันซ้ำได้

---

## ส่วนที่ 1 — สิ่งที่ทำเสร็จ

### 1.1 ตั้ง GPU measurement environment (จากศูนย์)
- สร้าง conda env `gpubench` (miniforge, Python 3.12.13) แยกขาดจาก Vitis AI container
- ลง torch 2.11.0**+cu128** (CUDA build) + torchvision + ultralytics 8.4.75
- ยืนยัน `torch.cuda.is_available() = True` เห็น GTX 1660 Ti Max-Q (gate ของการตั้ง env)

### 1.2 เขียน baseline measurement script (`gpu_baseline.py`)
ความสามารถ:
- โหลด `yolov8n.pt` (โมเดลเดียวกับ Phase 0 — apple-to-apple)
- preprocessing ตรงกับ Phase 0 calib เป๊ะ (plain resize 640 → BGR→RGB → /255 → CHW)
- วัด 3 โหมด: **PURE GPU** (batch=1), **END-TO-END** (ต่อเฟรม + H2D), **BATCH SWEEP** (หา peak)
- power/util/clock ผ่าน **NVML** (`pynvml`)
- **multi-run** (median-of-medians) กัน outlier จาก Max-Q throttle
- **OOM guard** ใน batch sweep (จับ VRAM เต็ม หยุดเอง)
- **autocast** สำหรับ FP16 (แทน manual `.half()`)
- **`--save`** เขียนผลลง JSON

### 1.3 ได้ตัวเลข baseline ทางการ (FP32)
| โหมด | throughput | power | ประสิทธิภาพ | uncertainty |
|---|---|---|---|---|
| **Real-time (batch=1)** | 31.5 FPS | 22.8 W | **1.38 FPS/W** | spread ±6 FPS |
| **Peak (batch=32)** | 264 img/s | 58.7 W | **4.50 img/s/W** | นิ่ง (util 96%) |

raw data: `~/thesis/gpu_baseline/results_fp32_final.json`

---

## ส่วนที่ 2 — บั๊ก/อุปสรรคที่ติด และวิธีแก้

> เรียงตามลำดับที่เจอ — ส่วนใหญ่เป็นปัญหา environment ไม่ใช่ logic

### บั๊กที่ 1 — Python 3.14 ไม่มี CUDA wheel
- **อาการ:** host default เป็น Python 3.14 (Ubuntu 25.10 "resolute"). PyTorch ลงได้แต่เป็น **CPU build** เท่านั้น → `cuda.is_available() = False`
- **สาเหตุ:** Python 3.14 ใหม่เกินไป PyTorch ยังไม่ build CUDA wheel สำหรับ cp314 (verify ผ่าน web + PyTorch issue)
- **วิธีที่ลองแล้วไม่เวิร์ค:** `apt install python3.12` → resolute ไม่มีใน repo. deadsnakes PPA → **ไม่ publish repo สำหรับ resolute** (apt update fail)
- **วิธีแก้:** ใช้ **miniforge** → ได้ Python 3.12 pre-built ใน env แยก ไม่แตะ system Python, ไม่ต้องคอมไพล์
- **บทเรียน:** "ลงสำเร็จ ≠ ได้ของที่ต้องการ" — ต้อง verify `cuda.is_available()` ไม่ใช่แค่ pip ผ่าน (pattern เดียวกับ `[ok] wrote` หลอกใน Phase 0)

### บั๊กที่ 2 — `/tmp` tmpfs เต็ม (pip OOM)
- **อาการ:** `pip install torch` → `[Errno 28] No space left on device` ทั้งที่ disk เหลือ 954GB
- **สาเหตุ:** `/tmp` เป็น **tmpfs (RAM) จำกัด 1.4GB** pip ใช้เป็น scratch unpack torch ก้อนใหญ่ → เต็ม
- **วิธีแก้:** `TMPDIR=~/thesis/gpu_baseline/piptmp pip install --no-cache-dir ...` ชี้ scratch ไป disk จริง
- **บทเรียน:** disk เหลือ ≠ ทุก mount เหลือ — เช็ค `df -h /tmp` ด้วย

### บั๊กที่ 3 — Windows C: เต็ม → WSL filesystem พัง (รุนแรงสุด)
- **อาการ:** ลง torch ไปครึ่งทาง → `Read-only file system` → `Bus error` → `Segmentation fault` กับคำสั่งพื้นฐาน (`touch`, `df`, `rm`)
- **สาเหตุ:** WSL `.vhdx` วางบน Windows C: ที่เหลือแค่ **1.22GB**. torch ~6GB ทำให้ `.vhdx` โตเกินที่ C: มี → filesystem พังกลางคัน (remount read-only เพื่อกันข้อมูลเสีย)
- **วิธีแก้:** `wsl --shutdown` (filesystem กลับมา writable หลัง remount) + เคลียร์ Windows C: ให้เหลือ >12GB (Disk Cleanup) ก่อนลง torch ใหม่
- **ผลกระทบ:** Phase 0 artifacts **ไม่หาย** (เขียนเสร็จก่อนดิสก์เต็มนานแล้ว)
- **บทเรียน:** WSL `df` โกหก (โชว์ provision 1007GB) — ตัวจริงคือพื้นที่ Windows host ที่ `.vhdx` วางอยู่

### บั๊กที่ 4 — ultralytics ลาก torch CPU มาทับ (ป้องกันไว้)
- **ความเสี่ยง:** `pip install ultralytics` ปกติ resolve torch dependency เอง อาจดึง CPU build มาทับ +cu128
- **วิธีแก้:** ลง torch ก่อน แล้ว `pip install ultralytics --no-deps` + ลง dep ที่เหลือเองแบบเลือกตัว
- **ยืนยัน:** หลังลง ultralytics เช็ค `torch.__version__` ยังเป็น `+cu128` ✅

### บั๊กที่ 5 — script bug: `summarize` หายตอนแก้ไฟล์
- **อาการ:** `NameError: name 'summarize' is not defined`
- **สาเหตุ:** ตอนแทรกฟังก์ชัน `time_batch_throughput` หัว `def summarize(...)` ถูกลบไปด้วย (ผิดพลาดตอน str_replace)
- **วิธีแก้:** เติมหัว `def summarize` กลับ + ตั้งกฎเช็ค `grep -c "^def "` ทุกครั้งหลังแก้ (verify before claiming)

### บั๊กที่ 6 — lock GPU clock ไม่ได้บน WSL
- **อาการ:** `--lock-clock 1335` → `Insufficient Permissions`
- **สาเหตุ:** WSL GPU เป็น paravirtualized ผ่าน Windows driver. `nvmlDeviceSetGpuLockedClocks` ต้อง admin บน Windows host ซึ่ง WSL ไม่มีสิทธิ์
- **วิธีแก้ (เปลี่ยนแนวทาง):** lock ไม่ได้ → ใช้ **multi-run median-of-medians** จัดการ variance แทน + รายงาน spread อย่างโปร่งใส
- **บทเรียน:** hard limit ของ environment — ยอมรับแล้วหาทางอื่น ไม่ดันทุรัง

### บั๊กที่ 7 (สำคัญสุดเชิงวิชาการ) — FP16 ช้ากว่า FP32 บน TU116
- **อาการ:** FP16 peak ~75 img/s vs FP32 264 img/s (ช้ากว่า ~3.5 เท่า) ทั้งที่สเปค TU116 บอก FP16 เร็วกว่า 2 เท่า
- **กระบวนการ debug (ตาม debug-mantra):**
  1. **Hypothesis #1:** manual `.half()` ทำให้เกิด FP16↔FP32 cast → แก้เป็น `autocast`
  2. **Disprove:** autocast ก็ช้าเท่าเดิม (75 vs 76) → hypothesis #1 **ผิด**
  3. **Hypothesis ใหม่:** WSL ไม่ให้ FP16 acceleration → **ทดสอบด้วย microbenchmark matmul ล้วน** (ตัด YOLO ออก)
  4. **ผล:** matmul **FP16 227ms vs FP32 33ms** (ช้ากว่า 7 เท่า) → ปัญหาอยู่ระดับ GPU+library ไม่ใช่ YOLO
  5. **Cross-reference + verify:** ค้นเว็บเจอ known issue ตรงรุ่น (PyTorch #121957, yolov5 #1866, NVIDIA forum 165310)
- **สาเหตุที่แท้จริง (verified):** GTX 1660 Ti (TU116) **ตัด Tensor Core ออก** cuBLAS/cuDNN ไม่มี optimized FP16 kernel สำหรับสถาปัตยกรรมนี้ → FP16 fall back ไป path ที่ช้ามาก (สเปค 11 TFLOPS เป็น theoretical ที่ software เข้าไม่ถึง). **ไม่ใช่บั๊ก WSL** (เกิดบน Windows native ด้วย)
- **ข้อสรุป:** ใช้ **FP32 อย่างเดียว** — เป็น finding เข้า thesis ได้ (edge-DPU-INT8 vs mobile-GPU-FP32)
- **บทเรียน:** disprove hypothesis ก่อน + microbenchmark แยกตัวแปร + cross-reference ทำให้ finding แข็งพอเข้าเล่ม

---

## ส่วนที่ 3 — limitation ที่ต้องเขียนใน thesis methodology

1. **Max-Q throttling** — clock ramp 360–1335 MHz, batch=1 variance สูง. ใช้ multi-run median + รายงานเป็นช่วง. lock clock ทำไม่ได้บน WSL
2. **GPU util ต่ำที่ batch=1 (33%)** — YOLOv8n เบาเกินเติม GPU ที่ batch เดียว = ลักษณะ latency-bound serving (ไม่ใช่ข้อบกพร่อง) → จึงต้องรายงาน peak คู่
3. **FP16 ใช้ไม่ได้บน TU116** — known issue (ไม่มี Tensor Core + ไม่มี FP16 kernel). baseline เป็น FP32

---

## ส่วนที่ 4 — สถานะ Phase (สำคัญ)

| Phase | สถานะ |
|---|---|
| Phase 0 (host quantize+compile) | ✅ เสร็จ — `DPU subgraph number 1` |
| GPU baseline (Phase 3 ทำล่วงหน้า) | ✅ เสร็จ — session นี้ |
| **Phase 1 (board bring-up)** | ⏳ **ยังไม่เริ่ม — รอบอร์ด KV260** |

> **ย้ำ:** GPU baseline ≠ Phase 1. Phase 1 = board bring-up (ต้องมีบอร์ดจริง). งานนี้เป็นงานวัดผล (Phase 3) ที่ทำบน host ได้โดยไม่ต้องใช้บอร์ด จึงดึงมาทำล่วงหน้า. **Phase 1 ปิดไม่ได้จนกว่าบอร์ดจะมาถึง**
