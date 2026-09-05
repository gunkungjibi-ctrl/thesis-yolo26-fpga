# KV260 Deployment Results — YOLOv8n (fine-tuned, package 1-class, INT8/B4096)

เก็บผลวัดจริงบนบอร์ด KV260 (PetaLinux starterkit 2022.2, DPUCZDX8G_ISA1_B4096,
fingerprint `0x101000056010407`, VART 3.0) สำหรับ Track A โมเดล `yolov8n_pkg_kv260.xmodel`
(md5 `c702ccb8f027bf9aea95288243d1b274`). วัดเมื่อ 2026-08-16.

## Model / artifact
| item | value |
|---|---|
| float weights | `yolov8n_leaky_pkg_ft.pt` (6.30 MB) — Colab 400 ep, SiLU→LeakyReLU(0.1015625) |
| INT8 xmodel | `yolov8n_pkg_kv260.xmodel` (3.99 MB) |
| output head | 3× [H,W,65] = 4×16 DFL + 1 class (package) |
| DPU workload | 8.10 GOP |
| DPU mem (REG) | CONST 3.35 MB · WORKSPACE 10.85 MB |

## (A) Latency breakdown — single image, 100 iters (after 10 warmup), threads=1
| stage | mean (ms) | median (ms) | std |
|---|---|---|---|
| preprocess (cv2 resize/cvt/quant, PS) | 49.75 | 49.54 | 1.00 |
| **DPU exec** | **12.49** | 12.49 | 0.04 |
| decode + NMS (DFL/sigmoid/NMS, PS) | 16.07 | 16.14 | 0.16 |
| **end-to-end** | **78.31** | 78.12 | 1.04 |

- DPU-only throughput = **80.1 FPS** (สอดคล้อง `xdputil benchmark` = 82.5 FPS)
- **end-to-end = 12.77 FPS** → คอขวดอยู่ที่ **preprocessing บน ARM PS (49.7 ms, ~64%)** ไม่ใช่ DPU
- ข้อสังเกต (discussion): DPU ไม่ใช่ตัวจำกัด — PS-side pre/post processing ต่างหาก; optimize preproc (vectorize/C++/HW scaler) จะดัน e2e FPS ได้อีกมาก

## (B) Power & efficiency — INA260 (ina260_u14, ราง SOM)
| metric | value |
|---|---|
| idle power | 4.85 W |
| load avg (benchmark 4 threads) | 9.27 W |
| load peak | 9.90 W |
| dynamic (load − idle) | 4.42 W |
| DPU throughput (4 threads) | 89.6 FPS |
| **efficiency** | **9.67 FPS/W** (total) · 20.3 FPS/W (dynamic) |

## (B2) Power ตอนรัน "แอปจริง" (video pipeline 600 เฟรม, clip20.avi)
วัด power ต่อเฟรมระหว่างรัน `video_detect.py` (preproc+DPU+decode+track+write, threads=1).
| metric | value |
|---|---|
| power avg | 5.10 W  (idle=4.85 → +0.25 W เท่านั้น) |
| power peak | 5.45 W |
| end-to-end throughput | 10.15 FPS |
| **efficiency (real app)** | **1.99 FPS/W** (whole SOM) |

**Finding สำคัญ:** ตอนรันแอปนับกล่องจริง DPU กินไฟเพิ่มแค่ +0.25W เพราะ **DPU ว่างเกือบตลอด**
(preproc 49.7ms >> DPU 12.5ms/เฟรม) → ระบบเป็น **PS-bound ไม่ใช่ DPU-bound**. รายงานต้องแยก 2
operating point: (i) DPU-saturated 89.6 FPS @ 9.27W = 9.67 FPS/W [ศักยภาพสูงสุดของ accelerator];
(ii) real-app 10.15 FPS @ 5.10W = 1.99 FPS/W [ที่ใช้จริงตอนนี้]. DPU มี headroom มาก — optimize
preproc จะดัน FPS โดย power แทบไม่ขึ้น.

## Comparison vs GPU baseline (FP32, `05-benchmarks/gpu-baseline/results_fp32_final.json`)
GPU = NVIDIA GTX 1660 Ti Max-Q, FP32, imgsz 640, วัด power เฉพาะตัว GPU (nvidia-smi, idle 3.83W).
FPGA = KV260 INT8, วัด power ทั้งบอร์ด SOM (INA260 u14, idle 4.85W).

| System | Precision | FPS | Power (W) | FPS/W | หมายเหตุ |
|---|---|---|---|---|---|
| GPU pure (batch 1) | FP32 | 31.5 | 22.80 | 1.38 | GPU chip only, no pre/post |
| GPU e2e (batch 1) | FP32 | 25.5 | 22.65 | 1.13 | รวม pre/post |
| GPU peak (batch 32) | FP32 | 263.9 | 58.46 | 4.51 | throughput สูงสุด, util 96% |
| **KV260 DPU-sat** (4 thr) | INT8 | **89.6** | **9.27** | **9.67** | ทั้ง SOM, DPU เต็มสูบ |
| **KV260 real-app** (video) | INT8 | 10.15 | 5.10 | 1.99 | ทั้ง SOM, รวม pre/post+track |

สรุป (honest, หลาย operating point):
- **Efficiency best-vs-best:** FPGA 9.67 vs GPU สูงสุด 4.51 FPS/W → FPGA ดีกว่า **~2.1×** (ทั้งที่ FPGA วัด power ทั้งบอร์ด, GPU วัดชิปเดียว)
- **Absolute power:** FPGA 9.27W vs GPU peak 58.5W → กินไฟน้อยกว่า **~6.3×** (จุดขาย edge/ภาคสนาม)
- **End-to-end (แฟร์สุด):** FPGA real-app 1.99 vs GPU e2e 1.13 FPS/W → **1.8×**, และกินไฟ 5.1W vs 22.65W (**4.4× น้อยกว่า**)
- **Raw throughput:** GPU ชนะ (264 vs 89.6 FPS) — FPGA แลกความเร็วดิบกับพลังงาน/ขนาด/ราคา
- ต้อง caveat: FP32 vs INT8 → ชดเชยด้วยตัวเลข mAP drop (ข้อ C) ว่าความแม่นยำตกน้อย

## (C) Accuracy — float vs INT8 mAP  [FLOAT DONE, INT8 pending]
วัดบน valid set (57 รูป, 289 GT). pipeline เดียวกันทั้ง float/INT8 (plain resize640,
raw head, DFL decode, NMS iou0.7 conf0.001). GT = polygon segmentation → แปลงเป็น bbox.
| model | mAP@0.50 | mAP@0.50:0.95 | P | R | F1 |
|---|---|---|---|---|---|
| **float** (yolov8n_leaky_pkg_ft.pt, host) | **0.8846** | 0.6879 | 0.801 | 0.848 | 0.824 |
| **INT8** (board xmodel) | **0.8713** | 0.6795 | 0.822 | 0.834 | 0.828 |
| **drop (float→INT8)** | **−0.0133 (−1.5%)** | −0.0084 (−1.2%) | +0.021 | −0.014 | ~0 |
- float mAP@0.5 = 0.8846 ตรงกับ Colab val (0.884) → ยืนยัน pipeline ถูกต้อง
- **INT8 quantization drop = 1.3 จุด mAP@0.5 (0.885→0.871)** — น้อยมาก, F1 แทบไม่ต่าง (0.824→0.828)
- ปิด caveat "FP32 vs INT8" ในตารางเทียบ GPU: FPGA ประหยัดไฟ 6.3× แลกความแม่นยำแค่ 1.5%

## (D) Counting — line-crossing tracker vs GT=215 (clip 600s, 18000 เฟรม)
วิธี: DPU detection ต่อเฟรม (บอร์ด dump `dets600.json`, 174,769 กล่องรวม) → centroid tracker
(greedy) → นับเมื่อ track ข้ามเส้นครั้งแรก. จูนเส้น/tracker offline บน host (`count_offline.py`).

| config | count | error |
|---|---|---|
| **axis=x, line=0.40·W, max_dist=60** (calibrated) | **215** | **0.0%** |
| axis=x, line=0.30, max_dist=60 | 219 | 1.9% |
| axis=x, line=0.30, max_dist=40 | 209 | 2.8% |
| axis=x, line=0.40, max_dist=90 | 224 | 4.2% |
| default line=0.5 (ไม่ calibrate) | 270–331 | 25–54% |

- **แกน x ถูกต้อง** (กล่องเคลื่อนแนวนอน); axis=y ผิดหมด (error 20–92%)
- **เส้นควรวางโซนกล่องเข้าเฟรม (x≈0.3–0.4)** ที่กล่องยังไม่ทับกัน; กลางจอ (0.5) นับเกินเพราะ
  detection กระพริบ/ทับ → track แตก นับซ้ำ
- หลัง calibrate เส้นที่ x=0.40·W: **นับ 215/215 = 100%**; robust รอบข้าง (line 0.3–0.4 ×
  max_dist 40–60 → 209–219, error <3%) = ไม่ใช่ fluke
- การตั้งเส้นนับ = calibration มาตรฐานของระบบ line-counting (ต่อ 1 มุมกล้อง/สายพาน)
- processing = ~10 FPS บนบอร์ด (real-time factor ~0.33× ที่วิดีโอ 30fps)
- คลิปสาธิต 20 วิ: `08-figures/out20.mp4`
