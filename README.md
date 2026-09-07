# Real-Time Video-Based Object Counting on FPGA Using YOLO26

Senior thesis / capstone project · อัปเดต 5 ก.ย. 2026

> 📊 สถานะละเอียด = `00-admin/timeline.md` (single source of truth) · แดชบอร์ด = `00-admin/dashboard.html`
> 📄 เอกสารเล่าให้อาจารย์ = `00-admin/advisor-brief/` (12 ไฟล์) หรือฉบับรวม `00-admin/ADVISOR_BRIEF_2026-08-20.md`

## สถานะปัจจุบัน — 18/20 milestone (90%)

**ระบบทำงานครบวงจรบนบอร์ดจริงแล้ว และเก็บผลวัดครบ 4 หมวด**

| งาน | สถานะ |
|---|---|
| Platform selection (KV260) | ✅ `01-docs/platform_selection.docx` |
| **Phase 0** — host quantize + compile | ✅ ผ่าน **ด้วย YOLOv8n** (`DPU subgraph number 1`, arch B4096) |
| Dataset + fine-tune (single-class `package`) | ✅ Roboflow `package-conveyo v2` · Colab T4 400 epochs |
| **Phase 1** — board bring-up (M9) | ✅ บอร์ดมาถึง 16 ส.ค. 2026 · fingerprint ตรง · `xdputil benchmark` 82.5 FPS Test PASS |
| VART host code + detection app | ✅ `04-deploy/board/` — C++ compile-check + `yolo_dpu_detect.py` เห็นกล่องจริงบน DPU |
| **Phase 2** — tracking + line-crossing counting (M10) | ✅ **นับ 215/215 = error 0.0%** เทียบ ground truth |
| **Phase 3** — benchmark เทียบ GPU baseline (M11–M12) | ✅ ครบ accuracy / latency / power / counting |
| **Track B** — YOLO26n compile gate (M2-B3) | ✅ compile จริงด้วย `vai_c_xir` แล้ว — ล้มที่ scale-multiply ใน attention block แรก (`nndct_elemwise_mul`), ปิด RQ1 ด้วย compile error ตามกติกาที่ตั้งไว้ |
| รายงาน + สไลด์ (M14) | 🟡 **งานหลักที่เหลือ** — ร่างบท Results เสร็จ + ใส่ผล M2-B3 เข้า Track B แล้ว · เหลือขัดสำนวน + ใส่รูป + บทที่เหลือ |

## ผลการทดลอง (วัดจริงบนบอร์ด)

| หมวด | ผล |
|---|---|
| Accuracy float | mAP@0.5 = **0.8846** · mAP@0.5:0.95 = 0.688 |
| Accuracy INT8 บน DPU | mAP@0.5 = **0.8713** → **drop เพียง 1.5%** |
| Throughput DPU ล้วน | **82.5 FPS** (1-thread) · 89.6 FPS (4-thread) |
| Throughput e2e app | 12.8 FPS (78.3 ms/เฟรม) |
| Power / efficiency | 9.27 W @ **9.67 FPS/W** (GPU ดีสุด 4.51 FPS/W @ 58.5 W) → **ดีกว่า 2.1× · กินไฟน้อยกว่า 6.3×** |
| Bottleneck | **PS-bound** — preproc 49.7 ms (63%) · DPU 12.5 ms (16%) · decode+NMS 16.1 ms (21%) |
| **Counting (โจทย์หลัก)** | GT = 215 → นับได้ **215 · error 0.0%** |

ผลดิบทั้งหมด: `05-benchmarks/results/kv260_results.md` · ร่างบท Results: `RESULTS_chapter_draft.md`

## ข้อค้นพบหลัก 3 ข้อ

1. **INT8 quantization แทบไม่ทำให้ความแม่นยำเสีย** (ตก 1.5%) แต่ประหยัดไฟ ~6 เท่า → คุ้มมากสำหรับงาน edge
2. **คอขวดไม่ได้อยู่ที่ตัวเร่ง AI แต่อยู่ที่ CPU** — DPU กินแค่ 16% ของ pipeline ส่วน preprocessing บน ARM กิน 63%
3. **YOLO26 map ลง DPUCZDX8G ไม่ได้ทั้งตัว** — NMS-free head *ไม่ใช่* ตัวปัญหา ยืนยันด้วย compile จริงว่าตกที่ **scale-multiply ก่อน softmax ใน attention block แรก** (`model.10`, op `nndct_elemwise_mul`) ไม่ใช่แค่ matmul/softmax ตามที่คาดจาก ONNX

## Target platform
- **Board:** AMD Kria KV260 Vision AI Starter Kit — Zynq UltraScale+ MPSoC
- **Detection:** DPUCZDX8G **B4096** บน programmable logic — fingerprint `0x101000056010407` (ยืนยันแล้วใน P1 + `xdputil query` บนบอร์ด)
- **Tracking / counting:** Arm Cortex-A53 (PS) — centroid tracker + line-crossing
- **Toolchain:** Vitis AI 3.0
- **Docker (pinned):** `xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106` — **ห้ามใช้ `:latest`** (เป็น 3.5 ซึ่งไม่มี KV260 prebuilt + board_setup/mpsoc)
- **Boot image:** **PetaLinux starter kit** `xilinx-kv260-starterkit-2022.2` (เปลี่ยนจากแผนเดิมที่จะใช้ Ubuntu — DPU B4096 + VART 3.0 ครบ ใช้ deploy ได้เลย)
- **Runtime:** VART (C++ compile-check ผ่าน · แอปที่ใช้วัดผลจริงเป็น Python VART)

## Pipeline
```
train/export → SiLU→LeakyReLU → fine-tune → quantize → compile (.xmodel) → deploy → track+count
```
ห้ามข้ามขั้น compile — `vai_c_xir` log คือ gate เดียวที่นับ ไม่ใช่ op histogram หรือ Inspector

## กลยุทธ์โมเดล: เดินสองเคสขนานกัน

| | **Track A — YOLOv8n** | **Track B — YOLO26n** |
|---|---|---|
| บทบาท | baseline ที่พิสูจน์แล้ว / safety net | โมเดลเป้าหมายตามชื่อโครงงาน |
| Compile gate | ✅ ผ่าน (1 subgraph, 65 ch single-class) | ✅ compile จริงแล้ว — **ไม่ผ่าน** ล้มที่ scale-multiply ใน attention (M2-B3) |
| Deploy บนบอร์ด | ✅ ครบวงจร + วัดผลครบ | — (ไม่ผ่าน compile gate จึง deploy ไม่ได้ — ผลลัพธ์ที่ตั้งใจวัด) |
| accuracy หลัง fine-tune | mAP50 = 0.884 | mAP50 = 0.831 |
| ความเสี่ยง (ก่อน compile) | ต่ำ | สูง — attention block map ไม่ได้ → **ยืนยันแล้วด้วย compile จริง** |

Track A ทำให้โครงงาน**จบได้แล้ว** · Track B คือส่วนที่เพิ่มคุณค่าเชิงวิชาการ
**ถ้า Track B ล้ม ผลนั้นยังเป็น contribution** — องค์ความรู้ว่า YOLO26 ติด op ไหนบน DPUCZDX8G เขียนเข้าเล่มได้เต็มๆ

## Contribution
- **Primary:** ระบบนับวัตถุ real-time บน heterogeneous FPGA SoC + ผลวัดเชิงประจักษ์ + องค์ความรู้ข้อจำกัดการ deploy YOLO โมเดลใหม่
- **Optional extension:** optimize preprocessing / custom accelerator สำหรับ bottleneck — **ไม่ผูกเป็นเงื่อนไขจบโปรเจค**
- **โปรเจคคู่ขนาน:** port ไป Efinix Ti375 (TFLite Micro + Sapphire RISC-V) เพื่อเทียบข้ามแพลตฟอร์ม — repo แยก `thesis-yolo26-fpga_efinix` (E3-A เสร็จ, gate แรก E1 รอเครื่องที่มี Efinity)

## Folder map
| Folder | ใช้ทำอะไร |
|---|---|
| `00-admin/` | timeline, dashboard, เอกสารส่งอาจารย์ (`advisor-brief/`) |
| `01-docs/` | รายงาน, architecture, platform selection |
| `02-dataset/` | detection dataset / counting ground-truth (`clip_600s_gt215.mp4`) / calib frames |
| `03-model/` | **Track A + Track B** — export → quantize → compile |
| `04-deploy/` | VART host code (PS), detection app, tracking/counting, สคริปต์วัดผลบนบอร์ด |
| `05-benchmarks/` | GPU baseline + ผลวัดบนบอร์ด + ร่างบท Results |
| `06-references/` | เปเปอร์, AMD docs |
| `07-notes/` | worklog, session summary, บั๊กที่เจอ |
| `08-figures/` | รูป detection บนบอร์ด + วิดีโอ demo |

## งานถัดไป
1. **M14** — finalize บทรายงาน + สไลด์ (งานหลักที่เหลืออยู่ตอนนี้)
2. **Efinix E1** — ติดตั้ง Efinity + build Sapphire SoC ที่เครื่องที่มีบอร์ด
3. *(optional)* **M13** — optimize preprocessing: 49.7 → 10 ms ⇒ e2e 12.8 → 25.9 FPS
   · 🟡 **7 ก.ย.:** PL accelerator ออกแบบเสร็จ `04-deploy/pl-preproc/` (HLS kernel bit-exact กับ cv2 + XRT host lib) · เหลือ synth/bitstream + วัดบนบอร์ด · พบว่าโหมด `lut` (SW ล้วน) น่าจะเก็บ gain ส่วนใหญ่ได้ก่อน — ต้องวัด 3 จุด numpy/lut/hw
