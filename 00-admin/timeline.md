# Timeline / Milestones

อัปเดต 23 ส.ค. 2026

> 🔗 **ไฟล์นี้ = single source of truth ของสถานะโปรเจค.** `00-admin/dashboard.html` ถูก generate จากตารางในไฟล์นี้ — แก้สถานะที่นี่ที่เดียว แล้วบอก Claude ว่า **"sync dashboard"** เพื่ออัปเดตแดชบอร์ดให้ตรงกัน (กติกาแปลงสถานะ: ✅ = เสร็จ · `ติดบล็อก?`=ใช่ → กลุ่มติดบล็อก · นอกนั้น = ทำได้เลย)

> ⚠️ **หมายเหตุการเปลี่ยนคอลัมน์ (23 ส.ค. 2026):** เดิมคอลัมน์ที่ 4 คือ `ต้องมีบอร์ด?` ใช้บอกว่างานติดรอบอร์ด — **บอร์ด KV260 มาถึงแล้ว 16 ส.ค. 2026** จึงเปลี่ยนความหมายเป็น `ติดบล็อก?` (มีอะไรขวางอยู่หรือไม่) สคริปต์ sync ยังอ่านคอลัมน์ตำแหน่งเดิม จึงทำงานได้ตามปกติ

---

## 🎯 สรุปสถานะ 3 บรรทัด

- **ระบบหลัก (KV260) ทำงานครบวงจรบนบอร์ดจริงแล้ว** และเก็บผลวัดครบ 4 หมวด: accuracy / latency / power / counting
- **โจทย์หลักผ่าน:** นับกล่องบนคลิป 600 วินาที ได้ **215/215 = error 0.0%** เทียบ ground truth
- **M2-B3 เสร็จแล้ว (2026-09-05)** — YOLO26n compile ไม่ผ่าน แต่รู้สาเหตุระดับ op เป๊ะ (attention scale-multiply) → ปิดคำถามวิจัย Track B
- เหลือ **M14** (finalize รายงาน) · M13 optional

### ตัวเลขผลการทดลองที่ได้แล้ว

| หมวด | ผลจริง |
|---|---|
| Accuracy — float | mAP@0.5 = **0.8846** · mAP@0.5:0.95 = 0.688 (valid 57 รูป) |
| Accuracy — INT8 บน DPU | mAP@0.5 = **0.8713** (drop **1.5%**) · mAP@0.5:0.95 = 0.6795 (drop 1.2%) |
| Throughput — DPU ล้วน | **82.5 FPS** single-thread · 89.6 FPS 4-thread |
| Throughput — e2e app | 12.8 FPS (78.3 ms/เฟรม) |
| Power | DPU-saturated 9.27 W @ **9.67 FPS/W** · แอปวิดีโอจริง 5.10 W @ 1.99 FPS/W |
| เทียบ GPU baseline | GPU ดีสุด 4.51 FPS/W @ 58.5 W → FPGA **ดีกว่า 2.1×** และกินไฟ **น้อยกว่า 6.3×** |
| Bottleneck | **PS-bound** — preproc 49.7 ms (63%) · DPU 12.5 ms (16%) · decode+NMS 16.1 ms (21%) |
| Counting | GT = 215 → นับได้ **215 (0.0% error)** · robust ที่ line 0.3–0.4×W, max_dist 40–60 |

---

## เสร็จแล้ว

| # | Milestone | หลักฐาน | สถานะ |
|---|---|---|---|
| M0 | เลือก target board + toolchain (KV260 / Vitis AI 3.0) | `01-docs/platform_selection.docx` | ✅ |
| M1 | ตั้ง Vitis AI 3.0 container + flow ครบ | `03-model/RUNBOOK_phase0_ORIGINAL.md` | ✅ |
| M2-A | **Track A: YOLOv8n compile ผ่าน** | `DPU subgraph number 1` | ✅ |
| M3-GPU | GPU baseline (FP32 คู่: 31.5 FPS @22.8W / 264 img/s @58.7W; batch32 peak 4.51 FPS/W) | `05-benchmarks/gpu-baseline/results_fp32_final.json` | ✅ |
| P0 | **กู้ artifact ออกจาก WSL มาไว้บนเดสก์ท็อป** | 162 ไฟล์ verify byte-exact → `03-model/phase0-wsl-recovered/` | ✅ |
| P1 | **เคลียร์ B3136 vs B4096 + identity ของ xmodel** | `07-notes/P1_arch_and_identity_resolution.md` — arch จริง = **B4096**; xmodel เดิมคือ YOLOv8n ไม่ใช่ YOLO26n | ✅ |
| M2-B1 | Track B: วิเคราะห์ op ของ YOLO26n | `FINDINGS_op_analysis.md` — NMS-free head **ไม่เป็นปัญหา** (end2end=False) แต่เจอ **attention 2 จุด** (`model.10` ~32%, `model.22` ~69%: MatMul+Softmax) ที่ DPU map ไม่ได้ | ✅ |
| M2-B2 | Track B: fine-tune หลัง SiLU→LeakyReLU | Colab T4 400 epochs — **yolov8n mAP50 = 0.884**, **yolo26n mAP50 = 0.831** → พิสูจน์ว่า activation swap ไม่ทำ accuracy ตกถ้าเทรนพอ | ✅ |
| M4 | VART host code (pre-staging น้ำหนักสูงสุด) | `04-deploy/board/` — C++ compile-check ผ่าน + **`yolo_dpu_detect.py`** (preproc → DPU → DFL decode → sigmoid → NMS) รันบนบอร์ดจริง | ✅ |
| M5 | Verify quantization fidelity | ปิดด้วยหลักฐานที่ถูกต้องกว่า raw-tensor cosine: **INT8 mAP drop เพียง 1.5%** (`infer_dump.py` + `eval_map.py`) · บันทึกการขุด root cause เดิมไว้ที่ `07-notes/M5_cossim_dropoff_diagnosis.md` | ✅ |
| M6 | Boot image + DPU firmware checklist | **เปลี่ยนจาก Ubuntu → PetaLinux starter kit** (`xilinx-kv260-starterkit-2022.2`) — มี DPU B4096 + VART 3.0 ครบ, `xdputil query` fingerprint `0x101000056010407` **ตรง xmodel เป๊ะ** | ✅ |
| M7 | Counting ground-truth dataset | `02-dataset/counting-eval/videos/clip_600s_gt215.mp4` (600 s, 18001 เฟรม) — **GT = 215 กล่อง** | ✅ |
| M8 | re-quantize เป็น single-class | `artifacts/yolov8n_pkg_B4096/yolov8n_pkg_kv260.xmodel` — output 3 หัว × **65 ch** (4×16 DFL + 1 class), `DPU subgraph number 1` | ✅ |
| M9 | **Phase 1 — board bring-up** | บอร์ดมาถึง 16 ส.ค. → `xmutil listapps` = `kv260-benchmark-b4096` active · fingerprint ตรง · `xdputil benchmark` = **82.5 FPS Test PASS** = first live DPU inference | ✅ |
| M10 | **Phase 2 — tracking + line-crossing counting** | `video_dump_dets.py` → `count_offline.py --sweep`: axis=x, line=0.40·W, max_dist=60 → **นับ 215/215 = 0% error** | ✅ |
| M11 | **Phase 3 — benchmark เทียบ GPU baseline** | `05-benchmarks/results/kv260_results.md` — ครบ 4 หมวด (accuracy / latency / power / counting) + เทียบ GPU 2 operating point | ✅ |
| M12 | Profiling หา bottleneck | `bench_latency.py` — **PS-bound**: preproc 49.7 ms = 63% ของ pipeline, DPU แค่ 16% | ✅ |
| M2-B3 | Track B: compile YOLO26n ด้วย `vai_c_xir` จริง | รันจริงบนเดสก์ท็อป (env ตรง spec) — pass 1 calibrate ผ่าน, pass 2 export XIR ล้มที่ op `nndct_elemwise_mul` ใน `Attention[attn]` (`model.10`, scale-multiply ก่อน softmax) — **ไม่ถึงขั้นรัน `vai_c_xir`**. ตรง node แรกของ attention ที่ M2-B1 ชี้ไว้ → ปิดคำถามด้วย "compile error" ตามตารางตัดสินในบรีฟ. รายละเอียด `artifacts/inspect_report/FINDINGS_op_analysis.md` ภาคผนวก M2-B3 | ✅ |

## งานถัดไป (เรียงตาม critical path)

| # | งาน | เกณฑ์ผ่าน | ติดบล็อก? | สถานะ |
|---|---|---|---|---|
| M14 | รายงาน + สไลด์ป้องกัน | ส่งครบ | ไม่ | 🟡 **กำลังทำ — งานหลักที่เหลืออยู่ตอนนี้** — ร่างบท Results เสร็จ (`RESULTS_chapter_draft.md`, 6 ตาราง) + ชุดเอกสารอาจารย์ `00-admin/advisor-brief/` (12 ไฟล์) เสร็จ · เหลือ ขัดสำนวน + ใส่รูป + บทอื่นๆ ของเล่ม + เขียนผล M2-B3 เข้าบท Track B |
| M13 | (Optional) custom accelerator / optimize preprocessing | speedup วัดได้เทียบ baseline | ไม่ | ☐ **optional — ไม่ผูกเป็นเงื่อนไขจบ** · เป้าที่ชัดที่สุดคือ preproc 49.7 ms (port เป็น C++ / NEON SIMD / hardware scaler ใน PL): ถ้าลดเหลือ 10 ms → e2e 78.3 → 38.6 ms = **12.8 → 25.9 FPS (+103%)** |

## หมายเหตุ

- **โครงงานจบได้แล้วด้วยผลที่มีอยู่** — Track A สำเร็จครบวงจรและมีผลวัดครบ 4 หมวด **ความเสี่ยงที่เหลือทั้งหมดเป็นความเสี่ยงต่อ "ส่วนขยาย" ไม่ใช่ต่อ "การจบโครงงาน"** — นี่คือผลตอบแทนของกลยุทธ์ two-track + การ front-load งานที่ไม่ต้องใช้บอร์ด
- **M2-B3 ล้มไม่ใช่ความล้มเหลวของโครงงาน** — Track A ผ่านแล้ว และผลว่า YOLO26 ติดตรงไหน (attention MatMul+Softmax) เขียนเข้าเล่มเป็น contribution ได้เต็มๆ
- **3 ข้อค้นพบที่จะเขียนเข้าเล่ม:** (1) INT8 quantization แทบไม่ทำให้ accuracy เสีย (ตก 1.5%) แต่ประหยัดไฟ 6 เท่า (2) **คอขวดไม่ได้อยู่ที่ตัวเร่ง AI แต่อยู่ที่ CPU** — DPU กินแค่ 16% ของ pipeline (3) YOLO26 map ลง DPUCZDX8G ไม่ได้ทั้งตัว และรู้แล้วว่าติดตรงไหนเป๊ะๆ
- **ความปลอดภัยหลักฐาน:** P0 ทำแล้ว (มีสำเนาบนเดสก์ท็อป) · ทำงานกับบอร์ดแบบ relay (user คุมบอร์ด 100% ไม่ใส่ SSH key)
- **บทเรียน environment ที่ต้องระวังตอนทำ M2-B3:** host RAM 5.88 GB + C: ต้องมีที่ว่างพอให้ pagefile ขยาย ไม่งั้น container ตาย OOM (`unexpected EOF` / exit 125) · สคริปต์ quantize ต้องมี monkeypatch `C2f.forward` chunk→slice และ wrapper ที่ return raw `cat(cv2,cv3)` ไม่งั้นล้ม `XIR don't support multi-outputs op`

---

## โปรเจคคู่ขนาน: Efinix Ti375 (cross-platform comparison)

> repo แยก: `thesis-yolo26-fpga_efinix` — ทำ counting แบบเดียวกันบน Efinix Ti375 (TFLite Micro + Sapphire RISC-V) เพื่อเทียบข้ามแพลตฟอร์ม
> **ตารางนี้ไม่ถูก parse เข้า dashboard** (dashboard แสดงเฉพาะ milestone ของ KV260)

| # | งาน | ต้องมีบอร์ด/Efinity? | สถานะ |
|---|---|---|---|
| E0 | ยืนยัน platform + toolchain | ไม่ | ☐ |
| E1 | ติดตั้ง Efinity + build Sapphire SoC + รัน TinyML demo | **ใช่** | ☐ **gate จริงอันแรก** — มีคู่มือ `RUN_ON_EFINITY_MACHINE.md` เตรียมไว้ |
| E2 | ตัดสิน Lite vs Standard + ประเมิน resource fit | ไม่ | ☐ **จุดตัดสินใหญ่** |
| E3-A | Track A: YOLOv8n → TFLite INT8 | ไม่ | ✅ **เสร็จ** (parity 59/59) |
| E3-B | Track B: YOLO26n → TFLite + ตรวจ op | ไม่ | ☐ |
| E4 | TinyML Generator → model data | ไม่ | ☐ |
| E5 | build RISC-V app | ไม่ | 🟡 โค้ดเขียนเสร็จ รอ compile |
| E6 | Verify quant fidelity | ไม่ | ☐ |
| E7 | Counting GT (reuse GT = 215) | ไม่ | ☐ |
| E8 | Single-class quantize | ไม่ | ☐ |
| E9 | Board bring-up | ใช่ | ☐ |
| E10 | Tracking + counting บน RISC-V | ใช่ | ☐ |
| E11 | Benchmark เทียบ GPU + KV260 | ใช่ | ☐ |
| E12 | Profiling | ใช่ | ☐ |
| E13 | (Optional) custom instruction | ใช่ | ☐ |
| E14 | รายงาน (มุม cross-platform) | — | ☐ |

---

## เอกสารที่เกี่ยวข้อง

- `00-admin/advisor-brief/` — ชุดเอกสารเล่าให้อาจารย์ที่ปรึกษา 12 ไฟล์ (20 ส.ค. 2026)
- `00-admin/ADVISOR_BRIEF_2026-08-20.md` — ฉบับรวมไฟล์เดียว
- `05-benchmarks/results/kv260_results.md` — ผลวัดดิบทั้งหมด
- `05-benchmarks/results/RESULTS_chapter_draft.md` — ร่างบท Results
- `07-notes/worklog.md` — บันทึกรายวัน
