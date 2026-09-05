# 11 — สถานะ Milestone ทั้งหมด และงานที่เหลือ

> **สรุปไฟล์นี้ใน 3 บรรทัด**
> โปรเจค KV260 เสร็จ 14 จาก 17 milestone — เหลือหลักๆ คือ compile YOLO26n เพื่อยืนยัน finding และ finalize บทรายงาน
> โปรเจค Efinix เพิ่งเริ่ม (E3-A เสร็จ) และ gate จริงอันแรก (E1) ยังต้องทำที่เครื่องที่มี Efinity
> ความเสี่ยงที่เหลือทั้งหมดเป็นความเสี่ยงต่อ "ส่วนขยาย" ไม่ใช่ต่อ "การจบโครงงาน"

---

## 1. สถานะโปรเจคหลัก (KV260) — M0 ถึง M14

### 1.1 ตารางสถานะ

| # | Milestone | เกณฑ์ผ่าน | สถานะ | หลักฐาน |
|---|---|---|---|---|
| M0 | เลือก target board + toolchain | มีเอกสารเปรียบเทียบ | ✅ | `01-docs/platform_selection.docx` |
| M1 | ตั้ง Vitis AI 3.0 container + flow ครบ | รัน flow ได้ครบวงจร | ✅ | `RUNBOOK_phase0_ORIGINAL.md` |
| M2-A | Track A: YOLOv8n compile ผ่าน | `DPU subgraph number 1` | ✅ | compile log |
| M3-GPU | GPU baseline (FP32 คู่) | มีตัวเลข 2 จุดปฏิบัติการ | ✅ | `results_fp32_final.json` |
| **P0** | กู้ artifact ออกจาก WSL | มีสำเนาครบ verify แล้ว | ✅ | 162 ไฟล์, byte-exact |
| **P1** | เคลียร์ B3136 vs B4096 + identity | หลักฐานตรงกันทุกแหล่ง | ✅ | `P1_arch_and_identity_resolution.md` |
| M2-B1 | Track B: วิเคราะห์ op ของ YOLO26n | รู้ว่า op ไหนตก CPU | ✅ | `FINDINGS_op_analysis.md` |
| M2-B2 | Track B: fine-tune หลัง SiLU→LeakyReLU | accuracy กลับมาใกล้เดิม | ✅ | v8n 0.884 / 26n 0.831 |
| **M2-B3** | **Track B: compile YOLO26n → gate** | รู้จำนวน subgraph จริง | ☐ **เหลือ** | — |
| M4 | VART host code | compile-check ผ่าน | ✅ | `04-deploy/board/` |
| M5 | Verify quantization fidelity | ผ่านเกณฑ์ | ✅ | ปิดด้วย mAP drop 1.5% |
| M6 | Ubuntu-on-Kria image checklist | overlay ตรง arch | ✅ | (เปลี่ยนเป็น PetaLinux — fingerprint ตรง) |
| M7 | Counting ground-truth dataset | annotate เสร็จ | ✅ | GT = 215 |
| M8 | re-quantize เป็น single-class | xmodel ใหม่ compile ผ่าน | ✅ | 65 channel |
| **M9** | **Board bring-up** | fingerprint ตรง + first inference | ✅ | 82.5 FPS Test PASS |
| **M10** | **Tracking + line-crossing counting** | นับถูกบนวิดีโอทดสอบ | ✅ | **215/215 = 0% error** |
| **M11** | **Benchmark เทียบ GPU baseline** | ครบ FPS/W, mAP, counting | ✅ | ครบ 4 หมวด |
| M12 | Profiling หา bottleneck | รู้ stage คอขวด | ✅ | PS-bound, preproc 63% |
| M13 | (Optional) custom accelerator | speedup วัดได้ | ☐ | **ไม่ผูกเป็นเงื่อนไขจบ** |
| M14 | รายงาน + สไลด์ป้องกัน | ส่งครบ | 🟡 | ร่างบท Results เสร็จ |

### 1.2 สรุปเชิงตัวเลข

```
เสร็จแล้ว:  14 milestone  (M0, M1, M2-A, M3, P0, P1, M2-B1, M2-B2, M4, M5, M6, M7, M8, M9, M10, M11, M12)
เหลือ:       M2-B3 (สำคัญ) · M13 (optional) · M14 (กำลังทำ)
```

### 1.3 ภาพความคืบหน้าตาม critical path

```
M0 ─► M1 ─► M2-A ─► [P0, P1] ─► M2-B1 ─► M2-B2 ─► M4..M8 ─► M9 ─► M10 ─► M11 ─► M12
 ✅    ✅     ✅         ✅          ✅        ✅        ✅       ✅     ✅      ✅      ✅
                                     │
                                     └──► M2-B3 ☐  ← เส้นทางที่ยังไม่ปิด
                                                      (ไม่บล็อกอะไร แต่ปิดคำถามวิจัยข้อ 1 ให้สมบูรณ์)
                                                                    │
                                                                    ▼
                                                                  M14 🟡
```

---

## 2. งานที่เหลือ เรียงตามความสำคัญ

### 🔴 ลำดับ 1 — ปิด Track B ให้สมบูรณ์ (M2-B3)

**สิ่งที่ต้องทำ:** รัน `vai_c_xir` กับ YOLO26n จริง เพื่อยืนยันว่ากราฟถูกตัดเป็นกี่ subgraph

| รายการ | สถานะ |
|---|---|
| ต้องใช้บอร์ดไหม | **ไม่ต้อง** — ทำในคอนเทนเนอร์ได้ |
| น้ำหนักโมเดล | ✅ พร้อม (`yolo26n_leaky_pkg_ft.pt`, mAP 0.831) |
| สคริปต์ | ✅ มีอยู่ (TARGET แก้เป็น B4096 แล้ว) |
| เวลาที่คาด | ระดับชั่วโมง (ไม่ใช่วัน) |

**ทำไมสำคัญ:**
> เราตั้งกติกาไว้เองว่า *"gate เดียวที่นับคือ `vai_c_xir` — ไม่ใช่ op histogram"*
> ถ้าไม่ทำตามกติกาตัวเอง จะโดนถามว่า **"แน่ใจได้อย่างไรว่ามันตัดเป็น 3 subgraph จริง"**
> แล้วเราจะตอบไม่ได้

**ผลลัพธ์ที่คาดหวัง (ทั้งสองแบบใช้ได้):**

| ถ้า compile ได้ | ถ้า compile ล้ม |
|---|---|
| เห็น `DPU subgraph number 3` (หรือใกล้เคียง) | เห็น error ระบุ op ที่มีปัญหา |
| → ยืนยัน mechanism ที่วิเคราะห์ไว้ | → ยืนยันตรงกว่าเดิม + ได้ error message เป็นหลักฐาน |
| + วัด latency penalty ได้เพิ่ม | |

---

### 🟠 ลำดับ 2 — Finalize บทรายงาน (M14)

| งาน | สถานะ |
|---|---|
| ร่างบท Results (6 ตาราง + key findings) | ✅ `RESULTS_chapter_draft.md` |
| ผลวัดดิบรวม | ✅ `kv260_results.md` |
| เอกสารสรุปสำหรับอาจารย์ | ✅ ชุดนี้ |
| ขัดสำนวน + ใส่รูป | ☐ เหลือ |
| บทอื่นๆ ของเล่ม | ☐ เหลือ |

---

### 🟡 ลำดับ 3 — วิดีโอ demo สำหรับนำเสนอ

มีไฟล์พร้อมแล้ว:
- `08-figures/out20.mp4` — คลิปสั้น 20 วินาที (นับได้ 9)
- `08-figures/out600_calibrated.mp4` — คลิปเต็มพร้อมเส้นที่ calibrate แล้ว
- `08-figures/out600_conf45_roi.mp4` — เวอร์ชันปรับ confidence + ROI
- `08-figures/detect_out_pkg_kv260.jpg` — รูป detection บนบอร์ด

**เหลือแค่:** เลือกว่าจะใช้ตัวไหนตอนนำเสนอ (แนะนำ `out20.mp4` เพราะสั้นและเห็นชัด)

---

### 🟡 ลำดับ 4 — เดินหน้าโปรเจค Efinix

```
E0 (platform selection doc)
   ↓
E1 (ติดตั้ง Efinity + build Sapphire SoC + รัน TinyML demo)  ← gate จริงอันแรก
   ↓
E2 (ตัดสิน Lite vs Standard + ประเมิน resource fit)          ← จุดตัดสินใหญ่
   ↓
E4 (TinyML Generator → model data)
   ↓
E5 (build RISC-V app — โค้ดเขียนเสร็จแล้ว)
   ↓
E9 (board bring-up)
```

**หมายเหตุ:** E1 และ E9 ต้องทำที่เครื่องที่มี Efinity + บอร์ด (คนละเครื่องกับที่ใช้อยู่)
→ มีคู่มือ `RUN_ON_EFINITY_MACHINE.md` เตรียมไว้แล้ว

---

### ⚪ ลำดับ 5 (optional) — Optimize preprocessing

**ที่มา:** จากข้อค้นพบว่าคอขวดคือ preprocessing 49.7 ms (63% ของ pipeline)

**ทางเลือก:**
| วิธี | ผลที่คาด | ความยาก |
|---|---|---|
| port preprocess เป็น C++ | ลดลงมาก (ตัด Python overhead) | ปานกลาง |
| ใช้ NEON SIMD บน ARM | ลดลงมาก | ปานกลาง-สูง |
| ใช้ hardware scaler ใน PL | ลดเกือบหมด | สูง |

**ผลตอบแทนที่คาดได้:**
```
ถ้าลด preproc จาก 49.7 → 10 ms:
   e2e: 78.31 → 38.56 ms  →  FPS: 12.77 → 25.9  (ดีขึ้น 103%)
   และ power แทบไม่เพิ่ม เพราะ DPU ยังมี headroom
```

> **คุณค่าเชิงวิชาการ:** ถ้าทำสำเร็จ จะเป็นการ**ปิดวงจร**ของข้อค้นพบ —
> จากที่ระบุว่าคอขวดอยู่ไหน → แก้ → วัดผลยืนยัน = เป็นงานที่สมบูรณ์มาก
> แต่**ไม่จำเป็นต่อการจบโครงงาน**

---

## 3. ความเสี่ยงที่เหลือ

### 3.1 ตารางความเสี่ยง

| ความเสี่ยง | โอกาส | ผลกระทบ | การรับมือ |
|---|---|---|---|
| M2-B3 compile แล้วผลไม่ตรงกับที่วิเคราะห์ | ต่ำ | **ต่ำ** — ก็ยังเป็น finding | รายงานผลจริงตามที่ได้ |
| Efinix รัน YOLO เต็มตัวไม่ไหว | **ปานกลาง** | ต่ำ — มี fallback 3 ทาง | ตัดสินที่ E2, มีแผนสำรอง |
| ไม่มีเวลาทำ optimize preprocessing | ปานกลาง | **ต่ำมาก** — เป็น optional | ระบุเป็นงานต่อยอดในเล่ม |
| บอร์ด KV260 เสียหาย | ต่ำ | สูง | ผลวัดเก็บครบแล้ว · ทำงานแบบ relay ไม่ให้เครื่องมืออื่นเข้าถึงบอร์ด |
| หลักฐานหาย | ต่ำ | สูง | P0 ทำแล้ว — มีสำเนาบนเดสก์ท็อป |

### 3.2 ข้อสังเกตสำคัญ ⭐

> **ความเสี่ยงที่เหลือทั้งหมดเป็นความเสี่ยงต่อ "ส่วนขยาย" ไม่ใช่ต่อ "การจบโครงงาน"**
>
> เพราะ Track A สำเร็จครบวงจรและมีผลวัดครบ 4 หมวดแล้ว
> ต่อให้ทุกอย่างที่เหลือล้มหมด โครงงานก็ยังจบได้ด้วยผลที่มีอยู่
>
> **นี่คือผลตอบแทนของกลยุทธ์ two-track และการ front-load งานที่ไม่ต้องใช้บอร์ด**

---

## 4. สถานะโปรเจค Efinix (E0–E14)

| # | งาน | ต้องมีบอร์ด? | สถานะ |
|---|---|---|---|
| E0 | ยืนยัน platform + toolchain | ไม่ | ☐ |
| **E1** | ติดตั้ง Efinity + Sapphire SoC + รัน demo | **ใช่** | ☐ **gate แรก** |
| E2 | ตัดสิน Lite vs Standard + resource fit | ไม่ | ☐ **จุดตัดสินใหญ่** |
| **E3-A** | Track A: YOLOv8n → TFLite INT8 | ไม่ | ✅ **เสร็จ** (parity 59/59) |
| E3-B | Track B: YOLO26n → TFLite + ตรวจ op | ไม่ | ☐ |
| E4 | TinyML Generator → model data | ไม่ | ☐ |
| E5 | RISC-V app | ไม่ | 🟡 **โค้ดเสร็จ รอ compile** |
| E6 | Verify quant fidelity | ไม่ | ☐ |
| E7 | Counting GT (reuse GT=215) | ไม่ | ☐ |
| E8 | Single-class quantize | ไม่ | ☐ |
| E9 | Board bring-up | ใช่ | ☐ |
| E10 | Tracking + counting บน RISC-V | ใช่ | ☐ |
| E11 | Benchmark เทียบ GPU + KV260 | ใช่ | ☐ |
| E12 | Profiling | ใช่ | ☐ |
| E13 | (Optional) custom instruction | ใช่ | ☐ |
| E14 | รายงาน (มุม cross-platform) | — | ☐ |

---

## 5. เอกสารที่ควรอัปเดตให้ตรงกัน

### 5.1 จุดที่ยังไม่ตรง

| ไฟล์ | ปัญหา | ความสำคัญ |
|---|---|---|
| `README.md` (root) | ยังเป็นข้อมูล 11 ส.ค. — เขียนว่ารอบอร์ด, Phase 1 ยังไม่เริ่ม, VART code ยังไม่เขียน | **ปานกลาง** — ถ้าอาจารย์เปิดดู repo จะขัดกับที่เล่า |
| `00-admin/timeline.md` | อัปเดตถึง 15 ส.ค. — M10/M11/M12 ยังเป็น ☐ ทั้งที่เสร็จแล้ว | **ปานกลาง** — เป็น single source of truth |
| `00-admin/dashboard.html` | generate จาก timeline → ตามไปด้วย | ต่ำ |
| `01-docs/architecture.md` | ยังมีช่องว่าง `_____` ที่ยังไม่เติม | ต่ำ |

### 5.2 ข้อเสนอ

ถ้าจะให้ repo สอดคล้องกับเรื่องที่เล่า ควรอัปเดต `timeline.md` ก่อน (เป็น source of truth)
แล้วค่อย sync `README.md` และ `dashboard.html` ตาม

---

## 6. Checklist ก่อนเข้าพบอาจารย์

```
□ อ่าน README.md ของชุดเอกสารนี้ (สรุป 1 หน้า)
□ อ่านไฟล์ 07 (ผลการทดลอง) — ส่วนที่มีน้ำหนักที่สุด
□ อ่านไฟล์ 12 (คำถาม-คำตอบ) — ซ้อมตอบ
□ เตรียมเปิด 08-figures/detect_out_pkg_kv260.jpg (รูป detection)
□ เตรียมเปิด 08-figures/out20.mp4 (วิดีโอ demo)
□ (ถ้ามีเวลา) รัน M2-B3 ให้จบ เพื่อปิดคำถามเรื่อง Track B
□ (ถ้ามีเวลา) อัปเดต timeline.md + README.md ให้ตรงกับสถานะจริง
```

---

## ไฟล์ที่เกี่ยวข้อง
- ไฟล์ก่อนหน้า: [10-engineering-problems.md](10-engineering-problems.md)
- ไฟล์ถัดไป: [12-qa.md](12-qa.md)
- timeline (source of truth): `00-admin/timeline.md`
- dashboard: `00-admin/dashboard.html`
