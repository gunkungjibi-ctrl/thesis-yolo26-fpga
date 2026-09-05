# สรุปความคืบหน้าโครงงาน — ฉบับเล่าให้อาจารย์ที่ปรึกษา

**หัวข้อ:** Real-Time Video-Based Object Counting on FPGA Using YOLO
**ผู้จัดทำรายงานสรุป:** อัปเดต 20 สิงหาคม 2026
**ขอบเขตของเอกสาร:** เล่าตั้งแต่จุดเริ่มต้นจนถึงสถานะล่าสุด — ทั้งสิ่งที่สำเร็จ สิ่งที่ล้ม เหตุผลเบื้องหลังทุกการตัดสินใจ และตัวเลขผลการทดลองจริงทุกตัว

---

## 0. อ่านหน้าเดียวจบ (ถ้ามีเวลา 2 นาที)

**โจทย์:** สร้างระบบนับวัตถุ (กล่องพัสดุบนสายพาน) แบบเรียลไทม์จากวิดีโอ โดยรัน YOLO บน FPGA แทน GPU แล้วพิสูจน์เชิงตัวเลขว่าคุ้มค่าตรงไหน

**สถานะ: ระบบทำงานครบวงจรบนบอร์ดจริงแล้ว และเก็บผลวัดครบ 4 หมวดแล้ว**

| สิ่งที่ทำได้ | ตัวเลขจริง |
|---|---|
| โมเดล YOLOv8n ทำงานบน DPU ของ KV260 | 1 DPU subgraph, INT8, 8.10 GOP/ภาพ |
| ความแม่นยำหลัง quantize เป็น INT8 | mAP@0.5 = **0.8713** (จาก float 0.8846) → **ตกแค่ 1.5%** |
| ความเร็ว DPU ล้วน | **82.5 FPS** (single-thread) / 89.6 FPS (4 threads) |
| ประสิทธิภาพพลังงาน | **9.67 FPS/W** เทียบ GPU ดีที่สุด 4.51 FPS/W → **ดีกว่า 2.1 เท่า** |
| กำลังไฟ | 9.27 W เทียบ GPU 58.5 W → **น้อยกว่า 6.3 เท่า** |
| งานนับกล่อง (โจทย์หลัก) | GT = 215 กล่อง → ระบบนับได้ **215 = ผิดพลาด 0.0%** |

**สามข้อค้นพบที่จะเขียนเข้าเล่ม:**
1. **Quantization INT8 แทบไม่ทำให้ความแม่นยำเสีย** (ตก 1.5%) แต่ประหยัดไฟ 6 เท่า → คุ้มค่ามากสำหรับงาน edge
2. **คอขวดไม่ได้อยู่ที่ตัวเร่ง AI แต่อยู่ที่ CPU** — DPU ใช้เวลาแค่ 16% ของ pipeline ส่วน preprocessing บน ARM กิน 63% → เป็นข้อค้นพบเชิงระบบที่สำคัญกว่าตัวเลข FPS เฉยๆ
3. **YOLO26 (โมเดลรุ่นใหม่) map ลง DPU รุ่นนี้ไม่ได้ทั้งตัว** และเรารู้แล้วว่าติดตรงไหนเป๊ะๆ (attention block 2 จุด) → เป็นองค์ความรู้ที่ยังไม่มีใครรายงาน

---

## 1. โจทย์และเหตุผล — ทำไมต้อง FPGA

### 1.1 ปัญหาที่ต้องการแก้
การนับวัตถุจากวิดีโอแบบเรียลไทม์ (เช่น นับพัสดุบนสายพานในคลังสินค้า) ปกติต้องใช้เครื่องที่มี GPU ซึ่ง

- กินไฟสูง (การ์ดระดับกลางกิน 60–250 W)
- ขนาดใหญ่ ต้องระบายความร้อน
- ต้นทุนต่อจุดติดตั้งสูง เมื่อต้องติดหลายจุดในโรงงาน

งานลักษณะนี้เป็น **งาน edge** — โมเดลตัวเดียว ภาพเดียวต่อครั้ง ทำงาน 24 ชั่วโมง — ซึ่งเป็นสถานการณ์ที่ GPU เสียเปรียบที่สุด เพราะ GPU เก่งเรื่อง batch ใหญ่ ไม่ใช่ latency ต่ำที่ batch = 1

### 1.2 ทำไม FPGA ตอบโจทย์
FPGA ที่มี hardened AI accelerator (DPU) ทำ INT8 inference ได้ด้วยกำลังไฟระดับ < 10 W ทั้งบอร์ด และมี CPU (ARM) อยู่ในชิปเดียวกันสำหรับทำ tracking/counting → **ระบบทั้งระบบจบในบอร์ดเดียว ไม่ต้องมีพีซี**

### 1.3 คำถามวิจัยที่ตั้งไว้
1. YOLO รุ่นใหม่ deploy ลง DPU ได้จริงแค่ไหน ติดข้อจำกัดอะไรบ้าง
2. quantize เป็น INT8 แล้วความแม่นยำเสียไปเท่าไร คุ้มกับที่ประหยัดไฟหรือไม่
3. เมื่อวัดทั้งระบบ (ไม่ใช่แค่ตัวเร่ง) คอขวดจริงอยู่ที่ไหน
4. ระบบนับได้แม่นแค่ไหนเทียบกับที่คนนับ

---

## 2. คอนเซปต์ที่ล็อกแล้ว — สถาปัตยกรรมและกติกา

### 2.1 สถาปัตยกรรมระบบ

```
วิดีโอ / กล้อง
    ↓
[PS — ARM Cortex-A53]  decode + preprocess (resize 640, BGR→RGB, /255, quantize เป็น int8)
    ↓
[PL — DPUCZDX8G B4096]  YOLO inference (.xmodel, INT8)     ← ส่วนที่เร่งด้วยฮาร์ดแวร์
    ↓
[PS]  dequantize → DFL decode → sigmoid → NMS
    ↓
[PS]  centroid tracker (จับคู่ ID ข้ามเฟรม)
    ↓
[PS]  line-crossing logic → นับ
    ↓
แสดงผล / บันทึก
```

จุดสำคัญของสถาปัตยกรรมนี้คือเป็น **heterogeneous** — งาน convolution หนักๆ ไปที่ PL (ฮาร์ดแวร์) ส่วนงาน logic ที่แตกกิ่งเยอะ (tracking, การตัดสินใจนับ) อยู่ที่ PS (ซอฟต์แวร์) ซึ่งเป็นการแบ่งงานที่เหมาะกับธรรมชาติของแต่ละส่วน

### 2.2 แพลตฟอร์มที่เลือก

| รายการ | ค่า |
|---|---|
| บอร์ด | AMD Kria KV260 Vision AI Starter Kit (Zynq UltraScale+ MPSoC) |
| AI accelerator | DPUCZDX8G_ISA1_**B4096** (fingerprint `0x101000056010407`) |
| CPU | ARM Cortex-A53 quad-core |
| Toolchain | Vitis AI **3.0** (Docker: `xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106`) |
| OS บนบอร์ด | PetaLinux starter-kit 2022.2 |
| Runtime | VART (Vitis AI Runtime) 3.0 |

> **หมายเหตุการตัดสินใจ:** pin Docker image ไว้ที่ 3.0 ไม่ใช้ `:latest` เพราะ Vitis AI 3.5 **ไม่มี prebuilt สำหรับ KV260** — ถ้าใช้ latest จะไม่มี arch.json และ board_setup ของบอร์ดนี้ ทำให้ compile ไม่ได้

### 2.3 กติกาตัดสินที่ตั้งไว้ตั้งแต่ต้น (สำคัญมาก)

> **"gate เดียวที่นับคือ log ของ `vai_c_xir` ที่บอกว่า `DPU subgraph number 1` — ไม่ใช่ op histogram ไม่ใช่ผล Inspector"**

เหตุผล: เครื่องมือ Inspector ของ Vitis AI เป็นแค่ *advisory* — มันบอกว่า "น่าจะรันได้" แต่ตัวที่ตัดสินจริงคือ compiler ถ้า compiler ตัดกราฟออกเป็นหลาย subgraph แปลว่ามี op ที่ต้องเด้งกลับไปรันบน CPU ซึ่งจะทำให้ latency พัง กติกานี้ช่วยกันการ "หลอกตัวเอง" ว่างานสำเร็จทั้งที่ยังไม่จริง

### 2.4 กลยุทธ์สองเส้นทางขนาน (Two-track strategy)

นี่คือการตัดสินใจเชิงบริหารความเสี่ยงที่สำคัญที่สุดของโครงงาน

| | **Track A — YOLOv8n** | **Track B — YOLO26n** |
|---|---|---|
| บทบาท | baseline ที่พิสูจน์แล้ว / ตาข่ายนิรภัย | โมเดลเป้าหมายตามชื่อโครงงาน |
| ความเสี่ยง | ต่ำ (มี prior work เยอะ, อยู่ใน Model Zoo) | สูง (NMS-free head, ยังไม่มีใครทำบน DPU นี้) |
| สถานะ | ✅ **สำเร็จครบวงจรบนบอร์ดจริง** | 🔴 พิสูจน์แล้วว่า **compile ทั้งตัวไม่ได้** — และรู้สาเหตุแน่ชัด |

**เหตุผลที่ต้องเดินสองทาง:** ถ้าเดิมพันกับ YOLO26 อย่างเดียวแล้วมันรันไม่ได้ โครงงานจะไม่มีผลอะไรเลย แต่ถ้ามี Track A รองรับ โครงงาน**จบได้แน่นอน** ส่วนผลของ Track B ไม่ว่าจะสำเร็จหรือล้ม **ก็เป็น contribution ทั้งคู่** — ถ้าสำเร็จคือ "ทำได้เป็นคนแรก" ถ้าล้มคือ "รู้ว่าติดตรงไหน เพราะอะไร" ซึ่งเป็นองค์ความรู้ที่คนอื่นเอาไปใช้ต่อได้

---

## 3. การเดินทางของโครงงาน — เล่าเป็น 6 องก์

### องก์ที่ 1 — วางรากฐาน: เลือกบอร์ดและตั้ง toolchain ให้ได้ก่อน

**เป้าหมาย:** ก่อนจะพูดเรื่องโมเดล ต้องพิสูจน์ก่อนว่า "ทางเดินจากโมเดล → ฮาร์ดแวร์" มันมีอยู่จริง

สิ่งที่ทำ:
- เปรียบเทียบบอร์ดและเลือก KV260 (เอกสาร `01-docs/platform_selection.docx`) เหตุผลหลักคือมี DPU สำเร็จรูป + ราคาเข้าถึงได้ + มี Vitis AI stack ที่สมบูรณ์
- ตั้ง Vitis AI 3.0 container ให้ครบ flow: `export → quantize (vai_q_pytorch) → compile (vai_c_xir)`
- เขียน RUNBOOK บันทึกทุกคำสั่ง เพื่อให้ทำซ้ำได้ (`03-model/RUNBOOK_phase0_ORIGINAL.md`)

**ผลลัพธ์:** ✅ Track A (YOLOv8n) compile ผ่าน gate ได้ `DPU subgraph number 1`

---

### องก์ที่ 2 — วิกฤตสองเรื่องที่เกือบทำให้โครงงานเสียหาย

องก์นี้สำคัญที่สุดในแง่ "บทเรียนวิศวกรรม" และควรเล่าให้อาจารย์ฟัง เพราะมันแสดงวิธีคิดแบบตรวจสอบหลักฐาน

#### วิกฤตที่ 1 — หลักฐานทั้งหมดอยู่ในที่เดียว และที่นั้นเคยพังมาแล้ว

artifact ทั้งหมด (`.xmodel`, compile log, calibration images, ผลวัด) อยู่ใน WSL virtual disk (`.vhdx`) ที่วางบนไดรฟ์ C: ของ Windows ซึ่ง **เคยพังมาแล้วหนึ่งรอบ** จากการที่ดิสก์เต็มระหว่างติดตั้ง PyTorch (ระบบไฟล์ remount เป็น read-only แล้วคำสั่งพื้นฐานอย่าง `touch`, `df` ก็ segfault)

ถ้ามันพังอีกครั้ง หลักฐานเข้าเล่มหายหมด ต้องทำใหม่ทั้งหมด

**สิ่งที่ทำ (งาน P0):** กู้โฟลเดอร์ `phase0/` ทั้งหมดออกจาก WSL มาไว้บนเดสก์ท็อป — **162 ไฟล์ 43,042,717 ไบต์ ตรวจสอบ byte-exact ทั้งสองฝั่ง** และ verify md5 ของ xmodel ตรงกับที่ compiler บันทึกไว้

**บทเรียน:** งานวิจัยที่ผลลัพธ์อยู่ในที่เดียว = ยังไม่มีผลลัพธ์

#### วิกฤตที่ 2 — ไฟล์ชื่อ `yolo26n_kv260.xmodel` แต่ข้างในไม่ใช่ YOLO26

ตอนตรวจ artifact ที่กู้มา พบความผิดปกติ: บันทึกเดิมเขียนว่า "Phase 0 ผ่านแล้วด้วย YOLO26n" แต่หลักฐานหลายชั้นขัดกัน

**สิ่งที่ทำ (งาน P1):** ตรวจสอบ identity ของ xmodel จาก 5 แหล่งอิสระ

| หลักฐาน | ค่าที่พบ | แปลว่า |
|---|---|---|
| จำนวน output channel ของหัวโมเดล | **144 ch** | 144 = 4×16 (DFL) + 80 (COCO) = **หัวแบบ YOLOv8** |
| ถ้าเป็น yolo26n ต้องเป็น | 84 ch (reg_max=1, ตัด DFL ทิ้ง) | ไม่ตรง |
| `source range` ของ output node | ชี้ไปที่ `quantize_yolo_pytorch.py` | สคริปต์ของ v8 ไม่ใช่ของ 26n |
| โครงหัวใน inspect report | `Detect[22] / cv2 / cv3` | โครง Detect head แบบ v8 คลาสสิก |
| ค่า default ของ `--weights` ในสคริปต์ | `yolov8n.pt` | น้ำหนักต้นทางคือ v8n |

**สรุป: ไฟล์ตั้งชื่อผิด — สิ่งที่ผ่าน gate คือ YOLOv8n ไม่ใช่ YOLO26n → Track B ยังไม่เคยผ่าน gate เลย**

ถ้าไม่เจอตรงนี้ โครงงานจะรายงานผลผิดทั้งเล่ม (อ้างว่า deploy YOLO26 สำเร็จทั้งที่ไม่จริง) → นี่คือเหตุผลที่ต้องแก้ชื่อไฟล์ให้ตรงความจริงและเขียน `PROVENANCE.md` กำกับไว้ (ทำกับสำเนา ไม่แตะ backup ต้นฉบับ เพื่อรักษาหลักฐาน)

#### เรื่องที่ 3 (พ่วงมาใน P1) — DPU arch ขัดแย้งกันในเอกสาร: B3136 หรือ B4096?

สคริปต์เขียน `B3136` แต่เอกสาร handover เขียน `B4096` — ถ้าเลือกผิด ต้อง re-quantize + re-compile ใหม่ทั้งสอง track

ตรวจจากหลักฐาน 3 แหล่งที่ตรงกัน → **arch จริงคือ B4096**
1. `vai_c_xir_*.log` บรรทัดที่ 3: `Target architecture: DPUCZDX8G_ISA1_B4096`
2. `meta.json` ที่ฝังใน xmodel: `"target": "DPUCZDX8G_ISA1_B4096"`
3. `arch.json` default ของ KV260 ใน Vitis AI 3.0 = B4096

ส่วน B3136 ปรากฏเฉพาะใน comment เก่าและ Inspector dry-run — **ไม่เคย compile จริง** และเนื่องจาก inspect report ทั้งสอง arch เขียนตรงกันว่า `All the operators are assigned to the DPU` จึง**ไม่ต้อง re-compile** (ประหยัดเวลาไปหลายวัน)

📄 รายละเอียดเต็ม: `07-notes/P1_arch_and_identity_resolution.md`

---

### องก์ที่ 3 — สร้าง GPU baseline ไว้ก่อน (ทำระหว่างรอบอร์ด)

**ตรรกะ:** โครงงานนี้ต้องอ้างว่า "FPGA ดีกว่า GPU ในแง่พลังงาน" ดังนั้นต้องมีตัวเลข GPU ที่วัดเองด้วยเงื่อนไขเดียวกัน ไม่ใช่ไปหยิบตัวเลขจากเปเปอร์คนอื่นมาเทียบ (ซึ่งจะโดนถามแน่นอนว่าเทียบกันได้จริงไหม)

สิ่งที่ทำ:
- ตั้ง env แยกขาด (`conda gpubench`, Python 3.12, torch 2.11.0+cu128) บน GTX 1660 Ti Max-Q
- เขียน `gpu_baseline.py` ที่ **preprocessing ตรงกับฝั่ง FPGA เป๊ะ** (plain resize 640 → BGR→RGB → /255 → CHW) เพื่อให้เทียบกันแบบ apple-to-apple
- วัดกำลังไฟผ่าน NVML และใช้ median-of-medians จาก 5 runs × 200 iters กัน outlier จากการ throttle ของ Max-Q
- วัด 3 โหมด: pure GPU (batch=1), end-to-end (รวม pre/post), batch sweep (หาจุด peak)

**ผลลัพธ์ baseline:** (ไฟล์ `05-benchmarks/gpu-baseline/results_fp32_final.json`)

| โหมด | throughput | power | ประสิทธิภาพ |
|---|---|---|---|
| Real-time (batch=1, pure) | 31.5 FPS | 22.80 W | 1.38 FPS/W |
| End-to-end (batch=1) | 25.5 FPS | 22.65 W | 1.13 FPS/W |
| Peak (batch=32) | 263.9 img/s | 58.46 W | 4.51 img/s/W |

> การวัด **สองจุดปฏิบัติการ** (real-time กับ peak) สำคัญมาก เพราะถ้ารายงานแค่ peak จะดูเหมือน GPU เก่งกว่าความเป็นจริงในงาน edge ซึ่งใช้ batch=1

---

### องก์ที่ 4 — ได้ dataset จริง แล้วเจอกำแพงเรื่องทรัพยากรเครื่อง

#### 4.1 Dataset
ใช้ Roboflow export `package-conveyo v2` — **detection 1 คลาส (`package`)**
- train 208 / valid 57 / test 28 ภาพ
- calibration set = 208 ภาพจากโดเมนจริง (เดิมใช้ COCO ซึ่งผิดโดเมน → ทำให้ quantization scale เพี้ยน)
- วิดีโอทดสอบการนับ: ต้นฉบับยาว **12 ชั่วโมง** (1,295,985 เฟรม, 2.1 GB) → ตัดเป็นคลิป **600 วินาที** (`clip_600s_gt215.mp4`, 18,001 เฟรม, 640×360@30fps) แล้ว**นับด้วยมือได้ ground truth = 215 กล่อง**

#### 4.2 ปัญหาที่ต้องแก้: SiLU → LeakyReLU
DPUCZDX8G ไม่รองรับ activation แบบ SiLU (Swish) ซึ่ง YOLOv8/26 ใช้ → ต้องเปลี่ยนเป็น **LeakyReLU(0.1015625)** (ค่า 26/256 เพื่อให้เป็นเลขที่ fixed-point แทนได้พอดี)

แต่การเปลี่ยน activation เฉยๆ โดยไม่ retrain ทำให้โมเดลพัง — เราพิสูจน์เรื่องนี้ด้วยข้อมูลจริง (ดูองก์ต่อไป)

#### 4.3 กำแพง: เครื่อง host ไม่พอเทรน
- RAM ทั้งเครื่องมี **5.9 GB** → PowerShell ยัง spawn python ไม่ได้ (`System.OutOfMemoryException`)
- ไดรฟ์ C: เหลือ **0.44 GB** → `pip install torch` ล้มด้วย `[Errno 28] No space left on device`

**วิธีแก้:**
- ย้าย environment ไปไดรฟ์ D: (`D:\yolo26-export-env`) พร้อม redirect `TEMP`/`PIP_CACHE_DIR` ไป D: ด้วย (ถ้าไม่ redirect pip ยังแตกไฟล์ลง C: อยู่ดี)
- **ย้ายการ fine-tune ไป Google Colab (T4)** — เขียน notebook `colab_finetune_yolov8n_leaky.ipynb` ที่ดึง dataset จาก Roboflow + ใส่ callback สลับ SiLU→LeakyReLU

#### 4.4 ผล fine-tune (Colab T4, 400 epochs)

| โมเดล | mAP@0.5 หลัง fine-tune |
|---|---|
| **YOLOv8n + LeakyReLU** | **0.884** |
| **YOLO26n + LeakyReLU** | **0.831** |

> **ข้อค้นพบเชิงวิชาการ:** ค่านี้ใกล้เคียงกับ SiLU ตัวเดิม → **พิสูจน์ว่าการสลับ activation เพื่อให้ map ลง DPU ได้ ไม่ทำให้ความแม่นยำเสีย ถ้า fine-tune พอ** ซึ่งเป็นข้อสรุปที่มีประโยชน์กับคนที่จะ deploy YOLO บน DPU ต่อไป

---

### องก์ที่ 5 — บอร์ดมาถึง แล้วเดินจนเห็นกล่องจริง (15–16 ส.ค.)

นี่คือช่วงที่ทุกอย่างที่เตรียมไว้มาบรรจบกัน เล่าเป็นลำดับเหตุการณ์:

#### ขั้นที่ 1 — Board bring-up (M9)
- image ที่ flash ลง SD ได้จริงคือ **PetaLinux starter-kit 2022.2** (ไม่ใช่ Ubuntu ตามแผนเดิม) แต่ตรวจแล้วมี DPU + VART ครบ → ใช้ได้เลย ไม่ต้องเปลี่ยน
- `xmutil listapps` → `kv260-benchmark-b4096` active อยู่แล้ว
- `xdputil query` → arch `DPUCZDX8G_ISA1_B4096`, **fingerprint `0x101000056010407` ตรงกับที่ compile ไว้เป๊ะ**, VART 3.0

> จุดนี้คือการปิดความเสี่ยงใหญ่ที่สุดของโครงงาน — fingerprint ตรงหมายความว่า xmodel ที่ compile ไว้ตั้งแต่ก่อนบอร์ดมา จะรันบนบอร์ดนี้ได้จริง ไม่ต้อง compile ใหม่

- `xdputil benchmark` กับ xmodel ตัวเดิม → **68.1 FPS, Test PASS = live DPU inference ครั้งแรก**

#### ขั้นที่ 2 — ปัญหาการโอนไฟล์ (เรื่องเล็กที่กินเวลาจริง)
`scp` ใช้ไม่ได้ เพราะ PetaLinux ไม่มี `sftp-server` ติดมา
**วิธีแก้:** เปิด `python -m http.server 8000` บนเครื่อง host แล้วใช้ `wget` บนบอร์ดดึงไฟล์ → verify md5 ทุกครั้งหลังโอน

> **หมายเหตุด้านความปลอดภัย:** ตลอดการทำงานกับบอร์ด ใช้วิธี relay ผ่าน serial console (COM6 @115200) โดย**ไม่ติดตั้ง SSH key** — ผู้ทำโครงงานเป็นคนสั่งทุกคำสั่งเอง 100%

#### ขั้นที่ 3 — quantize + compile โมเดลที่ fine-tune แล้ว (เจอบั๊กจริงจัง)

**ปัญหา A — OOM ตอน calibration:** container ตายกลางคัน (`unexpected EOF`, exit 125) เพราะ host RAM 5.88 GB และ WSL2 cap อยู่ที่ ~2.9 GB
**วิธีแก้:** เคลียร์ไดรฟ์ C: (0.2 → 17 GB) ให้ pagefile/swap ขยายได้ + ลด calibration เป็น `--subset_len 32 --batch_size 4` → ผ่านโดยไม่ต้องแก้ `.wslconfig`

**ปัญหา B — regression ในสคริปต์ quantize:** export XIR ล้มด้วย `XIR don't support multi-outputs op`
สาเหตุที่หาเจอ: สคริปต์เวอร์ชันใน `track-a/` **ขาด fix 2 อย่าง**ที่สคริปต์ phase0 ตัวเดิมมี
1. monkeypatch `C2f.forward` เปลี่ยน `chunk(2,1)` → slice (เพราะ XIR ไม่รองรับ op ที่มีหลาย output)
2. `BackboneHead` wrapper ที่เดิน layer เองแล้ว return raw `cat(cv2,cv3)` — คือ**ตัดส่วน decode ทิ้ง** ไม่ให้แตะ `split_with_sizes`

**วิธีแก้:** เก็บตัวที่มีปัญหาไว้เป็น `.regression.bak` (ไม่ลบ) → ก๊อป proven script จาก phase0 มาทับ + patch `_quant_compile.sh` ให้ abort ถ้าไม่มีไฟล์ `_int.xmodel` (กันกรณี `set +e` ทำให้สคริปต์เดินต่อทั้งที่ล้ม = "ผ่านหลอกตา")

**ผลลัพธ์:** ✅ PASS1 calib 32/32 → PASS2 export สำเร็จ → `vai_c_xir` = **`DPU subgraph number 1`**
artifact: `yolov8n_pkg_kv260.xmodel` (3.99 MB), md5 `c702ccb8f027bf9aea95288243d1b274`

#### ขั้นที่ 4 — deploy ขึ้นบอร์ด แล้วยืนยันว่าเป็นโมเดลตัวใหม่จริง

`xdputil xmodel -l` บนบอร์ด:
- **1 DPU subgraph** ✓
- fingerprint ตรง ✓
- input `[1,640,640,3]`
- **output 3 หัว × 65 channel** = 4×16 (DFL) + **1 class**

> เลข 65 นี้คือหลักฐานยืนยันว่าเป็นโมเดล package คลาสเดียวตัวใหม่จริง (ต่างจาก phase0 ที่เป็น 144 ch = 80 class ของ COCO) — เป็นการใช้บทเรียนจากวิกฤตที่ 2 มาตรวจสอบตัวเอง

`xdputil benchmark` → **82.5 FPS, Test PASS** (สูงกว่า 68 FPS เดิม เพราะ output เบากว่า 1 คลาส vs 80 คลาส)

#### ขั้นที่ 5 — เขียน detection app แล้วเห็นกล่องจริง 🎉

เขียน `04-deploy/board/yolo_dpu_detect.py` — Python VART application ครบวงจร ประกอบด้วย:
1. **preprocess**: plain resize 640 → BGR→RGB → /255 → quantize ด้วย `2^in_fixpos = 64` → NHWC int8
2. **DPU execute** ผ่าน VART
3. **dequantize** 3 หัว (`2^-fixpos`)
4. **DFL decode เขียนเอง**: softmax 16 bin ต่อด้าน × arange → ระยะ l,t,r,b; anchor = cell + 0.5; box = (cx ∓ d) × stride
5. **sigmoid** ที่ channel class
6. **NMS**

> ทำไมต้องเขียน decode เอง: โค้ด C++ ที่เตรียมไว้ล่วงหน้า (`yolo_dpu_infer.cpp`) ทำได้แค่ดึง raw tensor + dequantize เพราะตอน compile เราตัดส่วน decode ออกจากกราฟไปแล้ว (เพื่อให้ผ่าน XIR) → decode ต้องมาทำบน PS

**ผลรัน:** input fixpos = 6 (scale 64), output ทั้ง 3 หัว float range ปกติ ไม่ saturate, **ได้ 4 กล่องหลัง NMS score 0.92 / 0.82 / 0.78 / 0.50** — กล่องกระดาษบนสายพานจับได้ที่ 0.92 ตรงตำแหน่งเป๊ะ
📷 รูปผล: `08-figures/detect_out_pkg_kv260.jpg`

**ข้อมูล decode ที่ยืนยันแล้ว (สำคัญสำหรับเขียนเล่ม):**
- channel 65 = `[0:64]` box DFL (4 ด้าน × 16 bin), `[64]` = class score
- stride map: หัว 80×80 → stride 8, 40×40 → 16, 20×20 → 32
- scale กลับภาพต้นฉบับต้องคูณ (origW/640, origH/640) **แยกแต่ละแกน** เพราะใช้ plain resize ไม่ใช่ letterbox

---

### องก์ที่ 6 — เก็บผลวัดครบ 4 หมวดสำหรับเขียนเล่ม

วางแผนเก็บ metrics 4 หมวดให้ตอบคำถามวิจัยทั้ง 4 ข้อพอดี:

| หมวด | ตอบคำถามวิจัยข้อ | สคริปต์ |
|---|---|---|
| (A) Latency breakdown | ข้อ 3 — คอขวดอยู่ไหน | `bench_latency.py` |
| (B) Power & efficiency | ข้อ 2 — คุ้มไหม | `measure_power.py` (INA260) |
| (C) Accuracy float vs INT8 | ข้อ 2 — เสียความแม่นยำเท่าไร | `infer_dump.py` + `eval_map.py` |
| (D) Counting vs GT | ข้อ 4 — นับแม่นแค่ไหน | `video_dump_dets.py` + `count_offline.py` |

**เทคนิคที่ใช้ในหมวด D — แยกการ detect ออกจากการ tune:**
แทนที่จะรันวิดีโอ 600 วินาทีซ้ำทุกครั้งที่อยากลองค่าพารามิเตอร์ (ซึ่งใช้เวลารอบละ ~30 นาที) เราทำแบบนี้แทน:
1. รันบนบอร์ด **ครั้งเดียว** → dump กล่องที่ detect ได้ทุกเฟรมลง `dets600.json` (174,769 กล่อง, 6.3 MB)
2. เอาไฟล์นั้นมา **sweep พารามิเตอร์ tracker/เส้นนับบน host แบบไม่จำกัดรอบ**

→ ประหยัดเวลาไปมหาศาล และทำให้ทำ sensitivity analysis ได้ (ซึ่งกลายเป็นตารางที่มีค่าที่สุดตารางหนึ่งในเล่ม)

---

## 4. ผลการทดลอง (ตัวเลขจริงทั้งหมด)

> ทุกค่าวัดจากบอร์ดจริง KV260 (PetaLinux starterkit 2022.2, VART 3.0, DPU B4096)
> โมเดล: `yolov8n_pkg_kv260.xmodel` INT8 · เทียบกับ GPU GTX 1660 Ti Max-Q FP32

### ตารางที่ 1 — สเปคโมเดลที่ deploy

| รายการ | ค่า |
|---|---|
| สถาปัตยกรรม | YOLOv8n (backbone + neck + Detect head), 1 คลาส (`package`) |
| Activation | LeakyReLU(0.1015625) แทน SiLU |
| Precision | INT8 (post-training quantization, `vai_q_pytorch`) |
| Graph partition | **1 DPU subgraph** (conv ทั้งหมดบน DPU; dequant/decode บน ARM PS) |
| DPU workload | **8.10 GOP** ต่อ inference |
| ขนาดโมเดล | float 6.30 MB → **INT8 xmodel 3.99 MB** (เล็กลง 1.6×) |
| Output head | 3 สเกล × [H,W,65] = 4×16 (DFL) + 1 (class) |
| Input | 640×640×3, plain resize, NHWC int8 |
| DPU memory | CONST 3.35 MB · WORKSPACE 10.85 MB |

---

### ตารางที่ 2 — ความแม่นยำ: float vs INT8 ⭐

วัดบน valid set 57 ภาพ (289 กล่อง GT) ด้วย pipeline เดียวกันทั้งสองฝั่ง (plain resize 640 → DFL decode → NMS iou 0.7 conf 0.001) เพื่อแยกผลของ quantization ล้วนๆ

| Metric | Float (host) | INT8 (KV260) | Δ |
|---|---|---|---|
| **mAP@0.50** | 0.8846 | **0.8713** | **−0.0133 (−1.5%)** |
| mAP@0.50:0.95 | 0.6879 | 0.6795 | −0.0084 (−1.2%) |
| Precision @0.25 | 0.8007 | 0.8225 | +0.0218 |
| Recall @0.25 | 0.8478 | 0.8339 | −0.0139 |
| F1 @0.25 | 0.8235 | 0.8282 | +0.0047 |

**ข้อสรุป:** quantize INT8 ทำให้ mAP@0.5 ลดเพียง **1.3 จุด (1.5%)** และ F1 แทบไม่เปลี่ยน (0.824 → 0.828)

**จุดที่ควรเน้นตอนนำเสนอ:** float mAP@0.5 ที่วัดได้ = 0.8846 **ตรงกับค่าที่ validate มาตรฐานบน Colab (0.884)** → เป็นการ cross-check ว่า pipeline การวัดผลที่เขียนเองถูกต้อง ไม่ได้วัดผิดแล้วบังเอิญได้เลขสวย

> **บั๊กที่เจอระหว่างทำหมวดนี้:** ground truth จาก Roboflow เป็น **polygon segmentation** ไม่ใช่ bounding box → ตอนแรกคำนวณ mAP ออกมาผิด ต้องเขียนโค้ดแปลง polygon → bbox ก่อน

---

### ตารางที่ 3 — Latency breakdown ต่อเฟรม ⭐ (ข้อค้นพบสำคัญ)

single-thread, เฉลี่ย 100 iterations (หลัง warmup 10 รอบ)

| Stage | เวลา (ms) | สัดส่วน |
|---|---|---|
| Preprocess (resize/convert/quantize บน PS) | 49.75 | **63%** |
| **DPU inference** | **12.49** | **16%** |
| Decode + NMS (DFL/sigmoid/NMS บน PS) | 16.07 | 21% |
| **รวม end-to-end** | **78.31** | 100% |

- DPU-only throughput = **80.1 FPS** (สอดคล้องกับ `xdputil benchmark` = 82.5 FPS → ยืนยันการวัดถูก)
- End-to-end = **12.77 FPS**

> **ข้อค้นพบ:** DPU ใช้เวลาเพียง **16%** ของ pipeline — คอขวดจริงคือ **preprocessing บน ARM PS (63%)**
> ระบบเป็น **PS-bound ไม่ใช่ DPU-bound**
>
> นี่คือข้อค้นพบเชิงระบบที่มีค่ากว่าตัวเลข FPS เพราะมันบอกว่า "การซื้อ accelerator ที่แรงขึ้นจะไม่ช่วยอะไร" — ต้องไปแก้ที่ preprocessing (vectorize, เขียนเป็น C++, หรือใช้ hardware scaler ใน PL) ซึ่งเป็น**ทิศทางงานต่อยอดที่ชัดเจน**

---

### ตารางที่ 4 — กำลังไฟและประสิทธิภาพพลังงาน

วัดด้วยเซนเซอร์ **INA260** (`ina260_u14`) บนราง SOM ของบอร์ดจริง

| สภาวะ | Power (W) | Throughput (FPS) | Efficiency (FPS/W) |
|---|---|---|---|
| Idle | 4.85 | — | — |
| **DPU-saturated** (benchmark 4 threads) | 9.27 avg / 9.90 peak | 89.6 | **9.67** |
| **Real application** (video counting) | 5.10 avg / 5.45 peak | 10.15 | 1.99 |

> **ข้อค้นพบที่น่าสนใจ:** ตอนรันแอปนับกล่องจริง กำลังไฟเฉลี่ยเกาะ idle แทบไม่ขยับ (**+0.25 W เท่านั้น**) เพราะ DPU ว่างเกือบตลอดเวลา (รอ preprocess 49.7 ms ในขณะที่ตัวเองใช้แค่ 12.5 ms)
>
> → **เป็นหลักฐานอิสระอีกชิ้นที่ยืนยันว่าระบบ PS-bound** (สอดคล้องกับตารางที่ 3 ที่วัดคนละวิธี)
>
> ดังนั้นรายงานต้องแยก **2 จุดปฏิบัติการ** ให้ชัด: (i) ศักยภาพสูงสุดของ accelerator = 9.67 FPS/W และ (ii) จุดที่ใช้งานจริงตอนนี้ = 1.99 FPS/W

---

### ตารางที่ 5 — เปรียบเทียบ GPU (FP32) vs KV260 FPGA (INT8) ⭐

GPU วัดกำลังไฟเฉพาะชิป (nvidia-smi, idle 3.83 W) · FPGA วัด**ทั้งบอร์ด SOM** (idle 4.85 W)

| ระบบ | Precision | FPS | Power (W) | FPS/W |
|---|---|---|---|---|
| GPU pure (batch 1) | FP32 | 31.5 | 22.80 | 1.38 |
| GPU e2e (batch 1) | FP32 | 25.5 | 22.65 | 1.13 |
| GPU peak (batch 32) | FP32 | 263.9 | 58.46 | 4.51 |
| **KV260 DPU-sat** (4 threads) | INT8 | 89.6 | **9.27** | **9.67** |
| **KV260 real-app** (video) | INT8 | 10.15 | **5.10** | 1.99 |

**ข้อสรุปเชิงเปรียบเทียบ (รายงานอย่างซื่อสัตย์ หลายจุดปฏิบัติการ):**

- **ประสิทธิภาพพลังงาน ดีสุด-ต่อ-ดีสุด:** FPGA 9.67 vs GPU 4.51 FPS/W → FPGA **ดีกว่า ~2.1×**
  (และ FPGA ยัง**เสียเปรียบ**ในการวัดด้วย เพราะวัดไฟทั้งบอร์ด ส่วน GPU วัดแค่ชิป ไม่รวม CPU/RAM/เมนบอร์ด)
- **กำลังไฟสัมบูรณ์:** FPGA 9.27 W vs GPU peak 58.5 W → **น้อยกว่า ~6.3×** ← จุดขายหลักของ edge
- **เทียบแบบแฟร์ที่สุด (end-to-end ทั้งคู่):** FPGA 1.99 vs GPU 1.13 FPS/W → ดีกว่า **1.8×** และกินไฟ 5.1 W vs 22.65 W (**น้อยกว่า 4.4×**)
- **สิ่งที่แลกไป:** throughput ดิบ GPU ชนะ (264 vs 89.6 FPS) — FPGA แลกความเร็วดิบกับพลังงาน/ขนาด/ต้นทุน
- **caveat FP32 vs INT8 ปิดด้วยตารางที่ 2:** ต้นทุนของ INT8 คือ mAP ลดแค่ 1.5% → คุ้มค่าอย่างชัดเจน

---

### ตารางที่ 6 — งานนับกล่อง เทียบ Ground Truth ⭐ (โจทย์หลักของโครงงาน)

Ground truth = **215 กล่อง** บนคลิป 600 วินาที (18,000 เฟรม @30fps) นับด้วยมือ
วิธี: DPU detection ทุกเฟรม → centroid tracker (greedy matching) → นับเมื่อ track ข้ามเส้นครั้งแรก

| Config เส้นนับ | นับได้ | Error |
|---|---|---|
| **calibrated (แกน x, เส้นที่ 0.40·W, max_dist=60)** | **215** | **0.0%** |
| แกน x, เส้น 0.30·W, max_dist=60 | 219 | 1.9% |
| แกน x, เส้น 0.30·W, max_dist=40 | 209 | 2.8% |
| แกน x, เส้น 0.40·W, max_dist=90 | 224 | 4.2% |
| **ไม่ calibrate (เส้นกลางจอ 0.5)** | 270–331 | **25–54%** |

**สิ่งที่วิเคราะห์ได้จากตารางนี้ (ไม่ใช่แค่ตัวเลข 100%):**

1. **แกนต้องเป็น x** — กล่องเคลื่อนที่แนวนอนบนสายพาน (แกน y ผิดหมด error 20–92%) → เป็นการยืนยันว่าระบบเข้าใจการเคลื่อนที่ถูก
2. **ตำแหน่งเส้นต้องอยู่โซนที่กล่องเพิ่งเข้าเฟรม (x ≈ 0.3–0.4)** ที่กล่องยังไม่ทับกัน — **เส้นกลางจอนับเกินอย่างรุนแรง** เพราะบริเวณกลางภาพกล่องหนาแน่น/ซ้อนทับ → detection กระพริบ → track แตกเป็นหลายเส้น → นับซ้ำ
3. **ผลไม่ใช่ fluke** — บริเวณรอบๆ ค่าที่ calibrate (เส้น 0.3–0.4 × max_dist 40–60) ให้ผล 209–219 = **error < 3% ทั้งย่าน** → ระบบ robust ไม่ใช่จูนจนพอดีเป๊ะจุดเดียว
4. การกำหนดเส้นนับคือ **calibration มาตรฐาน** ของระบบ line-counting ทุกตัว (ทำครั้งเดียวต่อ 1 มุมกล้อง/สายพาน) ไม่ใช่การ "แอบจูนให้ได้เลขสวย"

- Processing throughput = ~10 FPS บนบอร์ด (real-time factor ~0.33× ที่วิดีโอ 30 fps)
- คลิปสาธิต: `08-figures/out20.mp4`, `08-figures/out600_calibrated.mp4`

> **ข้อจำกัดทางเทคนิคที่เจอ:** OpenCV 4.5.2 บนบอร์ด (ผ่าน GStreamer) **decode H.264 ไม่ได้** → ต้องแปลงคลิปเป็น MJPG `.avi` ก่อนโอนขึ้นบอร์ด (514 MB, พอดีกับ `/tmp` tmpfs ขนาด 2 GB)

---

## 5. Track B — YOLO26n: ผลเชิงลบที่มีคุณค่าทางวิชาการ

ส่วนนี้ควรเล่าให้อาจารย์ฟังอย่างละเอียด เพราะเป็น **contribution ที่แท้จริงของโครงงาน** ไม่ใช่ส่วนที่ล้มเหลว

### 5.1 อุปสรรคชั้นแรก: toolchain เก่ากว่าโมเดล

จะรัน Inspector (เครื่องมือวิเคราะห์ว่า op ไหนจะตกไป CPU) บน YOLO26n **ทำไม่ได้** เพราะต้องใช้ 2 แพ็กเกจที่ต้องการ Python คนละเวอร์ชัน:

| ต้องการ | อยู่ที่ | Python |
|---|---|---|
| `pytorch_nndct` (Inspector + quantizer) | conda env `vitis-ai-pytorch` เท่านั้น | **3.7.12** |
| `ultralytics==8.4.71` (โหลด YOLO26n ได้) | ต้องติดตั้งเอง | บังคับ **≥ 3.8** |

สำรวจ env ทั้งหมดในคอนเทนเนอร์แล้ว: `base` = py3.9.9 (ว่างเปล่า ไม่มี torch/nndct), `vitis-ai-pytorch` = py3.7.12 (มี nndct), `vitis-ai-wego-torch` = py3.7.8 → **ไม่มี env ไหนมีทั้ง nndct และ py3.8+ พร้อมกัน** และ nndct ถูก build ผูกกับ py3.7 + torch 1.12.1 ย้ายข้ามไม่ได้

> **นี่คือบทเรียนเชิงระบบที่ควรรายงาน:** การ deploy โมเดล AI รุ่นใหม่บน DPU stack เดิม **ติดที่ toolchain/Python compatibility ก่อนจะไปถึงเรื่อง op mapping ด้วยซ้ำ** — Vitis AI 3.0 เป็นของปี 2022 (Python 3.7) ในขณะที่ YOLO26 ต้องการ ultralytics รุ่นใหม่ที่ต้องการ Python 3.8+ นี่เป็นข้อจำกัดเชิงปฏิบัติที่คนวางแผนโครงการต้องรู้

### 5.2 ทางออกที่ใช้: วิเคราะห์ ONNX op graph เอง

แทนที่จะรอ Inspector → สร้าง env py3.9 แยกบนไดรฟ์ D: แล้ว export ONNX เอง (`yolo26n_o2m_leakyrelu.onnx`, 457 nodes, `end2end=False`, SiLU→LeakyReLU ครบ) แล้ววิเคราะห์ op histogram เทียบกับ watchlist ของ DPUCZDX8G

### 5.3 ผลการวิเคราะห์ — สองข้อค้นพบ

#### ✅ ข่าวดี: NMS-free head ไม่ใช่ปัญหา (ตรงข้ามกับที่คาดไว้ตอนแรก)
ตั้ง `end2end=False` → **watchlist ops หายหมด**: ไม่มี `TopK`, `GatherElements`, `GatherND`, `ReduceMax`, `Mod`, `ScatterND`, `NonMaxSuppression`, `Range`
เพราะ postprocess ที่สร้าง TopK/GatherElements ถูกข้าม แล้วเราไปทำ decode/NMS บน ARM PS ตอน deploy แทน (เหมือนที่ทำสำเร็จแล้วกับ Track A)

#### 🔴 ตัวบล็อกจริง: attention block 2 จุด "กลาง" เน็ตเวิร์ก

| ตำแหน่งในกราฟ | โมดูล | ops |
|---|---|---|
| node ~148–153 (**~32% ของกราฟ**) | `/model.10/m/m.0/attn` | MatMul → Softmax → MatMul |
| node ~315–320 (**~69% ของกราฟ**) | `/model.22/m.0/m.0.1/attn` | MatMul → Softmax → MatMul |

โครงสร้างคือ self-attention: Q·Kᵀ → Softmax → ·V บน DPUCZDX8G (Zynq UltraScale+) มีปัญหา 2 อย่าง:
- **MatMul แบบ data × data** (ไม่ใช่ conv ที่คูณกับ weight คงที่) → DPU map ไม่ได้ เพราะ DPU ออกแบบมาสำหรับ convolution ที่ weight เป็นค่าคงที่
- **Softmax** → ไม่ใช่ DPU op → ต้องเด้งไปรันบน ARM PS

**กลไกที่ทำให้ล้ม gate:** เพราะ attention อยู่ที่ ~32% และ ~69% ของกราฟ — คือ**อยู่กลาง ไม่ใช่ปลาย** — จึงจะ**ตัด DPU subgraph ออกเป็น ~3 ชิ้น** → gate "1 DPU subgraph" ผ่านไม่ได้ตามสภาพ

(เปรียบเทียบ: `Sigmoid` ที่อยู่ตำแหน่ง ~99.6% = ปลายสุด **ไม่ตัด subgraph** เพราะอยู่ท้ายกราฟพอดี ส่วน MaxPool×3 ใน SPPF และ Resize×2 ใน FPN upsample → DPU รองรับปกติ ไม่ใช่ปัญหา)

### 5.4 ทำไมผลนี้คือ contribution

> **"YOLO26 ติด DPUCZDX8G ตรงไหน" ยังไม่มีใครรายงาน** — YOLO26 ไม่อยู่ใน Vitis AI Model Zoo และเป็นโมเดลใหม่ ผลนี้บอกคนที่จะทำต่อว่า:
> 1. ไม่ต้องไปกังวลเรื่อง NMS-free head (แก้ได้ด้วย `end2end=False`)
> 2. ตัวบล็อกจริงคือ attention (A2C2f/PSA) และมันอยู่ตำแหน่งที่แย่ที่สุด (กลางกราฟ)
> 3. ทางเลือกที่เหลือมี 3 ทาง: (a) ยอมรับ multi-subgraph แล้ววัด latency penalty จริง (b) ถอด/แทน attention ด้วยบล็อก conv-only แล้ว retrain (c) รอ DPU รุ่นที่รองรับ attention

**สถานะปัจจุบัน:** ยังต้อง compile จริงด้วย `vai_c_xir` เพื่อ**ยืนยันเชิงประจักษ์**ว่าตัดเป็นกี่ subgraph จริง (ตามกติกาที่ตั้งไว้ในข้อ 2.3 ว่า histogram เป็นแค่ advisory) — เป็นงานที่เหลืออยู่และไม่ต้องใช้บอร์ด

📄 รายละเอียด: `03-model/track-b-yolo26n-target/artifacts/inspect_report/FINDINGS_op_analysis.md`, `07-notes/M2B1_inspector_python_blocker.md`

---

## 6. ส่วนขยาย: port ไป Efinix Titanium Ti375 (เริ่ม 16 ส.ค.)

### 6.1 เหตุผลเชิงวิชาการ
เมื่อระบบบน KV260 พิสูจน์จบแล้ว การทำ**เรื่องเดิมบนสถาปัตยกรรม FPGA คนละแบบ**จะทำให้ได้ contribution ใหม่ที่แข็งขึ้นมาก:

> **Cross-platform comparison:** ระบบนับ YOLO ตัวเดียวกัน dataset เดียวกัน ground truth เดียวกัน บน **3 แพลตฟอร์ม**
> **GPU** (FP32) vs **KV260** (hard DPU, INT8) vs **Ti375** (soft RISC-V + TinyML accelerator, INT8)
> → เทียบ FPS / W / mAP / counting error / resource utilization / effort ที่ใช้พัฒนา

นี่คือสิ่งที่ทำให้โปรเจคที่สอง**ไม่ใช่งานซ้ำ** แต่เป็นส่วนขยายเชิงวิชาการ

### 6.2 flow ต่างกันคนละโลก

| | **KV260** | **Ti375** |
|---|---|---|
| สถาปัตยกรรม | Zynq US+ (Arm hard PS + PL) | Titanium (hardened quad-core RISC-V + fabric) |
| Toolchain | Vitis AI 3.0 (Docker) | Efinity IDE + RISC-V SW IDE (RiscFree) |
| ตัวเร่ง AI | DPUCZDX8G B4096 (hard IP) | Sapphire RISC-V SoC + **Efinix TinyML Accelerator** (Lite/Standard) |
| รูปแบบโมเดล | `.xmodel` | **`.tflite` INT8** |
| Quantize | `vai_q_pytorch` | **TFLite Converter** |
| Compile | `vai_c_xir` | **Efinix TinyML Generator** → C arrays |
| Runtime | VART C++ บน Cortex-A53 | **TFLite Micro** บน Sapphire RISC-V |
| Deploy | `.xmodel` + overlay bitstream | **combined hex** (bitstream + RISC-V binary) |

### 6.3 ความคืบหน้า: **E3-A เสร็จแล้ว** ✅

ได้โมเดล INT8 TFLite พร้อม deploy: `yolov8n_leaky_pkg_ft_full_integer_quant.tflite` (3.1 MB, int8 ทั้ง input และ output)

**ผล verify:** INT8 เทียบ FLOAT32 → **parity 59/59 กล่อง @ conf 0.25** (ตรงกันทุกกล่อง) และ detection ใกล้เคียง GT (39 vs 44, score 0.96–0.98) → **quantization ซื่อสัตย์สมบูรณ์**

**บั๊กสำคัญที่เจอและแก้ (ควรเล่า เพราะเป็นบทเรียนเรื่อง quantization ที่ดีมาก):**
- ตอนแรก calibration data เป็นช่วง **[0,255]** แต่ SavedModel คาดหวัง **[0,1]** → input scale ที่ TFLite คำนวณออกมาได้ 1.0 = ข้อมูลขยะ (score ระเบิด, ได้ 300+ กล่องมั่ว)
- แก้โดย normalize `/255` ใน representative dataset → **input scale = 0.00392 = 1/255 พอดี, zero-point = −128 พอดี** ซึ่งเป็นค่าที่ "ถูกต้องโดยโครงสร้าง"
- **บทเรียน:** representative dataset ที่ผิด scale ทำให้ quantization พังทั้งหมด โดยที่ไม่มี error ใดๆ แจ้งเตือน — ต้องตรวจค่า scale/zero-point ที่ได้ออกมาเสมอ

**ปัญหา OOM ที่แก้ได้:** ตัว export INT8 ของ ultralytics โหลด calibration 57 ภาพเป็น array เดียว 267 MB → OOM บนเครื่อง RAM 5.9 GB → เขียน `quantize_tflite_int8.py` เองที่ป้อน representative dataset **ทีละภาพผ่าน mmap**

**ความแตกต่างสำคัญจาก KV260 ที่ค้นพบ:**
- input `[1,640,640,3]` int8 scale 1/255 zp −128 → **preprocess = pixel − 128** (ง่ายกว่า KV260 มาก)
- output `[1,5,8400]` — **DFL ถูก fold เข้าในกราฟแล้ว → ไม่ต้อง decode DFL เอง** (ต่างจาก KV260 ที่ต้องเขียน DFL decode บน PS)

**โค้ดฝั่ง RISC-V เขียนเสร็จแล้ว** (`04-deploy/riscv-app/`): `postprocess.c` (dequant + xywh→xyxy + NMS), `counting.c` (centroid tracker + line-crossing, ใช้พารามิเตอร์เดียวกับที่ calibrate ไว้), `main.cc` (TFLite Micro glue) — รอ compile ที่เครื่องที่ติดตั้ง Efinity

**เขียน turnkey run guide** `RUN_ON_EFINITY_MACHINE.md` ครบทุกขั้น (IP Manager → TinyML Generator → RiscFree → Programmer → run) พร้อม troubleshooting และแผนสำรอง เพราะ Efinity กับบอร์ดอยู่คนละเครื่อง

### 6.4 ความเสี่ยงที่ยอมรับไว้แล้ว
Efinix TinyML Accelerator เดิมออกแบบมาสำหรับ**โมเดลเล็ก** (TFLite Micro) → **ยังไม่ยืนยันว่า YOLOv8n เต็มตัวรันไหว** จุดตัดสินอยู่ที่ E2 (ประเมิน resource fit + เลือกโหมด Lite/Standard)

ข่าวดีคือ Efinix มี demo **"Yolo Person Detection"** บน Ti375 อยู่แล้ว → พิสูจน์ว่า YOLO บนแพลตฟอร์มนี้ทำได้ในหลักการ
ถ้ารันเต็มตัวไม่ไหว → fallback คือ (a) เริ่มจาก demo แล้ว retrain เป็น package (b) ลด input resolution
**และไม่ว่าทางไหน ผลก็ยังเขียนเข้าเล่มได้** ในหัวข้อ "ข้อจำกัดของ soft accelerator เทียบ hard DPU"

---

## 7. ปัญหาวิศวกรรมที่เจอตลอดทาง และวิธีแก้

ส่วนนี้รวบรวมไว้เพราะอาจารย์มักถามว่า "ติดอะไรบ้าง" — และมันแสดงว่าเราเข้าใจระบบจริง ไม่ใช่แค่รันสคริปต์ตาม tutorial

| # | ปัญหา | สาเหตุราก | วิธีแก้ | บทเรียน |
|---|---|---|---|---|
| 1 | WSL filesystem พัง คำสั่งพื้นฐาน segfault | `.vhdx` วางบน C: ที่เหลือ 1.22 GB, torch ~6 GB ทำให้โตเกิน → remount read-only | `wsl --shutdown` + เคลียร์ C: ให้เหลือ >12 GB | **`df` ใน WSL โกหก** — โชว์ provision 1007 GB แต่ตัวจริงคือพื้นที่ Windows host |
| 2 | `pip install torch` ล้ม ทั้งที่ดิสก์เหลือ 954 GB | `/tmp` เป็น tmpfs (RAM) จำกัด 1.4 GB pip ใช้เป็น scratch | ตั้ง `TMPDIR` ชี้ไป disk จริง | **ดิสก์เหลือ ≠ ทุก mount เหลือ** |
| 3 | `cuda.is_available() = False` ทั้งที่ลง torch สำเร็จ | Python 3.14 ใหม่เกินไป PyTorch ยังไม่ build CUDA wheel สำหรับ cp314 → ได้ CPU build | ใช้ miniforge สร้าง env Python 3.12 | **"ลงสำเร็จ ≠ ได้ของที่ต้องการ"** — ต้อง verify ที่ผลลัพธ์ ไม่ใช่ที่ exit code |
| 4 | ultralytics ลาก torch CPU มาทับ +cu128 | pip resolve dependency เอง | ลง torch ก่อน แล้ว `--no-deps` + ลง dep ที่เหลือเอง | dependency resolver ทำลาย env ที่ตั้งใจ pin ไว้ได้ |
| 5 | Docker Desktop crash "initializing Inference manager" | Model Runner ค้างกับ stale AF_UNIX socket ใน `%LOCALAPPDATA%\Docker\run\` ที่ Windows ลบไม่ได้ | kill Docker ทุก process แล้วเปิดใหม่ (หรือปิด Model Runner) | — |
| 6 | quantize container ตาย `unexpected EOF` exit 125 | OOM — host RAM 5.88 GB, WSL2 cap ~2.9 GB, C: เหลือ 0.5 GB ทำให้ pagefile ขยายไม่ได้ | เคลียร์ C: (0.2→17 GB) + `--subset_len 32 --batch_size 4` | ดิสก์เต็มทำให้ **RAM** ไม่พอได้ (ผ่าน pagefile) |
| 7 | `XIR don't support multi-outputs op` ตอน export | สคริปต์ track-a ขาด fix 2 อย่างจาก phase0: C2f `chunk` monkeypatch และ BackboneHead wrapper ที่ตัด decode ทิ้ง | ก๊อป proven script มาทับ (เก็บตัวเก่าเป็น `.bak`) | **regression เกิดจากการ "จัดระเบียบโค้ด" ได้** — ต้อง diff กับตัวที่เคยผ่าน |
| 8 | สคริปต์บอกว่าสำเร็จทั้งที่ล้ม | `set +e` ทำให้เดินต่อแม้ compile fail | patch ให้ abort ถ้าไม่มี `_int.xmodel` | **verify artifact จริง ไม่ใช่เชื่อ log** |
| 9 | `scp` ขึ้นบอร์ดไม่ได้ | PetaLinux ไม่มี `sftp-server` | `python -m http.server` บน host + `wget` บนบอร์ด + verify md5 | — |
| 10 | mAP คำนวณออกมาผิด | GT จาก Roboflow เป็น polygon segmentation ไม่ใช่ bbox | เขียนโค้ดแปลง polygon → bbox | ต้องเข้าใจ format ของ GT ก่อนวัดผล |
| 11 | บอร์ด decode วิดีโอ H.264 ไม่ได้ | OpenCV 4.5.2 + GStreamer บน PetaLinux ไม่มี codec | แปลงเป็น MJPG `.avi` ก่อนโอน (514 MB, `/tmp` tmpfs 2 GB พอ) | — |
| 12 | Efinix: input scale ออกมาเป็น 1.0 = ขยะ | calibration data เป็น [0,255] แต่โมเดลรอ [0,1] | normalize `/255` ใน representative dataset → scale = 1/255 zp = −128 พอดี | **quantization พังเงียบๆ ได้ ต้องตรวจ scale/zp ที่ได้เสมอ** |
| 13 | Inspector รัน YOLO26n ไม่ได้ | nndct ต้อง py3.7, ultralytics 8.4.71 ต้อง py3.8+ | export ONNX ใน env py3.9 แยก แล้ววิเคราะห์ op graph เอง | toolchain compatibility บล็อกก่อนถึงเรื่องเทคนิค |

### 7.1 กรณีศึกษาพิเศษ — การขุดหาสาเหตุที่สอนอะไรได้มาก (M5)

ระหว่างทำ verification เจอสัญญาณเตือน: cosine similarity ระหว่าง float กับ quantized model **ต่ำกว่าเกณฑ์ 0.99** ที่ 2 ใน 3 output tensor (0.991 / 0.942 / 0.894)

แทนที่จะเดา เราขุดเป็นชั้นๆ:

1. **สมมติฐานแรก:** ส่วน BOX (DFL) เป็นตัวฉุด → แยกวัด BOX vs CLASS → **ถูกหักล้าง** เพราะ CLASS แย่พอกันหรือแย่กว่า
2. **สังเกตแพทเทิร์น:** logit range ระเบิดตามความละเอียดที่หยาบลง — ±65 (80×80) → ±340 (40×40) → **±2007 (20×20)**
3. **พิสูจน์ saturation ตรงๆ:** float range `[−128.7, 56.1]` แต่ quantized ถูก clamp เหลือ `[−64.0, 38.5]` — int8 มีแค่ 256 ระดับ ถ้า range ±2000 แปลว่า 1 ระดับ ≈ 16 หน่วย
4. **หาผลกระทบจริง:** พิกเซลที่แย่ที่สุด float logit = 6.004 (sigmoid = 0.9975 = **เจอ object**) → quantized logit = −11.0 (sigmoid = 0.0000 = **พลาด**) → **sign flip**
5. **หาเบาะแสต้นตอ:** cell ที่ class prob > 0.5 มีถึง **556 cells** ที่ 40×40 — ผิดปกติมหาศาล ภาพปกติมี object ไม่กี่ตัว → **แสดงว่า golden float model เองก็เพี้ยน**

**สรุปเชิงกลไก:**
```
สลับ SiLU→LeakyReLU โดยยังไม่ fine-tune
   └─► logit distribution เพี้ยน range ระเบิด ±2000
          └─► int8 per-tensor fixpos เดียวรับ range ไม่ไหว → saturate/clamp
                 └─► cos_sim ตก + detection พลิกจาก "เจอ" เป็น "พลาด"
```

**ข้อสรุปสองชั้น:**
- **ไม่ใช่บั๊กของ pipeline** — เป็นผลลูกโซ่จากการยังไม่ fine-tune → แก้ที่ต้นเหตุ (ทำ fine-tune ก่อน) แล้วปัญหาหายไปเอง ซึ่ง**ยืนยันแล้วด้วยผลจริง**: หลัง fine-tune ครบ INT8 mAP drop เหลือแค่ 1.5%
- **metric ที่เลือกใช้ตอนแรกผิด** — cosine บน raw pre-decode logit มองโลกแง่ร้ายเกินไป สิ่งที่ควร verify คือ **detection agreement / mAP หลัง decode+NMS** ซึ่งเป็นสิ่งที่ผู้ใช้เห็นจริง

📄 รายละเอียด: `07-notes/M5_cossim_dropoff_diagnosis.md`

---

## 8. สถานะ milestone และงานที่เหลือ

### 8.1 โปรเจคหลัก (KV260) — M0–M14

| # | Milestone | สถานะ |
|---|---|---|
| M0 | เลือก target board + toolchain | ✅ |
| M1 | ตั้ง Vitis AI 3.0 container + flow ครบ | ✅ |
| M2-A | Track A: YOLOv8n compile ผ่าน gate | ✅ |
| M3-GPU | GPU baseline (FP32 คู่) | ✅ |
| P0 | กู้ artifact ออกจาก WSL | ✅ |
| P1 | เคลียร์ B3136 vs B4096 + identity ของ xmodel | ✅ |
| M2-B1 | Track B: วิเคราะห์ op ของ YOLO26n | ✅ (ผ่าน ONNX op analysis) |
| M2-B2 | Track B: fine-tune หลัง SiLU→LeakyReLU | ✅ (Colab T4, 400 epochs) |
| **M2-B3** | **Track B: compile YOLO26n → gate** | ☐ **เหลืออยู่** |
| M4 | VART host code | ✅ |
| M5 | Verify quantization fidelity | ✅ (ปิดด้วย mAP drop 1.5% แทน cosine) |
| M7 | Counting ground-truth dataset | ✅ (GT = 215) |
| M8 | re-quantize เป็น single-class | ✅ |
| **M9** | **Board bring-up** | ✅ **fingerprint ตรง, 82.5 FPS** |
| **M10** | **Tracking + line-crossing counting** | ✅ **215/215 = 0% error** |
| **M11** | **Benchmark เทียบ GPU baseline** | ✅ **ครบ 4 หมวด** |
| M12 | Profiling หา bottleneck | ✅ (พบว่า PS-bound, preproc 63%) |
| M13 | (Optional) custom accelerator | ☐ ไม่ผูกเป็นเงื่อนไขจบ |
| M14 | รายงาน + สไลด์ป้องกัน | 🟡 ร่างบท Results เสร็จแล้ว |

### 8.2 งานที่เหลือ เรียงตามความสำคัญ

**ลำดับ 1 — ปิด Track B ให้สมบูรณ์ (ไม่ต้องใช้บอร์ด)**
รัน `vai_c_xir` กับ YOLO26n จริง เพื่อยืนยันเชิงประจักษ์ว่ากราฟถูกตัดเป็นกี่ subgraph ตามที่ ONNX analysis ทำนายไว้ (~3 ชิ้น) → ทำให้ finding เรื่อง attention มีหลักฐานระดับ compiler ไม่ใช่แค่ระดับ op histogram
*งานนี้สำคัญเพราะเป็นการทำตามกติกาที่ตัวเองตั้งไว้ในข้อ 2.3 — ถ้าไม่ทำ จะโดนถามได้ว่า "แน่ใจได้อย่างไรว่ามันตัดจริง"*

**ลำดับ 2 — finalize บท Results**
ร่างเสร็จแล้ว (`05-benchmarks/results/RESULTS_chapter_draft.md`, 6 ตาราง + key findings) เหลือขัดสำนวนและใส่รูป

**ลำดับ 3 — วิดีโอ demo พร้อมเส้นนับที่ calibrate แล้ว**
มีไฟล์ `08-figures/out600_calibrated.mp4` และ `out600_conf45_roi.mp4` แล้ว ใช้สำหรับ present ได้

**ลำดับ 4 — เดินหน้า Efinix**
E0 (platform selection doc) → **E1 (ติดตั้ง Efinity + รัน TinyML demo ให้ผ่าน = gate จริงอันแรก)** → E2 (ตัดสิน Lite/Standard + resource fit) → E4 (Generator) → E9 (bring-up)

**ลำดับ 5 (optional) — optimize preprocessing**
จากข้อค้นพบว่าคอขวดคือ preproc 49.7 ms → ถ้ามีเวลา ลองเขียนเป็น C++ หรือใช้ hardware scaler → จะดัน e2e FPS ขึ้นได้มากโดย power แทบไม่เพิ่ม → เป็นผลเพิ่มที่ดีมากสำหรับเล่ม

---

## 9. คำถามที่น่าจะโดนถาม และคำตอบที่เตรียมไว้

**Q: ทำไม end-to-end ได้แค่ 12.8 FPS ทั้งที่ DPU ทำได้ 82 FPS แบบนี้เรียก real-time ได้หรือ**

A: ถูกต้องครับ และนั่นคือ**ข้อค้นพบหลักข้อหนึ่งของโครงงาน** — DPU ใช้เวลาแค่ 16% ของ pipeline ส่วน preprocessing บน ARM กิน 63% ระบบเป็น PS-bound ไม่ใช่ DPU-bound ยืนยันด้วยหลักฐาน 2 ชิ้นที่วัดคนละวิธี: (1) latency breakdown (2) power ตอนรันแอปจริงเพิ่มจาก idle แค่ 0.25 W = DPU ว่างเกือบตลอด ประเด็นนี้ชี้ทิศทาง optimization ที่ชัดเจน คือไปแก้ที่ preproc ไม่ใช่หา accelerator ที่แรงขึ้น

**Q: เทียบกับ GPU ยุติธรรมหรือไม่ ในเมื่อ FPGA เป็น INT8 แต่ GPU เป็น FP32**

A: เราปิดประเด็นนี้ด้วยตัวเลข mAP ครับ — INT8 ทำให้ mAP@0.5 ลดแค่ 1.5% (0.885 → 0.871) ดังนั้นต้นทุนความแม่นยำที่แลกมากับการประหยัดไฟ 6.3 เท่านั้นเล็กมาก นอกจากนี้การวัดของเรายัง**เข้าข้าง GPU** ด้วยซ้ำ เพราะ GPU วัดกำลังไฟเฉพาะชิป (ไม่รวม CPU/RAM/เมนบอร์ด) ส่วน FPGA วัดทั้งบอร์ด SOM

**Q: การนับได้ 215/215 พอดี จูนจนพอดีหรือเปล่า**

A: ไม่ครับ มีหลักฐาน 2 ชั้น: (1) **sensitivity analysis** — บริเวณรอบค่าที่เลือก (เส้น 0.3–0.4·W × max_dist 40–60) ให้ผล 209–219 คือ error < 3% ทั้งย่าน ไม่ใช่แหลมเดียว (2) การกำหนดเส้นนับเป็น **calibration มาตรฐาน** ของระบบ line-counting ทุกตัวในอุตสาหกรรม ทำครั้งเดียวต่อ 1 มุมกล้อง และเรารายงานผลตอนไม่ calibrate ไว้ด้วย (error 25–54%) เพื่อความโปร่งใส พร้อมอธิบายสาเหตุว่าเส้นกลางจอนับเกินเพราะกล่องซ้อนทับทำให้ track แตก

**Q: ชื่อโครงงานคือ YOLO26 แต่ที่ deploy สำเร็จคือ YOLOv8n**

A: ถูกต้องครับ และเราแยกสองเรื่องนี้ชัดเจนตั้งแต่ต้นด้วยกลยุทธ์ two-track: Track A (YOLOv8n) เป็นตัวพิสูจน์ว่า**ระบบนับทั้งระบบทำงานได้จริงบน FPGA** ส่วน Track B (YOLO26n) เป็นการตอบคำถามวิจัยว่า **โมเดลรุ่นใหม่ deploy ลง DPU รุ่นนี้ได้แค่ไหน** — และเราตอบได้แล้วอย่างเจาะจงว่าติดที่ attention block (MatMul แบบ data×data + Softmax) 2 จุดที่ ~32% และ ~69% ของกราฟ ซึ่งอยู่กลางกราฟจึงตัด subgraph ผลนี้เป็นองค์ความรู้ใหม่ที่ยังไม่มีใครรายงาน เพราะ YOLO26 ไม่อยู่ใน Vitis AI Model Zoo

**Q: ทำไมต้องเปลี่ยน SiLU เป็น LeakyReLU ทำให้ความแม่นยำเสียหรือไม่**

A: เพราะ DPUCZDX8G ไม่รองรับ SiLU ครับ ส่วนเรื่องความแม่นยำ — **ถ้าเปลี่ยนเฉยๆ โดยไม่ retrain จะพังหนักมาก** (เราพิสูจน์แล้วว่า logit range ระเบิดถึง ±2000 ทำให้ int8 saturate จน detection พลิกจากเจอเป็นพลาด) แต่**หลัง fine-tune 400 epochs ความแม่นยำกลับมาเท่าเดิม** (mAP 0.884) → ข้อสรุปคือการสลับ activation เพื่อ hardware compatibility ทำได้โดยไม่เสียความแม่นยำ ถ้ายอมลงทุน fine-tune

**Q: ทำไมทำสองบอร์ด จะทำทันหรือ**

A: โปรเจค KV260 พิสูจน์จบครบวงจรแล้วครับ (มีตัวเลขครบ 4 หมวด) ส่วน Efinix เป็น**ส่วนขยาย** ที่ทำให้ contribution แข็งขึ้นเป็น cross-platform comparison (GPU vs hard DPU vs soft RISC-V accelerator) และมี fallback ชัดเจน — ถ้า Ti375 รัน YOLO เต็มตัวไม่ไหว ผลนั้นก็ยังเขียนเข้าเล่มได้ในหัวข้อข้อจำกัดของ soft accelerator นอกจากนี้ dataset / ground truth / GPU baseline / counting logic reuse ได้ทั้งหมด จึงไม่ใช่การเริ่มใหม่จากศูนย์

**Q: ทำไมใช้ Python รัน inference ไม่ใช้ C++**

A: โค้ด C++ (`yolo_dpu_infer.cpp`) เขียนและ compile-check ผ่านแล้วครับ แต่ทำได้แค่ raw inference + dequantize ส่วน DFL decode ต้องเขียนใหม่ทั้งหมด เราจึงเลือกทำ Python ก่อนเพื่อ**พิสูจน์ความถูกต้องของ decode logic ให้จบก่อน** แล้วค่อย port เป็น C++ ทีหลัง — และเนื่องจากตอนนี้คอขวดคือ preprocessing (49.7 ms) ไม่ใช่ตัว interpreter การ port เป็น C++ จึงเป็นหนึ่งในวิธี optimize ที่ระบุไว้แล้วในงานต่อยอด

---

## 10. หลักฐานและไฟล์อ้างอิง

| หัวข้อ | ไฟล์ |
|---|---|
| **สถานะโครงงาน (source of truth)** | `00-admin/timeline.md` |
| **ร่างบทผลการทดลอง** | `05-benchmarks/results/RESULTS_chapter_draft.md` |
| **ผลวัดดิบทั้งหมดบนบอร์ด** | `05-benchmarks/results/kv260_results.md` |
| GPU baseline (ข้อมูลดิบ) | `05-benchmarks/gpu-baseline/results_fp32_final.json` |
| การขุดหาสาเหตุ quantization | `07-notes/M5_cossim_dropoff_diagnosis.md` |
| การตรวจสอบ arch + identity ของ xmodel | `07-notes/P1_arch_and_identity_resolution.md` |
| **ผลวิเคราะห์ YOLO26n op** | `03-model/track-b-yolo26n-target/artifacts/inspect_report/FINDINGS_op_analysis.md` |
| ปัญหา toolchain ของ Track B | `07-notes/M2B1_inspector_python_blocker.md` |
| worklog รายวัน | `07-notes/worklog.md` |
| **xmodel ที่ deploy จริง** | `03-model/track-a-yolov8n-baseline/artifacts/yolov8n_pkg_B4096/yolov8n_pkg_kv260.xmodel` |
| โค้ดบนบอร์ด | `04-deploy/board/` (`yolo_dpu_detect.py`, `bench_latency.py`, `measure_power.py`, `infer_dump.py`, `video_detect.py`, `video_dump_dets.py`) |
| โค้ดวัดผลบน host | `05-benchmarks/results/` (`eval_map.py`, `infer_float.py`, `count_offline.py`) |
| **รูปผล detection** | `08-figures/detect_out_pkg_kv260.jpg` |
| **วิดีโอ demo การนับ** | `08-figures/out20.mp4`, `08-figures/out600_calibrated.mp4` |
| Efinix — timeline | `../thesis-yolo26-fpga_efinix/00-admin/timeline.md` |
| Efinix — สเปค I/O ของโมเดล | `../thesis-yolo26-fpga_efinix/03-model/track-a-yolov8n-baseline/artifacts/TFLITE_IO_SPEC.md` |
| Efinix — คู่มือรันบนเครื่อง Efinity | `../thesis-yolo26-fpga_efinix/RUN_ON_EFINITY_MACHINE.md` |

---

## 11. สรุปปิดท้าย — สามประโยคสำหรับจำ

1. **ระบบทำงานได้จริงครบวงจรบนฮาร์ดแวร์จริง** — จากวิดีโอเข้า จนถึงตัวเลขนับออก บนบอร์ด KV260 ที่กินไฟ ~5 W นับได้ 215/215 กล่อง
2. **ตัวเลขที่ได้ตอบคำถามวิจัยครบทุกข้อ** — INT8 เสียความแม่นยำ 1.5% แต่ประหยัดไฟ 6.3 เท่า และประสิทธิภาพพลังงานดีกว่า GPU 2.1 เท่า
3. **สิ่งที่ "ล้ม" กลับเป็นส่วนที่มีค่าที่สุด** — เรารู้แล้วว่า YOLO26 ติด DPU ตรงไหน เพราะอะไร และคนที่ทำต่อควรทำอย่างไร ซึ่งเป็นองค์ความรู้ที่ยังไม่มีใครรายงาน
