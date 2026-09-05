# 09 — ส่วนขยาย: Port ไป Efinix Titanium Ti375

> **สรุปไฟล์นี้ใน 4 บรรทัด**
> เมื่อระบบบน KV260 พิสูจน์จบแล้ว จึง port ระบบเดียวกันไปบอร์ด Efinix Ti375 ที่ใช้สถาปัตยกรรมคนละแบบ
> เพื่อสร้าง contribution ใหม่ = **cross-platform comparison** ระหว่าง GPU vs hard DPU vs soft RISC-V accelerator
> สถานะ: **E3-A เสร็จแล้ว** — ได้โมเดล INT8 TFLite ที่ verify แล้วว่าตรงกับ float 59/59 กล่อง

---

## 1. เหตุผลเชิงวิชาการ

### 1.1 ปัญหาของการทำแค่แพลตฟอร์มเดียว

ถ้าโครงงานรายงานแค่ *"YOLO บน KV260 ได้ผลดี"* จะเกิดคำถามว่า:
- ผลนี้เป็นเพราะ **FPGA** ดี หรือเพราะ **DPU ของ AMD** ดี?
- ถ้าไปใช้ FPGA ยี่ห้ออื่นจะได้ผลแบบเดียวกันไหม?
- ข้อจำกัดที่เจอ (เช่น attention map ไม่ได้) เป็นข้อจำกัดของ FPGA โดยทั่วไป หรือเฉพาะ DPU ตัวนี้?

**คำถามเหล่านี้ตอบได้ก็ต่อเมื่อมีแพลตฟอร์มที่สองมาเทียบ**

### 1.2 Contribution ที่เพิ่มขึ้น

> **Cross-platform comparison:** ระบบนับ YOLO **ตัวเดียวกัน** dataset **เดียวกัน** ground truth **เดียวกัน**
> บน **3 แพลตฟอร์ม** ที่มีสถาปัตยกรรมต่างกันสิ้นเชิง

```
┌─────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
│  GPU            │   │  KV260              │   │  Ti375               │
│  GTX 1660 Ti    │   │  hard DPU (B4096)   │   │  soft RISC-V +       │
│  FP32           │   │  INT8               │   │  TinyML accel · INT8 │
│                 │   │                     │   │                      │
│ วงจรตายตัว      │   │ วงจรสำเร็จรูปที่    │   │ CPU ธรรมดา + วงจร    │
│ ทั่วไป          │   │ ออกแบบมาเพื่อ AI    │   │ ช่วยที่เรากำหนดเอง   │
└─────────────────┘   └─────────────────────┘   └──────────────────────┘
        │                       │                          │
        └───────────────────────┴──────────────────────────┘
                                ▼
        เทียบ: FPS / W / mAP / counting error / resource utilization / effort
```

**สิ่งที่ทำให้ repo นี้ไม่ใช่งานซ้ำ:** จุดขายไม่ใช่ "ทำ YOLO บนบอร์ดอีกตัว"
แต่คือ **"เปรียบเทียบสถาปัตยกรรม"** ซึ่งต้องมีอย่างน้อย 2 จุดถึงจะพูดได้

### 1.3 สิ่งที่ reuse ได้ (จึงไม่ใช่การเริ่มจากศูนย์)

| reuse ได้ | reuse ไม่ได้ |
|---|---|
| ✅ dataset (`02-dataset`) | ❌ toolchain (Vitis AI ↔ Efinity คนละโลก) |
| ✅ counting ground truth (GT = 215) | ❌ รูปแบบโมเดล (.xmodel ↔ .tflite) |
| ✅ GPU baseline | ❌ วิธี quantize |
| ✅ logic การ decode + counting (แนวคิด) | ❌ runtime (VART ↔ TFLite Micro) |
| ✅ ประสบการณ์เรื่อง quantization | ❌ วิธี deploy |

---

## 2. Flow ต่างกันคนละโลก

### 2.1 ตารางเปรียบเทียบเต็ม

| | **KV260 (repo เดิม)** | **Ti375 (repo ใหม่)** |
|---|---|---|
| เจ้าของ | AMD/Xilinx | Efinix |
| สถาปัตยกรรม | Zynq UltraScale+ MPSoC (Arm hard PS + PL) | Titanium Ti375 (hardened quad-core RISC-V + fabric) |
| Toolchain | **Vitis AI 3.0** (Docker) | **Efinity IDE** + RISC-V Embedded SW IDE (RiscFree) |
| ตัวเร่ง AI | **DPUCZDX8G** (hard-IP-style overlay, B4096) | **Sapphire RISC-V SoC + Efinix TinyML Accelerator** (Lite/Standard) |
| รูปแบบโมเดล | PyTorch → **`.xmodel`** | TensorFlow → **`.tflite` INT8** |
| Quantize | `vai_q_pytorch` (PTQ) | **TFLite Converter** (PTQ) |
| Compile / gen | `vai_c_xir` → gate `DPU subgraph number 1` | **Efinix TinyML Generator** (GUI) → model data (C arrays) |
| Runtime | **VART C++** บน Arm Cortex-A53 | **TFLite Micro (C++)** บน Sapphire RISC-V |
| Deploy artifact | `.xmodel` + overlay bitstream | **combined hex** (bitstream + RISC-V binary) |
| แหล่งอ้างอิง | Vitis AI Model Zoo | `github.com/Efinix-Inc/tinyml` |

### 2.2 ความต่างเชิงแนวคิดที่สำคัญที่สุด

```
KV260 — hard DPU:
   วงจรตัวเร่ง AI ถูกออกแบบมาแล้วโดย AMD
   เราแค่ "เอาโมเดลไปใส่ให้พอดีกับวงจรที่มี"
   → ข้อดี: เร็ว ไม่ต้องออกแบบ
   → ข้อเสีย: ถ้าโมเดลมี op ที่วงจรไม่รองรับ = จบ (กรณี YOLO26)

Ti375 — soft accelerator:
   วงจรตัวเร่งถูก generate ขึ้นมาตาม config ที่เราเลือก (Lite / Standard)
   → ข้อดี: ปรับได้ตามงาน · ทำ custom instruction เพิ่มได้
   → ข้อเสีย: ช้ากว่า hard IP · ต้องพิสูจน์ว่ารับโมเดลใหญ่ไหว
```

> **นี่คือ trade-off เชิงสถาปัตยกรรมที่การเปรียบเทียบจะทำให้เห็นชัดเป็นตัวเลข**

### 2.3 Gate ที่ตั้งไว้สำหรับฝั่ง Efinix

เทียบเคียงกับ `DPU subgraph number 1` ของ KV260:

> **"โมเดลรันได้จริงบน Sapphire + TinyML accelerator แล้ววัด latency/accuracy ได้"**
> — ไม่ใช่แค่แปลง `.tflite` ผ่าน

(ยึดหลักการเดียวกับไฟล์ 02 ข้อ 4: gate ต้องเป็นสิ่งที่เกิดขึ้นจริงบนฮาร์ดแวร์ ไม่ใช่การวิเคราะห์)

---

## 3. ความคืบหน้า: E3-A เสร็จแล้ว ✅

### 3.1 สิ่งที่ได้

| รายการ | ค่า |
|---|---|
| ไฟล์โมเดล | `yolov8n_leaky_pkg_ft_full_integer_quant.tflite` |
| ขนาด | **3.1 MB** |
| Precision | INT8 **ทั้ง input และ output** |
| ต้นทาง | `yolov8n_leaky_pkg_ft.pt` (ตัวเดียวกับที่ deploy บน KV260) |

### 3.2 ผลการ verify ⭐

**วิธี verify:** รัน INT8 กับ FLOAT32 บนภาพชุดเดียวกัน แล้วเทียบผลหลัง decode

| ตัวชี้วัด | ผล |
|---|---|
| **parity กล่อง @ conf 0.25** | **59/59 กล่อง ตรงกันทุกกล่อง** |
| detection เทียบ GT | 39 vs 44 |
| score ที่ได้ | 0.96–0.98 |

> **parity 59/59 = quantization ซื่อสัตย์สมบูรณ์**
> ไม่มีกล่องไหนหายหรือเพิ่มขึ้นมาจากการ quantize
>
> เทียบกับฝั่ง KV260 ที่วัดเป็น mAP drop 1.5% — วิธีวัดต่างกันแต่ให้ข้อสรุปเดียวกัน
> คือ **INT8 quantization ไม่ใช่ปัญหาสำหรับโมเดลนี้**

### 3.3 Environment ที่ใช้

ต่อยอดจาก `D:\yolo26-export-env` (py3.9) โดยลงเพิ่ม:
```
tensorflow==2.18.0  ·  onnx2tf==1.20.0  ·  tf_keras
```

**ปัญหาความเข้ากันได้ที่ต้องแก้:**
| ปัญหา | วิธีแก้ |
|---|---|
| py3.9 ติดเพดาน `onnx2tf ≤ 1.20` | ยอมใช้เวอร์ชันนั้น |
| `ai-edge-litert` ไม่มี wheel สำหรับ py3.9 | ไม่ใช้ |
| conflict: ml_dtypes 0.4.1 ของ TF ไม่มี `float4_e2m1fn` | downgrade `onnx → 1.17.0` |
| ไดรฟ์ C: เต็ม | ชี้ temp/cache ไปไดรฟ์ D: ทั้งหมด |

---

## 4. บั๊กสำคัญที่เจอและแก้ (บทเรียนเรื่อง quantization ที่ดีมาก)

### 4.1 บั๊กที่ 1 — calibration data ผิด scale ⭐

**อาการ:**
```
quantize ผ่านไม่มี error
      ↓
แต่ผลลัพธ์เป็นขยะ: score ระเบิด, ได้ 300+ กล่องมั่ว
```

**การวินิจฉัย:** ไปดูค่า scale ที่ TFLite Converter คำนวณออกมา
```
input scale = 1.0    ← ผิดปกติชัดเจน
```

**สาเหตุ:**
```
calibration data (.npy) อยู่ในช่วง [0, 255]   (ค่า pixel ดิบ)
แต่ SavedModel คาดหวังช่วง [0, 1]             (normalize แล้ว)
      ↓
TFLite Converter เห็นข้อมูลช่วง 0-255 → คำนวณ scale ให้ครอบคลุมช่วงนั้น
      ↓
แต่ตอนใช้งานจริง input เป็น 0-1 → ใช้แค่ 1/255 ของช่วงที่เตรียมไว้
      ↓
ความละเอียดที่ใช้ได้จริงเหลือแค่ ~1 ระดับ = ข้อมูลหายเกือบหมด
```

**วิธีแก้:** normalize `/255` ใน representative dataset

**ผลหลังแก้:**
```
input scale = 0.00392 = 1/255  ← ตรงพอดี
zero-point  = −128             ← ตรงพอดี
```

> **สังเกตว่าค่าที่ได้เป็นเลข "สวย" พอดี** (1/255 และ −128)
> นี่เป็นสัญญาณว่า quantization ถูกต้องโดยโครงสร้าง ไม่ใช่บังเอิญ

**บทเรียน ⭐:**
> ### representative dataset ที่ผิด scale ทำให้ quantization พังทั้งหมด
> ### **โดยที่ไม่มี error ใดๆ แจ้งเตือน**
> ### → ต้องตรวจค่า scale/zero-point ที่ได้ออกมาเสมอ

*(สังเกตความคล้ายกับปัญหา calibration set ผิดโดเมนในไฟล์ 05 ข้อ 1.3 — เป็นตระกูลปัญหาเดียวกัน
คือ **"ข้อมูลที่ใช้ calibrate ไม่ตรงกับข้อมูลที่จะเจอจริง"**)*

### 4.2 บั๊กที่ 2 — OOM ตอน export INT8

**อาการ:** ตัว export INT8 ของ ultralytics ตายเพราะ RAM ไม่พอ

**สาเหตุ:** onnx2tf โหลด calibration 57 ภาพเป็น array เดียวขนาด **267 MB**
บนเครื่องที่มี RAM 5.9 GB (และเหลือว่างน้อย)

**วิธีแก้:** เขียน `quantize_tflite_int8.py` เอง
```python
# แทนที่จะโหลดทั้งชุดเข้า RAM ทีเดียว
# → ป้อนทีละภาพผ่าน mmap
def representative_dataset():
    for path in image_paths:
        img = load_via_mmap(path)      # อ่านจากดิสก์ทีละภาพ
        yield [normalize(img) / 255.0]  # ← พร้อมแก้บั๊กที่ 1 ไปด้วย
```

---

## 5. ความแตกต่างสำคัญจาก KV260 ที่ค้นพบ

### 5.1 สเปค I/O ของโมเดล TFLite

| รายการ | ค่า |
|---|---|
| **Input** | `[1, 640, 640, 3]` int8 · scale = 1/255 · zp = −128 |
| **Preprocess ที่ถูกต้อง** | **`pixel − 128`** (ง่ายกว่า KV260 มาก) |
| **Output** | `[1, 5, 8400]` int8 · scale = 0.003985 · zp = −128 |
| output channel 0–3 | xywh (normalized → คูณ 640) |
| output channel 4 | score (ผ่าน sigmoid แล้ว) |

### 5.2 ความต่างที่ประหยัดงานไปมาก

```
KV260:                              Ti375:
output = 3 หัว × [H,W,65]           output = [1, 5, 8400]
         ↓                                   ↓
ต้อง DFL decode เอง                 DFL ถูก fold เข้าในกราฟแล้ว
(softmax 16 bin × 4 ด้าน            → ได้ xywh มาเลย
 + anchor + stride)                 → ไม่ต้อง decode เอง!
         ↓                                   ↓
16.1 ms ต่อเฟรม                     งาน postprocess เบากว่ามาก
```

> **นี่คือตัวอย่างที่ดีของสิ่งที่ cross-platform comparison จะเผยให้เห็น**
> — ความต่างของ toolchain ทำให้ **การแบ่งงานระหว่างฮาร์ดแวร์กับซอฟต์แวร์ต่างกัน**
> ซึ่งจะสะท้อนออกมาในตัวเลข latency breakdown ตอนวัดผล

### 5.3 preprocess ที่ถูกต้อง (จุดที่พลาดง่าย)

```
ขั้นที่ 1: หาร 255  →  ได้ช่วง [0, 1]      ← ห้ามลืม! ลืม = ได้ขยะ
ขั้นที่ 2: quantize  →  pixel − 128         ← เพราะ scale = 1/255, zp = −128
```

จดไว้ใน `TFLITE_IO_SPEC.md` เพื่อไม่ให้พลาดตอนเขียนโค้ดฝั่ง RISC-V

---

## 6. โค้ดฝั่ง RISC-V (เขียนเสร็จแล้ว รอ compile)

### 6.1 ไฟล์ที่เขียน

| ไฟล์ | หน้าที่ |
|---|---|
| `postprocess.c` / `.h` | dequantize + แปลง xywh → xyxy + NMS |
| `counting.c` / `.h` | centroid tracker + line-crossing (**ใช้พารามิเตอร์เดียวกับที่ calibrate ไว้บน KV260**) |
| `main.cc` | TFLite Micro glue — มี marker `<<<EFINIX TODO>>>` ตรงจุดที่ต้องเติมหลังได้ model data |

### 6.2 ทำไมยังไม่ compile

**Efinity IDE และบอร์ดอยู่คนละเครื่องกับเครื่องที่ทำงานอยู่นี้**
→ เครื่องนี้ทำหน้าที่ **เตรียมของ** เครื่องโน้นทำหน้าที่ **รัน**

### 6.3 คู่มือที่เขียนไว้: `RUN_ON_EFINITY_MACHINE.md`

เป็น turnkey run guide ที่ครอบคลุม:

```
STEP A  →  Efinity: สร้าง Sapphire SoC + TinyML Accelerator (IP Manager, GUI)
STEP B  →  TinyML Generator: แปลง .tflite → model data (GUI)
STEP C  →  RiscFree: build RISC-V app
STEP D  →  รวม combined hex + program บอร์ด
STEP E  →  รัน + ผลที่ควรเห็น
   +    →  Troubleshooting + แผนสำรอง
```

> **เหตุผลที่ต้องเขียนละเอียดขนาดนี้:** เพราะคนที่รันอาจไม่ใช่คนที่เตรียม
> และการทำงานข้ามเครื่องทำให้ debug ยาก → ต้องลดโอกาสผิดพลาดให้มากที่สุดตั้งแต่เอกสาร

---

## 7. ความเสี่ยงที่ยอมรับไว้แล้ว

### 7.1 ความเสี่ยงหลัก

> **Efinix TinyML Accelerator เดิมออกแบบมาสำหรับโมเดลเล็ก (TFLite Micro)**
> **→ ยังไม่ยืนยันว่า YOLOv8n เต็มตัวรันไหว**

จุดตัดสินอยู่ที่ **E2** — ประเมิน resource fit (LE / DSP / memory ของ Ti375) + เลือกโหมด Lite vs Standard

### 7.2 ข่าวดีที่ทำให้ความเสี่ยงลดลง

Efinix มี demo ชื่อ **"Yolo Person Detection"** บน Ti375 อยู่แล้วใน repo ทางการ
→ **พิสูจน์ว่า YOLO บนแพลตฟอร์มนี้ทำได้ในหลักการ** ไม่ใช่เรื่องที่เป็นไปไม่ได้

### 7.3 แผนสำรองถ้ารันเต็มตัวไม่ไหว

| แผนสำรอง | รายละเอียด |
|---|---|
| (a) เริ่มจาก demo | ใช้ "Yolo Person Detection" ของ Efinix เป็นฐาน แล้ว retrain เป็น `package` |
| (b) ลดขนาด input | ลด resolution จาก 640 → เล็กลง |
| (c) ตัด head | ลดจำนวน detection head |

### 7.4 และไม่ว่าทางไหน ผลก็ยังเขียนเข้าเล่มได้

> **"YOLO บน soft RISC-V accelerator ติดตรงไหน รับได้แค่ไหน เทียบกับ hard DPU"**
> เป็นหัวข้อที่มีคุณค่าทางวิชาการเท่ากับการรายงานว่ารันได้
>
> — เป็นหลักการเดียวกับ Track B ในไฟล์ 08: **ผลเชิงลบที่เจาะจง = contribution**

---

## 8. สถานะ milestone ฝั่ง Efinix (E0–E14)

| # | งาน | ต้องมีบอร์ด? | สถานะ |
|---|---|---|---|
| E0 | ยืนยัน platform + toolchain → เขียน platform_selection | ไม่ | ☐ |
| **E1** | **ติดตั้ง Efinity + build Sapphire SoC + รัน TinyML demo** | **ใช่** | ☐ **gate จริงอันแรก** |
| E2 | ตัดสิน Lite vs Standard + ประเมิน resource fit | ไม่ | ☐ **จุดตัดสินใหญ่** |
| **E3-A** | **Track A: YOLOv8n → TFLite INT8** | ไม่ | ✅ **เสร็จ** |
| E3-B | Track B: YOLO26n → TFLite + ตรวจ op | ไม่ | ☐ |
| E4 | TinyML Generator → model data + config accelerator | ไม่ | ☐ |
| E5 | RISC-V app (preprocess → invoke → decode → counting) | ไม่ | 🟡 **โค้ดเขียนเสร็จ รอ compile** |
| E6 | Verify quant fidelity (host emulation) | ไม่ | ☐ |
| E7 | Counting GT — **reuse GT=215 จาก KV260** | ไม่ | ☐ |
| E8 | Single-class package quantize | ไม่ | ☐ |
| E9 | Board bring-up — combined hex → program → first inference | ใช่ | ☐ |
| E10 | Tracking + counting บน RISC-V | ใช่ | ☐ |
| E11 | **Benchmark เทียบ GPU + เทียบ KV260** | ใช่ | ☐ |
| E12 | Profiling หา bottleneck | ใช่ | ☐ |
| E13 | (Optional) custom instruction เพิ่ม | ใช่ | ☐ |
| E14 | รายงาน + สไลด์ (เน้นมุม cross-platform) | — | ☐ |

**เส้นทางที่วางไว้:** E0 → **E1 (gate แรก)** → E2 (ตัดสินใจใหญ่) → E4 → E5 build → E9 bring-up

---

## 9. กฎความปลอดภัยที่ตั้งไว้

> **ห้ามแก้หรือลบของใน repo KV260 เดิมเด็ดขาด — repo นี้แยกโฟลเดอร์ขาดจากกัน**

เหตุผล: repo KV260 มีผลการทดลองที่สมบูรณ์แล้วและเป็นฐานของเล่มวิทยานิพนธ์
การเผลอแก้จะทำให้ผลที่รายงานไปแล้วไม่ตรงกับหลักฐาน

**ตำแหน่ง:**
- repo KV260: `C:\Users\Gunonemore\Desktop\thesis-yolo26-fpga`
- repo Efinix: `C:\Users\Gunonemore\Desktop\thesis-yolo26-fpga_efinix`

---

## 10. สรุป

| หัวข้อ | สถานะ |
|---|---|
| เหตุผลของโปรเจคที่ 2 | cross-platform comparison = contribution ใหม่ที่ไม่ใช่งานซ้ำ |
| flow | คนละโลกกับ KV260 — เขียนใหม่หมด (แต่ reuse dataset/GT/baseline ได้) |
| ความคืบหน้า | **E3-A เสร็จ** — INT8 tflite verify แล้ว parity 59/59 |
| โค้ด RISC-V | เขียนเสร็จ รอ compile ที่เครื่อง Efinity |
| เอกสารส่งต่อ | `RUN_ON_EFINITY_MACHINE.md` — turnkey guide |
| ความเสี่ยงหลัก | YOLO เต็มตัวอาจใหญ่เกินสำหรับ TinyML accelerator → ตัดสินที่ E2 |
| แผนสำรอง | มี 3 ทาง และทุกทางยังเขียนเข้าเล่มได้ |
| งานถัดไป | E0 → E1 (ติดตั้ง Efinity + รัน demo) |

---

## ไฟล์ที่เกี่ยวข้อง
- ไฟล์ก่อนหน้า: [08-track-b-yolo26.md](08-track-b-yolo26.md)
- ไฟล์ถัดไป: [10-engineering-problems.md](10-engineering-problems.md)
- timeline ฝั่ง Efinix: `../../../thesis-yolo26-fpga_efinix/00-admin/timeline.md`
- สเปค I/O: `../../../thesis-yolo26-fpga_efinix/03-model/track-a-yolov8n-baseline/artifacts/TFLITE_IO_SPEC.md`
- คู่มือรัน: `../../../thesis-yolo26-fpga_efinix/RUN_ON_EFINITY_MACHINE.md`
- โค้ด RISC-V: `../../../thesis-yolo26-fpga_efinix/04-deploy/riscv-app/`
