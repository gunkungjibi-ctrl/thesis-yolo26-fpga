# 06 — บอร์ดมาถึง → Bring-up → Deploy → เห็นกล่องจริง

> **สรุปไฟล์นี้ใน 3 บรรทัด**
> วินาทีที่บอร์ดมาถึงและ fingerprint ตรงกับที่ compile ไว้ล่วงหน้า = ความเสี่ยงใหญ่ที่สุดของโครงงานถูกปิด
> ระหว่างทาง quantize เจอ OOM และ regression ในสคริปต์ ต้องแก้ทั้งสองก่อนจะได้ xmodel ตัวจริง
> จบด้วยการเขียน DFL decode เองบน PS แล้วเห็นกล่องพัสดุถูกตรวจจับที่ score 0.92

---

## ลำดับเหตุการณ์ทั้งหมด (15–16 สิงหาคม 2026)

```
บอร์ดมาถึง → flash SD → bring-up → ยืนยัน fingerprint → รัน xmodel เก่า (68 FPS)
      ↓
fine-tune บน Colab เสร็จ → ได้ .pt ตัวใหม่
      ↓
quantize (เจอ OOM → แก้) → compile (เจอ regression → แก้) → ได้ xmodel ตัวใหม่
      ↓
โอนขึ้นบอร์ด → ยืนยัน 65 channel → benchmark 82.5 FPS
      ↓
เขียน detection app + DFL decode → เห็นกล่องจริง score 0.92 🎉
```

---

## ขั้นที่ 1 — Board Bring-up

### 1.1 สิ่งที่ได้มาจริง vs สิ่งที่วางแผนไว้

| ประเด็น | แผนเดิม | ของจริง | ผลกระทบ |
|---|---|---|---|
| OS | Ubuntu for Kria | **PetaLinux starter-kit 2022.2** | ต้องปรับวิธีทำงาน แต่ไม่กระทบผลลัพธ์ |
| ผู้ใช้ | `ubuntu` | `root` / `petalinux` (ไม่มี `ubuntu`) | ต้องเปลี่ยนคำสั่ง |
| home directory | `/home/ubuntu` | `/root` ไม่มีในบางบูต → ใช้ `/tmp` | ต้องเปลี่ยนที่วางไฟล์ |
| DPU | ต้องโหลดเอง | **`kv260-benchmark-b4096` active อยู่แล้ว** | ประหยัดขั้นตอน |
| VART | ต้องติดตั้ง | **มีมาแล้ว v3.0** | ประหยัดขั้นตอน |

> **การตัดสินใจ:** ไม่เปลี่ยนกลับไป Ubuntu เพราะ image ที่ได้มามี DPU + VART ครบและใช้ deploy ได้เลย
> การเปลี่ยน OS จะเสียเวลาโดยไม่ได้อะไรเพิ่ม

### 1.2 การตรวจสอบทีละขั้น

**ขั้น A — ดูว่ามี DPU overlay อะไรโหลดอยู่**
```bash
xmutil listapps
```
ผล: `kv260-benchmark-b4096` **active** → มี DPU B4096 พร้อมใช้งานแล้ว

**ขั้น B — ยืนยันตัวตนของ DPU** ⭐ (จุดตัดสินของโครงงาน)
```bash
xdputil query
```

| สิ่งที่ตรวจ | ผลที่ได้ | ตรงกับที่ compile ไว้? |
|---|---|---|
| arch | `DPUCZDX8G_ISA1_B4096` | ✅ ตรง |
| **fingerprint** | **`0x101000056010407`** | ✅ **ตรงเป๊ะ** |
| VART version | 3.0 | ✅ ตรง |

> ### ทำไมวินาทีนี้สำคัญที่สุดในโครงงาน
>
> เรา compile โมเดลไว้**ก่อนบอร์ดจะมาถึง** โดยอ้างอิงจากเอกสารเท่านั้น
> (และเคยมีปัญหาเอกสารขัดกันเรื่อง B3136 vs B4096 — ดูไฟล์ 03)
>
> ถ้า fingerprint ไม่ตรง → ต้อง re-quantize + re-compile ทั้งหมด = เสียเวลาหลายวันตอนที่มีบอร์ดอยู่ในมือ
>
> **การที่มันตรง แปลว่างานเตรียมล่วงหน้าทั้งหมดใช้ได้จริง** — เป็นผลตอบแทนของการทำ P1 อย่างละเอียด

**ขั้น C — โอนไฟล์ขึ้นบอร์ด (เจอปัญหาแรก)**

```
พยายาม: scp yolov8n_kv260_B4096.xmodel root@192.168.40.53:/tmp/
ผล:     ล้มเหลว — PetaLinux ไม่มี sftp-server ติดตั้งมา
        ลอง scp -O (legacy protocol) ก็ยังไม่ผ่าน เพราะ /root ไม่มี
```

**วิธีแก้ที่ใช้ได้:**
```bash
# ฝั่ง host (Windows, IP 192.168.40.219)
python -m http.server 8000

# ฝั่งบอร์ด
wget http://192.168.40.219:8000/yolov8n_kv260_B4096.xmodel -O /tmp/model.xmodel
md5sum /tmp/model.xmodel    # ← ต้องตรงกับฝั่ง host
```

ผล md5: `69adf8...` **ตรง** ✅

> **กติกาที่ยึด:** verify md5 ทุกครั้งหลังโอน — เพราะไฟล์เพี้ยนระหว่างโอนจะไม่มีอาการ
> จนกว่าจะรันแล้วได้ผลประหลาด ซึ่งตอนนั้นจะ debug ยากมาก

**ขั้น D — ตรวจโครงสร้างโมเดลบนบอร์ด**
```bash
xdputil xmodel /tmp/model.xmodel -l
```

| สิ่งที่ตรวจ | ผล |
|---|---|
| จำนวน DPU subgraph | **1** ✅ (compute 100%) |
| USER subgraph | input |
| CPU subgraph | output dequant (อยู่ปลายกราฟ = ปกติ) |
| output shape | `[40,40,144]` `[20,20,144]` `[80,80,144]` |

144 channel → ยืนยันว่าเป็นโมเดล YOLOv8 80 คลาสของ phase0 (ตามที่วิเคราะห์ไว้ในไฟล์ 03)

**ขั้น E — inference จริงครั้งแรก** 🎉
```bash
xdputil benchmark /tmp/model.xmodel 1
```
ผล: **FPS = 68.1** (single-thread), **Test PASS**

→ **M9 (board bring-up) ผ่านครบทุกเกณฑ์**

### 1.3 วิธีเชื่อมต่อบอร์ด (และเหตุผลด้านความปลอดภัย)

| ช่องทาง | รายละเอียด |
|---|---|
| Network | LAN — บอร์ด `192.168.40.53`, host `192.168.40.219` |
| Serial console | **COM6 @ 115200 8N1** ผ่าน PowerShell |
| SSH key | **ไม่ติดตั้ง** |

> **การตัดสินใจเชิงความปลอดภัย:** ทำงานแบบ **relay** — ผู้ทำโครงงานเป็นคนพิมพ์/ยิงทุกคำสั่งเอง 100%
> ไม่มีการติดตั้ง SSH key ที่ให้เครื่องมืออื่นเข้าถึงบอร์ดได้โดยตรง
> เพราะบอร์ดเป็นฮาร์ดแวร์ชิ้นเดียวที่มี ถ้าพังหรือถูกแก้ไขผิดพลาดจะไม่มีตัวสำรอง

**หมายเหตุทางเทคนิคที่เจอ:** PowerShell serial object (`$p`) ถือ COM6 แบบ exclusive
→ เปิดซ้ำจากหน้าต่างอื่นจะได้ `Access denied` ต้องกลับไปใช้ object เดิม
และการอ่าน output ต้องใช้ `$p.ReadExisting()` แบบ poll หลายรอบ (`ReadLine` timeout ไม่ทัน)

---

## ขั้นที่ 2 — Quantize + Compile โมเดลที่ fine-tune แล้ว (เจอบั๊ก 2 ตัว)

### 2.1 ปัญหา A — OOM ตอน calibration

**อาการ:**
```
container ตายกลางคัน: "unexpected EOF" / exit code 125
```

**การวินิจฉัย:**
```
รัน docker info เพื่อดูสถานะ → คำสั่งนี้เองก็ล้ม!
errno = 1450 = ERROR_NO_SYSTEM_RESOURCES
```
→ ไม่ใช่ปัญหาของสคริปต์ แต่เป็นปัญหาระดับระบบ

**สาเหตุที่แท้จริง:**
```
host RAM 5.88 GB
      ↓
WSL2 จำกัดตัวเองไว้ที่ ~2.9 GB (default = ครึ่งหนึ่งของ RAM)
      ↓
ปกติเมื่อ RAM ไม่พอ ระบบจะใช้ pagefile (swap) ช่วย
      ↓
แต่ไดรฟ์ C: เหลือแค่ 0.5 GB → pagefile ขยายไม่ได้
      ↓
ไม่มีที่ให้ swap → process ถูกฆ่า
```

> **จุดที่น่าสนใจ:** **ดิสก์เต็มทำให้ RAM ไม่พอได้** — เป็นความเชื่อมโยงที่ไม่ชัดในตอนแรก
> คนส่วนใหญ่จะไปแก้ที่ `.wslconfig` เพื่อเพิ่ม memory limit ซึ่งไม่ได้แก้ต้นเหตุ

**วิธีแก้ (2 อย่างพร้อมกัน):**
1. เคลียร์ไดรฟ์ C: จาก 0.2 GB → **17 GB** → pagefile/swap ขยายได้
2. ลด calibration workload: `--subset_len 32 --batch_size 4`

**ผล:** calibration pass 32/32 รอด — **ไม่ต้องแตะ `.wslconfig` เลย**

### 2.2 ปัญหา B — regression ในสคริปต์ quantize

**อาการ:**
```
XIR don't support multi-outputs op
```
เกิดที่ operation: `nndct_chunk` (×8 จาก C2f block) และ `split_with_sizes` (จาก Detect head)

**การวินิจฉัย: เทียบกับสคริปต์ที่เคยผ่าน**

สคริปต์ที่ใช้อยู่ (`track-a/quantize/quantize_yolov8n_pytorch.py`) ขาด fix 2 อย่าง
ที่สคริปต์ phase0 (`quantize_yolo_pytorch.py`) มีอยู่:

| fix ที่หายไป | ทำอะไร | ทำไมจำเป็น |
|---|---|---|
| **monkeypatch `C2f.forward`** | เปลี่ยน `chunk(2, 1)` → slice operation | XIR ไม่รองรับ op ที่ให้ output หลายตัว → ต้องเปลี่ยนเป็น slice ที่ให้ output ทีละตัว |
| **`BackboneHead` wrapper** | เดิน layer เอง แล้ว return raw `cat(cv2, cv3)` และ stride | **ตัดส่วน decode ทิ้ง** → ไม่แตะ `split_with_sizes` เลย |

**ทำไม regression นี้เกิดขึ้น:** เกิดจากการ "จัดระเบียบโค้ด" ตอนแยกโปรเจคเป็น 2 track
สคริปต์ถูกเขียนใหม่ให้สะอาดขึ้น แต่ fix ที่เคยใส่ไว้แบบ ad-hoc หายไป

**วิธีแก้:**
1. **backup ตัวที่มีปัญหา** เป็น `quantize_yolov8n_pytorch.py.regression.bak` (ไม่ลบ — เก็บเป็นหลักฐาน)
2. ก๊อป proven script จาก phase0 มาทับ (TARGET เป็น B4096 อยู่แล้ว)
3. **patch `_quant_compile.sh`** ให้ abort ถ้าไม่มีไฟล์ `_int.xmodel` ออกมา

**ข้อ 3 สำคัญเป็นพิเศษ:** สคริปต์เดิมใช้ `set +e` ซึ่งทำให้มันเดินต่อแม้ขั้นตอนก่อนหน้าจะล้ม
→ สคริปต์รายงานว่า "สำเร็จ" ทั้งที่ไม่มีไฟล์ผลลัพธ์ = **ผ่านหลอกตา**

> **บทเรียน:** สคริปต์ automation ต้องมี **gate ที่ตรวจ artifact จริง** ไม่ใช่แค่ดู exit code
> เป็นกติกาเดียวกับที่ใช้ในไฟล์ 03 (verify หลักฐาน ไม่เชื่อบันทึก)

### 2.3 ผลลัพธ์หลังแก้

```
PASS 1: calibration 32/32 ภาพ            ✅
PASS 2: export BackboneHead_int.xmodel   ✅ (12.3 MB, ไม่มี XIR error)
COMPILE: vai_c_xir                        ✅ DPU subgraph number 1 ← ผ่าน gate
```

| artifact | ค่า |
|---|---|
| ไฟล์ | `artifacts/yolov8n_pkg_B4096/yolov8n_pkg_kv260.xmodel` |
| ขนาด | **3.99 MB** (จาก float 6.30 MB → เล็กลง 1.58×) |
| arch | B4096 |
| kernel name | `subgraph_BackboneHead__BackboneHead_10884` |
| **md5** | **`c702ccb8f027bf9aea95288243d1b274`** |

> **การตรวจสอบที่สำคัญ:** md5 นี้ **ต่างจาก** phase0 (`69adf8...`)
> → ยืนยันว่าเป็นน้ำหนัก LeakyReLU ที่ fine-tune แล้วจริง ไม่ใช่ไฟล์เก่าที่ก๊อปมาผิด

---

## ขั้นที่ 3 — Deploy โมเดลตัวใหม่ขึ้นบอร์ด

### 3.1 การยืนยันตัวตนของโมเดล (ใช้บทเรียนจากไฟล์ 03)

โอนไฟล์ผ่าน HTTP → `md5sum` บนบอร์ด = `c702ccb8f027bf9aea95288243d1b274` **ตรง** ✅

```bash
xdputil xmodel /tmp/yolov8n_pkg_kv260.xmodel -l
```

| สิ่งที่ตรวจ | ผล | ตีความ |
|---|---|---|
| DPU subgraph | **1** | ✅ ผ่าน gate บนฮาร์ดแวร์จริง |
| fingerprint | `0x101000056010407` | ✅ ตรงกับบอร์ด |
| arch | B4096 | ✅ |
| input | `[1, 640, 640, 3]` | ✅ |
| **output** | **3 หัว × 65 channel** (`[80,80,65] [40,40,65] [20,20,65]`) | ⭐ ดูข้างล่าง |
| CPU subgraph idx 2–4 | output dequant | ปกติ (อยู่ปลายกราฟ) |

### 3.2 ทำไมเลข 65 คือหลักฐานสำคัญ

```
65 = 64 + 1
   = (4 ด้าน × 16 bin DFL) + (1 คลาส "package")

เทียบกับ phase0: 144 = (4 × 16) + 80 คลาส COCO
```

> **นี่คือการใช้บทเรียนจากวิกฤตที่ 2 (ไฟล์ 03) มาตรวจสอบตัวเอง**
> แทนที่จะเชื่อว่า "ก็โอนไฟล์ที่ compile มาแล้วนี่" เราตรวจจาก**โครงสร้างจริง**ว่าเป็นโมเดล 1 คลาสจริง
>
> ถ้าเผลอโอนไฟล์เก่าขึ้นไป เลข channel จะเป็น 144 และเราจะจับได้ทันที

### 3.3 Benchmark

```bash
xdputil benchmark /tmp/yolov8n_pkg_kv260.xmodel 1
```

| ผล | ค่า |
|---|---|
| FPS | **82.5** (single-thread) |
| เฟรมที่ประมวลผล | 4,952 เฟรม ใน 60 วินาที |
| สถานะ | **Test PASS** |

**เทียบกับ phase0 (68.1 FPS): เร็วขึ้น 21%**
สาเหตุ: output เบากว่า (65 vs 144 channel) → เขียนข้อมูลออกจาก DPU น้อยลง

> เป็นหลักฐานเชิงตัวเลขว่า **การลดจำนวนคลาสให้ตรงกับงานจริง มีผลต่อความเร็วจริง**
> ไม่ใช่แค่ทำให้โมเดลเล็กลงบนกระดาษ

---

## ขั้นที่ 4 — เขียน Detection App และ DFL Decode เอง

### 4.1 ทำไมต้องเขียน decode เอง

จำได้ว่าตอน quantize เราใช้ `BackboneHead` wrapper ที่**ตัดส่วน decode ทิ้ง** (ข้อ 2.2)
→ สิ่งที่ DPU คืนมาคือ **ตัวเลขดิบ** ไม่ใช่พิกัดกล่อง

โค้ด C++ ที่เตรียมไว้ล่วงหน้า (`yolo_dpu_infer.cpp`) ทำได้แค่:
- รัน inference
- dequantize

ส่วน decode เป็น `// TODO` → ต้องเขียนใหม่ทั้งหมด

**การตัดสินใจ:** เขียนเป็น **Python ก่อน** (`04-deploy/board/yolo_dpu_detect.py`)
เพื่อพิสูจน์ความถูกต้องของ decode logic ให้จบก่อน แล้วค่อย port เป็น C++ ทีหลัง

### 4.2 Pipeline ที่เขียน

```python
# 1. PREPROCESS
img = resize(img, 640, 640)          # plain resize ไม่ letterbox
img = BGR_to_RGB(img)
img = img / 255.0                     # normalize เป็น 0..1
img_int8 = round(img * 64)            # quantize: 2^in_fixpos = 2^6 = 64
                                      # → NHWC int8

# 2. DPU
outputs = dpu.execute(img_int8)       # ได้ 3 หัว, int8

# 3. DEQUANTIZE
for each head:
    out_float = out_int8 * (2 ** -fixpos)

# 4. DFL DECODE  ← ส่วนที่ต้องเขียนเอง (ดูข้อ 4.3)
# 5. SIGMOID ที่ channel class
# 6. NMS
```

### 4.3 DFL Decode อธิบายละเอียด

**DFL (Distribution Focal Loss) คืออะไร**

YOLO แบบเก่าทำนายระยะจากจุดกึ่งกลางไปขอบกล่องเป็น**ตัวเลขเดียว** (เช่น "ห่างไป 3.7 หน่วย")
YOLOv8 เปลี่ยนมาทำนายเป็น**การแจกแจงความน่าจะเป็นบน 16 ค่า** แทน

```
แทนที่จะบอกว่า:  ระยะ = 3.7

มันบอกว่า:  P(ระยะ=0)=0.01  P(ระยะ=1)=0.02  P(ระยะ=2)=0.05
            P(ระยะ=3)=0.35  P(ระยะ=4)=0.42  P(ระยะ=5)=0.10  ...  (16 ค่า)

แล้วเราคำนวณค่าคาดหวัง: Σ (i × P(i)) = 0×0.01 + 1×0.02 + ... = 3.7
```

**ข้อดี:** โมเดลแสดง "ความไม่แน่ใจ" ได้ → เทรนได้ดีกว่า
**ข้อเสีย:** ต้องมี 16 channel ต่อ 1 ด้าน × 4 ด้าน = **64 channel** สำหรับกล่องอย่างเดียว

**ขั้นตอน decode ที่เขียน**

```
สำหรับแต่ละตำแหน่ง (cell) บน feature map:

ขั้น 1: แยก channel
   ch[0:64]  = box (4 ด้าน × 16 bin)
   ch[64]    = class score (package)

ขั้น 2: สำหรับแต่ละด้าน (left, top, right, bottom):
   นำ 16 ค่ามา softmax  →  ได้ความน่าจะเป็นที่รวมกันได้ 1
   คูณกับ arange(16) = [0,1,2,...,15] แล้วรวม  →  ได้ระยะ d

ขั้น 3: หาจุดยึด (anchor)
   anchor = ตำแหน่ง cell + 0.5   (จุดกึ่งกลางของ cell)

ขั้น 4: แปลงเป็นพิกัดกล่อง
   x1 = (anchor_x - d_left)   × stride
   y1 = (anchor_y - d_top)    × stride
   x2 = (anchor_x + d_right)  × stride
   y2 = (anchor_y + d_bottom) × stride

ขั้น 5: คะแนน
   score = sigmoid(ch[64])
```

**ตาราง stride ที่ต้องใช้**

| หัว output | stride | ที่มา | ตรวจจับวัตถุขนาด |
|---|---|---|---|
| 80×80 | **8** | 640 ÷ 80 = 8 | เล็ก |
| 40×40 | **16** | 640 ÷ 40 = 16 | กลาง |
| 20×20 | **32** | 640 ÷ 20 = 32 | ใหญ่ |

**จุดที่พลาดง่ายที่สุด: การ scale กลับไปภาพต้นฉบับ**

```
เพราะใช้ plain resize (ไม่ใช่ letterbox) → สัดส่วนภาพถูกบิด
→ ต้อง scale กลับ แยกทีละแกน:

   x_original = x_640 × (origW / 640)
   y_original = y_640 × (origH / 640)

ถ้าใช้ตัวคูณเดียวกันทั้งสองแกน → กล่องจะเพี้ยน
(ภาพต้นฉบับ 640×360 → อัตราส่วนต่างกันมาก)
```

### 4.4 ผลการรันจริง

**การตรวจสอบระหว่างทาง:**

| สิ่งที่ตรวจ | ผล | ตีความ |
|---|---|---|
| input fixpos | 6 → scale = 64 | ✅ ตรงกับที่ใช้ใน preprocess |
| output float range ทั้ง 3 หัว | ปกติ **ไม่ saturate** | ✅ ยืนยันว่า fine-tune แก้ปัญหา logit ระเบิดได้จริง (เทียบไฟล์ 05 ข้อ 3) |
| จำนวนกล่องหลัง NMS | **4 กล่อง** | ✅ สมเหตุสมผลกับภาพทดสอบ |

**คะแนนที่ได้: 0.92 / 0.82 / 0.78 / 0.50**

> **กล่องกระดาษบนสายพานถูกจับที่ score 0.92 ตรงตำแหน่งเป๊ะ**
> = โมเดล package ทำ detection บน DPU เห็นกล่องจริงเป็นครั้งแรก 🎉

📷 รูปผล: `08-figures/detect_out_pkg_kv260.jpg`

**การดึงรูปกลับมาดู:** เปิด `python3 -m http.server 8001` บนบอร์ด แล้วใช้ `Invoke-WebRequest` จาก host ดึงมา
(ทิศทางกลับกันกับตอนโอนขึ้นไป)

### 4.5 ข้อมูล decode ที่ยืนยันแล้ว (สำหรับเขียนเข้าเล่ม)

| รายการ | ค่าที่ยืนยันแล้ว |
|---|---|
| การจัดเรียง channel | `[0:64]` = box DFL (4×16) · `[64]` = class |
| stride map | 80×80→8 · 40×40→16 · 20×20→32 |
| input quantization | `2^in_fixpos` โดย fixpos = 6 → คูณ 64 |
| output dequantization | `2^-fixpos` ต่อหัว |
| anchor | cell index + 0.5 |
| การ scale กลับ | `(origW/640, origH/640)` **แยกแกน** |
| การ resize | plain resize ไม่ letterbox |

---

## ขั้นที่ 5 — สรุปสิ่งที่พิสูจน์ได้จากองก์นี้

| สิ่งที่พิสูจน์ | หลักฐาน |
|---|---|
| งานเตรียมล่วงหน้าใช้ได้จริง | fingerprint ตรงเป๊ะตั้งแต่ครั้งแรก |
| โมเดล fine-tuned ผ่าน gate บนฮาร์ดแวร์จริง | 1 DPU subgraph บนบอร์ด + Test PASS |
| เป็นโมเดล 1 คลาสจริง | output 65 channel |
| การลดคลาสมีผลต่อความเร็วจริง | 68 → 82.5 FPS (+21%) |
| fine-tune แก้ปัญหา logit ระเบิดได้ | output float range ปกติ ไม่ saturate |
| decode ที่เขียนเองถูกต้อง | เห็นกล่องตรงตำแหน่ง score 0.92 |

---

## ไฟล์ที่เกี่ยวข้อง
- ไฟล์ก่อนหน้า: [05-dataset-and-finetune.md](05-dataset-and-finetune.md)
- ไฟล์ถัดไป: [07-results.md](07-results.md) — ผลการทดลองทั้งหมด
- โค้ด detection: `04-deploy/board/yolo_dpu_detect.py`
- โค้ด C++ (raw inference): `04-deploy/board/yolo_dpu_infer.cpp`
- artifact: `03-model/track-a-yolov8n-baseline/artifacts/yolov8n_pkg_B4096/`
- รูปผล: `08-figures/detect_out_pkg_kv260.jpg`
