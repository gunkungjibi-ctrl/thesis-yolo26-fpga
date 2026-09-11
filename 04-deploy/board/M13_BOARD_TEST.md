# M13 — Runbook เทสบนบอร์ด KV260

เป้าหมาย: เก็บผล M13 ให้ได้**มากที่สุดเท่าที่ทำได้วันนี้** โดยยังไม่ต้องมี Vitis / bitstream

| Stage | ต้องมีอะไร | ได้อะไร | เวลาโดยประมาณ |
|---|---|---|---|
| **A** | บอร์ด + Python/cv2 — **ไม่ต้องมีไฟล์สื่อ** | ✅ พิสูจน์ว่าสูตร HW ตรงกับ **cv2 บน ARM NEON** จริง (ก่อนลงทุน synth) | ~5 นาที |
| **B** | บอร์ด — **ไม่ต้องมีไฟล์สื่อ** | ✅ ตัวเลข preproc จริงบน A53: `numpy` vs `lut` | ~2 นาที |
| **C** | บอร์ด + DPU + `.xmodel` + คลิป (ต้องโอนจาก PC) | ✅ e2e FPS ใหม่ + ยืนยัน **COUNT ยัง 215** | ~30 นาที |
| **D** | เครื่องที่มี Vitis 2022.2 | ⏳ bitstream + โหมด `hw` | เป็นวัน |

**Stage A + B รันจากมือถือได้ล้วนๆ** (SSH + `wget` จาก GitHub — ดูข้อ 0B) ไม่ต้องมี PC เลย

---

## ⚡ ทางลัด — คำสั่งเดียวจบ Stage A + B

ถ้าไม่อยากไล่ทีละขั้น (โดยเฉพาะตอนพิมพ์บนมือถือ) `m13_quickstart.sh` ทำให้ครบตั้งแต่
เช็ค env → เช็คเน็ต → โหลดสคริปต์ → รัน Stage A0 + B0 → สรุปผล

```sh
wget -O m13.sh https://raw.githubusercontent.com/gunkungjibi-ctrl/thesis-yolo26-fpga/claude/accelerator-test-apuc1z/04-deploy/board/m13_quickstart.sh
sh m13.sh
```

แล้วก๊อป output ทั้งหมดส่งกลับมา · ถ้ามันฟ้องตรงไหน สคริปต์จะบอกคำสั่งแก้ให้เอง
(เขียนด้วย POSIX sh ล้วน — busybox ash ของ PetaLinux รันได้ · ไม่ต้องมีไฟล์สื่อบนบอร์ด)

> **Stage A คือด่านที่สำคัญที่สุด** — ที่ verify ไว้ตอนออกแบบเป็น OpenCV บน **x86 SIMD**
> ถ้า OpenCV ของบอร์ด (ARM/NEON, คนละเวอร์ชัน) ให้ค่าต่างแม้ 1 LSB สเปกของ kernel ต้องแก้
> **ก่อน** จะเสียเวลา synth ทั้งวัน → เทสนี้ราคาถูกมากแต่กันงานเสียเปล่าได้เยอะ

---

## 0. เอาสคริปต์ขึ้นบอร์ด — เลือกทางใดทางหนึ่ง

### 0A — มี PC ที่มี repo (ทางปกติ)

PetaLinux starter kit ไม่มี `sftp-server` → `scp` ใช้ไม่ได้ ใช้ HTTP เหมือนเดิม (M9)

```bash
# ---- ที่ host (เครื่องที่มี repo) ----
cd 04-deploy/board
./make_board_bundle.sh              # -> m13_board.tar.gz
python3 -m http.server 8000         # ปล่อยค้างไว้
ip addr | grep 'inet '              # จด IP (คราวก่อนคือ 192.168.40.219)
```

```bash
# ---- บนบอร์ด ----
cd /tmp && wget http://<HOST_IP>:8000/m13_board.tar.gz && tar xzf m13_board.tar.gz && cd m13
```

### 0B — ไม่มี PC (คุมบอร์ดจากมือถือ) — `wget` จาก GitHub ตรงๆ ⭐

repo เป็น **public** → บอร์ดโหลดเองได้ ไม่ต้องมีเครื่องกลาง ไม่ต้องใช้ token
**paste ก้อนนี้ทั้งก้อนลง SSH ได้เลย** (ต้องการแค่บอร์ดออกอินเทอร์เน็ตได้)

```sh
mkdir -p /tmp/m13 && cd /tmp/m13
B=claude/accelerator-test-apuc1z
R=https://raw.githubusercontent.com/gunkungjibi-ctrl/thesis-yolo26-fpga/$B
for f in 04-deploy/pl-preproc/host/preproc_tables.py \
         04-deploy/pl-preproc/golden/preproc_golden.py \
         04-deploy/board/preproc_lib.py \
         04-deploy/board/bench_preproc.py \
         04-deploy/board/bench_latency.py \
         04-deploy/board/video_detect.py ; do
  wget -q "$R/$f" -O "$(basename $f)" || echo "FAIL $f"
done
ls -l
```

> วางแบนแบบนี้ได้เพราะ `preproc_golden.py` หา `preproc_tables.py` ในโฟลเดอร์ตัวเองด้วย
> ถ้า `wget` ฟ้อง certificate (CA bundle ของ image เก่า) เติม `--no-check-certificate`
> — ยอมรับได้เพราะเป็น repo สาธารณะของเราเองบน LAN แต่อย่าใช้เป็นนิสัยกับ URL อื่น
> ถ้าบอร์ดออกเน็ตไม่ได้ (`wget -q --spider https://github.com` ไม่ผ่าน) → ต้องใช้ 0A

### ไฟล์สื่อ/โมเดล (ต้องมีเฉพาะ Stage C)

`.xmodel` และ `.mp4` **ไม่ได้อยู่ใน git** (ติด `.gitignore`) และ `/tmp` หายทุกครั้งที่รีบูต
→ Stage C ต้องโอนจาก PC เท่านั้น ส่วน **Stage A/B รันได้โดยไม่ต้องมีไฟล์พวกนี้เลย**

- `yolov8n_pkg_kv260.xmodel` · `clip_600s_gt215.mp4` (30 MB) · `test.jpg`

### เช็กสภาพแวดล้อมก่อนเริ่ม

```bash
python3 -c "import numpy, cv2; print('numpy', numpy.__version__, '| cv2', cv2.__version__)"
uname -m        # ต้องได้ aarch64
```

---

## Stage A — golden model vs cv2 ของบอร์ด ⭐

รันโมเดลซอฟต์แวร์ที่ **จำลอง kernel ทีละบิต** แล้วเทียบกับ `preprocess()` ตัวจริงของแอป

```bash
cd /tmp/m13

# A0: ⭐ ไม่ต้องมีไฟล์อะไรเลย (สร้างเฟรมเอง) — ทำข้อนี้ก่อนเสมอ
python3 preproc_golden.py check --sizes 640x360,640x640,1280x720,1920x1080 --n 2

# A1: รูปจริงจาก valid set (ถ้าโอนขึ้นมา) หรือรูปเทสรูปเดียวก็ได้
python3 preproc_golden.py sweep test.jpg

# A2: เฟรมจริงจากคลิปที่ใช้นับ — ขนาด 640x360 = เคสที่ deploy จริง (สำคัญสุด)
python3 preproc_golden.py sweep clip_600s_gt215.mp4 --n 30 --stride 200

# A3: จำลองกล้อง 1080p (upscale/downscale คนละอัตราส่วน)
python3 preproc_golden.py sweep clip_600s_gt215.mp4 --n 10 --stride 500 --src 1920x1080
```

**ผลที่ต้องได้** — A0 ต้องปิดท้ายด้วย `worst ... = 0` และคอลัมน์ `simd int8 (deploy)` ต้องเป็น `0/1228800 max=0` ทุกแถว (คอลัมน์ `scalar u8` มิสเยอะเป็นเรื่องปกติ — เป็นสูตรที่ *ไม่ได้* ใช้) · A1–A3 ต้องได้:
```
[sweep] 30 frames, src 640x360, cv2 4.x.x (aarch64)
[sweep] int8 mismatches 0 / 36864000 (0.000000%), max|diff|=0  -> PASS
```

| ผล | แปลว่า | ทำต่อ |
|---|---|---|
| `PASS` ทุกข้อ | สูตร fixed-point ที่ reverse ไว้ตรงกับ OpenCV ของบอร์ดด้วย → สเปก kernel ใช้ได้จริง | ไป Stage B |
| `FAIL` max=1, มิสไม่กี่ % | cv2 บนบอร์ดใช้ path ต่างกัน (เวอร์ชัน/NEON) | **หยุด** ส่ง output มาให้ผม — ผมปรับสูตรใน `preproc_tables.py`/kernel ให้ตรง แล้วเทสใหม่ (ไม่ต้อง synth ใหม่ เพราะยังไม่ได้ synth) |
| `FAIL` max ใหญ่ / มิสเยอะ | คนละ path เลย (อาจใช้ INTER_LINEAR_EXACT) | เหมือนบน — ส่ง log มา |

---

## Stage B — วัด preprocessing จริงบน Cortex-A53

ไม่ต้องใช้ DPU เลย

```bash
cd /tmp/m13

# B0: ⭐ ไม่ต้องมีไฟล์อะไรเลย — ขนาด 640x640 เท่ากับที่ M12 วัดไว้
python3 bench_preproc.py --synth 640x640 --modes numpy,lut --verify --iters 50

# B1: ขนาดเดียวกับที่ M12 วัดไว้ (รูป valid set 640x640) — เทียบกับ 49.7 ms เดิมได้ตรงๆ
python3 bench_preproc.py test.jpg --modes numpy,lut --verify --iters 50

# B2: ขนาดที่ deploy จริง (คลิป 640x360)
python3 bench_preproc.py clip_600s_gt215.mp4 --modes numpy,lut --verify --iters 50

# B3: กล้อง 1080p
python3 bench_preproc.py clip_600s_gt215.mp4 --src 1920x1080 --modes numpy,lut --verify --iters 30
```

**ผลที่ต้องได้:**
```
[bench_preproc] frame 640x640  fixpos=6  iters=50  cv2=4.x.x  arch=aarch64
  numpy    mean=  49.xxx  median=  49.xxx ...
  lut      mean=   x.xxx  median=   x.xxx ...
           verify vs numpy: mismatch=0/1228800  max|diff|=0  -> OK
[speedup] lut vs numpy: N.Nx  (49.xx -> x.xx ms)
```

สิ่งที่ต้องดู:
1. `numpy` ต้องได้ **~49.7 ms** → ยืนยันว่า reproduce ตัวเลข M12 ได้ (ถ้าต่างมาก แปลว่ารูปเทสคนละขนาดกับรอบก่อน)
2. `verify ... -> OK` ทุกข้อ → `lut` ให้ผลตรงกับโค้ดเดิมทุกไบต์ ⇒ **ใช้แทนได้โดยไม่ต้องวัด mAP ใหม่**
3. `speedup` = gain ที่ได้ฟรีโดยไม่ต้องแตะ PL (บน x86 ได้ 8.7× — บน A53 คาด 5–10×)

> ถ้า `verify` FAIL แม้แต่ข้อเดียว **อย่าใช้โหมด `lut` ต่อ** ส่ง log มา (แปลว่า `cv2.LUT`
> ของ build นี้ทำ rounding ต่างจาก float path)

---

## Stage C — วัดทั้ง pipeline (ต้องมี DPU + xmodel)

โหลด DPU ก่อน (ถ้ายังไม่ได้โหลดหลังรีบูต):
```bash
sudo xmutil listapps
sudo xmutil loadapp kv260-benchmark-b4096   # ถ้ายังไม่ active
xdputil query | grep -i -E 'fingerprint|DPU'   # ต้องได้ 0x101000056010407
```

### C1 — latency breakdown ใหม่

```bash
cd /tmp/m13
python3 bench_latency.py yolov8n_pkg_kv260.xmodel test.jpg --preproc numpy --iters 100  # baseline เดิม
python3 bench_latency.py yolov8n_pkg_kv260.xmodel test.jpg --preproc lut   --iters 100  # ใหม่
```

เทียบกับของเดิม: preproc 49.75 / DPU 12.49 / decode+NMS 16.07 / e2e 78.31 ms = 12.77 FPS

### C2 — เกณฑ์ผ่านจริงของ M13: นับกล่องต้องยังถูก

```bash
python3 video_detect.py yolov8n_pkg_kv260.xmodel clip_600s_gt215.mp4 out.avi \
        --no_video --preproc lut
```

**เกณฑ์ผ่าน (ทั้งสองข้อพร้อมกัน):**
- `COUNT=215` — ต้องเท่าเดิมเป๊ะ ⇒ ผลนับ error 0.0% ยังใช้ได้
- `proc=... fps` สูงกว่า 10.15 FPS เดิมอย่างมีนัยสำคัญ

สคริปต์จะพิมพ์ power ให้ด้วย (INA260) → ได้ FPS/W จุดใหม่มาลงตารางเลย

> ⚠️ คลิปยาว 600 วิ / 18001 เฟรม — ที่ ~10 FPS ใช้เวลา ~30 นาที
> ถ้าอยากลองเร็วๆ ก่อน ใส่ `--max_frames 1000` (แต่ COUNT จะไม่ใช่ 215 — ใช้ดูว่าไม่พังเฉยๆ)

---

## Stage D — โหมด `hw` (ต้องมี Vitis 2022.2 — ยังทำไม่ได้วันนี้)

ขั้นตอนเต็มอยู่ใน `../pl-preproc/README.md` หัวข้อ 6 ย่อ ๆ คือ
`vitis_hls -f run_hls.tcl` → link เข้า overlay B4096 → firmware app ใหม่ →
`./build_hw_lib.sh` บนบอร์ด → รัน Stage B/C ซ้ำด้วย `--preproc hw`

**ห้ามแก้ `dpu_conf.vh`** ไม่งั้น fingerprint เปลี่ยน → xmodel เดิมใช้ไม่ได้ ต้อง compile ใหม่ทั้งหมด

---

## ตารางกรอกผล (ส่งกลับมาให้ผมเขียนลง timeline / บท Results)

```
=== Stage A ===
cv2 version / arch บนบอร์ด :
A0 synthetic 4 ขนาด        :  worst ____ (ต้อง 0)         PASS/FAIL
A1 test.jpg                :  mismatches ____ / ____  max ____  PASS/FAIL
A2 clip 640x360 (30 เฟรม)  :  mismatches ____ / ____  max ____  PASS/FAIL
A3 1920x1080 (10 เฟรม)     :  mismatches ____ / ____  max ____  PASS/FAIL

=== Stage B ===              numpy (ms)   lut (ms)   speedup   verify
B0 synth 640x640           :  ______      ______     ______x   OK/FAIL
B1 640x640                 :  ______      ______     ______x   OK/FAIL
B2 640x360                 :  ______      ______     ______x   OK/FAIL
B3 1920x1080               :  ______      ______     ______x   OK/FAIL

=== Stage C ===
C1 --preproc numpy         : pre ____ / DPU ____ / dec ____ / e2e ____ ms = ____ FPS
C1 --preproc lut           : pre ____ / DPU ____ / dec ____ / e2e ____ ms = ____ FPS
C2 video --preproc lut     : COUNT ____ (ต้อง 215) · ____ FPS · ____ W · ____ FPS/W
```

---

## ถ้าเจอปัญหา

| อาการ | สาเหตุที่น่าจะเป็น | ทางแก้ |
|---|---|---|
| `ModuleNotFoundError: preproc_lib` | รันจากคนละโฟลเดอร์ | `cd /tmp/m13` ก่อน |
| `ModuleNotFoundError: preproc_tables` | แตกไฟล์ไม่ครบ | เช็ก `ls` ว่ามี `preproc_tables.py` |
| Stage B `numpy` ได้ไม่ถึง 49 ms | รูปเทสเล็กกว่า 640×640 | ใช้รูปจาก valid set หรือใส่ `--src 640x640` |
| `no DpuController found` | ยังไม่ `xmutil loadapp` | ดูหัว Stage C |
| video_detect ช้ามาก / RAM หมด | ลืมใส่ `--no_video` | ใส่ `--no_video` |
| โหมด `hw` บอก `falling back to 'lut'` | ยังไม่มี xclbin (ปกติ — Stage D ยังไม่ทำ) | ใช้ `lut` ไปก่อน |
