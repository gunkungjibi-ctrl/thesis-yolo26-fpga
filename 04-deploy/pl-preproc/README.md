# 04-deploy/pl-preproc — PL preprocessing accelerator (M13)

ตัวเร่ง preprocessing บน programmable logic ของ KV260 สำหรับคอขวดที่ M12 ชี้ไว้:
**preproc 49.7 ms = 63% ของ pipeline** ในขณะที่ DPU ใช้แค่ 12.5 ms

> สถานะ (7 ก.ย. 2026): **ออกแบบ + พิสูจน์ความถูกต้องเชิงตัวเลขบน host เสร็จ** (bit-exact กับ cv2 ทุกไบต์)
> · **ยังไม่ได้ synth/สร้าง bitstream** — ต้องใช้ Vitis HLS + Vitis 2022.2 บนเครื่องที่มี tool
> · ตัวเลข latency ของ HW ในไฟล์นี้เป็น **ค่าประมาณจาก cycle count** ยังไม่ได้วัดจริง

## 1. งานที่ kernel ทำ (แทน `preprocess()` ทั้งก้อน)

```
DDR: BGR uint8 [H][W][3]  (เฟรมจาก cv2 / กล้อง; W ≤ 1920, W % 8 == 0)
        │  AXI4 master (64-bit) burst-read ทีละแถว (line buffer 2 แถว, reuse ถ้าซ้ำ)
        ▼
  bilinear resize → 640×640     ← สูตร fixed-point ตรงกับ cv2.resize(INTER_LINEAR) บน ARM NEON
        ▼
  BGR → RGB (สลับลำดับตอนเขียน)
        ▼
  quantize ผ่าน LUT 256 ค่า      ← LUT = round(x/255 · 2^fixpos) สร้างจากนิพจน์เดียวกับแอป
        ▼
DDR: int8 NHWC [1][640][640][3]  (1,228,800 B)  → ส่งให้ VART ตรง ๆ
```

ตารางพิกัด/น้ำหนัก (xofs, alpha, yofs, beta) และ LUT **host เป็นคนคำนวณ** (8 KB, int16×4096)
kernel อ่านครั้งเดียวตอนเริ่มเฟรม → ไม่มี float / หาร ใน PL และเปลี่ยนขนาดเฟรม/fixpos ได้โดยไม่ต้อง synth ใหม่

## 2. ทำไมต้อง bit-exact กับ cv2 และทำได้อย่างไร

ผลวัดทั้งหมด (mAP 0.8713, นับ 215/215) ได้จาก input ที่ preprocess ด้วย cv2 บน ARM
ถ้า HW ให้ค่าต่างแม้ 1 LSB ต้องวัด accuracy ใหม่ทั้งหมด → เป้าคือ **ตรงทุกไบต์** เพื่อให้ผลเดิมใช้ต่อได้เลย

สิ่งที่ค้นพบระหว่างทำ (สำคัญกับคนที่จะทำ HW scaler เทียบ cv2):

| ประเด็น | ที่ OpenCV ทำจริง (`resize.cpp`) | ผลถ้าทำตามสูตร "ตำรา" |
|---|---|---|
| vertical blend | path SIMD (NEON/SSE): `((h>>4)·b)>>16` ต่อแถว แล้ว `(t0+t1+2)>>2` | สูตร scalar `(h0·b0+h1·b1+2^21)>>22` ต่างถึง **~10% ของพิกเซล** (±1) |
| ขอบแกน x | clamp น้ำหนักเป็น 2048/0 และ index ให้อยู่ในภาพ | — |
| ขอบแกน y | **ไม่** clamp น้ำหนัก (เช่น 92/1956) แต่ clip index แถวตอนอ่าน | ถ้าทำเหมือนแกน x จะพลาด ±1 ที่แถวบน/ล่างตอน upscale (เช่น 640×360→640) |
| coefficient | `float` 11-bit fixed, round-half-even (`lrint`) | ใช้ double หรือ round-half-up จะเพี้ยนบางค่า |

**ผลตรวจบน host (cv2 5.0 x86, SIMD path เดียวกับ NEON ในเชิงตัวเลข):**

| การทดสอบ | ผล |
|---|---|
| golden model vs cv2 — 11 ขนาดต้นทาง (2×2 … 1920×1080) × synthetic frames | **0 mismatch** ทั้ง uint8 และ int8 |
| golden model vs cv2 — valid set 57 รูป × 3 ขนาด (640×640, 640×360, 1920×1080) | **0 / 210,124,800 ค่า** |
| C model ของ kernel (g++) vs golden — 8 ขนาด + รูปจริง | **PASS bit-exact ทุกเคส** (`make test`) |
| `preproc_lib.py` โหมด `lut` vs โค้ดเดิม | 0 mismatch |

## 3. ไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `hls/preproc_accel.h` | I/O contract + layout ของ param buffer (ใช้ร่วม kernel / tb / host) |
| `hls/preproc_accel.cpp` | **kernel Vitis HLS** (C มาตรฐาน + pragma — g++ compile ได้ตรง ๆ) |
| `hls/tb_preproc.cpp` | C testbench เทียบ output กับ golden ทุกไบต์ (ใช้ทั้ง g++ และ csim/cosim) |
| `hls/run_hls.tcl` | Vitis HLS 2022.2: csim → csynth → export `.xo` (หรือ IP-XACT) part `xck26-sfvc784-2LV-c` 300 MHz |
| `hls/Makefile` | `make vectors` / `make test` / `make hls` |
| `golden/preproc_golden.py` | bit-exact software model + `check` (vs cv2) / `gen` (test vector) / `sweep` (dataset) |
| `host/preproc_tables.py` | สร้างตารางพิกัด/น้ำหนัก/LUT แบบเดียวกับ cv2 |
| `host/preproc_xrt.cpp` + `Makefile` | host driver ผ่าน XRT native API → `libpreproc_xrt.so` (build บนบอร์ด) |
| `host/preproc_accel.py` | ctypes wrapper: `HwPreproc(xclbin)(bgr, in_scale)` → int8 [1,640,640,3] |
| `vitis/preproc_link.cfg` | v++ link config: ต่อ kernel เข้า overlay DPU B4096 (HP3, 300 MHz) |
| `../board/preproc_lib.py` | เลือกโหมด `numpy` / `lut` / `hw` ให้ `bench_latency.py`, `video_detect.py` (`--preproc`) |
| `../board/bench_preproc.py` | วัด preproc แยกโหมด + `--verify` ว่าทุกโหมดให้ผลตรงกัน |

## 4. สถาปัตยกรรม kernel (สั้น ๆ)

- **Interface:** 3 × AXI4 master 64-bit (`src`, `params`, `dst`) + AXI-Lite (`src_w`, `src_h`, pointers) — Vitis kernel มาตรฐาน
- **ต่อ output row (640 แถว):**
  1. เช็ก line buffer 2 แถว (`rows[2][1920]` 32-bit packed pixel, cyclic-partition 8 bank) — ถ้าแถว `sy0/sy1` ยังไม่มีค่อย burst-read (3 word = 8 pixel/iteration, II=1)
     · upscale (640×360 → 640): แต่ละแถวต้นทางอ่าน**ครั้งเดียว** (reuse) · downscale ≤ 2×: อ่าน ≤ 1 แถวใหม่ต่อ output row
  2. interpolate 640 pixel × 3 channel ขนาน, `PIPELINE II=1` — 4 read จาก dual-port BRAM/cycle, LUT partition complete (3 lookup/cycle)
  3. pack 8 byte/word → burst-write 240 words
- **Datapath ต่อ channel** (int ล้วน, ไม่มี DSP กว้าง): `h = p0·a0 + p1·a1` (≤2^19) → `t = ((h>>4)·b)>>16` → `u8 = sat((t0+t1+2)>>2)` → `lut[u8]`
- **ทรัพยากรโดยประมาณ (รอ csynth ยืนยัน):** BRAM ≈ 10–14 (line buffer 15 KB + tables 6.4 KB) · DSP ≈ 12–24 · LUT/FF หลักพัน — เล็กมากเทียบ DPU B4096

### ประมาณการ latency (300 MHz, ยังไม่ได้วัด)

| ต้นทาง | cycles ≈ compute 640×(640+240+~20) + loads | เวลา PL | รวม host memcpy in/out (~1 ms A53) |
|---|---|---|---|
| 640×360 (คลิปนับจริง) | 576k + 360×240 = ~662k | ~2.2 ms | **~3 ms** |
| 640×640 (รูป valid set) | 576k + 640×240 = ~730k | ~2.4 ms | ~3.5 ms |
| 1920×1080 (กล้อง 1080p) | 576k + 1080×720 = ~1.35M | ~4.5 ms | ~5.5 ms |

เทียบ baseline 49.7 ms → ถ้าได้ตามนี้ e2e ≈ 78.3 − 49.7 + 3 ≈ **32 ms ≈ 31 FPS** (จาก 12.8) โดย DPU ยังเหลือ headroom

## 5. ⛔ ผลวัดบนบอร์ดจริง (11 ก.ย. 2026) — **อย่าเพิ่ง synth**

วัดแล้วบนบอร์ด (`../board/m13_quickstart.sh`, cv2 4.5.2/aarch64) ผลเต็มอยู่ที่
`05-benchmarks/results/kv260_results.md` หัวข้อ **(E)** สรุปที่กระทบไฟล์นี้โดยตรง:

| | preproc 640×360 | e2e |
|---|---|---|
| `numpy` (baseline เดิม) | 53.88 ms | 12.77 FPS |
| **`lut` (ซอฟต์แวร์ล้วน, bit-exact)** | **6.73 ms** | **~31 FPS** |
| PL accelerator (ประมาณการในไฟล์นี้) | ~3 ms | +~11% จาก `lut` |

**49.7 ms ที่ M12 รายงาน ~90% คือ numpy float path ไม่ใช่ resize** — float path สร้าง temporary
`float32` 4.9 MB หลายก้อนจนชน memory bandwidth ของ A53 ส่วน resize จริงแค่ ~3.1 ms
พอเปลี่ยน quantize เป็น LUT (ผลตรงกันทุกไบต์) ก็เก็บ gain ไปเกือบหมดแล้ว

⇒ **PL accelerator เหลือที่ให้เพิ่มอีกแค่ ~11% ไม่คุ้มกับงาน synth + bitstream + firmware app**
⇒ คอขวดใหม่หลังใช้ `lut` คือ **decode+NMS 16.07 ms (50% ของ pipeline)** — เป็นเป้าที่ถูกต้องกว่า

### แล้วโค้ดในโฟลเดอร์นี้ยังมีค่าตรงไหน

1. **วิธี verify** — golden model + C testbench พิสูจน์ bit-exact กับ cv2 ได้ทั้ง x86 SIMD และ
   ARM NEON (0 mismatch ทั้ง 4 ขนาด) เป็นวิธีที่ใช้ซ้ำได้กับ HW block อื่น
2. **ข้อค้นพบ E3** — การทำ golden model นี่แหละที่ทำให้เห็นว่าต้นทุนอยู่ที่ quantize ไม่ใช่ resize
   ซึ่งเป็นเหตุผลที่ทำให้ *ไม่ต้อง* ทำ HW เขียนเข้าเล่มได้เต็ม ๆ
3. **สเปกพร้อมใช้** ถ้าวันหนึ่งต้องรับกล้อง MIPI เข้า PL ตรง ๆ (preproc ไม่ต้องผ่าน PS เลย)
   หรือต้องปลด A53 ไปทำงานอื่น — kernel ตัวนี้ synth ได้ทันทีโดยไม่ต้องออกแบบใหม่

**เงื่อนไขที่จะกลับมาทำ HW:** ถ้า decode+NMS ถูก optimize จนเหลือระดับเดียวกับ DPU แล้ว
preproc กลับมาเป็นสัดส่วนที่มีนัย หรือถ้าย้ายไปรับภาพจาก MIPI/ISP ใน PL

## 6. ขั้นตอนบนเครื่องที่มี Vitis 2022.2 (ยังไม่ได้ทำ)

```bash
# (1) HLS: csim ต้องผ่านทุกเคสก่อน แล้ว csynth + export .xo
cd 04-deploy/pl-preproc/hls && make vectors && make test        # g++ (ทำแล้ว ผ่าน)
vitis_hls -f run_hls.tcl                                        # → preproc_accel.xo + รายงาน timing/resource
vitis_hls -f run_hls.tcl -tclargs cosim                         # RTL co-sim (optional, ช้า)

# (2) link เข้า overlay DPU B4096 (kria-vitis-platforms, branch xlnx_rel_v2022.2)
git clone https://github.com/Xilinx/kria-vitis-platforms && cd kria-vitis-platforms/kv260
#   overlays/examples/benchmark = DPUCZDX8G B4096 ตัวเดียวกับ firmware kv260-benchmark-b4096
#   → เพิ่ม preproc_accel.xo + preproc_link.cfg ใน Makefile ของ overlay (ห้ามแก้ dpu_conf.vh)
make overlay OVERLAY=benchmark   # ได้ .xclbin (มี DPU + preproc_accel) + .bit.bin

# (3) ทำ firmware app ใหม่ "kv260-yolo-preproc": .bit.bin + .dtbo + shell.json + .xclbin
#     วางที่ /lib/firmware/xilinx/kv260-yolo-preproc/ บนบอร์ด แล้ว
sudo xmutil unloadapp && sudo xmutil loadapp kv260-yolo-preproc
xdputil query          # fingerprint ต้องยัง 0x101000056010407 (ไม่งั้น xmodel เดิมใช้ไม่ได้)

# (4) host lib + วัด
cd 04-deploy/pl-preproc/host && make                            # libpreproc_xrt.so (บนบอร์ด)
cd ../../board
python3 bench_preproc.py frame.jpg --src 640x360 --verify      # numpy / lut / hw ต้อง mismatch=0
python3 bench_latency.py yolov8n_pkg_kv260.xmodel test.jpg --preproc lut
python3 bench_latency.py yolov8n_pkg_kv260.xmodel test.jpg --preproc hw
python3 video_detect.py yolov8n_pkg_kv260.xmodel clip_600s_gt215.mp4 out.avi --no_video --preproc hw
#   → COUNT ต้องยัง 215 และ FPS ต้องขึ้น = เกณฑ์ผ่าน M13
```

## 7. ความเสี่ยง / สิ่งที่ยังไม่ได้ยืนยัน

| ประเด็น | สถานะ |
|---|---|
| csynth ได้ II=1 จริง + timing 300 MHz | ยังไม่รัน — ถ้าไม่ผ่านลดเป็น 275/250 MHz ใน `preproc_link.cfg` ได้โดยไม่แก้โค้ด |
| ชื่อ SP tag (HP3) ของ platform kv260 | ตรวจด้วย `platforminfo` ก่อน link |
| XRT บน PetaLinux starter kit เปิด kernel ผ่าน `xrt::kernel` ได้ (ควบคู่ VART ที่ใช้ XRT อยู่แล้ว) | คาดว่าได้ (smartcam ของ Xilinx ใช้แบบเดียวกัน) — ยังไม่ทดสอบ |
| `preproc_xrt.cpp` | compile-check ด้วย stub header เท่านั้น (ไม่มี XRT บน host) |
| zero-copy กับ VART input buffer | v1 ยัง memcpy 1.2 MB ออก (~0.5 ms) — ทำภายหลังถ้าจำเป็น |
| เฟรมกว้างไม่หาร 8 | kernel ไม่รับ → `preproc_lib` fallback เป็น `lut` อัตโนมัติ |
