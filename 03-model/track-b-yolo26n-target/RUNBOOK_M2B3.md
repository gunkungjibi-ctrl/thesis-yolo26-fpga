# RUNBOOK — M2-B3: compile YOLO26n ด้วย `vai_c_xir` (Track B gate)

อัปเดต 5 ก.ย. 2026 · เขียนขึ้นหลังปลดล็อก blocker ที่ค้างมาตั้งแต่ 15 ส.ค.

> **สถานะ:** สคริปต์พร้อมและผ่านการ dry-run หมดแล้ว **ยกเว้นขั้นสุดท้ายที่ต้องใช้คอนเทนเนอร์ Vitis AI จริง**
> ขั้นตอนในไฟล์นี้ต้องรันบน **เครื่องที่มี Docker + image Vitis AI 3.0** (เดสก์ท็อปของคุณ) — cloud session ดึง image ไม่ได้ (egress policy บล็อก `production.cloudfront.docker.com`)

---

## blocker ที่ปลดล็อกไปแล้ว (อ่านก่อน ไม่งั้นจะงงว่าทำไมสคริปต์เปลี่ยน)

`07-notes/M2B1_inspector_python_blocker.md` บันทึกไว้ว่า M2-B3 รันไม่ได้เพราะ:

| ต้องการ | อยู่ที่ | Python |
|---|---|---|
| `pytorch_nndct` (ตัวที่ผลิต `_int.xmodel`) | conda env `vitis-ai-pytorch` เท่านั้น | **3.7.12** |
| `ultralytics==8.4.71` (สร้างกราฟ YOLO26n) | ต้องลงเอง | **≥ 3.8** |

`quantize_yolo26n_pytorch.py` เรียก `from ultralytics import YOLO` → **import ไม่ได้ใน env เดียวที่มี nndct** สคริปต์ตายก่อนถึง `vai_c_xir` ด้วยซ้ำ

**ทางแก้ที่ใช้:** ตัด ultralytics ออกจากคอนเทนเนอร์ทั้งหมด → `quantize/yolo26n_dpu.py` เขียนกราฟ YOLO26n ด้วย PyTorch ล้วน (syntax py3.7, ใช้เฉพาะ op ที่มีใน torch 1.12) น้ำหนักส่งเข้าเป็น plain state_dict

**ทำไมเชื่อได้ว่ากราฟตรง:** `export_yolo26n_state_dict.py` โหลด state_dict เข้ากราฟ standalone แบบ `strict` แล้ว**เทียบตัวเลข output กับ ultralytics ทุกครั้งที่รัน** ถ้าไม่ตรงมันจะ abort — รันกับ `yolo26n_leaky_pkg_ft.pt` ตัวจริงแล้วได้ **`max|diff| = 0.000e+00` (bit-exact)** จาก 708 tensors

**ของแถมที่ได้มาด้วย:** กราฟนี้เขียนโดยใช้ **slicing แทน `chunk`/`split`** ตั้งแต่ต้น → ตัดปัญหา `XIR don't support multi-outputs op` ที่เคยทำ Track A ล้ม (worklog 16 ส.ค.) ทิ้งไปเลย ตรวจ traced graph แล้วไม่มี `aten::chunk` / `aten::split` / `aten::split_with_sizes` / `aten::unbind` เหลือสักตัว

---

## ขั้นที่ 0 — เตรียมก่อนเริ่ม

**ประเมินเวลารวม 1–3 ชม.** ส่วนใหญ่คือรอ quantize บน CPU · **ไม่ต้องใช้บอร์ด**

```powershell
cd <repo>
git pull origin claude/project-status-summary-iytv45
```

เช็กว่ามีไฟล์ครบ (อยู่ใน `03-model/track-b-yolo26n-target/`):

| ไฟล์ | ขนาด | หน้าที่ |
|---|---|---|
| `quantize/yolo26n_dpu.py` | 16 KB | กราฟ YOLO26n standalone |
| `quantize/yolo26n_pkg_state_dict.pt` | 10 MB | น้ำหนัก verify แล้ว bit-exact ✅ |
| `quantize/quantize_yolo26n_dpu.py` | 7 KB | ตัวรัน nndct 2 pass |
| `compile/compile_yolo26n.sh` | 4 KB | `vai_c_xir` = gate |

**calib images** — ต้องเป็นภาพสายพานจริง 32+ รูป ใช้ `02-dataset/calib/images/` (208 รูป บนเครื่องคุณ)
ถ้าหาไม่เจอ ใช้ `02-dataset/detection/train/images/` แทนได้ — **เนื้อหาเดียวกัน** และอยู่ใน repo

---

## ขั้นที่ 1 — บนเครื่อง host (Windows, env py3.9+ ที่ D:) — **ข้ามได้**

> ⏭️ **ขั้นนี้รันไปแล้วและ commit ผลไว้ให้**: `quantize/yolo26n_pkg_state_dict.pt`
> สร้างจาก `finetune/yolo26n_leaky_pkg_ft.pt` ตัวจริง ผ่าน check ครบ **`max|diff| = 0.000e+00`** (bit-exact)
> ถ้าไม่ได้เปลี่ยน checkpoint ก็ **ข้ามไปขั้นที่ 2 ได้เลย**
> รันซ้ำเมื่อ fine-tune ใหม่ หรืออยากตรวจเองอีกรอบ

ใช้ venv เดิมที่ `D:\yolo26-export-env` (มี ultralytics 8.4.71 อยู่แล้ว)

```bash
cd 03-model/track-b-yolo26n-target/quantize

python export_yolo26n_state_dict.py \
    --weights <path>/yolo26n_leaky_pkg_ft.pt \
    --nc 1 \
    --out yolo26n_pkg_state_dict.pt
```

**สิ่งที่ต้องเห็นก่อนไปต่อ** (ผลจริงจากน้ำหนัก fine-tuned ตัวจริง):
```
[ok] state_dict loaded 1:1 into yolo26n_dpu.YOLO26nBackboneHead (708 tensors)
[ok] BatchNorm (eps, momentum) matches: [(0.001, 0.03)]
[ok] LeakyReLU slope matches: [0.1015625]
[ok] standalone graph matches ultralytics, max|diff| = 0.000e+00
```
ถ้า abort ตรงนี้ **อย่าไปต่อ** — แปลว่า checkpoint ไม่ใช่ YOLO26n scale n หรือโครงต่างจากที่ resolve ไว้

> ### 🐛 บทเรียนจากการ bring-up ไฟล์นี้ — ทำไมถึงต้องมี check ตัวนี้
>
> รอบแรกที่รันกับ checkpoint จริง **ผลต่างถึง 2.7** ทั้งที่ state_dict โหลดครบ 708 tensors แบบ `strict=True`
> ไล่ทีละเลเยอร์แล้วเจอว่าเพี้ยนตั้งแต่ **layer 0 (Conv ตัวแรกสุด)** — conv output ตรงกัน 0.0 แต่หลัง BN ต่างไป 5.7
>
> สาเหตุ: **ultralytics `initialize_weights()` ตั้ง `BatchNorm2d.eps = 1e-3` (PyTorch default = `1e-5`)**
> `eps` **ไม่ใช่พารามิเตอร์** → ไม่อยู่ใน state_dict → โหลด strict ผ่านฉลุยแต่คำนวณคนละค่า
>
> ถ้าไม่มี numerical check ตัวนี้ เราจะ quantize กราฟที่ผิดโดยไม่มีอะไรเตือนเลย แล้วได้ `.xmodel` ที่ compile ผ่าน
> แต่ผลลัพธ์ผิด — และจะไปโผล่เป็น "INT8 accuracy ตก" ตอนอยู่บนบอร์ด ซึ่งไล่ย้อนยากมาก
> ตอนนี้สคริปต์เช็ค `eps`/`momentum`/slope แยกต่างหากด้วย เพื่อให้บอกสาเหตุได้ ไม่ใช่แค่บอกว่าต่าง

> ยังไม่ต้องมีน้ำหนัก fine-tuned ก็ซ้อมได้: ใส่ `--weights yolo26n.yaml --nc 80` จะได้กราฟเปล่า
> แต่ **ห้ามเอาผล accuracy จากน้ำหนักเปล่าไปเขียนรายงาน** — ใช้ซ้อม flow เท่านั้น
> (compile gate ไม่สนค่า weight เพราะการแบ่ง subgraph ขึ้นกับ topology ล้วน)

## ขั้นที่ 2 — เปิดคอนเทนเนอร์

```bash
docker run -it --rm \
  -v "$PWD":/workspace \
  -v "<path ไป 02-dataset/calib/images>":/calib \
  xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106 bash

conda activate vitis-ai-pytorch
python -c "import sys, torch, pytorch_nndct; print(sys.version, torch.__version__)"
cd /workspace
```

ต้องเห็น `3.7.12 ... 1.12.1` — ถ้าไม่ใช่ แปลว่า activate ผิด env

> ⚠️ **ห้ามใช้ tag `:latest`** (เป็น 3.5 ซึ่งไม่มี KV260 prebuilt + board_setup/mpsoc)

## ขั้นที่ 3 — Inspector (ทางเลือก, advisory)

```bash
python quantize_yolo26n_dpu.py --state_dict yolo26n_pkg_state_dict.pt --nc 1 --inspect
```

คาดว่าจะเห็น attention 2 จุด (`model.10` C2PSA, `model.22` C3k2 `attn=True`) โดนมาร์กเป็น CPU/USER
**นี่ไม่ใช่ gate** — เป็นแค่การยืนยันล่วงหน้าว่า M2-B1 อ่านถูก

## ขั้นที่ 4 — quantize สองรอบ

```bash
# pass 1 — calibrate ด้วยภาพสายพานจริง
python quantize_yolo26n_dpu.py \
    --state_dict yolo26n_pkg_state_dict.pt --nc 1 \
    --calib_dir /calib --quant_mode calib --subset_len 32

# pass 2 — export XIR xmodel
python quantize_yolo26n_dpu.py \
    --state_dict yolo26n_pkg_state_dict.pt --nc 1 \
    --calib_dir /calib --quant_mode test \
    --subset_len 1 --batch_size 1 --deploy
```

ได้ `quantize_result/YOLO26nBackboneHead_int.xmodel`

**ระวัง RAM:** Track A เคยตายตรงนี้ด้วย `unexpected EOF` / exit 125 = OOM ไม่ใช่บั๊กสคริปต์
(host RAM 5.88 GB, WSL2 cap ~2.9 GB) → เคลียร์ C: ให้ pagefile ขยายได้ก่อน `subset_len 32` เคยรอด

## ขั้นที่ 5 — compile = **gate จริง**

```bash
cd /workspace
INT_XMODEL=quantize_result/YOLO26nBackboneHead_int.xmodel \
  bash ../compile/compile_yolo26n.sh
```

สคริปต์จะ grep บรรทัด `DPU subgraph number N` ออกมาให้เอง แล้วสรุป verdict

## ขั้นที่ 6 — ดู partition จริง (นี่คือความจริง ไม่ใช่ summary ใน log)

```bash
xdputil xmodel yolo26n_kv260.xmodel -l
```

op ไหนขึ้น device `USER` / `CPU` = ตกไปรันบน ARM PS

---

## ตารางตัดสิน — ผลออกทางไหนก็เป็น finding

| ผล | แปลว่า | ทำต่อ |
|---|---|---|
| `DPU subgraph number 1` | YOLO26n map ลง B4096 ได้ทั้งตัว | Track B ขึ้นเป็นโมเดลหลัก · Track A เป็น baseline เทียบในเล่ม |
| 2–4 subgraph | แตกตรง attention ตามที่ M2-B1 ทำนาย | วัด latency penalty จริงบนบอร์ด → เขียนเป็น finding เชิงปริมาณ |
| compile error | มี op ที่ XIR/compiler ปฏิเสธตรงๆ | **จด error message + ชื่อ op ให้ครบ** = contribution ของ track นี้ |

**กติกาที่ตั้งไว้เอง:** gate เดียวที่นับคือ `vai_c_xir` log — ไม่ใช่ op histogram ไม่ใช่ Inspector
ผลออกทางไหน **ก็ไม่ใช่ความล้มเหลวของโครงงาน** เพราะ Track A ผ่านครบวงจรและวัดผลครบ 4 หมวดแล้ว

---

## ข้อค้นพบใหม่ที่เจอระหว่างเตรียม M2-B3 (มีผลกับตอนเขียน deploy app)

**YOLO26 ไม่มี DFL** — `yolo26.yaml` ตั้ง `reg_max: 1` ทำให้ `Detect.dfl = nn.Identity()`

| | Track A (YOLOv8n) | Track B (YOLO26n) |
|---|---|---|
| `reg_max` | 16 | **1** |
| ch/หัว (single class) | 4×16 + 1 = **65** | 4×1 + 1 = **5** |
| decode ฝั่ง PS | ต้อง softmax 16 bin ต่อด้าน แล้วคูณ arange | **ไม่ต้องทำ DFL เลย** — อ่าน l,t,r,b ตรงๆ |

→ ถ้า Track B ผ่าน gate **`yolo_dpu_detect.py` ใช้ซ้ำไม่ได้ทันที** ต้องแก้ decoder
แต่ decode จะถูกลงมาก (ตัด softmax+matmul ต่อ anchor ทิ้ง) ซึ่งช่วยเรื่อง bottleneck ที่เป็น PS-bound อยู่แล้ว

**หมายเหตุ one2one:** YOLO26 มี 2 หัว — `cv2/cv3` (one-to-many, ใช้คู่กับ NMS) และ `one2one_cv2/cv3` (NMS-free)
ค่า default ของ flow นี้คือหัว **o2m + NMS บน PS** ให้เหมือน Track A เพื่อให้เทียบกันได้ตรงๆ
อยาก export หัว NMS-free ให้ใส่ `--use_one2one` ทั้งตอน export state_dict และตอน quantize

---

## ไฟล์ที่เกี่ยวข้อง

| ไฟล์ | รันที่ไหน | ทำอะไร |
|---|---|---|
| `quantize/yolo26n_dpu.py` | ทั้งสองที่ | กราฟ YOLO26n standalone (py3.7 + torch 1.12 safe, ไม่มี chunk/split) |
| `quantize/yolo26n_pkg_state_dict.pt` | **container** | น้ำหนักที่ verify แล้ว (bit-exact) — สร้างไว้ให้แล้ว |
| `quantize/export_yolo26n_state_dict.py` | **host** py3.9+ | .pt → plain state_dict + พิสูจน์ว่ากราฟตรง ultralytics |
| `quantize/quantize_yolo26n_dpu.py` | **container** py3.7 | calib + export `_int.xmodel` |
| `compile/compile_yolo26n.sh` | **container** | `vai_c_xir` = gate + parse subgraph count |
| `quantize/quantize_yolo26n_pytorch.py` | — | **ตัวเดิม ใช้ไม่ได้** (ติด ultralytics ↔ py3.7) เก็บไว้อ้างอิงเฉยๆ |
