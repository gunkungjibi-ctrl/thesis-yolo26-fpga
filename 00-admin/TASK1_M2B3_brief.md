# งาน 1 — M2-B3: รัน `vai_c_xir` กับ YOLO26n จริง

อัปเดต 5 ก.ย. 2026

> เอกสารนี้ = สรุปงานลำดับ 1 แบบลงมือทำได้เลย
> รายละเอียดเชิงเทคนิคเต็ม + เหตุผลเบื้องหลัง = `03-model/track-b-yolo26n-target/RUNBOOK_M2B3.md`

**เป้าหมาย:** ตอบคำถามวิจัย Track B ให้จบว่า YOLO26n compile ลง DPUCZDX8G B4096 ได้กี่ subgraph
**ไม่ต้องใช้บอร์ด · ประเมิน 1–3 ชม. (ส่วนใหญ่คือรอ quantize บน CPU)**

---

## ก่อนเริ่ม — ดึงโค้ดใหม่ลงมา

```powershell
cd <repo>
git pull origin claude/project-status-summary-iytv45
```

ไฟล์ที่ต้องมีครบ (อยู่ใน `03-model/track-b-yolo26n-target/`):

| ไฟล์ | ขนาด | หน้าที่ |
|---|---|---|
| `quantize/yolo26n_dpu.py` | 16 KB | กราฟ YOLO26n standalone (py3.7 + torch 1.12) |
| `quantize/yolo26n_pkg_state_dict.pt` | 10 MB | น้ำหนัก verify แล้ว bit-exact ✅ |
| `quantize/quantize_yolo26n_dpu.py` | 7 KB | ตัวรัน nndct 2 pass |
| `compile/compile_yolo26n.sh` | 4 KB | `vai_c_xir` = gate |

> **ไม่ต้องรันขั้น export** — รันให้แล้วด้วย `yolo26n_leaky_pkg_ft.pt` ตัวจริง ได้ `max|diff| = 0.000e+00`

**เตรียม calib images** — ต้องเป็นภาพสายพานจริง 32+ รูป ใช้ `02-dataset/calib/images/` บนเครื่องคุณ (208 รูป)
ถ้าหาไม่เจอ ใช้ `02-dataset/detection/train/images/` แทนได้ (เนื้อหาเดียวกัน อยู่ใน repo)

---

## ขั้น 2 — เปิดคอนเทนเนอร์

```powershell
docker run -it --rm `
  -v "${PWD}:/workspace" `
  -v "${PWD}\02-dataset\detection\train\images:/calib" `
  xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106 bash
```

> ⚠️ **ห้ามใช้ `:latest`** — เป็น 3.5 ซึ่งไม่มี KV260 prebuilt

ในคอนเทนเนอร์:

```bash
conda activate vitis-ai-pytorch
python -c "import sys, torch, pytorch_nndct; print(sys.version.split()[0], torch.__version__)"
```

**ต้องเห็น `3.7.12 1.12.1`** ถ้าไม่ใช่ = activate ผิด env อย่าไปต่อ

```bash
cd /workspace/03-model/track-b-yolo26n-target/quantize
ls /calib | wc -l          # ต้องได้ 208
```

---

## ขั้น 3 — Inspector (ทางเลือก, ~5 นาที)

```bash
python quantize_yolo26n_dpu.py \
    --state_dict yolo26n_pkg_state_dict.pt --nc 1 --inspect
```

คาดว่าจะเห็น `model.10` (C2PSA) และ `model.22` (C3k2 attn) โดนมาร์ก CPU/USER
**นี่ไม่ใช่ gate** — เป็นการยืนยันล่วงหน้าว่า M2-B1 อ่านถูก ข้ามได้ถ้าอยากประหยัดเวลา

---

## ขั้น 4 — quantize 2 pass (ตัวกินเวลาที่สุด)

```bash
# pass 1 — calibrate
python quantize_yolo26n_dpu.py \
    --state_dict yolo26n_pkg_state_dict.pt --nc 1 \
    --calib_dir /calib --quant_mode calib --subset_len 32

# pass 2 — export XIR xmodel
python quantize_yolo26n_dpu.py \
    --state_dict yolo26n_pkg_state_dict.pt --nc 1 \
    --calib_dir /calib --quant_mode test \
    --subset_len 1 --batch_size 1 --deploy
```

**สิ่งที่ต้องเห็น:**

```
[ok] loaded yolo26n_pkg_state_dict.pt into YOLO26nBackboneHead(nc=1)
[info] head is DFL-free (reg_max=1) -> 5 channels per stride
[ok] jit.trace pre-flight passed
...
[ok] exported quantize_result/YOLO26nBackboneHead_int.xmodel
```

**🔴 landmine ที่เคยเจอกับ Track A:**

| อาการ | สาเหตุจริง | แก้ |
|---|---|---|
| `unexpected EOF` / exit 125 | **OOM** ไม่ใช่บั๊กสคริปต์ (host RAM 5.88 GB, WSL2 cap ~2.9 GB) | เคลียร์ C: ให้ pagefile ขยายได้ก่อน · ลด `--subset_len` เหลือ 16 |
| `XIR don't support multi-outputs op` | chunk/split | **ไม่ควรเจอแล้ว** — `yolo26n_dpu.py` เขียนด้วย slicing ตั้งแต่ต้น ตรวจ traced graph แล้วสะอาด ถ้าเจอ ส่ง log มา |

---

## ขั้น 5 — compile = **gate จริง**

```bash
bash ../compile/compile_yolo26n.sh
```

สคริปต์ grep บรรทัด `DPU subgraph number N` ออกมาสรุปให้เอง:

```
PARSED: DPU subgraph number = <N>
VERDICT: ...
```

log เต็มเก็บที่ `vai_c_xir_yolo26n_kv260.log`

---

## ขั้น 6 — ดู partition จริง (นี่คือความจริง ไม่ใช่ summary)

```bash
xdputil xmodel yolo26n_kv260.xmodel -l
```

op ไหนขึ้น device **`USER` / `CPU`** = ตกไปรันบน ARM PS

---

## ตารางตัดสิน — ผลออกทางไหนก็เป็น finding

| ผล | แปลว่า | ทำต่อ |
|---|---|---|
| `subgraph number 1` | YOLO26n map ลง B4096 ได้ทั้งตัว 🎉 | Track B ขึ้นเป็นโมเดลหลัก · Track A เป็น baseline เทียบในเล่ม · **ต้องเขียน decoder ใหม่** (5 ch ไม่มี DFL) |
| **2–4 subgraph** ← คาดว่าอันนี้ | แตกตรง attention ตามที่ M2-B1 ทำนาย | วัด latency penalty จริงบนบอร์ด → เขียนเป็น finding **เชิงปริมาณ** ไม่ใช่แค่คาดการณ์ |
| compile error | มี op ที่ compiler ปฏิเสธตรงๆ | **จด error + ชื่อ op ให้ครบ** = contribution ของ track นี้ |

**สิ่งที่คาดว่าจะเจอ** จาก traced graph ของ checkpoint จริง: **softmax 2 ตัว + matmul(data×data) 4 ตัว**
= attention 2 จุด × (2 matmul + 1 softmax) — ตรงกับ M2-B1 เป๊ะ
ทั้งคู่อยู่ **กลางกราฟ** (~32% และ ~69%) จึงน่าจะตัดเป็น ~3 subgraph

---

## ส่งอะไรกลับมา

1. `vai_c_xir_yolo26n_kv260.log` (ทั้งไฟล์)
2. output ของ `xdputil xmodel yolo26n_kv260.xmodel -l`
3. ถ้าพังตรงไหน — error message เต็มๆ + คำสั่งที่รัน

แล้วจะ:

- อ่านผล + สรุปว่าแตกตรงไหนเพราะอะไร
- เขียนเข้า `FINDINGS_op_analysis.md` + `timeline.md` + `worklog.md`
- อัปเดต README/dashboard ให้เป็น 18/20
- ถ้าผ่าน gate → ร่าง decoder ใหม่สำหรับหัว 5 ch ให้เลย
