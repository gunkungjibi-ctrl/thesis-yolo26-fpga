# Provenance — yolov8n_kv260_B4096

artifact deploy ของ **Track A (YOLOv8n baseline)** สำหรับ KV260 / arch **DPUCZDX8G_ISA1_B4096**

## ที่มา
สำเนา (copy-and-rename) จาก backup `03-model/phase0-wsl-recovered/` ซึ่งกู้จาก WSL `~/thesis/phase0/` (P0, 2026-08-11)
**ต้นฉบับใน phase0-wsl-recovered/ ไม่ถูกแก้ไข** — เก็บเป็นหลักฐานดิบ

## ตารางเปลี่ยนชื่อ (rename map)
| ต้นฉบับ (phase0-wsl-recovered/) | ที่นี่ |
|---|---|
| `yolo26n_kv260.xmodel` | `yolov8n_kv260_B4096.xmodel` |
| `vai_c_xir_yolo26n_kv260.log` | `vai_c_xir_yolov8n_kv260_B4096.log` |
| `meta.json` (`filename` = yolo26n_kv260.xmodel) | `meta.json` (`filename` = yolov8n_kv260_B4096.xmodel) |
| `md5sum.txt` | `md5sum.txt` (คงเดิม) |

## เนื้อไฟล์ไม่เปลี่ยน (แค่ชื่อ)
- xmodel md5 = `69adf8bcd1276feeff6de220bcb39b3e` — เท่ากับต้นฉบับและ `md5sum.txt` ที่ compiler บันทึก verify แล้ว 2026-08-11
- `meta.json` แก้เฉพาะ field `filename` ให้ตรงชื่อไฟล์ใหม่ (จำเป็นต่อ VART runner); field `kernel` (`subgraph_BackboneHead__BackboneHead_10889`) เป็นชื่อ subgraph **ภายในกราฟ** ไม่เกี่ยวกับชื่อไฟล์ จึงคงไว้; `target` ยัง `DPUCZDX8G_ISA1_B4096`

## ทำไมชื่อเดิม "yolo26n" ถึงผิด → ที่ถูกคือ YOLOv8n
ดูหลักฐานเต็มใน `07-notes/P1_arch_and_identity_resolution.md`. โดยย่อ:
- output head = **144 channels** (`[1,20,20,144]`, `[1,40,40,144]`) = 4×16 DFL + 80 COCO = หัว v8
  (yolo26n จะเป็น 84 ch เพราะ reg_max=1 ตัด DFL — ตาม `quantize_yolo_pytorch.py:105-106`)
- `source range` ของ output node ชี้ `quantize_yolo_pytorch.py` (สคริปต์ v8, `--weights` default `yolov8n.pt`)
- โครงหัว `Detect[22]/cv2/cv3` = Detect head แบบ v8

→ นี่คือ **Track A / YOLOv8n / B4096** YOLO26n (Track B) ยังไม่มี artifact ที่ผ่าน gate

## สถานะ / ข้อควรระวังตอน deploy (Phase 1)
- Phase 0 gate ผ่าน: `vai_c_xir` รายงาน `DPU subgraph number 1` (ดู log ในโฟลเดอร์นี้)
- ⚠️ Fingerprint: ก่อนเชื่อผลรันบนบอร์ด รัน `xdputil query` เทียบ fingerprint ของ DPU overlay จริงกับ arch.json (handover จด `0x101000056010407`)
- decode (DFL+sigmoid) + NMS = งานฝั่ง PS ไม่อยู่ในกราฟนี้ (spec I/O ดู `../../README.md`)
