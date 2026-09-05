# Track A — YOLOv8n baseline (PROVEN)

**สถานะ: Phase 0 ✅ ผ่าน** — `vai_c_xir` รายงาน `DPU subgraph number 1` (มิ.ย. 2026)
นี่คือ safety net ของโครงงาน ถ้า Track B ล้ม โครงงานยังส่งได้ด้วย track นี้

## ทำไมถึงเป็น v8n ไม่ใช่ 26n
`quantize_yolov8n_pytorch.py` (เดิมชื่อ `quantize_yolo_pytorch.py`) ตั้ง default weights = `yolov8n.pt`,
fallback = `yolov8n.yaml`, wrapper class = `BackboneHead` และ compile script คู่กันชี้
`quantize_result/BackboneHead_int.xmodel` — สอดคล้องกันทั้งสาย
ยืนยันซ้ำจาก output shape `[1,40,40,144]` → 144 = 64 DFL + 80 class = **หัวแบบ v8 (มี DFL) ไม่ใช่ NMS-free head ของ YOLO26**

## Spec ที่ล็อกไว้ (ใช้เขียน VART host code)
- input `[1, 640, 640, 3]` **NHWC** (ไม่ใช่ NCHW), fixpos 6 → int8 = `round(float × 2^6)` clamp [-128,127]
- output 3 tensors: `[1,40,40,144]` fixpos 0 · `[1,20,20,144]` fixpos -2 · `[1,80,80,144]` fixpos 1
- dequantize (× 2^-fixpos) + decode (DFL + sigmoid) + NMS = **งานฝั่ง PS ทั้งหมด** ไม่อยู่ในกราฟที่ quantize
- fingerprint ที่จดไว้: `0x101000056010407` (⚠️ ดูประเด็น B3136/B4096 ใน `../README.md`)

## Preprocessing (ต้องตรงกับ calib เป๊ะ ห้ามเปลี่ยน)
```
cv2.resize(im, (640,640))     # plain resize ไม่ letterbox — ตรงกับ calib
cv2.cvtColor(BGR2RGB)
.astype(float32) / 255.0
transpose → CHW (PyTorch) / NHWC (VART)
```
ถ้าเปลี่ยน preprocessing ต้อง re-calibrate ใหม่ ไม่งั้น inference ให้ผลขยะแบบ**เงียบ ไม่มี error ฟ้อง**

## วิธีรันซ้ำ
```bash
docker pull xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106   # อย่าใช้ :latest (เป็น 3.5)
docker run -it --rm -v "$PWD":/workspace -w /workspace \
  xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106 bash
conda activate vitis-ai-pytorch

# pass 1 — calibrate
python quantize/quantize_yolov8n_pytorch.py --weights yolov8n.pt \
  --calib_dir calib_images --quant_mode calib --subset_len 200
# pass 2 — export XIR xmodel
python quantize/quantize_yolov8n_pytorch.py --weights yolov8n.pt \
  --calib_dir calib_images --quant_mode test --subset_len 1 --batch_size 1 --deploy
# compile
bash compile/compile_yolov8n.sh
```

## TODO
- [ ] ดึง `.xmodel` + `vai_c_xir_*.log` + `quant_info.json` จาก WSL `~/thesis/phase0/` มาไว้ `artifacts/`
- [ ] ยืนยัน DPU arch จริงจาก log (B3136 หรือ B4096)
- [ ] re-quantize เป็น single-class "package" ก่อน Phase 3 (ตอนนี้เป็น 80-class COCO)
