# System Architecture

## Target
- Board: _____ (เช่น Kria KV260 / ZCU104 = Zynq UltraScale+ MPSoC, หรือ VCK190 = Versal)
- Vitis version: _____
- Vitis AI version: _____
- DPU IP + arch: _____ (เช่น DPUCZDX8G, B4096) → กำหนด `arch.json` ที่ใช้ตอน compile

## Dataflow
```
Camera / video file
   ↓
[PS] decode + preprocess (resize, normalize)
   ↓
[PL] DPU IP — YOLO inference (.xmodel)
   ↓  (op ที่ไม่รองรับ → fall back มารันบน PS: ต้องบันทึกว่ามีอะไรบ้าง)
[PS] post-process (decode boxes / NMS ถ้ามี)
   ↓
[PS] tracker (assign ID ข้ามเฟรม)
   ↓
[PS] line-crossing logic → count
   ↓
Display / log
```

## จุดที่ต้องวัด
- % ของงานที่รันบน PL จริง (ไม่ใช่แค่ "รันบน FPGA")
- latency แยกรายสเตจ: preprocess / DPU / post / track
