# 02-dataset

## detection/ — Roboflow `package-conveyo` v2 (YOLOv8 detection)
1 class = `package` · 640×640 · แปลงจาก polygon → bbox โดย Roboflow

| split | images | labels |
|---|---|---|
| train | 208 | 208 |
| valid | 57 | 57 |
| test  | 28 | 28 |

- `data.yaml` — path เป็น **absolute** แล้ว (แก้จาก `../` ของ Roboflow ที่ resolve ผิด layout นี้)
- ใช้กับ fine-tune: `finetune_leakyrelu.py --data 02-dataset/detection/data.yaml`

## calib/images/ — 208 รูป (ก๊อปจาก detection/train/images)
ภาพสายพานจริง → ใช้ทำ post-training quantization calibration (M2-B4 / M8)
แทนที่ COCO `calib_images` เดิมที่ผิดโดเมน

## counting-eval/ — งานนับ (M7 / M10)
- `videos/video_for_test.mp4` — วิดีโอเทสสำหรับวัดการนับ
- `annotations/` — (ยังว่าง) ต้องทำ ground-truth: เส้นนับ + จำนวนจริง เพื่อวัด MAE

## raw/ — ไฟล์ต้นฉบับ (zip export สำรองไว้)

## scripts/ — สคริปต์เตรียมข้อมูล
