# 04-deploy

- `board/` — โค้ดที่รันบน PS (Arm): VART/Vitis AI Runtime, tracker, line-crossing counter
- `host-app/` — สคริปต์ฝั่ง host, การส่งไฟล์ขึ้นบอร์ด, การเก็บ log
- `platform/` — boot image, ไฟล์ `.bit`/`.xclbin`, device tree, บันทึกว่าใช้ DPU arch อะไร
