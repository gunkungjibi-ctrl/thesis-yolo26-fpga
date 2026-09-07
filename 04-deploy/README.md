# 04-deploy

- `board/` — โค้ดที่รันบน PS (Arm): VART/Vitis AI Runtime, tracker, line-crossing counter · `preproc_lib.py` + `bench_preproc.py` = โหมด preprocessing 3 แบบ (M13)
- `pl-preproc/` — **M13** PL preprocessing accelerator (Vitis HLS kernel + golden model + XRT host lib) — resize/cvt/quantize ใน PL, bit-exact กับ cv2
- `host-app/` — สคริปต์ฝั่ง host, การส่งไฟล์ขึ้นบอร์ด, การเก็บ log
- `platform/` — boot image, ไฟล์ `.bit`/`.xclbin`, device tree, บันทึกว่าใช้ DPU arch อะไร
