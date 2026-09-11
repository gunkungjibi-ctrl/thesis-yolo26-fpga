# Work log

รูปแบบ: วันที่ / ทำอะไร / ผลลัพธ์ / ติดอะไร / ขั้นต่อไป

## 2026-08-11
- สร้างโครงโปรเจค แล้วรวมของเดิมจาก `Desktop/FPGA Master` เข้ามา (คัดลอก ไม่ย้าย — ต้นทางยังอยู่ครบ 12 รายการ)
- **finding สำคัญ:** Phase 0 ที่จดว่า "✅ ผ่าน" ทำด้วย **YOLOv8n ไม่ใช่ YOLO26n**
  หลักฐาน: `quantize_yolo_pytorch.py` default `yolov8n.pt` / fallback `yolov8n.yaml` / class `BackboneHead`,
  compile script คู่กันชี้ `BackboneHead_int.xmodel`, และ output `[1,40,40,144]` = 64 DFL + 80 cls = หัวแบบ v8
  → YOLO26 (NMS-free head) **ยังไม่เคยผ่าน gate**
- จัด `03-model/` ใหม่เป็นสอง track: A = YOLOv8n (proven, safety net), B = YOLO26n (target, ทำต่อ)
- เขียน `track-b/compile/compile_yolo26n.sh` ใหม่ (ของเดิมไม่มีตัวที่ชี้ `YOLO26nBackboneHead_int.xmodel`)
- **ประเด็นค้าง B3136 vs B4096:** สคริปต์ทั้งสอง track ตั้ง `TARGET = DPUCZDX8G_ISA1_B3136`
  แต่ handover/pre-staging ระบุ B4096 + fingerprint `0x101000056010407` และ platform_selection.docx ก็บอก B4096
  → ยังไม่แก้ให้ track ไหน เพราะยังไม่รู้ว่าอันไหนถูก ต้องเปิด `vai_c_xir_*.log` ตัวจริงก่อน
- ขั้นต่อไป: **P0** (กู้ artifact ออกจาก WSL) → **P1** (เคลียร์ B3136/B4096) ตาม `00-admin/timeline.md`
- **P0 เสร็จ** (2026-08-11): กู้ `phase0/` ทั้งหมด → `03-model/phase0-wsl-recovered/` (162 ไฟล์, verify byte-exact)
- **P1 เสร็จ** (2026-08-11): เคลียร์แล้ว 2 ประเด็น — ดูรายละเอียด `07-notes/P1_arch_and_identity_resolution.md`
  1. **arch จริง = B4096** (ยืนยัน 3 แหล่ง: `vai_c_xir_*.log`, `meta.json`, KV260 arch.json) → B3136 เป็นแค่ comment/inspect dry-run เก่า **ไม่ต้อง re-compile** (inspect ผ่านทั้งสอง arch)
  2. **ยืนยัน worklog ข้างบน:** xmodel ที่ผ่าน gate = **YOLOv8n** จริง (output 144 ch = 4×16 DFL + 80; source range ชี้ `quantize_yolo_pytorch.py`; `--weights` default `yolov8n.pt`) — ชื่อไฟล์ `yolo26n_kv260.xmodel` ตั้งผิด → **YOLO26n (Track B) ยังไม่ผ่าน gate**
- **เปลี่ยนชื่อ artifact เสร็จ** (2026-08-11): copy-and-rename จาก `phase0-wsl-recovered/` → `track-a-yolov8n-baseline/artifacts/yolov8n_kv260_B4096/` (ไม่แตะ backup ต้นฉบับ)
  - `yolo26n_kv260.xmodel` → `yolov8n_kv260_B4096.xmodel`; log/meta ตามด้วย; แก้ `filename` ใน meta.json ให้ตรง
  - md5 xmodel ยัง `69adf8...` (verify แล้ว เนื้อไม่เปลี่ยน); เขียน `PROVENANCE.md` กำกับ rename map + หลักฐาน v8n
- ขั้นต่อไป: เดินหน้า Track B (quantize/compile yolo26n ให้ผ่าน gate, คาดหัว 84 ch) → งานอิสระจากบอร์ด M4–M8
- **M5 เขียน compare script + ขุด WARN เสร็จ** (2026-08-11):
  - เขียน `verify_task2/compare_dpu_vs_golden.py` (โหลด `.bin` จาก M4 → match ด้วยขนาด spatial → cos_sim/max_abs_err)
  - พยายามรัน `yolo_dpu_infer` (M4) จริงในคอนเทนเนอร์ x86 → **abort** `no DpuController found for DPUCZDX8G` = ต้องมี DPU จริง (ผูก M9) ไม่ใช่บั๊ก
  - ขุด part-B WARN (cos_sim 0.991/0.942/0.894) ด้วย `diag_*.py` → **root cause = SiLU→LeakyReLU ยังไม่ fine-tune (M2-B2)** → logit range ระเบิด ±2000 ที่ scale หยาบ → int8 saturate (พิสูจน์: quant clamp `[-64,38.5]` vs float `[-128.7,56.1]`; worst pixel logit 6.0→-11.0 พลิกจากเจอเป็นพลาด)
  - เขียน `07-notes/M5_cossim_dropoff_diagnosis.md`; อัปเดต timeline ให้ **M5 block ด้วย M2-B2**
  - หมายเหตุ env: ต้อง `pip install --no-deps 'ultralytics<8.1'` ในคอนเทนเนอร์ก่อนรัน diag (ไม่ persist เพราะ `--rm`)

## 2026-08-15
- **Dataset พร้อมแล้ว** — วาง Roboflow export `package-conveyo v2` (YOLOv8 detection, 1 class `package`) เข้า `02-dataset/`:
  - `detection/` train 208 / valid 57 / test 28 (+ labels ครบ) · `data.yaml` แก้ path เป็น absolute แล้ว
  - `calib/images/` 208 รูปโดเมนจริง (ก๊อปจาก train) แทน COCO calib เดิมที่ผิดโดเมน → ปลดล็อก M2-B4/M8
  - `counting-eval/videos/video_for_test.mp4` — **วิดีโอ 12 ชม.** (43,200 s, 1,295,985 เฟรม @30fps, 2.1GB) ตรงกับชื่อ dataset `clip_12h`. ยาวเกินกว่าจะ annotate ทั้งหมด → ต้องตัดคลิปสั้น (2–5 นาที) ทำ GT นับ (M7)
  - zip สำรองไว้ `02-dataset/raw/`
- **แก้ TARGET arch** ใน `track-b/quantize/quantize_yolo26n_pytorch.py`: B3136 → **B4096** (2 จุด) ให้ตรง P1
- **M2-B1 (Inspector) BLOCKED** — nndct(py3.7) ↔ ultralytics 8.4.71(ต้อง py3.8+) อยู่ร่วม env ไม่ได้. คอนเทนเนอร์มี base py3.9 (ว่างเปล่า) / vitis-ai-pytorch py3.7 (มี nndct). รายละเอียด+ทางเลือก: `07-notes/M2B1_inspector_python_blocker.md`. ทางที่แนะนำ = export ONNX ใน env py3.9 แยก แล้ววิเคราะห์ op histogram เอง (ตรง README ขั้น 2)
- **Docker factory-reset โดยไม่ตั้งใจ แต่ image รอด** — `xilinx/vitis-ai-pytorch-cpu:...3.0.0.106` (13.7GB) ยังอยู่ครบหลัง reset (ต่างจากที่โน้ตเก่ากังวลว่าจะโดนลบ)
- ขั้นต่อไป: ตัดสินใจทางเลือก M2-B1 (แนะนำ export ONNX py3.9) → M2-B2 fine-tune ด้วย dataset ใหม่ → M2-B4 quantize → M2-B3 compile gate

## 2026-08-15 (ต่อ) — M2-B1 เสร็จ + ตัดคลิป + disk/safety
- **C: เต็ม (เหลือ 0.44GB)** ทำ pip install torch ล้ม → ย้าย venv ไป **D: (`D:\yolo26-export-env`, ว่าง 40GB)** + redirect TEMP/PIP_CACHE_DIR ไป D: → ติดตั้งสำเร็จ (ultralytics 8.4.71 / torch 2.8.0+cpu / onnx 1.19.1). บันทึก memory [[disk-space-constraint]]
- **ตัดคลิปเทสสำเร็จ:** `02-dataset/counting-eval/videos/clip_600s_gt215.mp4` (ffmpeg stream-copy, 00:00:00 +600s, 640×360@30fps, 18001 เฟรม, 30.2MB). **GT = 215 กล่อง** (นับโดย user). วิดีโอจริงคือ 640×360 (mp4probe เดิมอ่าน resolution เพี้ยน)
- **M2-B1 เสร็จผ่าน ONNX op analysis:**
  - export `yolo26n_o2m_leakyrelu.onnx` (457 nodes, end2end=False, SiLU→LeakyReLU ครบ)
  - **NMS-free head ไม่เป็นปัญหา** — watchlist ops (TopK/GatherElements/...) หายหมด
  - **ตัวบล็อกจริง = attention 2 จุดกลางเน็ต** (`model.10` ~32%, `model.22` ~69%: MatMul+Softmax = Q·Kᵀ→softmax→·V) ที่ DPUCZDX8G map ไม่ได้ → ตัด subgraph ~3 ชิ้น → gate "1 subgraph" ผ่านยาก
  - เขียน `artifacts/inspect_report/FINDINGS_op_analysis.md` + `op_histogram.json`. ยังต้อง compile จริง (`vai_c_xir`) ยืนยัน subgraph split
- **user ย้ำกฎถาวร:** ห้าม assistant ลบไฟล์เองเด็ดขาด, งานเสี่ยงต้องขอก่อน, ความปลอดภัยข้อมูล/ตัว user มาก่อน → memory [[safety-first-no-deletion]]
- ขั้นต่อไป: (1) M2-B2 fine-tune LeakyReLU ด้วย dataset package ใหม่ (2) ตัดสินใจทางเลือก attention (ยอม multi-subgraph vs ถอด attention) (3) annotate เส้นนับใน clip_600s สำหรับ GT=215

## 2026-08-16 — 🎉 บอร์ดมาถึง + M9 board bring-up ผ่าน!
- **บอร์ด KV260 มาถึง** (เซ็ตอัพ SD boot image เมื่อวาน) → ปลดล็อก M9+. image ที่ flash = **PetaLinux starter kit (xilinx-kv260-starterkit-2022.2) ไม่ใช่ Ubuntu** ตามแผนเดิม — แต่มี DPU B4096 + VART 3.0 ครบ, ใช้ deploy ได้เลย (user id = root/petalinux ไม่มี ubuntu; `/root` ไม่มี ใช้ `/tmp`)
- เชื่อมต่อ: LAN + serial (user สคริปต์ serial ผ่าน PowerShell `$p.WriteLine`). **relay mode — user คุมบอร์ด 100% ไม่ใส่ SSH key** (ตาม [[safety-first-no-deletion]])
- **M9 ผ่านครบ:**
  - `xmutil listapps` → `kv260-benchmark-b4096` active (DPU B4096 โหลดอยู่แล้ว)
  - `xdputil query` → arch `DPUCZDX8G_ISA1_B4096`, **fingerprint `0x101000056010407` ตรง P1 เป๊ะ**, VART 3.0
  - transfer xmodel: scp ล้ม (PetaLinux ไม่มี sftp-server; `scp -O` ก็งอแง /root ไม่มี) → **ใช้ HTTP: python http.server บน host (192.168.40.219:8000) + wget บนบอร์ด → /tmp** สำเร็จ (md5 ตรง 69adf8...)
  - `xdputil xmodel -l` → **1 DPU subgraph** (compute 100%), USER=input, CPU=output dequant (ปลายกราฟ), output `[40,40,144][20,20,144][80,80,144]` = YOLOv8 ✓
  - `xdputil benchmark <xmodel> 1` → **FPS = 68.1** (single-thread, Test PASS) = **first live DPU inference!**
- **fine-tune ย้ายไป Colab** — host RAM แค่ 5.9GB เทรน local ไม่ได้ (OutOfMemory ตอน spawn python). สร้าง notebook `03-model/track-a-yolov8n-baseline/finetune/colab_finetune_yolov8n_leaky.ipynb` (T4, ultralytics 8.4.71, Roboflow pull, SiLU→LeakyReLU callback). ดู [[disk-space-constraint]]
- ค้าง/ทำต่อ: (1) รัน Colab → ได้ `yolov8n_leaky_pkg_ft.pt` (2) quantize+compile ในคอนเทนเนอร์ → xmodel package (3) deploy บนบอร์ด + VART detection app → เห็นกล่อง package จริง (4) M10 counting เทียบ GT=215 บน clip_600s

## 2026-08-16 (ต่อ) — quantize+compile โมเดล fine-tuned ผ่าน gate
- **OOM ตอน quantize แก้ได้ด้วยการเคลียร์ C:** — รอบแรก container ตาย `unexpected EOF`/exit125 = OOM (host RAM 5.88GB, WSL2 default cap ~2.9GB, free 0.5GB). Diag: `docker info` เองก็ล้ม `errno=1450` (ERROR_NO_SYSTEM_RESOURCES). user เคลียร์ C: (0.2→17GB) → pagefile/swap ขยายได้ → calib pass 32 imgs รอด ไม่ต้องแตะ `.wslconfig`
- **เจอ regression ในสคริปต์ quantize ของ track-a** — ตัวที่ใช้ export ล้ม `XIR don't support multi-outputs op` (C2f `nndct_chunk` ×8 + Detect `split_with_sizes`). ตัว phase0 `quantize_yolo_pytorch.py` ที่เคย export ผ่าน มี 2 fix ที่ track-a หายไป: (a) monkeypatch `C2f.forward` chunk(2,1)→slice (b) BackboneHead wrapper เดิน layer เองแล้ว return raw `cat(cv2,cv3)`/stride (ตัด decode ทิ้ง ไม่แตะ split_with_sizes)
- **แก้:** backup ตัว regression เป็น `.regression.bak` (ไม่ลบ) → ก๊อป proven script จาก phase0 ทับ `quantize/quantize_yolov8n_pytorch.py` (TARGET B4096 อยู่แล้ว) + patch `_quant_compile.sh` ให้ abort ถ้าไม่มี `_int.xmodel` (กัน `set +e` หลอกตา)
- **รันใหม่ผ่านครบ:** PASS1 calib 32/32 → PASS2 export `BackboneHead_int.xmodel` (12.3MB, ไม่มี XIR error) → `vai_c_xir` = **DPU subgraph number 1 = ผ่าน Phase 0 gate**, arch B4096
  - artifact: `artifacts/yolov8n_pkg_B4096/yolov8n_pkg_kv260.xmodel` (3.99MB) + meta.json (kernel `subgraph_BackboneHead__BackboneHead_10884`) + md5 `c702ccb8f027bf9aea95288243d1b274` (ต่างจาก phase0 SiLU `69adf8...` → เป็นน้ำหนัก LeakyReLU fine-tuned จริง)
- ทำต่อ: deploy `yolov8n_pkg_kv260.xmodel` ขึ้นบอร์ด (HTTP transfer, relay mode) → `xdputil xmodel -l` + `benchmark` ยืนยัน → VART detection app เห็นกล่อง package จริง → M10 counting เทียบ GT=215

## 2026-08-16 (ต่อ) — 🎉 deploy โมเดล package fine-tuned บนบอร์ด สำเร็จ end-to-end
- โอน `yolov8n_pkg_kv260.xmodel` ขึ้นบอร์ดผ่าน HTTP (host `python -m http.server 8000` @192.168.40.219 + `wget` → /tmp). serial console = **COM6 @115200 8N1** (prompt `root@xilinx-kv260-starterkit-20222`, บูตนี้ user=root มี home `~`). helper `sendlong` อ่านผ่าน `$p.ReadExisting()` poll หลายรอบ (ReadLine timeout ไม่ทัน). ระวัง: `$p` ถือ COM6 exclusive → เปิดซ้ำจากอีกหน้าต่าง = `Access denied`, ต้องกลับไป reuse `$p` เดิม
- **md5 บนบอร์ด `c702ccb8f027bf9aea95288243d1b274` ตรง** ✓ (โอนไม่เพี้ยน)
- `xdputil xmodel -l`: **1 DPU subgraph**, fingerprint `0x101000056010407` ✓ ตรงบอร์ด, arch B4096, input `[1,640,640,3]`
  - **output 3 หัว × 65 ch** (`[40,40,65][80,80,65][20,20,65]`) = 4×16 DFL + **1 class** → ยืนยันเป็นโมเดล package คลาสเดียวจริง (ต่าง phase0 SiLU 80-class 144ch). subgraph CPU idx2–4 = output dequant ปกติ
- **`xdputil benchmark` = 82.5 FPS** (single-thread, 4952 frames/60s), **Test PASS** — สูงกว่า phase0 (68) เพราะ output เบากว่า (1 vs 80 class). = โมเดล package ตัวจริงรันบน DPU ครั้งแรก
- ทำต่อ: (1) VART detection app (preproc→DPU→DFL decode+sigmoid→NMS บน PS) รันภาพ/วิดีโอจริง เห็นกล่อง package (2) M10 line-counting เทียบ GT=215 บน `clip_600s_gt215.mp4`

## 2026-08-16 (ต่อ) — 🎉 VART detection เห็นกล่อง package บน DPU จริง
- เขียน **`04-deploy/board/yolo_dpu_detect.py`** — Python VART app ครบวงจร: preproc (plain resize640/BGR→RGB//255/quant `2^in_fixpos=64`, NHWC int8) → DPU → dequant 3 หัว (`2^-fixpos`) → **DFL decode เอง** (softmax 16bin/side @ arange → l,t,r,b; anchor=cell+0.5; box=(cx∓d)*stride) → sigmoid(cls) → NMS. C++ M4 `yolo_dpu_infer.cpp` ทำแค่ raw+dequant (decode=TODO) จึงต้องเขียน decode ใหม่. รองรับ cv2 หรือ fallback PIL
- โอนสคริปต์+รูปเทสผ่าน HTTP → รันบนบอร์ด: `input fixpos=6 scale=64`, 3 หัว float_range ปกติ (ไม่ saturate), **4 กล่องหลัง NMS** (score 0.92/0.82/0.78/0.50)
- ดึงรูปกลับ: เปิด `python3 -m http.server 8001` บนบอร์ด (board IP `192.168.40.53`) → host `Invoke-WebRequest` → `08-figures/detect_out_pkg_kv260.jpg`. กล่องกระดาษบนสายพานจับที่ 0.92 ตรงเป๊ะ = **โมเดล package ทำ detection บน DPU เห็นกล่องจริงครั้งแรก**
- หมายเหตุ decode ที่ยืนยันใช้ได้: channel 65 = `[0:64]`=box DFL(4×16), `[64]`=cls(package); stride map 80→8/40→16/20→32; scale กลับภาพต้นฉบับด้วย (origW/640, origH/640) ต่อแกน (plain resize ไม่ letterbox)
- ทำต่อ: M10 line-counting บน `clip_600s_gt215.mp4` เทียบ GT=215 (ต่อยอด decode+NMS ตัวนี้ + tracker/เส้นนับบน PS)

## 2026-08-16 (ต่อ) — เก็บ metrics สำหรับรายงาน (A latency / B power / C mAP / D counting)
เก็บผลรวมใน `05-benchmarks/results/kv260_results.md` + สคริปต์ `04-deploy/board/` และ `05-benchmarks/results/`
- **(A) Latency** (`bench_latency.py`): preproc **49.7ms** (คอขวด PS) / DPU 12.5ms / decode+NMS 16.1ms / e2e 78.3ms → **12.8 FPS e2e** (DPU-only 80). PS-bound
- **(B) Power** (`measure_power.py`, INA260 ina260_u14 SOM): idle 4.85W / benchmark 4-thr 9.27W / **89.6 FPS @ 9.67 FPS·W⁻¹**. GPU FP32 pure=1.38 → ~7× (caveat: FP32vsINT8, GPU ชิปเดียว/FPGA ทั้ง SOM, batch)
- **(B2) Power แอปวิดีโอจริง** (sampler ใน `video_detect.py`): avg **5.10W** (idle+0.25!) / e2e 10.15 FPS / **1.99 FPS·W⁻¹** → DPU ว่างเกือบตลอด (preproc ครอง). รายงาน 2 operating point. GPU e2e=1.13 → real-app กินไฟน้อยกว่า 4.4× & FPS/W ดีกว่า 1.8×
- **(C) Accuracy** (`infer_float.py`/`eval_map.py`): **float mAP@0.5=0.8846** (ตรง Colab 0.884), mAP@0.5:0.95=0.688, F1=0.82 (valid 57รูป). แก้บั๊ก **GT เป็น polygon segmentation → แปลง polygon→bbox**. **INT8 mAP/drop ยังไม่ dump**
- **(D) Counting** (`video_detect.py`/`make_clip.py`): centroid tracker+line-crossing. คลิป20วิ → **COUNT=9**, proc 10.2FPS. output `08-figures/out20.mp4`. **600วิ vs GT=215 ยังไม่รัน** (รอจูนเส้น)
- Transfer: host clip→MJPG .avi (บอร์ด opencv 4.5.2 อ่านได้); ดึง output ผ่าน board `http.server 8001`+host `Invoke-WebRequest`; avi→mp4 ด้วย env D:
- **(C เสร็จ) INT8 mAP** (`infer_dump.py` board, valid 57รูป, conf0.001/iou0.7): **mAP@0.5=0.8713** (float 0.8846 → **drop 1.5%**), mAP@0.5:0.95=0.6795 (drop 1.2%), F1=0.828. → quantization รักษาความแม่นยำเกือบสมบูรณ์
- **(D เสร็จ) Counting vs GT=215** — บอร์ด opencv 4.5.2 (GStreamer) decode H.264 mp4 **ไม่ได้** → ต้อง MJPG .avi เต็ม (514MB, /tmp tmpfs 2GB พอ). แผนฉลาด: `video_dump_dets.py` dump กล่องต่อเฟรม (`dets600.json` 6.3MB, 174,769 กล่อง) ครั้งเดียว → `count_offline.py --sweep` จูน tracker/เส้นบน host ไม่จำกัด. **ผล: axis=x line=0.40·W max_dist=60 → นับ 215/215 = 0% error**, robust (line0.3–0.4×dist40–60 → 209–219, <3%). แกน y ผิดหมด; เส้นกลางจอ(0.5) นับเกิน 270–373 เพราะ track แตกบริเวณกล่องทับ. เพิ่ม `--no_video` flag ให้ counting run ประหยัด RAM. nohup redirect → stdout buffered ต้องใช้ `python3 -u` ดู progress สด
- **(GPU batch sweep)** จาก `results_fp32_final.json`: batch1 pure 31.5FPS@22.8W=1.38 / batch32 peak 263.9FPS@58.5W=4.51 FPS/W. FPGA DPU-sat 9.67 → ดีกว่า best GPU 2.1×; กินไฟ 9.27 vs 58.5W = 6.3× น้อยกว่า
- **ร่างบท Results** → `05-benchmarks/results/RESULTS_chapter_draft.md` (6 ตาราง + key findings). ผลรวม → `kv260_results.md`
- ครบ 4 metric (accuracy/latency/power/counting) + เทียบ GPU. **ทำต่อ:** finalize บท, (option) สร้างวิดีโอ demo เส้น calibrated, quantize+compile yolo26n (Track B)

## 2026-08-23 — sync เอกสารให้ตรงสถานะจริง
- **timeline.md เขียนใหม่** (source of truth เดิมค้างที่ 15 ส.ค.): ย้าย M2-B1/M2-B2/M5–M12 ขึ้นตาราง "เสร็จแล้ว" พร้อมหลักฐานตัวเลขจริง · เหลือ M2-B3 / M14 / M13(optional) · เพิ่มตารางสรุปผลวัด 4 หมวด + ตาราง Efinix E0–E14 (ไม่ถูก parse เข้า dashboard)
- **เปลี่ยนความหมายคอลัมน์ที่ 4** `ต้องมีบอร์ด?` → `ติดบล็อก?` เพราะบอร์ดมาแล้ว 16 ส.ค. (สคริปต์ sync อ่านตำแหน่งเดิม ทำงานได้ปกติ) → dashboard ไม่โชว์ blocked ปลอมอีก
- **รัน `sync_dashboard.ps1`** → `done=17 ready=3 blocked=0 total=20 (85%)` · แก้ข้อความ `t:`/`g:` ที่สคริปต์ไม่ได้ generate ให้เป็นหลักฐานจริง (mAP/FPS/GT) + KPI cards + risk register + next actions + เพิ่มการ์ด "ผลวัดบนบอร์ด KV260" 6 ช่อง · verify render ในเบราว์เซอร์แล้ว
- **README.md เขียนใหม่** (เดิมค้าง 11 ส.ค. ยังเขียนว่ารอบอร์ด/VART ยังไม่เริ่ม): สถานะ 17/20, ตารางผลวัด, ข้อค้นพบ 3 ข้อ, boot image = PetaLinux, arch B4096 ยืนยันแล้ว, ตัด warning เรื่อง artifact ค้างใน WSL (P0 ปิดแล้ว)
- สำรองไฟล์เดิมไว้ `.bak-20260823` ทั้ง 3 ไฟล์ (ไม่ลบอะไร)
- ทำต่อ: **M2-B3** รัน `vai_c_xir` กับ yolo26n จริง → ปิดคำถาม Track B

## 2026-09-05 — ปลดล็อก blocker M2-B3 (ทำจาก cloud session, ยังไม่ได้รัน vai_c_xir)

**บริบทเครื่อง:** session นี้รันบน cloud container ของ Anthropic (Linux x86_64, 4 core, RAM 15GB, disk 30GB) ไม่ใช่เดสก์ท็อปและไม่ใช่บอร์ด

- **ดึง image Vitis AI ไม่ได้** — dockerd รันได้ แต่ `docker pull xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106` ตายที่ blob:
  `production.cloudfront.docker.com:443 — 403 (policy denial)` (manifest ผ่าน, blob โดนบล็อก) → **รัน `vai_c_xir` ที่นี่ไม่ได้** ต้องรันบนเครื่องที่มี Docker + image
- **แต่เจอว่า blocker ตัวจริงไม่ใช่เรื่อง docker** — `quantize_yolo26n_pytorch.py` เรียก `from ultralytics import YOLO` ซึ่ง import ไม่ได้ใน env py3.7 ที่มี nndct → **ต่อให้เปิดคอนเทนเนอร์บนเดสก์ท็อปตอนนี้ก็ตายก่อนถึง gate** (คือ blocker ที่จดไว้ตั้งแต่ 15 ส.ค. และยังไม่เคยแก้)
- **แก้ด้วยการตัด ultralytics ออกจากคอนเทนเนอร์ทั้งหมด** → เขียน `quantize/yolo26n_dpu.py`: กราฟ YOLO26n เป็น PyTorch ล้วน
  - syntax ผ่าน `ast.parse(feature_version=(3,7))` · ใช้เฉพาะ op ที่มีใน torch 1.12 · ชื่อ module ตรง ultralytics เป๊ะเพื่อให้ state_dict โหลด `strict=True` ได้ 1:1
  - **เขียนด้วย slicing แทน `chunk`/`split` ตั้งแต่ต้น** → ตัดปัญหา `XIR don't support multi-outputs op` ที่ทำ Track A ล้ม. ตรวจ traced graph แล้ว **ไม่มี** `aten::chunk`/`split`/`split_with_sizes`/`unbind` เหลือเลย
  - เก็บ attention 2 จุดไว้ครบตามเดิม — มันคือตัวที่กำลังทดสอบ ไม่ไปแก้
- **พิสูจน์ว่ากราฟตรงจริง** (`export_yolo26n_state_dict.py` ทำ check นี้ทุกครั้งที่รัน ไม่ตรง = abort):
  **708 tensors โหลดครบ · `max|diff|` เทียบ ultralytics = `3.8e-06`** (noise float32) ทั้ง 3 หัว
- **ยืนยัน M2-B1 จาก source โดยตรง** (ไม่ใช่แค่ชื่อ node ใน ONNX): `model.10` = `C2PSA` → `PSABlock.attn`; `model.22` = `C3k2(..., attn=True)` → `Sequential(Bottleneck, PSABlock)` → `.attn` ตรงกับ path `/model.22/m.0/m.0.1/attn` เป๊ะ
- **🔴 ข้อค้นพบใหม่ที่ ONNX histogram ไม่ได้ชี้: YOLO26 ไม่มี DFL** — `yolo26.yaml` ตั้ง `reg_max: 1` → `Detect.dfl = nn.Identity()`
  → single-class = **5 ch/หัว** ไม่ใช่ 65 ch แบบ Track A → **`yolo_dpu_detect.py` ใช้กับ YOLO26n ไม่ได้ทันที ต้องแก้ decoder** (แต่ decode จะเบาลงมาก = ดีต่อคอขวดที่เป็น PS-bound)
- **หมายเหตุ 2 หัว:** `end2end: True` ทำให้มีทั้ง `cv2/cv3` (o2m, คู่กับ NMS) และ `one2one_cv2/cv3` (NMS-free). flow นี้ default เป็น o2m ให้เทียบ Track A ได้ตรง · เลือกอีกหัวด้วย `--use_one2one`
- dry-run ทั้ง flow ด้วย nndct stub → calib loop + jit.trace + export path ผ่านหมด ไม่มีบั๊ก
- เขียน `RUNBOOK_M2B3.md` (6 ขั้น + ตารางตัดสิน) · เพิ่ม parse `DPU subgraph number N` อัตโนมัติใน `compile_yolo26n.sh` · เคลียร์คอมเมนต์ B3136/B4096 ที่ค้างใน compile script (P1 ปิดไปแล้ว) · ใส่หัวเตือนใน `quantize_yolo26n_pytorch.py` ว่าใช้ไม่ได้
## 2026-09-05 (ต่อ) — user push น้ำหนักขึ้น git แล้ว → verify ด้วย checkpoint จริง เจอบั๊กเงียบ

user เอา `.pt`/`.onnx` ทั้งหมด (ใหญ่สุด 13MB) ขึ้น git (commit `691c084`, ตัด 2 บรรทัดออกจาก `.gitignore`) → merge เข้ามาแล้วรัน verify ด้วย `yolo26n_leaky_pkg_ft.pt` ตัวจริง

- **รอบแรกไม่ผ่าน — และดีที่ไม่ผ่าน:** `max|diff| = 2.7` ทั้งที่ state_dict โหลดครบ 708 tensors แบบ `strict=True`
  (ตอนทดสอบด้วย `yolo26n.yaml` เปล่าได้ 3.8e-06 จึงไม่เห็นปัญหา — บั๊กโผล่เฉพาะกับน้ำหนักจริง)
- **ไล่ทีละเลเยอร์เจอว่าเพี้ยนตั้งแต่ layer 0** (Conv ตัวแรกสุด): conv output ตรงกัน `0.0` แต่หลัง BN ต่าง `5.7`
- **root cause: `BatchNorm2d.eps`** — ultralytics `initialize_weights()` ตั้ง `eps=1e-3, momentum=0.03` ทับ default ของ PyTorch (`1e-5`, `0.1`)
  ตรวจ checkpoint: BN ทั้ง **114 ตัวเป็น `eps=0.001, momentum=0.03`** ครบ
  **`eps` ไม่ใช่พารามิเตอร์ → ไม่อยู่ใน state_dict → `strict=True` ผ่านฉลุยแต่คำนวณคนละค่า**
- **แก้:** เพิ่ม `BN_EPS=1e-3` / `BN_MOMENTUM=0.03` ใน `yolo26n_dpu.py` + เพิ่ม check แยกใน `export_yolo26n_state_dict.py`
  ที่เทียบ `(eps, momentum)` ของ BN และ slope ของ LeakyReLU ตรงๆ เพื่อให้ครั้งหน้า**บอกสาเหตุได้ ไม่ใช่แค่บอกว่าต่าง**
- **ผลหลังแก้: `max|diff| = 0.000e+00` — bit-exact ทั้ง 3 หัว** กับน้ำหนัก fine-tuned ตัวจริง
- **ความสำคัญ:** ถ้าไม่มี numerical check นี้ เราจะ quantize กราฟที่ผิดเงียบๆ ได้ `.xmodel` ที่ compile ผ่านแต่ผลผิด
  แล้วจะไปโผล่เป็น "INT8 accuracy ตก" ตอนอยู่บนบอร์ด ซึ่งไล่ย้อนยากมาก (เทียบ M5 ที่เคยเสียเวลาขุด cos_sim)
- **cross-check ที่ได้เพิ่ม:** traced graph ของ checkpoint จริงมี **softmax 2 + matmul(data×data) 4** = attention 2 จุด × (2 matmul + 1 softmax) ตรงกับ M2-B1 เป๊ะ
- commit `yolo26n_pkg_state_dict.pt` (verify แล้ว) เข้า repo → ขั้นที่ 1 ของ runbook ข้ามได้ เข้าคอนเทนเนอร์ได้เลย
- dry-run flow เต็มด้วยน้ำหนักจริง + calib จริง 8 รูป (nndct stub) → ผ่านทั้ง 2 pass

## 2026-09-05 (ต่ออีกที) — M2-B3 รันจริงบนเดสก์ท็อป: **ปิดคำถาม Track B แล้ว (compile error)**

เปิด Docker Desktop บนเดสก์ท็อป (image + env ตรง spec: py3.7.12 / torch 1.12.1) รัน `quantize_yolo26n_dpu.py` ด้วย `yolo26n_pkg_state_dict.pt` ตัวจริง + calib 208 รูปจาก `02-dataset/calib/images/`

- **Pass 1 (calibrate, subset_len=32) ผ่านสมบูรณ์** — forward ครบ 32/32
- **Pass 2 (`--quant_mode test --deploy`) ล้ม — ไม่ถึงขั้นรัน `vai_c_xir` ด้วยซ้ำ**
  พังตอน nndct แปลง traced graph → XIR graph: `AddXopError` ที่ op `nndct_elemwise_mul` ตรง
  `C2PSA[model]/.../Attention[attn]/22573` — คือ `attn = (q.transpose(-2,-1) @ k) * self.scale` ใน `Attention.forward`
  (`self.scale` เป็น python float คงที่ ไม่ใช่ tensor — nndct's XIR converter สร้าง fixed-point binary op จาก tensor×scalar-constant ตัวนี้ไม่ได้ โยน `'Caught an unknown exception!'`)
- **นี่คือ node แรกของ attention block ที่ M2-B1 ชี้ไว้พอดี** (`model.10`) — ยืนยัน mechanism เดิม (attention คือตัวบล็อก) เพียงแต่จุดที่ตกจริงคือ scale-multiply ก่อน softmax ไม่ใช่ matmul/softmax เอง
- **ตามตารางตัดสินในบรีฟ** ผลนี้ตกแถว "compile error — op ที่ compiler ปฏิเสธตรงๆ" = ปิดคำถามวิจัย Track B ได้แล้ว (ไม่ต้องรอ `vai_c_xir` log เพราะพังก่อนถึงขั้นนั้น)
- log เต็ม: `quantize/export_xmodel_FAIL_2026-09-05.log` (pass 1: `quantize_calib_pass1_2026-09-05.log`) · เขียนเข้า `FINDINGS_op_analysis.md` ภาคผนวก M2-B3 แล้ว
- **M2-B3 เสร็จ (ผลคือ compile error ที่ระบุ op ได้ชัด)** → เหลือ M14 (finalize รายงาน) เป็นงานหลักที่เหลือ
- **ทำต่อ:** รัน `RUNBOOK_M2B3.md` ขั้น 2–6 บนเครื่องที่มี Docker → ได้ `vai_c_xir` log = ปิด M2-B3

## 2026-09-07 — M13 (optional): ออกแบบ PL preprocessing accelerator + verify บน host

โจทย์จาก M12: preproc 49.7 ms (63%) บน PS → ทำตัวเร่งใน PL แทน `preprocess()` ทั้งก้อน (resize → BGR→RGB → /255 → quantize int8) · โค้ดทั้งหมดใน `04-deploy/pl-preproc/` (README อธิบายละเอียด)

- **kernel Vitis HLS** `hls/preproc_accel.cpp` — เขียนด้วย C มาตรฐาน + pragma (g++ compile ได้) · line buffer 2 แถว reuse · II=1 ต่อ pixel (3 ch ขนาน) · ตารางพิกัด/น้ำหนัก/LUT ให้ host คำนวณส่งเป็น int16×4096 → ใน PL ไม่มี float
- **เป้า = bit-exact กับ cv2** (ไม่งั้นต้องวัด mAP/นับใหม่หมด) → ต้อง reverse สูตรจริงของ `resize.cpp`:
  1. vertical blend ที่ cv2 ใช้จริงคือ path SIMD `((h>>4)·b)>>16` แล้ว `(t0+t1+2)>>2` — สูตร scalar ตามเอกสาร `(…+2^21)>>22` ต่าง ~10% ของพิกเซล (±1)
  2. ขอบแกน x clamp น้ำหนักเป็น 2048/0 แต่**แกน y ไม่ clamp** (ใช้ 92/1956 แล้ว clip index แถว) — รอบแรกทำเหมือนกันสองแกนเลยพลาด ±1 ที่แถวบน/ล่างตอน upscale (จับได้จากเคส 1032×582)
- **ผล verify:** golden model vs cv2 = **0 mismatch** ทั้ง u8 และ int8 (11 ขนาดต้นทาง synthetic + valid set 57 รูป × 640×640/640×360/1920×1080 = 210M ค่า) · C model ของ kernel vs golden = PASS ทุกไบต์ 8 เคส (`make test`)
- **host side:** `host/preproc_xrt.cpp` (XRT native API → `.so`, compile-check ด้วย stub header) + `preproc_accel.py` (ctypes) + `board/preproc_lib.py` เลือกโหมด `numpy/lut/hw` · เพิ่ม `--preproc` ใน `bench_latency.py` / `video_detect.py` (default = โค้ดเดิม ไม่กระทบผลที่วัดไว้) · `bench_preproc.py --verify` วัดแยกโหมดและเช็กว่าตรงกัน
- **ข้อค้นพบสำคัญ (ต้องซื่อสัตย์ในเล่ม):** รูป valid set เป็น 640×640 อยู่แล้ว → 49.7 ms ที่วัดเป็น **numpy float path ไม่ใช่ resize** · โหมด `lut` (cv2.LUT แทน float) ให้ผลตรงกันทุกไบต์และเร็วกว่า 8.7× บน x86 → บน A53 น่าจะเหลือ ~4–8 ms **โดยไม่ต้องแตะ PL** → M13 ต้องรายงาน 3 จุด numpy → lut → hw และ HW ต้องชนะ `lut` ถึงจะอ้างได้
- **ประมาณการ HW (ยังไม่วัด):** ~2.2 ms PL @300 MHz สำหรับ 640×360 (+ ~1 ms memcpy) → e2e ≈ 32 ms ≈ 31 FPS
- **ยังไม่ได้ทำ (ต้องมี tool/บอร์ด):** `vitis_hls -f run_hls.tcl` (csynth/timing/resource) → link เข้า overlay `kv260-benchmark-b4096` ผ่าน kria-vitis-platforms (`vitis/preproc_link.cfg`, ห้ามแก้ dpu_conf.vh ให้ fingerprint เดิม) → firmware app ใหม่ → build `.so` บนบอร์ด → วัด `bench_preproc.py --verify` + `video_detect.py --preproc hw` (COUNT ต้องยัง 215)
- **เตรียมชุดเทสบนบอร์ด** (`04-deploy/board/M13_BOARD_TEST.md` + `make_board_bundle.sh`) แบ่ง 4 stage: A = golden vs cv2 **ของบอร์ด** (ตรวจว่าสูตรที่ reverse จาก x86 SIMD ตรงกับ ARM NEON ด้วย — ด่านที่ต้องผ่านก่อนลงทุน synth), B = วัด numpy vs lut บน A53, C = e2e + COUNT ต้องยัง 215, D = โหมด hw (รอ Vitis)
- ปรับให้รันบนบอร์ดสะดวก: golden model vectorize horizontal pass (เดิม loop ต่อแถว ช้าบน A53), `sweep` รับไฟล์วิดีโอ + ใช้ขนาดเฟรมจริงได้ (`--stride` กระจายเฟรม), import แบบ flat dir ได้, `bench_preproc --verify` ใช้ numpy เป็น reference เสมอและ exit non-zero เมื่อ FAIL

## 2026-09-11 — M13 วัดบนบอร์ดจริง: ได้ speedup 12× ด้วยซอฟต์แวร์ล้วน และ **ปิดคำถามว่าไม่ต้องทำ HW**

user รัน `m13_quickstart.sh` บนบอร์ด (SSH จากมือถือ) — env: PetaLinux 2022.2, kernel 5.15.36-xilinx, Python 3.9.9, numpy 1.21.2, **cv2 4.5.2 aarch64**. ผลเต็ม: `05-benchmarks/results/kv260_results.md` หัวข้อ (E)

- **Stage A0 ผ่าน — ด่านสำคัญที่สุดของงานนี้:** golden model ที่ reverse สูตร `resize.cpp` มาจาก **x86 SIMD (cv2 5.0.0)** ให้ผล **bit-exact กับ cv2 4.5.2 บน ARM NEON ด้วย** — 0 mismatch ทั้ง 4 ขนาด (640×360 / 640×640 / 1280×720 / 1920×1080) ⇒ สเปก HLS kernel ถูกต้องสำหรับเป้าหมายจริง
  - คอลัมน์ `scalar` ยืนยันข้ามสถาปัตยกรรมด้วยว่าสูตร "ตามตำรา" พลาด ~10% ของพิกเซล (±1 LSB) บน ARM เหมือนกัน → การไปดู path SIMD จริงใน `resize.cpp` ไม่ได้เป็นรายละเอียดของ x86 อย่างเดียว
- **Stage B0 ผ่าน:** `numpy` → `lut` ได้ **44.36 → 3.65 ms (12.1×)** ที่ 640×640 และ **53.88 → 6.73 ms (8.0×)** ที่ 640×360 โดย `--verify` = **0 mismatch ทั้งสองขนาด** ⇒ สลับได้เลยโดยไม่ต้องวัด mAP ใหม่
- **ข้อค้นพบชี้ขาด (E3):** ทั้งสองโหมดใช้ `cv2.resize`+`cvtColor` ชุดเดียวกัน ต่างแค่ขั้น quantize → ส่วนต่าง 41–47 ms **คือต้นทุนของ numpy float path ล้วน ๆ** ส่วน resize จริงแค่ ~3.1 ms
  → **"preproc 49.7 ms" ที่ M12 รายงาน ~90% ไม่ใช่ resize แต่เป็น float path** ที่สร้าง temporary `float32` 4.9 MB หลายก้อนจนชน memory bandwidth ของ A53 (แคชเล็ก)
  → คำอธิบายเดิมในบรีฟที่ว่า "ARM ทำทีละค่าไม่ได้ใช้ SIMD เต็มที่" **ไม่ตรงกับสาเหตุจริง** ต้องแก้ตอนเขียนเล่ม
- **ผลต่อการตัดสินใจ M13:** e2e คาด 78.31 → 32.21 ms = **12.77 → ~31 FPS (+143%)** ด้วยการแก้โค้ด ~10 บรรทัด
  PL accelerator (ประมาณ ~3 ms) จะเพิ่มได้อีกแค่ **~11%** แลกกับ synth+bitstream+firmware app → **ตัดสินใจไม่ทำต่อ** เขียน rationale ไว้ใน `04-deploy/pl-preproc/README.md` ข้อ 5
  คอขวดใหม่ = **decode+NMS 16.07 ms (50% ของ pipeline ใหม่)** = เป้า optimize ที่ถูกต้องถัดไป
- **งานออกแบบ kernel ไม่สูญเปล่า:** ได้ (1) วิธี verify bit-exact กับ cv2 ข้ามสถาปัตยกรรม ใช้ซ้ำได้ (2) ข้อค้นพบ E3 ซึ่งมาจากการทำ golden model นี่เอง และเป็นตัวที่บอกว่าไม่ต้องทำ HW (3) สเปกพร้อม synth ถ้าวันหลังต้องรับ MIPI เข้า PL ตรง
- **ขั้นต่อไป:** Stage C บนบอร์ด (ต้องโอน `.xmodel` + `clip_600s_gt215.mp4` จาก PC เพราะติด `.gitignore`) — `bench_latency --preproc lut` ยืนยัน e2e จริง + `video_detect --preproc lut` ต้องได้ **COUNT=215** เท่าเดิม แล้วเก็บ power/FPS-per-W จุดใหม่
