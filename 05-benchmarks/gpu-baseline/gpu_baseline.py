#!/usr/bin/env python3
"""
GPU baseline measurement for thesis: Real-Time Video-Based Object Counting on FPGA
----------------------------------------------------------------------------------
วัด FPS + power (FPS/Watt) ของ YOLOv8n บน GPU (GTX 1660 Ti) เพื่อเป็น high-power
baseline เทียบกับ KV260 (edge DPU, ~5W).

หลักการสำคัญ (ตรงกับ methodology ของ thesis):
  - โมเดลเดียวกับ Phase 0: yolov8n.pt  (apple-to-apple)
  - preprocessing ตรงกับ calib เป๊ะ: plain resize 640, BGR->RGB, /255, CHW
    (อ้างจาก quantize_yolo_pytorch.py บรรทัด load_calib_batch)
  - วัด PURE GPU inference แยกจาก preprocessing (เทียบ DPU compute share)
  - power อ่านผ่าน NVML (nvidia-ml-py) แม่นกว่า parse nvidia-smi
  - วัด idle power baseline แยก เพื่อแยก dynamic power ของ inference
  - warmup ก่อนวัด (GPU clock ramp) + วัดหลายรอบ รายงาน median + variance
    (Max-Q throttling จะเห็นเป็น variance สูง)

การใช้งาน:
  python gpu_baseline.py --weights yolov8n.pt --images ~/thesis/phase0/calib_images
  python gpu_baseline.py --weights yolov8n.pt --synthetic        # โหมด synthetic
  python gpu_baseline.py --weights yolov8n.pt --images <dir> --half  # FP16
"""

import argparse
import glob
import os
import statistics
import time

import cv2
import numpy as np
import torch


# ----------------------------------------------------------------------------
# NVML power sampling
# ----------------------------------------------------------------------------
class PowerSampler:
    """อ่าน GPU power ผ่าน NVML. ถ้า NVML ใช้ไม่ได้ จะ degrade เป็น None เงียบ ๆ."""

    def __init__(self, gpu_index=0):
        self.handle = None
        self.samples = []
        self.util_samples = []
        self.clock_samples = []
        try:
            import pynvml
            self.pynvml = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception as e:
            print(f"[warn] NVML init failed ({e}); power will be unavailable")
            self.pynvml = None

    def read_watt(self):
        if self.handle is None:
            return None
        try:
            # nvmlDeviceGetPowerUsage คืนค่าเป็น milliwatt
            return self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except Exception:
            return None

    def read_util(self):
        if self.handle is None:
            return None
        try:
            return self.pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
        except Exception:
            return None

    def read_clock(self):
        if self.handle is None:
            return None
        try:
            return self.pynvml.nvmlDeviceGetClockInfo(
                self.handle, self.pynvml.NVML_CLOCK_SM)
        except Exception:
            return None

    def sample(self):
        w = self.read_watt()
        if w is not None:
            self.samples.append(w)
        u = self.read_util()
        if u is not None:
            self.util_samples.append(u)
        c = self.read_clock()
        if c is not None:
            self.clock_samples.append(c)

    def reset(self):
        self.samples = []
        self.util_samples = []
        self.clock_samples = []

    def stats(self):
        if not self.samples:
            return None
        s = {
            "mean_w": statistics.mean(self.samples),
            "max_w": max(self.samples),
            "min_w": min(self.samples),
            "n": len(self.samples),
        }
        if self.util_samples:
            s["mean_util"] = statistics.mean(self.util_samples)
            s["max_util"] = max(self.util_samples)
        if self.clock_samples:
            s["mean_clock"] = statistics.mean(self.clock_samples)
            s["max_clock"] = max(self.clock_samples)
            s["min_clock"] = min(self.clock_samples)
        return s

    def shutdown(self):
        if self.pynvml is not None:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass


# ----------------------------------------------------------------------------
# Preprocessing — ตรงกับ Phase 0 calib เป๊ะ
# ----------------------------------------------------------------------------
def preprocess_bgr(im_bgr, imgsz=640):
    """
    Replicate load_calib_batch จาก quantize_yolo_pytorch.py:
        cv2.resize -> BGR2RGB -> float32/255 -> CHW
    คืน numpy CHW (ยังไม่ batch, ยังไม่ขึ้น GPU)
    """
    im = cv2.resize(im_bgr, (imgsz, imgsz))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(im, (2, 0, 1))  # HWC -> CHW


def load_images(img_dir, imgsz, limit=None):
    files = sorted(glob.glob(os.path.join(img_dir, "*")))
    if limit:
        files = files[:limit]
    imgs = []
    for p in files:
        im = cv2.imread(p)
        if im is None:
            continue
        imgs.append(preprocess_bgr(im, imgsz))
    return imgs


# ----------------------------------------------------------------------------
# Build the backbone model (raw conv head, ตรงแนวคิดกับ BackboneHead ของ Phase 0)
# ----------------------------------------------------------------------------
def build_model(weights, half, device):
    """
    โหลด YOLOv8n ผ่าน ultralytics แล้วใช้ underlying nn.Module ตรง ๆ
    เพื่อวัด pure forward (ไม่ผ่าน .predict() ที่ปน NMS/postprocess บน CPU)

    หมายเหตุ FP16: ไม่ใช้ model.half() ทั้งก้อน เพราะ YOLOv8 detect head
    (DFL/anchor/stride) มี op ที่ต้อง FP32 -> half ทั้งโมเดลทำให้เกิด
    FP16<->FP32 cast ทุก forward (ช้ากว่า FP32 ล้วนบน TU116).
    วิธีถูกคือเก็บโมเดล FP32 แล้วใช้ torch.autocast ตอน forward (AMP)
    ให้ PyTorch เลือก precision per-op เอง.
    """
    from ultralytics import YOLO
    yolo = YOLO(weights)
    model = yolo.model.to(device).eval()
    return model


# ----------------------------------------------------------------------------
# Timing helpers
# ----------------------------------------------------------------------------
def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_pure_inference(model, batch, device, n_iters, power, half=False):
    """
    วัด pure GPU inference. batch อยู่บน GPU แล้ว (ไม่นับเวลา H2D copy / preprocess)
    FP16 ใช้ torch.autocast (AMP) ไม่ใช่ half ทั้งโมเดล
    คืน list ของ latency ต่อ iteration (วินาที)
    """
    latencies = []
    power.reset()
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16,
                              enabled=half)
    with torch.no_grad():
        for _ in range(n_iters):
            cuda_sync(device)
            t0 = time.perf_counter()
            with autocast:
                _ = model(batch)
            cuda_sync(device)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            power.sample()
    return latencies


def time_end_to_end(model, np_imgs, device, half, power):
    """
    วัด end-to-end ต่อเฟรม: preprocess(บน CPU แล้ว) -> H2D -> inference
    np_imgs = list ของ CHW numpy (preprocess แล้ว)
    input เป็น FP32 เสมอ (autocast จัดการ cast ภายใน)
    """
    latencies = []
    power.reset()
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16,
                              enabled=half)
    with torch.no_grad():
        for chw in np_imgs:
            cuda_sync(device)
            t0 = time.perf_counter()
            t = torch.from_numpy(chw).unsqueeze(0).to(device, dtype=torch.float32)
            with autocast:
                _ = model(t)
            cuda_sync(device)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            power.sample()
    return latencies


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def time_batch_throughput(model, device, half, imgsz, batch_sizes, n_iters, power):
    """
    วัด peak throughput ที่ batch ต่าง ๆ — หา batch ที่ GPU อิ่มตัว
    คืน dict: batch -> {fps_per_image, latency_ms, power, util, oom}
    มี OOM guard: ถ้า batch ไหน OOM หยุดเพิ่มและรายงาน
    """
    results = {}
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16,
                              enabled=half)
    for bs in batch_sizes:
        try:
            batch = torch.rand(bs, 3, imgsz, imgsz, device=device,
                               dtype=torch.float32)
            # warmup เฉพาะ batch นี้
            with torch.no_grad():
                for _ in range(10):
                    with autocast:
                        _ = model(batch)
            cuda_sync(device)
            power.reset()
            lat = []
            with torch.no_grad():
                for _ in range(n_iters):
                    cuda_sync(device)
                    t0 = time.perf_counter()
                    with autocast:
                        _ = model(batch)
                    cuda_sync(device)
                    lat.append(time.perf_counter() - t0)
                    power.sample()
            med = statistics.median(lat)
            ps = power.stats()
            results[bs] = {
                "latency_ms": med * 1000,
                "fps_total": bs / med,           # ภาพต่อวินาที (ทั้ง batch)
                "fps_per_call": 1.0 / med,
                "power_w": ps["mean_w"] if ps else None,
                "util": ps.get("mean_util") if ps else None,
                "oom": False,
            }
            del batch
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            results[bs] = {"oom": True}
            torch.cuda.empty_cache()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                results[bs] = {"oom": True}
                torch.cuda.empty_cache()
                break
            raise
    return results


def summarize(name, latencies, power_stats, idle_w):
    lat_ms = [l * 1000 for l in latencies]
    med = statistics.median(lat_ms)
    mean = statistics.mean(lat_ms)
    p95 = sorted(lat_ms)[int(len(lat_ms) * 0.95)] if len(lat_ms) > 1 else lat_ms[0]
    stdev = statistics.stdev(lat_ms) if len(lat_ms) > 1 else 0.0
    fps = 1000.0 / med

    print(f"\n=== {name} ===")
    print(f"  frames/iters     : {len(latencies)}")
    print(f"  latency median   : {med:.2f} ms")
    print(f"  latency mean     : {mean:.2f} ms")
    print(f"  latency p95      : {p95:.2f} ms")
    print(f"  latency stdev    : {stdev:.2f} ms   (สูง = throttling/variance)")
    print(f"  throughput       : {fps:.1f} FPS  (จาก median latency)")

    if power_stats:
        pw = power_stats["mean_w"]
        print(f"  power mean       : {pw:.1f} W  (max {power_stats['max_w']:.1f}, "
              f"min {power_stats['min_w']:.1f}, n={power_stats['n']})")
        if "mean_util" in power_stats:
            print(f"  GPU util         : {power_stats['mean_util']:.0f}% mean, "
                  f"{power_stats['max_util']:.0f}% max  "
                  f"(<50% = GPU ไม่เต็ม, batch เล็กไป)")
        if "mean_clock" in power_stats:
            print(f"  SM clock         : {power_stats['mean_clock']:.0f} MHz mean "
                  f"(max {power_stats['max_clock']:.0f}, min {power_stats['min_clock']:.0f}"
                  f" -> ต่างมาก = throttle/ramp)")
        print(f"  FPS/Watt         : {fps / pw:.2f}   <-- เมตริกหลักเทียบ KV260")
        if idle_w is not None:
            dyn = pw - idle_w
            if dyn > 0:
                print(f"  dynamic power    : {dyn:.1f} W  (หัก idle {idle_w:.1f} W)")
                print(f"  FPS/Watt(dyn)    : {fps / dyn:.2f}   (เฉพาะ inference power)")
    else:
        print("  power            : N/A (NVML ไม่พร้อม)")
    return {"fps": fps, "median_ms": med,
            "power_w": power_stats["mean_w"] if power_stats else None}


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--images", default=None, help="dir ของรูป calib (COCO128)")
    ap.add_argument("--synthetic", action="store_true",
                    help="ใช้ random tensor แทนรูปจริง")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200,
                    help="จำนวน iteration สำหรับ pure-inference mode")
    ap.add_argument("--limit", type=int, default=128,
                    help="จำกัดจำนวนรูปจาก dir")
    ap.add_argument("--sweep", action="store_true",
                    help="วัด batch sweep (peak throughput) เพิ่มจาก batch=1")
    ap.add_argument("--lock-clock", type=int, default=None,
                    help="ล็อก SM clock (MHz) ตัด ramp variance ถ้า driver ยอม")
    ap.add_argument("--runs", type=int, default=1,
                    help="วัด PURE inference ซ้ำ N รอบ เอา median-of-medians (กัน outlier บน Max-Q)")
    ap.add_argument("--save", default=None,
                    help="เขียนผลลงไฟล์ JSON (เช่น results_fp32.json)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[FAIL] CUDA ไม่พร้อม — baseline นี้ต้องใช้ GPU")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[info] device      : {gpu_name}")
    print(f"[info] torch       : {torch.__version__}")
    print(f"[info] precision   : {'FP16' if args.half else 'FP32'}")
    print(f"[info] imgsz       : {args.imgsz}")

    power = PowerSampler(0)

    # --- ลองล็อก GPU clock ถ้าขอ (ตัด ramp variance) ---
    if args.lock_clock and power.handle is not None:
        try:
            power.pynvml.nvmlDeviceSetGpuLockedClocks(
                power.handle, args.lock_clock, args.lock_clock)
            print(f"[info] locked SM clock to {args.lock_clock} MHz")
        except Exception as e:
            print(f"[warn] lock clock ไม่สำเร็จ ({e}) — "
                  f"Max-Q/WSL อาจไม่ให้ lock (ต้อง permission)")

    # --- idle power baseline (วัดก่อนโหลดงาน) ---
    idle_w = None
    if power.handle is not None:
        idle_samples = []
        for _ in range(20):
            w = power.read_watt()
            if w is not None:
                idle_samples.append(w)
            time.sleep(0.05)
        if idle_samples:
            idle_w = statistics.median(idle_samples)
            print(f"[info] idle power  : {idle_w:.1f} W")

    # --- build model ---
    model = build_model(args.weights, args.half, device)
    # input เป็น FP32 เสมอ — FP16 ทำผ่าน autocast (ดู build_model docstring)
    dtype = torch.float32

    # --- เตรียม input ---
    if args.synthetic or not args.images:
        if not args.synthetic:
            print("[warn] ไม่ได้ระบุ --images, ใช้ synthetic แทน")
        print("[info] input       : SYNTHETIC random tensor")
        np_imgs = None
        batch = torch.rand(1, 3, args.imgsz, args.imgsz,
                           device=device, dtype=dtype)
    else:
        print(f"[info] input       : รูปจริงจาก {args.images}")
        np_imgs = load_images(args.images, args.imgsz, args.limit)
        if not np_imgs:
            print("[warn] หารูปไม่เจอ -> fallback synthetic")
            batch = torch.rand(1, 3, args.imgsz, args.imgsz,
                               device=device, dtype=dtype)
            np_imgs = None
        else:
            print(f"[info] loaded      : {len(np_imgs)} รูป (preprocess แล้ว)")
            # batch สำหรับ pure-inference = รูปแรกค้างบน GPU
            batch = torch.from_numpy(np_imgs[0]).unsqueeze(0).to(device, dtype=dtype)

    # --- warmup (GPU clock ramp + cudnn autotune) ---
    print(f"[info] warmup      : {args.warmup} iters ...")
    _warmup_ac = torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=args.half)
    with torch.no_grad():
        for _ in range(args.warmup):
            with _warmup_ac:
                _ = model(batch)
    cuda_sync(device)

    results = {}

    # --- (A) PURE GPU inference: รูปเดียวค้างบน GPU, วน n iters ---
    if args.runs > 1:
        # multi-run: วัด N รอบ เอา median-of-medians + spread (กัน outlier Max-Q)
        run_meds = []
        run_fps = []
        run_pw = []
        last_stats = None
        for r in range(args.runs):
            lat_r = time_pure_inference(model, batch, device, args.iters, power,
                                        half=args.half)
            med_r = statistics.median([l * 1000 for l in lat_r])
            run_meds.append(med_r)
            run_fps.append(1000.0 / med_r)
            st = power.stats()
            last_stats = st
            if st:
                run_pw.append(st["mean_w"])
            print(f"  run {r+1}/{args.runs}: median {med_r:.2f} ms "
                  f"-> {1000.0/med_r:.1f} FPS"
                  + (f", {st['mean_w']:.1f} W" if st else ""))
        med_of_med = statistics.median(run_meds)
        fps_stable = 1000.0 / med_of_med
        fps_spread = max(run_fps) - min(run_fps)
        pw_stable = statistics.median(run_pw) if run_pw else None
        print(f"\n=== PURE GPU (multi-run, {args.runs} รอบ x {args.iters} iters) ===")
        print(f"  median-of-medians: {med_of_med:.2f} ms -> {fps_stable:.1f} FPS")
        print(f"  FPS spread       : {fps_spread:.1f} FPS "
              f"(min {min(run_fps):.1f}, max {max(run_fps):.1f})")
        print(f"  run-to-run stdev : "
              f"{statistics.stdev(run_fps) if len(run_fps)>1 else 0:.1f} FPS")
        if pw_stable:
            print(f"  power median     : {pw_stable:.1f} W")
            print(f"  FPS/Watt         : {fps_stable/pw_stable:.2f}   "
                  f"<-- เมตริกหลัก (เสถียรกว่ารอบเดียว)")
            if idle_w:
                dyn = pw_stable - idle_w
                if dyn > 0:
                    print(f"  FPS/Watt(dyn)    : {fps_stable/dyn:.2f}")
        results["pure"] = {"fps": fps_stable, "median_ms": med_of_med,
                           "power_w": pw_stable}
    else:
        lat_pure = time_pure_inference(model, batch, device, args.iters, power,
                                       half=args.half)
        results["pure"] = summarize(
            f"PURE GPU inference ({args.iters} iters, รูปค้างบน GPU)",
            lat_pure, power.stats(), idle_w)

    # --- (B) END-TO-END ต่อเฟรม: H2D + inference (เฉพาะตอนมีรูปจริง) ---
    if np_imgs is not None:
        lat_e2e = time_end_to_end(model, np_imgs, device, args.half, power)
        results["e2e"] = summarize(
            f"END-TO-END ต่อเฟรม (H2D copy + inference, {len(np_imgs)} รูป)",
            lat_e2e, power.stats(), idle_w)

    # --- สรุปเทียบ ---
    print("\n" + "=" * 60)
    print("สรุปสำหรับ thesis (เทียบกับ KV260):")
    print(f"  GPU                : {gpu_name}")
    print(f"  precision          : {'FP16' if args.half else 'FP32'}")
    if "pure" in results and results["pure"]["power_w"]:
        r = results["pure"]
        print(f"  pure inference     : {r['fps']:.1f} FPS @ {r['power_w']:.1f} W "
              f"= {r['fps']/r['power_w']:.2f} FPS/W")
    print("  หมายเหตุ: 1660 Ti Max-Q ไม่มี Tensor Core -> เทียบเป็น")
    print("           edge-DPU-INT8 vs mobile-GPU-FP -> ระบุให้ชัดในเล่ม")
    print("=" * 60)

    # --- (C) BATCH SWEEP: peak throughput (ถ้าขอ) ---
    if args.sweep:
        print("\n" + "=" * 60)
        print("BATCH SWEEP (peak throughput — หา batch ที่ GPU อิ่มตัว)")
        print("=" * 60)
        sweep = time_batch_throughput(
            model, device, args.half, args.imgsz,
            [1, 4, 8, 16, 32, 64], n_iters=50, power=power)
        print(f"  {'batch':>6} {'img/s':>9} {'lat/call':>10} "
              f"{'power':>8} {'util':>6} {'img/s/W':>9}")
        best = None
        for bs, r in sweep.items():
            if r.get("oom"):
                print(f"  {bs:>6}  OOM (VRAM เต็ม — หยุด sweep)")
                break
            pw = r["power_w"]
            ips_w = (r["fps_total"] / pw) if pw else 0
            util = r["util"] if r["util"] is not None else 0
            print(f"  {bs:>6} {r['fps_total']:>9.1f} "
                  f"{r['latency_ms']:>8.1f}ms {pw:>6.1f}W "
                  f"{util:>5.0f}% {ips_w:>9.2f}")
            if best is None or r["fps_total"] > best[1]:
                best = (bs, r["fps_total"], pw, ips_w)
        if best:
            print(f"\n  peak: batch={best[0]} -> {best[1]:.1f} img/s @ "
                  f"{best[2]:.1f}W = {best[3]:.2f} img/s/W")
            print("  (เทียบกับ batch=1 latency-bound ด้านบน — รายงานคู่ใน thesis)")
            results["sweep"] = {str(bs): v for bs, v in sweep.items()}
            results["peak"] = {"batch": best[0], "img_s": best[1],
                               "power_w": best[2], "img_s_per_w": best[3]}

    # --- เขียนผลลง JSON ถ้าขอ ---
    if args.save:
        import json
        payload = {
            "gpu": gpu_name,
            "torch": torch.__version__,
            "precision": "FP16" if args.half else "FP32",
            "imgsz": args.imgsz,
            "idle_power_w": idle_w,
            "runs": args.runs,
            "iters": args.iters,
            "results": results,
        }
        with open(args.save, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n[ok] saved results -> {args.save}")

    # --- reset clock ถ้า lock ไว้ ---
    if args.lock_clock and power.handle is not None:
        try:
            power.pynvml.nvmlDeviceResetGpuLockedClocks(power.handle)
        except Exception:
            pass

    power.shutdown()


if __name__ == "__main__":
    main()
