#!/usr/bin/env python3
# ============================================================================
# bench_preproc.py — M13: วัด preprocessing แยกโหมด numpy / lut / hw + ตรวจว่าผลตรงกันทุกไบต์
# ----------------------------------------------------------------------------
# ไม่ต้องใช้ DPU/VART → รันได้ทั้งบนบอร์ดและบน host (โหมด hw จะ fallback ถ้าไม่มี xclbin)
#
# usage:
#   python3 bench_preproc.py <image_or_video> [--modes numpy,lut,hw] [--iters 100]
#                            [--fixpos 6] [--src WxH] [--verify]
#   --src   : บังคับขนาดเฟรมต้นทางก่อนวัด (เช่น 640x360 = คลิปนับจริง, 1920x1080 = กล้อง)
#   --verify: เทียบ output ทุกโหมดกับ numpy (baseline) → ต้อง mismatch = 0
# ============================================================================
import sys, time, argparse
import numpy as np
import cv2
from preproc_lib import make_preprocessor, MODES


def load_frame(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is not None:
        return im
    cap = cv2.VideoCapture(path)
    ok, fr = cap.read(); cap.release()
    if not ok:
        sys.exit("cannot read %s" % path)
    return fr


def stats(name, arr):
    a = np.array(arr) * 1000.0
    print("  %-8s mean=%8.3f  median=%8.3f  std=%7.3f  min=%8.3f  max=%8.3f ms"
          % (name, a.mean(), np.median(a), a.std(), a.min(), a.max()))
    return float(np.median(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--fixpos", type=int, default=6, help="in_fixpos ของ xmodel (yolov8n pkg = 6)")
    ap.add_argument("--src", default=None, help="WxH บังคับขนาดต้นทาง")
    ap.add_argument("--xclbin", default=None)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    bgr = load_frame(args.inp)
    if args.src:
        w, h = (int(v) for v in args.src.split("x"))
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    bgr = np.ascontiguousarray(bgr)
    in_scale = float(2 ** args.fixpos)
    print("[bench_preproc] frame %dx%d  fixpos=%d  iters=%d" % (bgr.shape[1], bgr.shape[0], args.fixpos, args.iters))

    ref = None; med = {}
    for mode in args.modes.split(","):
        f = make_preprocessor(mode, args.xclbin)
        t = []
        for i in range(args.warmup + args.iters):
            s = time.perf_counter(); out = f(bgr, in_scale); e = time.perf_counter()
            if i >= args.warmup:
                t.append(e - s)
        med[mode] = stats(mode, t)
        if args.verify:
            if ref is None:
                ref = out
            else:
                d = np.abs(out.astype(np.int16) - ref.astype(np.int16))
                print("           verify vs numpy: mismatch=%d/%d  max|diff|=%d  -> %s"
                      % (int((d != 0).sum()), d.size, int(d.max()), "OK" if d.max() == 0 else "FAIL"))
    if "numpy" in med:
        for m, v in med.items():
            if m != "numpy":
                print("[speedup] %s vs numpy: %.1fx  (%.2f -> %.2f ms)" % (m, med["numpy"] / v, med["numpy"], v))


if __name__ == "__main__":
    main()
