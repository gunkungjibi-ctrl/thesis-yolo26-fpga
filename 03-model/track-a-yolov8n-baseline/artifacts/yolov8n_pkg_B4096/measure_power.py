#!/usr/bin/env python3
# ============================================================================
# measure_power.py  —  KV260 SOM power (INA260 u14) idle vs DPU load + FPS/W
# ----------------------------------------------------------------------------
# อ่าน power1_input ของ hwmon ชื่อ "ina260_u14" (ราง SOM, หน่วย µW) —
# sample idle ก่อน แล้วรัน `xdputil benchmark <xmodel> <threads>` เป็น subprocess
# ระหว่างนั้น sample power ต่อเนื่อง -> รายงาน idle/load avg+max, delta, และ
# FPS/W (ดึง FPS จาก output ของ benchmark).
#
# usage: python3 measure_power.py <model.xmodel> [--threads 4] [--idle_s 5]
# ============================================================================
import os, sys, time, glob, re, subprocess, threading
import argparse


def find_ina260():
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            nm = open(os.path.join(h, "name")).read().strip()
        except Exception:
            continue
        if nm == "ina260_u14":
            p = os.path.join(h, "power1_input")
            if os.path.exists(p):
                return p
    # fallback: first power1_input ที่เจอ
    c = glob.glob("/sys/class/hwmon/hwmon*/power1_input")
    return c[0] if c else None


def read_w(path):
    try:
        return int(open(path).read().strip()) / 1e6  # µW -> W
    except Exception:
        return None


def sample_loop(path, stop_evt, out, period=0.1):
    while not stop_evt.is_set():
        w = read_w(path)
        if w is not None:
            out.append(w)
        time.sleep(period)


def summ(name, arr):
    if not arr:
        print("  %-12s (no samples)" % name); return None, None
    a = sorted(arr); n = len(a)
    avg = sum(a) / n; mx = a[-1]; mn = a[0]
    print("  %-12s avg=%.3f W  max=%.3f W  min=%.3f W  (n=%d)" % (name, avg, mx, mn, n))
    return avg, mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xmodel")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--idle_s", type=float, default=5.0)
    args = ap.parse_args()

    path = find_ina260()
    if not path:
        sys.exit("[FATAL] no INA260 power sensor found")
    print("[info] power sensor: %s" % path)

    # 1) idle
    print("[phase] sampling idle for %.1f s ..." % args.idle_s)
    idle = []
    t_end = time.time() + args.idle_s
    while time.time() < t_end:
        w = read_w(path)
        if w is not None:
            idle.append(w)
        time.sleep(0.1)
    idle_avg, _ = summ("idle", idle)

    # 2) load: benchmark subprocess + sampling thread
    print("[phase] running xdputil benchmark (threads=%d) under power sampling ..."
          % args.threads)
    load = []
    stop = threading.Event()
    th = threading.Thread(target=sample_loop, args=(path, stop, load, 0.1))
    th.start()
    proc = subprocess.Popen(
        ["xdputil", "benchmark", args.xmodel, str(args.threads)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    out_txt = proc.communicate()[0]
    stop.set(); th.join()
    load_avg, load_max = summ("load", load)

    # 3) ดึง FPS จาก benchmark output
    m = re.search(r"FPS\s*=\s*([0-9.]+)", out_txt)
    fps = float(m.group(1)) if m else None
    if fps is None:
        print("[warn] could not parse FPS; benchmark tail:")
        print("\n".join(out_txt.strip().splitlines()[-5:]))

    print("\n==================== POWER / EFFICIENCY ====================")
    if idle_avg is not None:
        print("  idle power        : %.3f W" % idle_avg)
    if load_avg is not None:
        print("  load power (avg)  : %.3f W" % load_avg)
    if load_max is not None:
        print("  load power (peak) : %.3f W" % load_max)
    if idle_avg is not None and load_avg is not None:
        print("  dynamic (load-idle): %.3f W" % (load_avg - idle_avg))
    if fps is not None:
        print("  DPU throughput    : %.2f FPS (threads=%d)" % (fps, args.threads))
        if load_avg:
            print("  efficiency        : %.2f FPS/W (vs load avg)" % (fps / load_avg))
        if idle_avg is not None and (load_avg - idle_avg) > 0:
            print("  efficiency (dyn)  : %.2f FPS/W (vs dynamic power)"
                  % (fps / (load_avg - idle_avg)))
    print("============================================================")


if __name__ == "__main__":
    main()
