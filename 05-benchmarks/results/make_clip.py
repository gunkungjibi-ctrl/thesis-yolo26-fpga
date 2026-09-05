#!/usr/bin/env python3
# make_clip.py (HOST, cv2) — ตัดคลิปสั้นจาก source แล้วเขียนเป็น MJPG .avi
# ให้บอร์ดอ่านได้ชัวร์ (ไม่ต้องมี H.264 decoder).
# usage: python make_clip.py <src.mp4> <out.avi> [--seconds 20] [--start 0]
import sys, argparse, cv2

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--start", type=float, default=0.0)
a = ap.parse_args()

cap = cv2.VideoCapture(a.src)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
start_f = int(a.start * fps); n_f = int(a.seconds * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
print("src fps=%.2f size=%dx%d -> writing %d frames (%.1fs) MJPG" % (fps, W, H, n_f, a.seconds))
k = 0
while k < n_f:
    ok, fr = cap.read()
    if not ok:
        break
    vw.write(fr); k += 1
cap.release(); vw.release()
print("[ok] wrote %s (%d frames)" % (a.out, k))
