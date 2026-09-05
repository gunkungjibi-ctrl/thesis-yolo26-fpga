#!/usr/bin/env python3
# render_offline.py (HOST) — เรนเดอร์วิดีโอผลนับจาก dets JSON + ต้นฉบับ (calibrated line)
# วาดกล่อง+track id+เส้นนับ+COUNT ทับเฟรมต้นฉบับ แล้วเขียน mp4. ไม่ต้องรันบอร์ด.
# usage: python render_offline.py <src.avi> <dets.json> <out.mp4>
#          [--axis x|y] [--line 0.40] [--max_dist 60] [--max_missed 15]
import sys, json, argparse
import cv2

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from count_offline import Tracker, centroids  # reuse tracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dets"); ap.add_argument("out")
    ap.add_argument("--axis", choices=["x", "y"], default="x")
    ap.add_argument("--line", type=float, default=0.40)
    ap.add_argument("--max_dist", type=float, default=60)
    ap.add_argument("--max_missed", type=int, default=15)
    ap.add_argument("--conf_min", type=float, default=0.25)
    ap.add_argument("--roi_top", type=float, default=0.0,
                    help="ตัดกล่องที่ center-y < roi_top*H (โซนบน=คน/เครื่องจักร)")
    a = ap.parse_args()

    data = json.load(open(a.dets))
    W, H, fps = data["w"], data["h"], data["fps"]
    roi_y = int(a.roi_top * H)

    def keep(b):
        return b[4] >= a.conf_min and (b[1] + b[3]) / 2 >= roi_y

    frames_boxes = [[b for b in fb if keep(b)] for fb in data["frames"]]
    line_px = int(a.line * (W if a.axis == "x" else H))
    tr = Tracker(a.axis, line_px, a.max_dist, a.max_missed)

    cap = cv2.VideoCapture(a.src)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or n >= len(frames_boxes):
            break
        boxes = frames_boxes[n]
        cents = centroids(boxes)
        # เก็บ count ก่อน step เพื่อรู้ว่าเฟรมนี้มีการนับเพิ่มไหม (ไว้ debug)
        tr.step(cents)
        # วาดเส้นนับ (แดง) + เส้น ROI (เหลือง, ตัดโซนบน)
        if a.axis == "x":
            cv2.line(frame, (line_px, roi_y), (line_px, H), (0, 0, 255), 2)
        else:
            cv2.line(frame, (0, line_px), (W, line_px), (0, 0, 255), 2)
        if roi_y > 0:
            cv2.line(frame, (0, roi_y), (W, roi_y), (0, 255, 255), 1)
            cv2.putText(frame, "ROI", (W - 45, roi_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        for b in boxes:
            x1, y1, x2, y2, s = int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "%.2f" % s, (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "COUNT: %d" % tr.count, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "frame %d  line x=%d" % (n + 1, line_px), (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(frame)
        n += 1
        if n % 1000 == 0:
            print("  rendered %d/%d  count=%d" % (n, len(frames_boxes), tr.count), flush=True)
    cap.release(); vw.release()
    print("[ok] wrote %s  frames=%d  final COUNT=%d" % (a.out, n, tr.count))


if __name__ == "__main__":
    main()
