#!/usr/bin/env python3
# ============================================================================
# count_offline.py (HOST) — tracking + line-crossing counting จาก dets JSON
# ----------------------------------------------------------------------------
# อ่านผล detection ต่อเฟรม (จาก video_dump_dets.py) แล้วทำ centroid tracker +
# line-crossing count บน host — จูนเส้น/พารามิเตอร์ได้ไม่จำกัด ไม่ต้องรันบอร์ดซ้ำ.
# sweep ได้หลายตำแหน่งเส้น/สองแกน เพื่อหาค่าที่เข้าใกล้ GT.
#
# usage:
#   python count_offline.py <dets.json> [--axis x|y] [--line 0.5]
#             [--max_dist 60] [--max_missed 15] [--gt 215]
#   python count_offline.py <dets.json> --sweep      # ลองหลายเส้น/แกน เทียบ GT
# ============================================================================
import sys, json, argparse
import numpy as np


class Tracker:
    def __init__(self, axis, line_pos, max_dist=60, max_missed=15):
        self.axis = axis; self.line = line_pos
        self.max_dist = max_dist; self.max_missed = max_missed
        self.tracks = {}; self.next_id = 1; self.count = 0

    def _side(self, cx, cy):
        v = cx if self.axis == "x" else cy
        return 1 if v >= self.line else -1

    def step(self, cents):
        assigned = set()
        for tid, tr in list(self.tracks.items()):
            best, bd = None, self.max_dist
            for i, (cx, cy) in enumerate(cents):
                if i in assigned:
                    continue
                d = ((cx - tr["cx"]) ** 2 + (cy - tr["cy"]) ** 2) ** 0.5
                if d < bd:
                    bd = d; best = i
            if best is not None:
                cx, cy = cents[best]; assigned.add(best)
                side = self._side(cx, cy)
                if (not tr["counted"]) and side != tr["prev_side"]:
                    self.count += 1; tr["counted"] = True
                tr.update(cx=cx, cy=cy, missed=0, prev_side=side)
            else:
                tr["missed"] += 1
                if tr["missed"] > self.max_missed:
                    del self.tracks[tid]
        for i, (cx, cy) in enumerate(cents):
            if i in assigned:
                continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = dict(cx=cx, cy=cy, missed=0, counted=False,
                                    prev_side=self._side(cx, cy))


def centroids(frame_boxes):
    return [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in frame_boxes]


def run(data, axis, line_frac, max_dist, max_missed):
    W, H = data["w"], data["h"]
    line_px = line_frac * (W if axis == "x" else H)
    tr = Tracker(axis, line_px, max_dist, max_missed)
    for fb in data["frames"]:
        tr.step(centroids(fb))
    return tr.count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dets"); ap.add_argument("--axis", choices=["x", "y"], default="x")
    ap.add_argument("--line", type=float, default=0.5)
    ap.add_argument("--max_dist", type=float, default=60)
    ap.add_argument("--max_missed", type=int, default=15)
    ap.add_argument("--gt", type=int, default=215)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    data = json.load(open(args.dets))
    print("video %dx%d fps=%.1f frames=%d" % (data["w"], data["h"], data["fps"], len(data["frames"])))

    if args.sweep:
        print("\n=== SWEEP (เทียบ GT=%d) ===" % args.gt)
        for axis in ("x", "y"):
            for lf in (0.3, 0.4, 0.5, 0.6, 0.7):
                for md in (40, 60, 90):
                    c = run(data, axis, lf, md, args.max_missed)
                    err = abs(c - args.gt) / args.gt * 100
                    flag = "  <== ใกล้สุด" if err < 10 else ""
                    print("  axis=%s line=%.1f max_dist=%3d -> count=%4d  err=%5.1f%%%s"
                          % (axis, lf, md, c, err, flag))
        return

    c = run(data, args.axis, args.line, args.max_dist, args.max_missed)
    err = abs(c - args.gt) / args.gt * 100
    print("\naxis=%s line=%.2f max_dist=%.0f max_missed=%d" %
          (args.axis, args.line, args.max_dist, args.max_missed))
    print("COUNT = %d   GT = %d   error = %.1f%%" % (c, args.gt, err))


if __name__ == "__main__":
    main()
