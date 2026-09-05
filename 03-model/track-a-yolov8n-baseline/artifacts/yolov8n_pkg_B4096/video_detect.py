#!/usr/bin/env python3
# ============================================================================
# video_detect.py  —  DPU detection บนวิดีโอ + centroid tracker + line-count
# ----------------------------------------------------------------------------
# อ่านวิดีโอ -> ต่อเฟรม: preprocess -> DPU -> DFL decode + sigmoid + NMS ->
# วาดกล่อง + track id -> centroid tracker (greedy nearest) -> นับเมื่อ centroid
# ของ track ข้ามเส้น (ครั้งแรกต่อ track) -> overlay count -> เขียน MJPG .avi
#
# usage: python3 video_detect.py <model.xmodel> <in.avi> <out.avi>
#          [--conf 0.25] [--iou 0.45] [--axis x|y] [--line 0.5] [--max_frames 0]
#   --line = ตำแหน่งเส้นเป็นสัดส่วน 0..1 ของแกน (x: แนวตั้ง, y: แนวนอน)
# ============================================================================
import sys, os, glob, argparse, time
import numpy as np
import xir, vart
import cv2

REG_MAX, NC, IMGSZ = 16, 1, 640


def find_power_sensor():
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if open(os.path.join(h, "name")).read().strip() == "ina260_u14":
                p = os.path.join(h, "power1_input")
                if os.path.exists(p):
                    return p
        except Exception:
            pass
    c = glob.glob("/sys/class/hwmon/hwmon*/power1_input")
    return c[0] if c else None


def read_w(path):
    try:
        return int(open(path).read().strip()) / 1e6
    except Exception:
        return None


def get_dpu(g):
    subs = g.get_root_subgraph().toposort_child_subgraph()
    d = [s for s in subs if s.has_attr("device") and s.get_attr("device") == "DPU"]
    assert len(d) == 1
    return d[0]


def softmax(v, axis=-1):
    v = v - v.max(axis, keepdims=True); e = np.exp(v); return e / e.sum(axis, keepdims=True)


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def decode_head(feat, stride, conf_th):
    H, W, C = feat.shape
    box = feat[:, :, :4 * REG_MAX].reshape(H, W, 4, REG_MAX)
    cls = feat[:, :, 4 * REG_MAX:]
    sc = sigmoid(cls).max(-1)
    ys, xs = np.where(sc >= conf_th)
    if len(xs) == 0:
        return []
    bins = np.arange(REG_MAX, dtype=np.float32); out = []
    for y, x in zip(ys, xs):
        d = softmax(box[y, x], -1) @ bins
        cx, cy = x + 0.5, y + 0.5
        out.append(((cx - d[0]) * stride, (cy - d[1]) * stride,
                    (cx + d[2]) * stride, (cy + d[3]) * stride, float(sc[y, x])))
    return out


def nms(dets, iou_th):
    if not dets:
        return []
    a = np.array([d[:4] for d in dets], np.float32); s = np.array([d[4] for d in dets], np.float32)
    x1, y1, x2, y2 = a.T
    ar = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = s.argsort()[::-1]; keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ovr = inter / (ar[i] + ar[order[1:]] - inter + 1e-9)
        order = order[1:][ovr <= iou_th]
    return [dets[i] for i in keep]


class Tracker:
    """centroid tracker + line crossing counter (นับครั้งแรกที่ track ข้ามเส้น)."""
    def __init__(self, axis, line_pos, max_dist=60, max_missed=15):
        self.axis = axis; self.line = line_pos
        self.max_dist = max_dist; self.max_missed = max_missed
        self.tracks = {}   # id -> dict(cx,cy,missed,counted,prev_side)
        self.next_id = 1; self.count = 0

    def _side(self, cx, cy):
        v = cx if self.axis == "x" else cy
        return 1 if v >= self.line else -1

    def update(self, cents):
        # greedy match cents -> existing tracks
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
                tr["id"] = tid
            else:
                tr["missed"] += 1
                if tr["missed"] > self.max_missed:
                    del self.tracks[tid]
        # new tracks for unassigned detections
        out_ids = {}
        for i, (cx, cy) in enumerate(cents):
            if i in assigned:
                # หา track id ที่ถือ centroid นี้
                for tid, tr in self.tracks.items():
                    if abs(tr["cx"] - cx) < 1 and abs(tr["cy"] - cy) < 1:
                        out_ids[i] = tid; break
                continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = dict(cx=cx, cy=cy, missed=0, counted=False,
                                    prev_side=self._side(cx, cy), id=tid)
            out_ids[i] = tid
        return out_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xmodel"); ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--axis", choices=["x", "y"], default="x")
    ap.add_argument("--line", type=float, default=0.5)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--no_video", action="store_true",
                    help="ไม่เขียน output video (นับอย่างเดียว เร็ว+ประหยัด RAM)")
    args = ap.parse_args()

    g = xir.Graph.deserialize(args.xmodel)
    runner = vart.Runner.create_runner(get_dpu(g), "run")
    it, ot = runner.get_input_tensors(), runner.get_output_tensors()
    in_scale = float(2 ** int(it[0].get_attr("fix_point")))
    out_fp = [int(t.get_attr("fix_point")) for t in ot]
    stride_of = {IMGSZ // 8: 8, IMGSZ // 16: 16, IMGSZ // 32: 32}

    cap = cv2.VideoCapture(args.inp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_px = int(args.line * (W if args.axis == "x" else H))
    tr = Tracker(args.axis, line_px)
    vw = None if args.no_video else cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    print("[info] in=%s %dx%d fps=%.1f  line %s=%d  write_video=%s"
          % (args.inp, W, H, fps, args.axis, line_px, not args.no_video))

    psensor = find_power_sensor()
    pw_samples = []
    print("[info] power sensor: %s" % psensor)

    nfr = 0; t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        nfr += 1
        if psensor is not None:
            w = read_w(psensor)
            if w is not None:
                pw_samples.append(w)
        rgb = cv2.cvtColor(cv2.resize(frame, (IMGSZ, IMGSZ)), cv2.COLOR_BGR2RGB)
        inp = np.clip(np.round(rgb.astype(np.float32) / 255.0 * in_scale), -128, 127).astype(np.int8)[None]
        obuf = [np.empty(tuple(t.dims), np.int8) for t in ot]
        job = runner.execute_async([inp], obuf); runner.wait(job)
        dets = []
        for t, buf, fp in zip(ot, obuf, out_fp):
            deq = buf.astype(np.float32) * (2.0 ** (-fp))
            dets.extend(decode_head(deq[0], stride_of[buf.shape[1]], args.conf))
        kept = nms(dets, args.iou)
        sx, sy = W / IMGSZ, H / IMGSZ
        cents = []
        pix = []
        for (x1, y1, x2, y2, s) in kept:
            X1, Y1, X2, Y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
            cents.append(((X1 + X2) / 2.0, (Y1 + Y2) / 2.0)); pix.append((X1, Y1, X2, Y2, s))
        ids = tr.update(cents)
        # draw
        if args.axis == "x":
            cv2.line(frame, (line_px, 0), (line_px, H), (0, 0, 255), 2)
        else:
            cv2.line(frame, (0, line_px), (W, line_px), (0, 0, 255), 2)
        for i, (X1, Y1, X2, Y2, s) in enumerate(pix):
            cv2.rectangle(frame, (X1, Y1), (X2, Y2), (0, 255, 0), 2)
            cv2.putText(frame, "#%d %.2f" % (ids.get(i, 0), s), (X1, max(0, Y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "COUNT: %d" % tr.count, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "frame %d" % nfr, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if vw is not None:
            vw.write(frame)
        if args.max_frames and nfr >= args.max_frames:
            break
        if nfr % 50 == 0:
            print("  frame %d  count=%d  (%.1f fps proc)" % (nfr, tr.count, nfr / (time.time() - t0)))
    cap.release()
    if vw is not None:
        vw.release()
    dt = time.time() - t0
    fps_e2e = nfr / dt
    print("[ok] wrote %s  frames=%d  COUNT=%d  proc=%.2f fps (%.1fs)"
          % (args.out, nfr, tr.count, fps_e2e, dt))
    if pw_samples:
        avg = sum(pw_samples) / len(pw_samples)
        mx = max(pw_samples); mn = min(pw_samples)
        print("========= REAL-WORKLOAD POWER (video pipeline) =========")
        print("  power avg=%.3f W  peak=%.3f W  min=%.3f W  (n=%d)"
              % (avg, mx, mn, len(pw_samples)))
        print("  end-to-end throughput = %.2f FPS" % fps_e2e)
        print("  efficiency = %.2f FPS/W  (real application, whole SOM)" % (fps_e2e / avg))
        print("========================================================")


if __name__ == "__main__":
    main()
