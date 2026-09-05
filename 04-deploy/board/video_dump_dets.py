#!/usr/bin/env python3
# ============================================================================
# video_dump_dets.py  —  รัน DPU detection ทั้งวิดีโอ -> dump กล่องต่อเฟรมเป็น JSON
# ----------------------------------------------------------------------------
# บอร์ดรัน "ครั้งเดียว" ได้ผล detection ครบทุกเฟรม แล้วเอาไปทำ tracking/counting/
# จูนเส้นบน host แบบออฟไลน์ได้ไม่จำกัด (ไม่ต้องรันบอร์ด 30 นาทีซ้ำทุกครั้ง).
#
# output JSON: {"w":W,"h":H,"fps":fps,"frames":[[[x1,y1,x2,y2,score],...], ...]}
#   frames[i] = list ของกล่องในเฟรม i (พิกัดพิกเซลภาพจริง, หลัง NMS)
#
# usage: python3 video_dump_dets.py <model.xmodel> <in.avi> <out.json>
#          [--conf 0.25] [--iou 0.45] [--max_frames 0]
# ============================================================================
import sys, json, argparse, time
import numpy as np
import xir, vart
import cv2

REG_MAX, NC, IMGSZ = 16, 1, 640


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xmodel"); ap.add_argument("inp"); ap.add_argument("out_json")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max_frames", type=int, default=0)
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
    sx, sy = W / IMGSZ, H / IMGSZ
    frames = []
    nfr = 0; t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        nfr += 1
        rgb = cv2.cvtColor(cv2.resize(frame, (IMGSZ, IMGSZ)), cv2.COLOR_BGR2RGB)
        inp = np.clip(np.round(rgb.astype(np.float32) / 255.0 * in_scale), -128, 127).astype(np.int8)[None]
        obuf = [np.empty(tuple(t.dims), np.int8) for t in ot]
        job = runner.execute_async([inp], obuf); runner.wait(job)
        dets = []
        for t, buf, fp in zip(ot, obuf, out_fp):
            deq = buf.astype(np.float32) * (2.0 ** (-fp))
            dets.extend(decode_head(deq[0], stride_of[buf.shape[1]], args.conf))
        kept = nms(dets, args.iou)
        frames.append([[round(x1 * sx, 1), round(y1 * sy, 1), round(x2 * sx, 1),
                        round(y2 * sy, 1), round(s, 3)] for (x1, y1, x2, y2, s) in kept])
        if args.max_frames and nfr >= args.max_frames:
            break
        if nfr % 500 == 0:
            print("  frame %d  (%.1f fps proc)" % (nfr, nfr / (time.time() - t0)), flush=True)
    cap.release()
    json.dump({"w": W, "h": H, "fps": fps, "frames": frames}, open(args.out_json, "w"))
    tot = sum(len(f) for f in frames)
    print("[ok] wrote %s  frames=%d  total_boxes=%d  proc=%.2f fps"
          % (args.out_json, nfr, tot, nfr / (time.time() - t0)), flush=True)


if __name__ == "__main__":
    main()
