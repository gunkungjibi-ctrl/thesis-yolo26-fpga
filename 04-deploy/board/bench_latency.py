#!/usr/bin/env python3
# ============================================================================
# bench_latency.py  —  latency breakdown (preproc / DPU / decode+NMS) + e2e FPS
# ----------------------------------------------------------------------------
# วัดเวลาต่อ stage ของ pipeline detection บน KV260 เพื่อลงตาราง Results:
#   - preprocess (cv2 resize/cvt/quant, บน PS)
#   - DPU execute (execute_async + wait)
#   - decode + NMS (DFL softmax, sigmoid, NMS บน PS)
#   - end-to-end (รวมทุก stage, ยกเว้น imread/imwrite disk)
# warmup N รอบ แล้ววัด M รอบ รายงาน mean/median/std + FPS
#
# usage: python3 bench_latency.py <model.xmodel> <image> [--warmup 10] [--iters 100]
# ============================================================================
import sys, time, argparse
import numpy as np
import xir, vart
try:
    import cv2; _CV2 = True
except Exception:
    _CV2 = False
    from PIL import Image

REG_MAX, NC, IMGSZ = 16, 1, 640


def get_dpu(g):
    subs = g.get_root_subgraph().toposort_child_subgraph()
    d = [s for s in subs if s.has_attr("device") and s.get_attr("device") == "DPU"]
    assert len(d) == 1
    return d[0]


def softmax(v, axis=-1):
    v = v - v.max(axis, keepdims=True); e = np.exp(v)
    return e / e.sum(axis, keepdims=True)


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
    bins = np.arange(REG_MAX, dtype=np.float32)
    out = []
    for y, x in zip(ys, xs):
        d = softmax(box[y, x], -1) @ bins
        cx, cy = x + 0.5, y + 0.5
        out.append(((cx - d[0]) * stride, (cy - d[1]) * stride,
                    (cx + d[2]) * stride, (cy + d[3]) * stride, float(sc[y, x])))
    return out


def nms(dets, iou_th=0.45):
    if not dets:
        return []
    a = np.array([d[:4] for d in dets], np.float32)
    s = np.array([d[4] for d in dets], np.float32)
    x1, y1, x2, y2 = a.T
    ar = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = s.argsort()[::-1]; keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None); h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        ovr = inter / (ar[i] + ar[order[1:]] - inter + 1e-9)
        order = order[1:][ovr <= iou_th]
    return [dets[i] for i in keep]


def load_bgr(p):
    if _CV2:
        return cv2.imread(p, cv2.IMREAD_COLOR)
    return np.array(Image.open(p).convert("RGB"))[:, :, ::-1].copy()


def preprocess(bgr, in_scale):
    if _CV2:
        r = cv2.cvtColor(cv2.resize(bgr, (IMGSZ, IMGSZ)), cv2.COLOR_BGR2RGB)
    else:
        r = np.array(Image.fromarray(bgr[:, :, ::-1]).resize((IMGSZ, IMGSZ)))
    q = np.clip(np.round(r.astype(np.float32) / 255.0 * in_scale), -128, 127).astype(np.int8)
    return q[np.newaxis, ...]


def stats(name, arr):
    a = np.array(arr) * 1000.0  # ms
    print("  %-14s mean=%7.3f  median=%7.3f  std=%6.3f  min=%7.3f  max=%7.3f ms"
          % (name, a.mean(), np.median(a), a.std(), a.min(), a.max()))
    return a.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xmodel"); ap.add_argument("image")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--preproc", default="numpy", choices=("numpy", "lut", "hw"),
                    help="M13: numpy=โค้ดเดิม (baseline), lut=cv2.LUT, hw=PL accelerator (ดู preproc_lib.py)")
    ap.add_argument("--xclbin", default=None, help="xclbin ที่มี preproc_accel (โหมด hw)")
    args = ap.parse_args()
    if args.preproc != "numpy":
        from preproc_lib import make_preprocessor
        pp = make_preprocessor(args.preproc, args.xclbin)
    else:
        pp = preprocess
    print("[info] preproc mode = %s" % args.preproc)

    g = xir.Graph.deserialize(args.xmodel)
    runner = vart.Runner.create_runner(get_dpu(g), "run")
    it, ot = runner.get_input_tensors(), runner.get_output_tensors()
    in_scale = float(2 ** int(it[0].get_attr("fix_point")))
    out_fp = [int(t.get_attr("fix_point")) for t in ot]
    stride_of = {IMGSZ // 8: 8, IMGSZ // 16: 16, IMGSZ // 32: 32}

    bgr = load_bgr(args.image)
    t_pre, t_dpu, t_dec, t_e2e = [], [], [], []

    for i in range(args.warmup + args.iters):
        rec = i >= args.warmup
        s0 = time.perf_counter()
        inp = pp(bgr, in_scale)
        s1 = time.perf_counter()
        obuf = [np.empty(tuple(t.dims), np.int8) for t in ot]
        job = runner.execute_async([inp], obuf)
        runner.wait(job)
        s2 = time.perf_counter()
        dets = []
        for t, buf, fp in zip(ot, obuf, out_fp):
            deq = buf.astype(np.float32) * (2.0 ** (-fp))
            stride = stride_of[buf.shape[1]]
            dets.extend(decode_head(deq[0], stride, args.conf))
        _ = nms(dets)
        s3 = time.perf_counter()
        if rec:
            t_pre.append(s1 - s0); t_dpu.append(s2 - s1)
            t_dec.append(s3 - s2); t_e2e.append(s3 - s0)

    print("[latency] over %d iters (after %d warmup), image=%s"
          % (args.iters, args.warmup, args.image))
    stats("preprocess", t_pre)
    stats("DPU exec", t_dpu)
    stats("decode+NMS", t_dec)
    e2e = stats("end-to-end", t_e2e)
    print("[throughput] end-to-end FPS = %.2f  (DPU-only FPS = %.2f)"
          % (1000.0 / e2e, 1.0 / np.mean(t_dpu)))


if __name__ == "__main__":
    main()
