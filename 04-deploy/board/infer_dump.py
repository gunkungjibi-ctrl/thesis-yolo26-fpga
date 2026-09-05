#!/usr/bin/env python3
# ============================================================================
# infer_dump.py  —  รัน DPU detection ทั้งโฟลเดอร์ -> dump predictions เป็น JSON
# ----------------------------------------------------------------------------
# ใช้เก็บ prediction ของโมเดล INT8 บนบอร์ด เพื่อส่งกลับ host ไปคำนวณ mAP
# (host เทียบกับ GT label + รัน float ด้วย pipeline เดียวกันหา accuracy drop).
#
# output JSON: { "<image_basename>": [[x1,y1,x2,y2,score], ...], ... }
#   พิกัด = พิกเซลของภาพต้นฉบับ (xyxy), score = sigmoid(cls) ของ class package
# preprocess/decode ตรงกับ yolo_dpu_detect.py เป๊ะ (plain resize640, /255, DFL).
#
# usage: python3 infer_dump.py <model.xmodel> <images_dir> <out.json> [--conf 0.001] [--iou 0.7]
#   NOTE: conf ต่ำ (0.001) สำหรับคำนวณ mAP (ต้องได้ทุก candidate); iou 0.7 แบบ ultralytics
# ============================================================================
import os, sys, glob, json, argparse
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


def nms(dets, iou_th):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xmodel"); ap.add_argument("images_dir"); ap.add_argument("out_json")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=300)
    args = ap.parse_args()

    g = xir.Graph.deserialize(args.xmodel)
    runner = vart.Runner.create_runner(get_dpu(g), "run")
    it, ot = runner.get_input_tensors(), runner.get_output_tensors()
    in_scale = float(2 ** int(it[0].get_attr("fix_point")))
    out_fp = [int(t.get_attr("fix_point")) for t in ot]
    stride_of = {IMGSZ // 8: 8, IMGSZ // 16: 16, IMGSZ // 32: 32}

    files = sorted(glob.glob(os.path.join(args.images_dir, "*.jpg")) +
                   glob.glob(os.path.join(args.images_dir, "*.png")))
    print("[info] %d images in %s" % (len(files), args.images_dir))
    results = {}
    for n, fp in enumerate(files):
        bgr = load_bgr(fp)
        if bgr is None:
            print("[warn] skip unreadable %s" % fp); continue
        oh, ow = bgr.shape[:2]
        inp = preprocess(bgr, in_scale)
        obuf = [np.empty(tuple(t.dims), np.int8) for t in ot]
        job = runner.execute_async([inp], obuf); runner.wait(job)
        dets = []
        for t, buf, fpx in zip(ot, obuf, out_fp):
            deq = buf.astype(np.float32) * (2.0 ** (-fpx))
            dets.extend(decode_head(deq[0], stride_of[buf.shape[1]], args.conf))
        kept = nms(dets, args.iou)[:args.max_det]
        sx, sy = ow / IMGSZ, oh / IMGSZ
        boxes = [[float(x1 * sx), float(y1 * sy), float(x2 * sx), float(y2 * sy), float(s)]
                 for (x1, y1, x2, y2, s) in kept]
        results[os.path.basename(fp)] = boxes
        if (n + 1) % 10 == 0:
            print("  %d/%d" % (n + 1, len(files)))
    with open(args.out_json, "w") as f:
        json.dump(results, f)
    tot = sum(len(v) for v in results.values())
    print("[ok] wrote %s  (%d images, %d total boxes)" % (args.out_json, len(results), tot))


if __name__ == "__main__":
    main()
