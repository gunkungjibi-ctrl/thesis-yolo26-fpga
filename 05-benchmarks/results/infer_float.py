#!/usr/bin/env python3
# ============================================================================
# infer_float.py  (HOST, torch env)  —  float baseline ด้วย pipeline เดียวกับบอร์ด
# ----------------------------------------------------------------------------
# โหลด yolov8n_leaky_pkg_ft.pt -> raw-head wrapper (เหมือน quantize BackboneHead:
# C2f.chunk->slice, เดิน layer เอง, return cat(cv2,cv3)/stride) -> plain resize640,
# /255, decode DFL+sigmoid+NMS เดียวกับ infer_dump.py -> predictions JSON.
# ใช้เทียบกับ INT8 board เพื่อแยกผลของ quantization ล้วน ๆ (accuracy drop).
#
# usage: python infer_float.py <weights.pt> <images_dir> <out.json> [--conf 0.001] [--iou 0.7]
# run with the D: torch env python (ultralytics 8.4.71, torch cpu).
# ============================================================================
import os, sys, glob, json, argparse
import numpy as np
import torch, torch.nn as nn
import cv2

REG_MAX, NC, IMGSZ = 16, 1, 640


class BackboneHead(nn.Module):
    def __init__(self, ultra_model):
        super().__init__()
        self.model = ultra_model
        det = self.model.model[-1]; self.det = det; self.save = self.model.save
        det.end2end = False

    def forward(self, x):
        y = []
        for m in self.model.model:
            if m is self.det:
                break
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else \
                    [x if j == -1 else y[j] for j in m.f]
            x = m(x); y.append(x if m.i in self.save else None)
        det = self.det
        feats = [x if j == -1 else y[j] for j in det.f]
        return tuple(torch.cat((det.cv2[i](feats[i]), det.cv3[i](feats[i])), 1)
                     for i in range(det.nl))


def build(weights):
    from ultralytics import YOLO
    from ultralytics.nn.modules.block import C2f
    def _c2f(self, x):
        t = self.cv1(x)
        y = [t[:, :self.c, :, :], t[:, self.c:, :, :]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    C2f.forward = _c2f
    y = YOLO(weights); m = BackboneHead(y.model); m.eval()
    return m


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
    ap.add_argument("weights"); ap.add_argument("images_dir"); ap.add_argument("out_json")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=300)
    args = ap.parse_args()

    model = build(args.weights)
    stride_of = {IMGSZ // 8: 8, IMGSZ // 16: 16, IMGSZ // 32: 32}
    files = sorted(glob.glob(os.path.join(args.images_dir, "*.jpg")) +
                   glob.glob(os.path.join(args.images_dir, "*.png")))
    print("[info] %d images" % len(files))
    results = {}
    for n, fp in enumerate(files):
        bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        oh, ow = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (IMGSZ, IMGSZ)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        with torch.no_grad():
            outs = model(x)
        dets = []
        for t in outs:
            feat = t[0].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
            dets.extend(decode_head(feat, stride_of[feat.shape[0]], args.conf))
        kept = nms(dets, args.iou)[:args.max_det]
        sx, sy = ow / IMGSZ, oh / IMGSZ
        results[os.path.basename(fp)] = [
            [float(a * sx), float(b * sy), float(c * sx), float(d * sy), float(s)]
            for (a, b, c, d, s) in kept]
        if (n + 1) % 10 == 0:
            print("  %d/%d" % (n + 1, len(files)))
    json.dump(results, open(args.out_json, "w"))
    print("[ok] wrote %s (%d imgs, %d boxes)"
          % (args.out_json, len(results), sum(len(v) for v in results.values())))


if __name__ == "__main__":
    main()
