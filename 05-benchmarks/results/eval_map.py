#!/usr/bin/env python3
# ============================================================================
# eval_map.py  (HOST)  —  คำนวณ mAP50 / mAP50-95 จาก predictions JSON + GT labels
# ----------------------------------------------------------------------------
# single class (package). COCO-style AP (101-point interp) ที่ IoU 0.50:0.95:0.05.
# predictions JSON: { "<img>.jpg": [[x1,y1,x2,y2,score], ...] }  (พิกัดพิกเซลภาพต้นฉบับ)
# GT labels: YOLO txt ต่อภาพ (class cx cy w h  แบบ normalized 0..1)
#
# usage: python eval_map.py <pred.json> <images_dir> <labels_dir>
# ============================================================================
import os, sys, json, glob
import numpy as np
from PIL import Image


def load_gt(images_dir, labels_dir):
    gt = {}
    for ip in glob.glob(os.path.join(images_dir, "*.jpg")) + \
              glob.glob(os.path.join(images_dir, "*.png")):
        base = os.path.basename(ip)
        w, h = Image.open(ip).size
        lp = os.path.join(labels_dir, os.path.splitext(base)[0] + ".txt")
        boxes = []
        if os.path.exists(lp):
            for line in open(lp):
                p = line.split()
                if len(p) < 5:
                    continue
                coords = list(map(float, p[1:]))
                if len(coords) == 4:
                    # YOLO bbox format: cx cy w h
                    cx, cy, bw, bh = coords
                    boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                  (cx + bw / 2) * w, (cy + bh / 2) * h])
                else:
                    # polygon segmentation: x1 y1 x2 y2 ... -> bounding box
                    xs = np.array(coords[0::2]) * w
                    ys = np.array(coords[1::2]) * h
                    boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
        gt[base] = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), np.float32)
    return gt


def iou_matrix(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), np.float32)
    px1, py1, px2, py2 = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
    gx1, gy1, gx2, gy2 = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    ix1 = np.maximum(px1, gx1); iy1 = np.maximum(py1, gy1)
    ix2 = np.minimum(px2, gx2); iy2 = np.minimum(py2, gy2)
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    pa = (px2 - px1) * (py2 - py1); ga = (gx2 - gx1) * (gy2 - gy1)
    return inter / (pa + ga - inter + 1e-9)


def ap_at(preds_by_img, gt, iou_th):
    # รวมทุก prediction ทั้ง dataset เรียงตาม score จากมากไปน้อย
    entries = []  # (score, img, box_idx)
    n_gt = 0
    for img, g in gt.items():
        n_gt += len(g)
    for img, boxes in preds_by_img.items():
        for b in boxes:
            entries.append((b[4], img, b[:4]))
    entries.sort(key=lambda e: -e[0])
    if n_gt == 0:
        return 0.0
    matched = {img: np.zeros(len(g), bool) for img, g in gt.items()}
    tp = np.zeros(len(entries)); fp = np.zeros(len(entries))
    for i, (sc, img, box) in enumerate(entries):
        g = gt.get(img, np.zeros((0, 4), np.float32))
        if len(g) == 0:
            fp[i] = 1; continue
        ious = iou_matrix(np.array([box], np.float32), g)[0]
        j = int(np.argmax(ious))
        if ious[j] >= iou_th and not matched[img][j]:
            tp[i] = 1; matched[img][j] = True
        else:
            fp[i] = 1
    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    rec = tp_c / (n_gt + 1e-9)
    prec = tp_c / (tp_c + fp_c + 1e-9)
    # COCO 101-point interpolation
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        p = prec[rec >= r].max() if np.any(rec >= r) else 0.0
        ap += p / 101.0
    return ap


def prec_rec_f1(preds_by_img, gt, iou_th=0.5, conf_th=0.25):
    tp = fp = 0; n_gt = sum(len(g) for g in gt.values())
    for img, g in gt.items():
        boxes = [b for b in preds_by_img.get(img, []) if b[4] >= conf_th]
        boxes.sort(key=lambda b: -b[4])
        used = np.zeros(len(g), bool)
        for b in boxes:
            if len(g) == 0:
                fp += 1; continue
            ious = iou_matrix(np.array([b[:4]], np.float32), g)[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_th and not used[j]:
                tp += 1; used[j] = True
            else:
                fp += 1
    fn = n_gt - tp
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return p, r, f1, tp, fp, fn, n_gt


def main():
    pred_json, images_dir, labels_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    preds = json.load(open(pred_json))
    gt = load_gt(images_dir, labels_dir)
    # เก็บเฉพาะภาพที่มี GT entry
    preds = {k: [list(map(float, b)) for b in v] for k, v in preds.items()}

    ap50 = ap_at(preds, gt, 0.5)
    ths = np.arange(0.5, 0.96, 0.05)
    aps = [ap_at(preds, gt, t) for t in ths]
    map5095 = float(np.mean(aps))
    p, r, f1, tp, fp, fn, n_gt = prec_rec_f1(preds, gt, 0.5, 0.25)

    print("=== mAP (%s) ===" % os.path.basename(pred_json))
    print("  images=%d  GT boxes=%d" % (len(gt), n_gt))
    print("  mAP@0.50      = %.4f" % ap50)
    print("  mAP@0.50:0.95 = %.4f" % map5095)
    print("  @conf0.25/IoU0.5:  P=%.4f  R=%.4f  F1=%.4f  (TP=%d FP=%d FN=%d)"
          % (p, r, f1, tp, fp, fn))


if __name__ == "__main__":
    main()
