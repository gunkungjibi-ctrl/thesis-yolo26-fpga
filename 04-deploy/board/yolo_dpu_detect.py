#!/usr/bin/env python3
# ============================================================================
# yolo_dpu_detect.py  —  M9+: VART Python detection on KV260 DPU (B4096)
# ----------------------------------------------------------------------------
# ครบวงจรบน PS (Arm): โหลด .xmodel -> DPU runner -> preprocess ภาพ -> รัน DPU ->
# ดึง raw head 3 tensors -> DFL decode + sigmoid + NMS -> วาดกล่อง -> เซฟรูป
#
# โมเดล = yolov8n_pkg_kv260.xmodel (fine-tuned, package 1-class):
#   input : [1,640,640,3] NHWC int8   (fixpos อ่านจาก tensor)
#   output: 3 tensors [1,H,W,65] NHWC int8  (H,W = 80/40/20)
#           channel 65 = 64 DFL box (4 side x 16 bin) + 1 class(package)
#
# preprocess ต้องตรงกับ calibration เป๊ะ: plain resize 640, BGR->RGB, /255,
#   quantize int8 = round(x * 2^in_fixpos), NHWC (ไม่ transpose).
#
# usage:
#   python3 yolo_dpu_detect.py <model.xmodel> <image> [out.jpg] \
#           [--conf 0.25] [--iou 0.45] [--imgsz 640]
# ============================================================================
import os, sys, math, argparse
import numpy as np

try:
    import xir, vart
except Exception as e:
    sys.exit("[FATAL] import xir/vart failed (run on the board, VART 3.0): %r" % e)

# opencv ใช้อ่าน/เขียน/วาดรูป; ถ้าบอร์ดไม่มี cv2 จะ fallback เป็น PIL
_USE_CV2 = True
try:
    import cv2
except Exception:
    _USE_CV2 = False
    from PIL import Image, ImageDraw


# ---------------------------------------------------------------- model / dpu
def get_dpu_subgraph(graph):
    root = graph.get_root_subgraph()
    subs = root.toposort_child_subgraph()
    dpu = [s for s in subs
           if s.has_attr("device") and s.get_attr("device") == "DPU"]
    assert len(dpu) == 1, "expected exactly 1 DPU subgraph, got %d" % len(dpu)
    return dpu[0]


def fixpos_of(tensor):
    # VART: fixpos เก็บใน attr "fix_point"
    fp = tensor.get_attr("fix_point")
    return int(fp)


# ---------------------------------------------------------------- image io
def imread(path):
    if _USE_CV2:
        im = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        if im is None:
            sys.exit("[FATAL] cannot read image: %s" % path)
        return im  # HxWx3 BGR uint8
    im = Image.open(path).convert("RGB")
    arr = np.array(im)[:, :, ::-1].copy()  # RGB->BGR to match cv2 convention
    return arr


def imwrite(path, bgr):
    if _USE_CV2:
        cv2.imwrite(path, bgr)
        return
    rgb = bgr[:, :, ::-1]
    Image.fromarray(rgb).save(path)


def draw_box(bgr, x1, y1, x2, y2, label):
    if _USE_CV2:
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(bgr, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        # PIL path: bgr เป็น ndarray, แปลงชั่วคราว
        pass  # (จัดการรวมทีเดียวใน main สำหรับ PIL)


# ---------------------------------------------------------------- preprocess
def preprocess(bgr, imgsz, in_scale):
    if _USE_CV2:
        resized = cv2.resize(bgr, (imgsz, imgsz))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    else:
        from PIL import Image as _I
        rgb0 = bgr[:, :, ::-1]
        resized = np.array(_I.fromarray(rgb0).resize((imgsz, imgsz)))
        rgb = resized
    x = rgb.astype(np.float32) / 255.0          # [0,1]
    q = np.round(x * in_scale)                  # quantize by 2^fixpos
    q = np.clip(q, -128, 127).astype(np.int8)
    return q[np.newaxis, ...]                    # [1,H,W,3] NHWC int8


# ---------------------------------------------------------------- decode
def softmax(v, axis=-1):
    v = v - np.max(v, axis=axis, keepdims=True)
    e = np.exp(v)
    return e / np.sum(e, axis=axis, keepdims=True)


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def decode_head(feat, stride, reg_max, nc, conf_th):
    """feat: [H,W,C] float (dequantized). C = 4*reg_max + nc.
    คืน list ของ (x1,y1,x2,y2,score,cls) ในสเกลพิกเซลของ input 640."""
    H, W, C = feat.shape
    box = feat[:, :, :4 * reg_max].reshape(H, W, 4, reg_max)
    cls = feat[:, :, 4 * reg_max:]                          # [H,W,nc]

    # class prob (sigmoid) + คัดเบื้องต้นด้วย conf
    prob = sigmoid(cls)                                     # [H,W,nc]
    cls_id = np.argmax(prob, axis=-1)                       # [H,W]
    cls_sc = np.max(prob, axis=-1)                          # [H,W]
    ys, xs = np.where(cls_sc >= conf_th)
    if len(xs) == 0:
        return []

    # DFL: softmax 16 bin ต่อ side -> ระยะ (คูณ arange)
    bins = np.arange(reg_max, dtype=np.float32)
    out = []
    for y, x in zip(ys, xs):
        d = softmax(box[y, x], axis=-1) @ bins              # [4] = l,t,r,b
        cx, cy = x + 0.5, y + 0.5                           # anchor center (grid)
        x1 = (cx - d[0]) * stride
        y1 = (cy - d[1]) * stride
        x2 = (cx + d[2]) * stride
        y2 = (cy + d[3]) * stride
        out.append((x1, y1, x2, y2, float(cls_sc[y, x]), int(cls_id[y, x])))
    return out


def nms(dets, iou_th):
    if not dets:
        return []
    d = np.array([[a, b, c, e] for (a, b, c, e, s, k) in dets], dtype=np.float32)
    scores = np.array([s for (*_, s, k) in dets], dtype=np.float32)
    x1, y1, x2, y2 = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][ovr <= iou_th]
    return [dets[i] for i in keep]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xmodel")
    ap.add_argument("image")
    ap.add_argument("out", nargs="?", default="detect_out.jpg")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--reg_max", type=int, default=16)
    ap.add_argument("--nc", type=int, default=1)
    args = ap.parse_args()

    g = xir.Graph.deserialize(args.xmodel)
    sub = get_dpu_subgraph(g)
    runner = vart.Runner.create_runner(sub, "run")
    it = runner.get_input_tensors()
    ot = runner.get_output_tensors()
    assert len(it) == 1, "expected 1 input tensor"
    ishape = tuple(it[0].dims)                 # (1,640,640,3)
    in_fp = fixpos_of(it[0])
    in_scale = float(2 ** in_fp)
    print("[info] input %s shape=%s in_fixpos=%d scale=%.1f"
          % (it[0].name, ishape, in_fp, in_scale))

    bgr = imread(args.image)
    orig_h, orig_w = bgr.shape[:2]
    inp = preprocess(bgr, args.imgsz, in_scale)   # [1,H,W,3] int8

    # เตรียม output buffers (int8) ตาม shape ของแต่ละ tensor
    out_bufs = [np.empty(tuple(t.dims), dtype=np.int8) for t in ot]
    job = runner.execute_async([inp], out_bufs)
    runner.wait(job)

    # dequantize + decode ทุกหัว (แมป stride จากขนาด spatial)
    stride_of = {args.imgsz // 8: 8, args.imgsz // 16: 16, args.imgsz // 32: 32}
    all_dets = []
    for t, buf in zip(ot, out_bufs):
        fp = fixpos_of(t)
        deq = buf.astype(np.float32) * (2.0 ** (-fp))  # int8 -> float
        _, Hf, Wf, Cf = buf.shape
        stride = stride_of.get(Hf)
        assert stride is not None, "unexpected feat size %d" % Hf
        assert Cf == 4 * args.reg_max + args.nc, \
            "channel %d != 4*%d+%d" % (Cf, args.reg_max, args.nc)
        dets = decode_head(deq[0], stride, args.reg_max, args.nc, args.conf)
        print("[info] head %dx%d fixpos=%d float_range=[%.3f,%.3f] cand=%d"
              % (Hf, Wf, fp, deq.min(), deq.max(), len(dets)))
        all_dets.extend(dets)

    kept = nms(all_dets, args.iou)
    print("[result] boxes after NMS = %d" % len(kept))

    # scale กลับสู่ภาพต้นฉบับ (plain resize -> คูณอัตราส่วนต่อแกน)
    sx, sy = orig_w / args.imgsz, orig_h / args.imgsz
    if not _USE_CV2:
        from PIL import Image as _I, ImageDraw as _D
        pim = _I.fromarray(bgr[:, :, ::-1]); dr = _D.Draw(pim)
    for (x1, y1, x2, y2, s, k) in kept:
        X1, Y1 = int(x1 * sx), int(y1 * sy)
        X2, Y2 = int(x2 * sx), int(y2 * sy)
        X1 = max(0, min(orig_w - 1, X1)); X2 = max(0, min(orig_w - 1, X2))
        Y1 = max(0, min(orig_h - 1, Y1)); Y2 = max(0, min(orig_h - 1, Y2))
        label = "package %.2f" % s
        print("   box (%d,%d)-(%d,%d) score=%.3f" % (X1, Y1, X2, Y2, s))
        if _USE_CV2:
            draw_box(bgr, X1, Y1, X2, Y2, label)
        else:
            dr.rectangle([X1, Y1, X2, Y2], outline=(0, 255, 0), width=2)
            dr.text((X1, max(0, Y1 - 10)), label, fill=(0, 255, 0))

    if not _USE_CV2:
        np_out = np.array(pim)[:, :, ::-1]
        imwrite(args.out, np_out)
    else:
        imwrite(args.out, bgr)
    print("[ok] wrote %s (%d boxes)" % (args.out, len(kept)))


if __name__ == "__main__":
    main()
