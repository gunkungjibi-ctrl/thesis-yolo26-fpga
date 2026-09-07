#!/usr/bin/env python3
# ============================================================================
# preproc_golden.py  —  bit-exact software model ของ preproc_accel (PL kernel)
# ----------------------------------------------------------------------------
# ใช้ 3 อย่าง:
#   1. เป็น "ความจริง" ที่ C testbench (hls/tb_preproc.cpp) ต้องตรงทุกไบต์
#   2. วัดว่าโมเดล (= ฮาร์ดแวร์) ต่างจาก cv2.resize ตัวจริงกี่ค่า (เป้า: 0 ในโดเมน int8)
#   3. gen test vector (.bin) ให้ csim/cosim
#
# datapath ต่อ output pixel (ต่อ channel), ทุกค่าเป็นจำนวนเต็ม:
#   h0 = p[sy0][sx]*a0 + p[sy0][sx1]*a1            (<= 255*2048 = 2^19 ish)
#   h1 = p[sy1][sx]*a0 + p[sy1][sx1]*a1
#   -- "simd" (ทางที่ OpenCV NEON/SSE ใช้จริงเมื่อความกว้างแถวหาร 16 ลงตัว):
#   v  = (((h0>>4)*b0)>>16) + (((h1>>4)*b1)>>16);  u8 = sat8((v + 2) >> 2)
#   -- "scalar" (สูตรอ้างอิงใน resize.cpp, ใช้กับหาง/แถวที่ไม่หาร 16):
#   u8 = sat8((h0*b0 + h1*b1 + (1<<21)) >> 22)
#   q  = lut[u8]  (int8) ;  ลำดับ channel ออก = R,G,B (สลับจาก BGR)
#
# usage:
#   python3 preproc_golden.py check   [--sizes 640x360,1280x720,1920x1080] [--n 8]
#   python3 preproc_golden.py gen <out_dir> [--src WxH] [--image path] [--fixpos 6]
#   python3 preproc_golden.py sweep <image_dir> [--src WxH]
# ============================================================================
import os, sys, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "host"))
import preproc_tables as T  # noqa: E402

try:
    import cv2
except Exception:
    cv2 = None


def hpass(row_u32_or_u8, xo, a0, a1, src_w):
    """row: [src_w,3] uint8 -> [640,3] int32 (horizontal linear, fixed-point)."""
    sx0 = xo.astype(np.int64)
    sx1 = np.minimum(sx0 + 1, src_w - 1)
    r = row_u32_or_u8.astype(np.int32)
    return r[sx0] * a0[:, None].astype(np.int32) + r[sx1] * a1[:, None].astype(np.int32)


def model(bgr, params, mode="simd"):
    """โมเดลของ kernel ทั้งตัว: bgr [H,W,3] uint8 -> [640,640,3] int8 (RGB, quantized)."""
    src_h, src_w = bgr.shape[:2]
    p = params.astype(np.int32)
    xo = p[T.OFF_XOFS:T.OFF_XOFS + 640]
    a0 = p[T.OFF_ALPHA0:T.OFF_ALPHA0 + 640]
    a1 = p[T.OFF_ALPHA1:T.OFF_ALPHA1 + 640]
    yo = p[T.OFF_YOFS:T.OFF_YOFS + 640]
    b0 = p[T.OFF_BETA0:T.OFF_BETA0 + 640]
    b1 = p[T.OFF_BETA1:T.OFF_BETA1 + 640]
    lut = p[T.OFF_LUT:T.OFF_LUT + 256].astype(np.int8)

    # horizontal pass ครั้งเดียวต่อแถวต้นทาง (kernel ทำสดต่อ output row แต่ผลเท่ากัน)
    H = np.empty((src_h, 640, 3), dtype=np.int32)
    for y in range(src_h):
        H[y] = hpass(bgr[y], xo, a0, a1, src_w)

    sy0 = np.clip(yo, 0, src_h - 1)             # yofs อาจเป็น -1 (ขอบบนตอน upscale) — clip เหมือน cv2
    sy1 = np.clip(yo + 1, 0, src_h - 1)
    h0 = H[sy0]                     # [640,640,3]
    h1 = H[sy1]
    B0 = b0[:, None, None]; B1 = b1[:, None, None]
    if mode == "simd":
        v = ((h0 >> 4) * B0 >> 16) + ((h1 >> 4) * B1 >> 16)
        u8 = np.clip((v + 2) >> 2, 0, 255).astype(np.uint8)
    elif mode == "scalar":
        v = (h0.astype(np.int64) * B0 + h1.astype(np.int64) * B1 + (1 << 21)) >> 22
        u8 = np.clip(v, 0, 255).astype(np.uint8)
    else:
        raise ValueError(mode)
    rgb = u8[:, :, ::-1]            # BGR -> RGB
    return lut[rgb]                 # int8 [640,640,3]


def reference_cv2(bgr, in_scale):
    """preprocess() ตัวจริงของแอป (yolo_dpu_detect.py) — ต้องมี cv2."""
    r = cv2.cvtColor(cv2.resize(bgr, (640, 640)), cv2.COLOR_BGR2RGB)
    return np.clip(np.round(r.astype(np.float32) / 255.0 * in_scale), -128, 127).astype(np.int8)


def reference_cv2_u8(bgr):
    return cv2.resize(bgr, (640, 640))


def synth_frame(w, h, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([(xx * 255 // max(w - 1, 1)), (yy * 255 // max(h - 1, 1)),
                     ((xx + yy) * 255 // max(w + h - 2, 1))], -1).astype(np.int32)
    noise = rng.integers(-40, 41, size=(h, w, 3))
    im = np.clip(base + noise, 0, 255).astype(np.uint8)
    # แถบขาวดำคม ๆ ให้ interpolation ทำงานหนัก
    im[h // 3:h // 3 + 3, :, :] = 255
    im[:, w // 2:w // 2 + 2, :] = 0
    return im


def load_source(path, w, h):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        sys.exit("cannot read %s" % path)
    if (im.shape[1], im.shape[0]) != (w, h):
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(im)


def compare(a, b):
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return int((d != 0).sum()), int(d.max()), a.size


def cmd_check(args):
    if cv2 is None:
        sys.exit("check ต้องมี cv2")
    in_scale = float(2 ** args.fixpos)
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    worst = 0
    print("compare model vs cv2  (mismatch/total, max|diff|)   fixpos=%d" % args.fixpos)
    print("%-10s %-4s | %-26s | %-26s | %-26s" % ("src", "seed", "simd u8", "scalar u8", "simd int8 (deploy)"))
    for (w, h) in sizes:
        prm = T.build_params(w, h, in_scale)
        for seed in range(args.n):
            im = synth_frame(w, h, seed)
            ref_u8 = reference_cv2_u8(im)
            ref_q = reference_cv2(im, in_scale)
            lut = prm[T.OFF_LUT:T.OFF_LUT + 256].astype(np.int8)
            m_s = model(im, prm, "simd"); m_c = model(im, prm, "scalar")
            # เทียบใน u8 domain: ถอด LUT ไม่ได้ → เทียบ u8 ก่อน quantize ด้วยการรัน model แบบ lut=identity
            ident = prm.copy(); ident[T.OFF_LUT:T.OFF_LUT + 256] = np.arange(256, dtype=np.int16) - 128
            u8_s = (model(im, ident, "simd").astype(np.int16) + 128).astype(np.uint8)[:, :, ::-1]
            u8_c = (model(im, ident, "scalar").astype(np.int16) + 128).astype(np.uint8)[:, :, ::-1]
            cs = compare(u8_s, ref_u8); cc = compare(u8_c, ref_u8); cq = compare(m_s, ref_q)
            worst = max(worst, cq[0])
            print("%-10s %-4d | %8d/%-8d max=%d | %8d/%-8d max=%d | %8d/%-8d max=%d"
                  % ("%dx%d" % (w, h), seed, cs[0], cs[2], cs[1], cc[0], cc[2], cc[1], cq[0], cq[2], cq[1]))
    print("[result] worst int8-domain mismatch (simd model vs cv2 app path) = %d" % worst)
    return 0 if worst == 0 else 1


def cmd_gen(args):
    os.makedirs(args.out_dir, exist_ok=True)
    w, h = (int(v) for v in args.src.split("x"))
    in_scale = float(2 ** args.fixpos)
    im = load_source(args.image, w, h) if args.image else synth_frame(w, h, 0)
    prm = T.build_params(w, h, in_scale)
    gold = model(im, prm, "simd")
    im.tofile(os.path.join(args.out_dir, "src_%dx%d_bgr.bin" % (w, h)))
    prm.tofile(os.path.join(args.out_dir, "params_%dx%d.bin" % (w, h)))
    gold.tofile(os.path.join(args.out_dir, "golden_%dx%d.bin" % (w, h)))
    with open(os.path.join(args.out_dir, "meta_%dx%d.txt" % (w, h)), "w") as f:
        f.write("%d %d %d\n" % (w, h, args.fixpos))
    if cv2 is not None:
        ref = reference_cv2(im, in_scale)
        n, mx, tot = compare(gold, ref)
        print("[gen] %dx%d -> 640x640  model vs cv2 app-path: %d/%d mismatch (max %d)" % (w, h, n, tot, mx))
    print("[gen] wrote src/params/golden to %s" % args.out_dir)
    return 0


def cmd_sweep(args):
    """รันโมเดลกับรูปจริงทั้งโฟลเดอร์ (ย่อ/ขยายเป็นขนาดต้นทางที่กำหนดก่อน) เทียบ cv2 path."""
    if cv2 is None:
        sys.exit("sweep ต้องมี cv2")
    w, h = (int(v) for v in args.src.split("x"))
    in_scale = float(2 ** args.fixpos)
    prm = T.build_params(w, h, in_scale)
    files = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))[:args.n]
    tot_mis = 0; tot = 0; worst = 0
    for f in files:
        im = load_source(f, w, h)
        n, mx, sz = compare(model(im, prm, "simd"), reference_cv2(im, in_scale))
        tot_mis += n; tot += sz; worst = max(worst, mx)
    print("[sweep] %d images, src %dx%d: int8 mismatches %d / %d (%.6f%%), max|diff|=%d"
          % (len(files), w, h, tot_mis, tot, 100.0 * tot_mis / max(tot, 1), worst))
    return 0 if tot_mis == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("--sizes", default="640x360,640x640,1280x720,1920x1080")
    c.add_argument("--n", type=int, default=4); c.add_argument("--fixpos", type=int, default=6)
    g = sub.add_parser("gen"); g.add_argument("out_dir"); g.add_argument("--src", default="640x360")
    g.add_argument("--image", default=None); g.add_argument("--fixpos", type=int, default=6)
    s = sub.add_parser("sweep"); s.add_argument("image_dir"); s.add_argument("--src", default="640x360")
    s.add_argument("--n", type=int, default=1000); s.add_argument("--fixpos", type=int, default=6)
    args = ap.parse_args()
    sys.exit({"check": cmd_check, "gen": cmd_gen, "sweep": cmd_sweep}[args.cmd](args))


if __name__ == "__main__":
    main()
