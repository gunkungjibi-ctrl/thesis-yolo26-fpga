#!/usr/bin/env python3
# ============================================================================
# preproc_tables.py  —  สร้างตารางพารามิเตอร์ให้ preproc_accel (PL) และ golden model
# ----------------------------------------------------------------------------
# ตัวเร่งไม่คำนวณพิกัด/น้ำหนักเอง (ประหยัด DSP + ทำให้ bit-exact กับ cv2 ได้ง่าย)
# host เป็นคนคำนวณแบบเดียวกับ cv2.resize(INTER_LINEAR) ทุกขั้น แล้วส่งเป็นบัฟเฟอร์
# int16 ขนาด 4096 ค่า (8 KB) ให้ kernel อ่านครั้งเดียวตอนเริ่มงาน
#
# layout (หน่วย = int16, ต้องตรงกับ hls/preproc_accel.h):
#   [   0..639 ]  xofs    : พิกัด x ต้นทาง (clamp แล้ว 0..src_w-1)
#   [ 640..1279]  alpha0  : น้ำหนัก 1-fx  (fixed-point 11 bit, ผลรวม 2048)
#   [1280..1919]  alpha1  : น้ำหนัก fx
#   [1920..2559]  yofs    : แถวต้นทาง *ไม่ clamp* (-1..src_h-1) — kernel clip เอง
#   [2560..3199]  beta0
#   [3200..3839]  beta1
#   [3840..4095]  lut     : uint8 -> int8 quantized  (เก็บใน int16 ค่า -128..127)
#
# สูตรพิกัด = cv2 resize.cpp (fixed-point path ของ 8-bit INTER_LINEAR):
#   fx = float((dx + 0.5) * (src/dst) - 0.5); sx = floor(fx); fx -= sx
#   แกน x เท่านั้น:  sx < 0 -> fx = 0, sx = 0 ;  sx >= src-1 -> fx = 0, sx = src-1
#   แกน y:           ไม่แตะ fy — cv2 clip index แถวตอนอ่าน (clip(sy,0,h-1)) แทน
#   alpha = saturate_cast<short>((1-fx) * 2048), saturate_cast<short>(fx * 2048)
# ============================================================================
import numpy as np

DST_W = 640
DST_H = 640
COEF_BITS = 11
COEF_SCALE = 1 << COEF_BITS            # 2048  (INTER_RESIZE_COEF_SCALE ของ OpenCV)
PARAM_WORDS = 4096
OFF_XOFS, OFF_ALPHA0, OFF_ALPHA1 = 0, 640, 1280
OFF_YOFS, OFF_BETA0, OFF_BETA1 = 1920, 2560, 3200
OFF_LUT = 3840


def linear_coeffs(src_len, dst_len, clamp_border):
    """คืน (ofs int32[dst], c0 int16[dst], c1 int16[dst]) ตาม cv2 INTER_LINEAR fixed-point.

    clamp_border=True  = แกน x ของ cv2: ที่ขอบบังคับ fx=0 (น้ำหนัก 2048/0) และ sx อยู่ในภาพ
    clamp_border=False = แกน y ของ cv2: *ไม่* แตะน้ำหนัก (เช่น 92/1956) แต่ไป clip index แถว
                         ตอนอ่านแทน → ofs อาจเป็น -1 หรือ src_len-1 (kernel เป็นคน clip)
    สองแกนไม่สมมาตรจริง ๆ ใน resize.cpp — ถ้าทำเหมือนกันจะพลาด 1 LSB ที่แถวบน/ล่างตอน upscale
    """
    scale = 1.0 / (float(dst_len) / float(src_len))     # cv2: scale_x = 1./inv_scale_x
    d = np.arange(dst_len, dtype=np.float64)
    f = ((d + 0.5) * scale - 0.5).astype(np.float32)    # (float)(double expr)
    s = np.floor(f).astype(np.int32)                     # cvFloor
    f = (f - s.astype(np.float32)).astype(np.float32)    # fx -= sx  (float)
    if clamp_border:
        lo = s < 0
        hi = s >= src_len - 1
        f[lo] = 0.0; s[lo] = 0
        f[hi] = 0.0; s[hi] = src_len - 1
    c0 = (np.float32(1.0) - f).astype(np.float32) * np.float32(COEF_SCALE)
    c1 = f * np.float32(COEF_SCALE)
    # saturate_cast<short>(float) = cvRound = round-half-even (lrint)
    c0 = np.clip(np.rint(c0), -32768, 32767).astype(np.int16)
    c1 = np.clip(np.rint(c1), -32768, 32767).astype(np.int16)
    return s, c0, c1


def quant_lut(in_scale):
    """LUT uint8 -> int8 ใช้นิพจน์เดียวกับ yolo_dpu_detect.py ทุกตัวอักษร (float32 path)."""
    r = np.arange(256, dtype=np.uint8)
    q = np.clip(np.round(r.astype(np.float32) / 255.0 * in_scale), -128, 127).astype(np.int8)
    return q


def build_params(src_w, src_h, in_scale, dst_w=DST_W, dst_h=DST_H):
    assert dst_w == DST_W and dst_h == DST_H, "kernel ถูก fix ที่ 640x640"
    p = np.zeros(PARAM_WORDS, dtype=np.int16)
    xo, a0, a1 = linear_coeffs(src_w, dst_w, clamp_border=True)
    yo, b0, b1 = linear_coeffs(src_h, dst_h, clamp_border=False)
    assert 0 <= xo.min() and xo.max() < src_w
    assert -1 <= yo.min() and yo.max() <= src_h - 1      # kernel clip เป็น [0, src_h-1]
    p[OFF_XOFS:OFF_XOFS + dst_w] = xo.astype(np.int16)
    p[OFF_ALPHA0:OFF_ALPHA0 + dst_w] = a0
    p[OFF_ALPHA1:OFF_ALPHA1 + dst_w] = a1
    p[OFF_YOFS:OFF_YOFS + dst_h] = yo.astype(np.int16)
    p[OFF_BETA0:OFF_BETA0 + dst_h] = b0
    p[OFF_BETA1:OFF_BETA1 + dst_h] = b1
    p[OFF_LUT:OFF_LUT + 256] = quant_lut(in_scale).astype(np.int16)
    return p


def hw_supported(src_w, src_h):
    """ข้อจำกัดของ kernel v1: กว้าง <= 1920, กว้างหาร 8 ลงตัว (bus 64 bit), สูง <= 4096."""
    return 8 <= src_w <= 1920 and src_w % 8 == 0 and 2 <= src_h <= 4096


if __name__ == "__main__":
    import sys
    w, h = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (640, 360)
    p = build_params(w, h, 64.0)
    xo, a0, a1 = p[:640], p[640:1280], p[1280:1920]
    print("src %dx%d -> 640x640  xofs[0:4]=%s a0[0:4]=%s a1[0:4]=%s" % (w, h, xo[:4], a0[:4], a1[:4]))
    print("alpha sum unique:", np.unique(a0.astype(np.int32) + a1))
    print("lut[0:8]=%s lut[248:256]=%s" % (p[3840:3848], p[4088:4096]))
