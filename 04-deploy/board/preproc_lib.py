#!/usr/bin/env python3
# ============================================================================
# preproc_lib.py — preprocessing 3 แบบให้เลือกใน bench_latency.py / video_detect.py (M13)
# ----------------------------------------------------------------------------
#   numpy : โค้ดเดิมทุกตัวอักษร (baseline ที่วัดได้ 49.7 ms) — cv2.resize → cvtColor →
#           float32 /255 ×scale → np.round → clip → int8
#   lut   : เหมือน numpy แต่แทนช่วง float ด้วย cv2.LUT 256 ค่า (ผลเท่ากันทุกไบต์ —
#           เป็น "software baseline ที่ optimize แล้ว" ที่ต้องเทียบก่อนอ้างว่า HW เร็วกว่า)
#   hw    : PL accelerator (04-deploy/pl-preproc) ผ่าน XRT — resize+cvt+quant ทั้งก้อนใน PL
#           ถ้าเปิดไม่ได้ / เฟรมไม่รองรับ → fallback เป็น lut แล้วพิมพ์เตือน
#
# ทุกโหมดคืน np.int8 [1,640,640,3] NHWC ค่าตรงกัน bit-exact (พิสูจน์ด้วย bench_preproc.py --verify)
# ============================================================================
import os, sys
import numpy as np
import cv2

IMGSZ = 640


def _preprocess_numpy(bgr, in_scale):
    r = cv2.cvtColor(cv2.resize(bgr, (IMGSZ, IMGSZ)), cv2.COLOR_BGR2RGB)
    q = np.clip(np.round(r.astype(np.float32) / 255.0 * in_scale), -128, 127).astype(np.int8)
    return q[np.newaxis, ...]


class _LutPreproc:
    def __init__(self):
        self._scale = None; self._lut = None

    def _table(self, in_scale):
        if self._scale != in_scale:
            r = np.arange(256, dtype=np.uint8)
            q = np.clip(np.round(r.astype(np.float32) / 255.0 * in_scale), -128, 127).astype(np.int8)
            self._lut = q.view(np.uint8)          # cv2.LUT ต้องการ uint8 → เก็บ bit pattern เดิม
            self._scale = in_scale
        return self._lut

    def __call__(self, bgr, in_scale):
        lut = self._table(in_scale)
        r = cv2.cvtColor(cv2.resize(bgr, (IMGSZ, IMGSZ)), cv2.COLOR_BGR2RGB)
        q = cv2.LUT(r, lut).view(np.int8)
        return q[np.newaxis, ...]


class _HwPreproc:
    def __init__(self, xclbin=None):
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, "..", "pl-preproc", "host"))
        from preproc_accel import HwPreproc, DEFAULT_XCLBIN
        self._hw = HwPreproc(xclbin or DEFAULT_XCLBIN)
        self._sw = _LutPreproc()
        self._warned = False

    def __call__(self, bgr, in_scale):
        out = self._hw(bgr, in_scale)
        if out is None:
            if not self._warned:
                print("[preproc_lib] frame %dx%d not HW-supported (width%%8) -> lut fallback"
                      % (bgr.shape[1], bgr.shape[0]))
                self._warned = True
            return self._sw(bgr, in_scale)
        return out


def make_preprocessor(mode, xclbin=None):
    """คืน callable f(bgr, in_scale) -> int8 [1,640,640,3]."""
    if mode == "numpy":
        return _preprocess_numpy
    if mode == "lut":
        return _LutPreproc()
    if mode == "hw":
        try:
            return _HwPreproc(xclbin)
        except Exception as e:
            print("[preproc_lib] HW preproc unavailable (%r) -> falling back to 'lut'" % (e,))
            return _LutPreproc()
    raise ValueError("unknown preproc mode %r" % mode)


MODES = ("numpy", "lut", "hw")
