#!/usr/bin/env python3
# ============================================================================
# preproc_accel.py — Python wrapper (ctypes) ของ PL preprocessing accelerator
# ----------------------------------------------------------------------------
# ใช้แทน preprocess(bgr, in_scale) ของแอป:
#     hw = HwPreproc(xclbin)             # เปิดครั้งเดียว
#     inp = hw(bgr, in_scale)            # -> np.int8 [1,640,640,3] (เหมือน preprocess เดิมทุกไบต์)
#
# ต้องมี libpreproc_xrt.so (make -C host บนบอร์ด) + xclbin ที่มี kernel preproc_accel
# ถ้าเฟรมกว้างไม่หาร 8 ลงตัว / กว้างเกิน 1920 → คืน None (ให้ caller fallback SW)
# ============================================================================
import os, ctypes
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
import preproc_tables as T  # noqa: E402

DEFAULT_XCLBIN = os.environ.get(
    "PP_XCLBIN", "/lib/firmware/xilinx/kv260-yolo-preproc/kv260-yolo-preproc.xclbin")
DEFAULT_LIB = os.environ.get("PP_LIB", os.path.join(HERE, "libpreproc_xrt.so"))


class HwPreproc:
    def __init__(self, xclbin=DEFAULT_XCLBIN, lib=DEFAULT_LIB, kernel="preproc_accel",
                 max_src=(1920, 1080)):
        self._lib = ctypes.CDLL(lib)
        self._lib.pp_open.restype = ctypes.c_void_p
        self._lib.pp_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self._lib.pp_set_params.restype = ctypes.c_int
        self._lib.pp_set_params.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._lib.pp_run.restype = ctypes.c_int
        self._lib.pp_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self._lib.pp_close.argtypes = [ctypes.c_void_p]
        cap = max_src[0] * max_src[1] * 3
        self._h = self._lib.pp_open(xclbin.encode(), kernel.encode(), cap)
        if not self._h:
            raise RuntimeError("pp_open failed (xclbin=%s)" % xclbin)
        self._key = None
        self._params = None
        self._out = np.empty((1, T.DST_H, T.DST_W, 3), dtype=np.int8)

    def _ensure_params(self, w, h, in_scale):
        key = (w, h, float(in_scale))
        if key != self._key:
            self._params = np.ascontiguousarray(T.build_params(w, h, in_scale))
            rc = self._lib.pp_set_params(self._h, self._params.ctypes.data)
            if rc != 0:
                raise RuntimeError("pp_set_params rc=%d" % rc)
            self._key = key

    def __call__(self, bgr, in_scale):
        h, w = bgr.shape[:2]
        if not T.hw_supported(w, h):
            return None
        if not bgr.flags["C_CONTIGUOUS"] or bgr.dtype != np.uint8:
            bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
        self._ensure_params(w, h, in_scale)
        rc = self._lib.pp_run(self._h, bgr.ctypes.data, w, h, self._out.ctypes.data)
        if rc != 0:
            raise RuntimeError("pp_run rc=%d" % rc)
        return self._out.copy()      # VART ถือ buffer ไว้ระหว่าง execute_async → ให้สำเนา

    def close(self):
        if self._h:
            self._lib.pp_close(self._h); self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
