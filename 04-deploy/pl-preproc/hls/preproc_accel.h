// ============================================================================
// preproc_accel.h — I/O contract ของ PL preprocessing accelerator (KV260)
// ----------------------------------------------------------------------------
// งานของ kernel (แทน preprocess() ใน yolo_dpu_detect.py ทั้งก้อน):
//   BGR uint8 [src_h][src_w][3]  --bilinear resize (สูตร cv2 INTER_LINEAR)-->
//   640x640 --BGR->RGB--> LUT quantize (round(x/255 * 2^fixpos)) --> int8 NHWC
//   เขียนลง DDR ตรงรูปแบบที่ VART ต้องการ [1][640][640][3] int8
//
// ไฟล์นี้ใช้ร่วมกัน 3 ที่: kernel (Vitis HLS), testbench (g++), host lib (XRT)
// ============================================================================
#ifndef PREPROC_ACCEL_H
#define PREPROC_ACCEL_H
#include <stdint.h>

#define PP_DST_W        640
#define PP_DST_H        640
#define PP_MAX_SRC_W    1920            // line buffer กว้างสุด (1080p)
#define PP_MAX_SRC_H    4096            // แค่เช็ก range (ไม่ได้ buffer ทั้งเฟรม)
#define PP_DST_ROW_BYTES (PP_DST_W * 3)          // 1920
#define PP_DST_ROW_WORDS (PP_DST_ROW_BYTES / 8)  // 240  (bus 64 bit)
#define PP_DST_BYTES    (PP_DST_H * PP_DST_ROW_BYTES)   // 1,228,800

// param buffer: int16 x 4096 (ดู host/preproc_tables.py)
#define PP_PARAM_WORDS  4096
#define PP_OFF_XOFS     0
#define PP_OFF_ALPHA0   640
#define PP_OFF_ALPHA1   1280
#define PP_OFF_YOFS     1920            // int16 ไม่ clamp (-1..src_h-1): kernel clip เอง
#define PP_OFF_BETA0    2560
#define PP_OFF_BETA1    3200
#define PP_OFF_LUT      3840

#define PP_COEF_BITS    11              // INTER_RESIZE_COEF_BITS ของ OpenCV

// return code ของ kernel (อ่านผ่าน s_axilite "return" ไม่ได้ใน Vitis flow — ใช้ตอน csim)
#define PP_OK           0
#define PP_ERR_SIZE     1

#ifdef __cplusplus
extern "C" {
#endif
// src  : BGR uint8 ต่อเนื่อง (stride = src_w*3) มองเป็น 64-bit words  → src_w % 8 == 0
// params: int16[4096]
// dst  : int8 [640][640][3] มองเป็น 64-bit words (PP_DST_BYTES/8 = 153,600 words)
void preproc_accel(const uint64_t *src, const int16_t *params, uint64_t *dst,
                   int src_w, int src_h);
#ifdef __cplusplus
}
#endif
#endif
