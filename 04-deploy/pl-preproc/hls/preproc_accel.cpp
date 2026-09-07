// ============================================================================
// preproc_accel.cpp — Vitis HLS kernel: YOLO preprocessing บน PL ของ KV260
// ----------------------------------------------------------------------------
// แทน preprocess() ของ yolo_dpu_detect.py / video_detect.py ทั้งก้อน:
//   resize (bilinear, bit-exact กับ cv2.resize INTER_LINEAR บน ARM NEON)
//   -> BGR→RGB -> quantize ผ่าน LUT (round(x/255·2^fixpos)) -> int8 NHWC ลง DDR
//
// โครงสร้าง (ต่อ output row):
//   [1] ดูว่าแถวต้นทาง sy0, sy1 อยู่ใน line buffer 2 แถวหรือยัง → ถ้าไม่ burst-read
//   [2] คำนวณ 640 พิกเซล (3 ch ขนาน) II=1 → เก็บลง row byte buffer
//   [3] pack 8 byte/word → burst-write 240 words
//
// เขียนด้วย C มาตรฐาน (ไม่พึ่ง ap_int/hls_stream) เพื่อให้ g++ compile testbench
// ได้เหมือนกันทุกไบต์ — pragma HLS จะถูกเมินโดย g++ (-Wno-unknown-pragmas)
// ============================================================================
#include "preproc_accel.h"

#define ROW_BANKS 8   // cyclic partition ของ line buffer (รับ 8 pixel/cycle ตอนโหลด)

static inline uint8_t sat_u8(int v) {
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

// โหลดแถว sy ของภาพต้นทางเข้า line buffer: 3 words (24 byte) = 8 pixel
// เก็บ pixel เป็น 32 bit: [7:0]=B [15:8]=G [23:16]=R
static void load_row(const uint64_t *src, uint32_t row[PP_MAX_SRC_W],
                     int sy, int src_w) {
#pragma HLS INLINE   // ให้ rows[i0] (partitioned, index dynamic) กลายเป็น mux ใน caller
    const int words = (src_w * 3) >> 3;              // src_w % 8 == 0 → หาร 3 ลงตัว
    const uint64_t *base = src + (uint64_t)sy * (uint64_t)words;
    uint64_t acc0 = 0, acc1 = 0;                     // เก็บ 2 word ก่อนหน้า
    int phase = 0;
    int px = 0;
LOAD_ROW:
    for (int k = 0; k < words; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=240 max=720
        uint64_t w = base[k];
        if (phase == 0) {
            acc0 = w; phase = 1;
        } else if (phase == 1) {
            acc1 = w; phase = 2;
        } else {
            // 24 byte = acc0 | acc1 | w → 8 pixel (little-endian byte order)
            uint8_t b[24];
#pragma HLS ARRAY_PARTITION variable=b complete
            for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
                b[i]      = (uint8_t)(acc0 >> (8 * i));
                b[8 + i]  = (uint8_t)(acc1 >> (8 * i));
                b[16 + i] = (uint8_t)(w    >> (8 * i));
            }
            for (int p = 0; p < 8; p++) {
#pragma HLS UNROLL
                row[px + p] = (uint32_t)b[3 * p] | ((uint32_t)b[3 * p + 1] << 8)
                            | ((uint32_t)b[3 * p + 2] << 16);
            }
            px += 8;
            phase = 0;
        }
    }
}

// interpolation ต่อ 1 channel — สูตรตรงกับ OpenCV VResizeLinearVec_32s8u (NEON/SSE)
static inline uint8_t interp_ch(int p00, int p01, int p10, int p11,
                                int a0, int a1, int b0, int b1) {
    int h0 = p00 * a0 + p01 * a1;                    // <= 255*2048 < 2^19
    int h1 = p10 * a0 + p11 * a1;
    int t0 = ((h0 >> 4) * b0) >> 16;                 // (h>>4) <= 32640 (int16), *b <= 2^26
    int t1 = ((h1 >> 4) * b1) >> 16;
    return sat_u8((t0 + t1 + 2) >> 2);
}

extern "C" void preproc_accel(const uint64_t *src, const int16_t *params, uint64_t *dst,
                              int src_w, int src_h) {
#pragma HLS INTERFACE m_axi     port=src    offset=slave bundle=gmem0 max_read_burst_length=64 num_read_outstanding=8
#pragma HLS INTERFACE m_axi     port=params offset=slave bundle=gmem1 max_read_burst_length=64
#pragma HLS INTERFACE m_axi     port=dst    offset=slave bundle=gmem2 max_write_burst_length=64 num_write_outstanding=8
#pragma HLS INTERFACE s_axilite port=src
#pragma HLS INTERFACE s_axilite port=params
#pragma HLS INTERFACE s_axilite port=dst
#pragma HLS INTERFACE s_axilite port=src_w
#pragma HLS INTERFACE s_axilite port=src_h
#pragma HLS INTERFACE s_axilite port=return

    // ---- ตารางพารามิเตอร์ (อ่านครั้งเดียว 8 KB) ----
    uint16_t xofs[PP_DST_W];
    int16_t  alpha0[PP_DST_W], alpha1[PP_DST_W];
    int16_t  yofs[PP_DST_H];                         // -1..src_h-1 (ไม่ clamp — เหมือน cv2)
    int16_t  beta0[PP_DST_H], beta1[PP_DST_H];
    int8_t   lut[256];
#pragma HLS ARRAY_PARTITION variable=lut complete   // อ่าน 3 ค่า/cycle
#pragma HLS BIND_STORAGE variable=xofs   type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=alpha0 type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=alpha1 type=ram_1p impl=bram

    if (src_w < 8 || src_w > PP_MAX_SRC_W || (src_w & 7) != 0 ||
        src_h < 2 || src_h > PP_MAX_SRC_H) {
        return;                                      // host ต้องเช็กก่อน (hw_supported)
    }

LOAD_PARAMS:
    for (int i = 0; i < PP_PARAM_WORDS; i++) {
#pragma HLS PIPELINE II=1
        int16_t v = params[i];
        if      (i < PP_OFF_ALPHA0) xofs[i - PP_OFF_XOFS]     = (uint16_t)v;
        else if (i < PP_OFF_ALPHA1) alpha0[i - PP_OFF_ALPHA0] = v;
        else if (i < PP_OFF_YOFS)   alpha1[i - PP_OFF_ALPHA1] = v;
        else if (i < PP_OFF_BETA0)  yofs[i - PP_OFF_YOFS]     = v;
        else if (i < PP_OFF_BETA1)  beta0[i - PP_OFF_BETA0]   = v;
        else if (i < PP_OFF_LUT)    beta1[i - PP_OFF_BETA1]   = v;
        else                        lut[i - PP_OFF_LUT]       = (int8_t)v;
    }

    // ---- line buffer 2 แถว (pixel packed 32 bit) ----
    uint32_t rows[2][PP_MAX_SRC_W];
#pragma HLS ARRAY_PARTITION variable=rows dim=1 complete
#pragma HLS ARRAY_PARTITION variable=rows dim=2 cyclic factor=ROW_BANKS
#pragma HLS BIND_STORAGE variable=rows type=ram_t2p impl=bram
    int held[2] = {-1, -1};

    // ---- output row buffer (byte) ----
    uint8_t obuf[PP_DST_ROW_BYTES];
#pragma HLS ARRAY_PARTITION variable=obuf cyclic factor=8

    const int last_x = src_w - 1;
    const int last_y = src_h - 1;

OUT_ROWS:
    for (int dy = 0; dy < PP_DST_H; dy++) {
        // cv2: clip(sy, 0, h-1) ทั้งสองแถว — แถวบนสุดตอน upscale ได้ sy=-1 → ใช้แถว 0 สองครั้ง
        const int yraw = yofs[dy];
        const int sy0 = yraw < 0 ? 0 : (yraw > last_y ? last_y : yraw);
        const int sy1 = yraw + 1 < 0 ? 0 : (yraw + 1 > last_y ? last_y : yraw + 1);
        const int b0 = beta0[dy], b1 = beta1[dy];

        // [1] จัดแถวต้นทางเข้า buffer (reuse ถ้ามีอยู่แล้ว — กรณี upscale/ scale<2 ประหยัดมาก)
        int i0, i1;
        if      (held[0] == sy0) i0 = 0;
        else if (held[1] == sy0) i0 = 1;
        else { i0 = (held[0] == sy1) ? 1 : 0; load_row(src, rows[i0], sy0, src_w); held[i0] = sy0; }
        if      (held[0] == sy1) i1 = 0;
        else if (held[1] == sy1) i1 = 1;
        else { i1 = 1 - i0;                          load_row(src, rows[i1], sy1, src_w); held[i1] = sy1; }

        // [2] interpolate 640 pixel × 3 ch
OUT_PIX:
        for (int dx = 0; dx < PP_DST_W; dx++) {
#pragma HLS PIPELINE II=1
            const int sx0 = xofs[dx];
            const int sx1 = (sx0 + 1 > last_x) ? last_x : sx0 + 1;
            const int a0 = alpha0[dx], a1 = alpha1[dx];
            const uint32_t p00 = rows[i0][sx0], p01 = rows[i0][sx1];
            const uint32_t p10 = rows[i1][sx0], p11 = rows[i1][sx1];
            uint8_t u8[3];
#pragma HLS ARRAY_PARTITION variable=u8 complete
            for (int c = 0; c < 3; c++) {
#pragma HLS UNROLL
                const int sh = 8 * c;
                u8[c] = interp_ch((p00 >> sh) & 0xff, (p01 >> sh) & 0xff,
                                  (p10 >> sh) & 0xff, (p11 >> sh) & 0xff, a0, a1, b0, b1);
            }
            // BGR (c=0,1,2) → RGB
            obuf[3 * dx + 0] = (uint8_t)lut[u8[2]];
            obuf[3 * dx + 1] = (uint8_t)lut[u8[1]];
            obuf[3 * dx + 2] = (uint8_t)lut[u8[0]];
        }

        // [3] pack + burst write
        uint64_t *drow = dst + (uint64_t)dy * PP_DST_ROW_WORDS;
OUT_WRITE:
        for (int w = 0; w < PP_DST_ROW_WORDS; w++) {
#pragma HLS PIPELINE II=1
            uint64_t v = 0;
            for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
                v |= (uint64_t)obuf[8 * w + i] << (8 * i);
            }
            drow[w] = v;
        }
    }
}
