// ============================================================================
// preproc_xrt.cpp — host driver ของ preproc_accel ผ่าน XRT native C++ API (บอร์ด KV260)
// ----------------------------------------------------------------------------
// build บนบอร์ด (PetaLinux starter kit มี g++ + XRT อยู่แล้ว):  make -C host
// ได้ libpreproc_xrt.so ที่ preproc_accel.py เรียกผ่าน ctypes
//
// การทำงาน: จอง BO 3 ก้อน (src / params / dst) ครั้งเดียว → ต่อเฟรม:
//   memcpy เฟรม BGR ลง src BO → sync to device → run kernel → wait → sync from device
//   → memcpy ออกเป็น int8 [1,640,640,3] ให้ VART
// (zero-copy กับ VART ยังไม่ทำใน v1 — memcpy 1.2 MB บน A53 ≈ 0.5 ms)
//
// หมายเหตุ Kria: xclbin ที่ใช้ต้องเป็นตัวเดียวกับที่ xmutil loadapp โหลดอยู่
// (มี DPU B4096 + preproc_accel อยู่ด้วยกัน) ดู ../vitis/README ใน pl-preproc/README.md
// ============================================================================
#include <cstdio>
#include <cstring>
#include <string>
#include <stdexcept>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include "../hls/preproc_accel.h"

struct PPCtx {
    xrt::device dev;
    xrt::uuid   uuid;
    xrt::kernel krn;
    xrt::bo     bo_src, bo_par, bo_dst;
    size_t      src_cap = 0;
    int         last_w = -1, last_h = -1;
};

extern "C" {

// คืน handle หรือ NULL (พิมพ์สาเหตุลง stderr)
void *pp_open(const char *xclbin_path, const char *kernel_name, int max_src_bytes) {
    try {
        PPCtx *c = new PPCtx();
        c->dev  = xrt::device(0);
        c->uuid = c->dev.load_xclbin(xclbin_path);
        c->krn  = xrt::kernel(c->dev, c->uuid, kernel_name);
        c->src_cap = ((size_t)max_src_bytes + 63) & ~(size_t)63;
        // group_id(N) = memory bank ที่ argument N ต่ออยู่ (ตาม connectivity ใน .cfg)
        c->bo_src = xrt::bo(c->dev, c->src_cap,                 c->krn.group_id(0));
        c->bo_par = xrt::bo(c->dev, PP_PARAM_WORDS * 2,         c->krn.group_id(1));
        c->bo_dst = xrt::bo(c->dev, (size_t)PP_DST_BYTES,       c->krn.group_id(2));
        return c;
    } catch (const std::exception &e) {
        fprintf(stderr, "[preproc_xrt] open failed: %s\n", e.what());
        return nullptr;
    }
}

// โหลดตาราง (4096 x int16) — เรียกใหม่เมื่อขนาดเฟรมหรือ fixpos เปลี่ยน
int pp_set_params(void *h, const int16_t *params) {
    if (!h) return -1;
    PPCtx *c = (PPCtx *)h;
    try {
        c->bo_par.write(params, PP_PARAM_WORDS * 2, 0);
        c->bo_par.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "[preproc_xrt] set_params failed: %s\n", e.what());
        return -2;
    }
}

// รัน 1 เฟรม: src = BGR uint8 [h][w][3] ต่อเนื่อง, dst = int8 [640][640][3]
// คืน 0 = ok, <0 = error
int pp_run(void *h, const uint8_t *src, int w, int hgt, int8_t *dst) {
    if (!h) return -1;
    PPCtx *c = (PPCtx *)h;
    const size_t nbytes = (size_t)w * hgt * 3;
    if (w < 8 || w > PP_MAX_SRC_W || (w & 7) || hgt < 2 || hgt > PP_MAX_SRC_H || nbytes > c->src_cap) {
        fprintf(stderr, "[preproc_xrt] unsupported frame %dx%d (cap %zu)\n", w, hgt, c->src_cap);
        return -3;
    }
    try {
        c->bo_src.write(src, nbytes, 0);
        c->bo_src.sync(XCL_BO_SYNC_BO_TO_DEVICE, nbytes, 0);
        xrt::run r = c->krn(c->bo_src, c->bo_par, c->bo_dst, w, hgt);
        r.wait();
        c->bo_dst.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        c->bo_dst.read(dst, (size_t)PP_DST_BYTES, 0);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "[preproc_xrt] run failed: %s\n", e.what());
        return -4;
    }
}

void pp_close(void *h) {
    delete (PPCtx *)h;
}

} // extern "C"
