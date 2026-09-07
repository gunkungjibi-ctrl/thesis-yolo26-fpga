// ============================================================================
// tb_preproc.cpp — C testbench ของ preproc_accel (ใช้ได้ทั้ง g++ และ Vitis HLS csim/cosim)
// ----------------------------------------------------------------------------
// อ่าน test vector ที่ golden/preproc_golden.py gen ไว้:
//   src_WxH_bgr.bin  params_WxH.bin  golden_WxH.bin  meta_WxH.txt ("W H fixpos")
// รัน kernel แล้วเทียบ output ทุกไบต์กับ golden (bit-exact เท่านั้นถึงผ่าน)
//
// usage: tb_preproc <vector_dir> [WxH ...]      (ไม่ระบุ = ทุก meta_*.txt ในโฟลเดอร์)
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dirent.h>
#include "preproc_accel.h"

static bool read_file(const std::string &p, std::vector<uint8_t> &buf) {
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    buf.resize((size_t)n);
    size_t r = fread(buf.data(), 1, (size_t)n, f);
    fclose(f);
    return r == (size_t)n;
}

static int run_case(const std::string &dir, const std::string &tag) {
    std::vector<uint8_t> src, prm, gold;
    if (!read_file(dir + "/src_" + tag + "_bgr.bin", src) ||
        !read_file(dir + "/params_" + tag + ".bin", prm) ||
        !read_file(dir + "/golden_" + tag + ".bin", gold)) {
        printf("[%s] missing vector files\n", tag.c_str()); return 1;
    }
    int w = 0, h = 0, fixpos = 0;
    FILE *m = fopen((dir + "/meta_" + tag + ".txt").c_str(), "r");
    if (!m || fscanf(m, "%d %d %d", &w, &h, &fixpos) != 3) { printf("[%s] bad meta\n", tag.c_str()); return 1; }
    fclose(m);
    if (src.size() != (size_t)w * h * 3 || prm.size() != PP_PARAM_WORDS * 2 || gold.size() != PP_DST_BYTES) {
        printf("[%s] size mismatch src=%zu prm=%zu gold=%zu\n", tag.c_str(), src.size(), prm.size(), gold.size());
        return 1;
    }
    // ทำ src ให้ align 8 byte (m_axi 64 bit)
    std::vector<uint64_t> src64((src.size() + 7) / 8, 0);
    memcpy(src64.data(), src.data(), src.size());
    std::vector<uint64_t> dst64(PP_DST_BYTES / 8, 0xEEEEEEEEEEEEEEEEull);
    std::vector<int16_t> prm16(PP_PARAM_WORDS);
    memcpy(prm16.data(), prm.data(), prm.size());

    preproc_accel(src64.data(), prm16.data(), dst64.data(), w, h);

    const uint8_t *out = (const uint8_t *)dst64.data();
    size_t mism = 0; int maxd = 0; size_t first = (size_t)-1;
    for (size_t i = 0; i < (size_t)PP_DST_BYTES; i++) {
        int d = abs((int)(int8_t)out[i] - (int)(int8_t)gold[i]);
        if (d) { mism++; if (first == (size_t)-1) first = i; if (d > maxd) maxd = d; }
    }
    if (mism == 0) {
        printf("[%s] PASS  %dx%d -> 640x640 fixpos=%d  (%d bytes bit-exact)\n", tag.c_str(), w, h, fixpos, PP_DST_BYTES);
        return 0;
    }
    size_t px = first / 3;
    printf("[%s] FAIL  mismatches=%zu max|diff|=%d  first at byte %zu (y=%zu x=%zu c=%zu) hw=%d gold=%d\n",
           tag.c_str(), mism, maxd, first, px / PP_DST_W, px % PP_DST_W, first % 3,
           (int)(int8_t)out[first], (int)(int8_t)gold[first]);
    return 1;
}

int main(int argc, char **argv) {
    std::string dir = argc > 1 ? argv[1] : "vectors";
    std::vector<std::string> tags;
    for (int i = 2; i < argc; i++) tags.push_back(argv[i]);
    if (tags.empty()) {
        DIR *d = opendir(dir.c_str());
        if (!d) { printf("cannot open %s\n", dir.c_str()); return 1; }
        while (dirent *e = readdir(d)) {
            std::string n = e->d_name;
            if (n.rfind("meta_", 0) == 0 && n.size() > 9) tags.push_back(n.substr(5, n.size() - 9));
        }
        closedir(d);
    }
    if (tags.empty()) { printf("no vectors in %s (run golden/preproc_golden.py gen)\n", dir.c_str()); return 1; }
    int fails = 0;
    for (auto &t : tags) fails += run_case(dir, t);
    printf("==== %zu case(s), %d failed ====\n", tags.size(), fails);
    return fails ? 1 : 0;
}
