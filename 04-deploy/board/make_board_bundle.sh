#!/usr/bin/env bash
# ============================================================================
# make_board_bundle.sh — รวมไฟล์ M13 ที่ต้องใช้บนบอร์ดเป็น tar เดียว (flat layout)
# ----------------------------------------------------------------------------
# PetaLinux starter kit ไม่มี sftp-server → scp ใช้ไม่ได้ ต้องโอนผ่าน HTTP ทีละไฟล์
# สคริปต์นี้ทำเป็นไฟล์เดียวให้ wget รอบเดียวจบ
#
#   ./make_board_bundle.sh                  -> m13_board.tar.gz (ใน 04-deploy/board/)
#   python3 -m http.server 8000             # ที่ host
#   # บนบอร์ด:
#   cd /tmp && wget http://<HOST_IP>:8000/m13_board.tar.gz && tar xzf m13_board.tar.gz
#
# หมายเหตุ: วางแบน (ทุกไฟล์ในโฟลเดอร์เดียว) ได้เพราะ preproc_golden.py หา
# preproc_tables.py ทั้งใน ../host และในโฟลเดอร์ตัวเอง
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PL="$HERE/../pl-preproc"
OUT="${1:-$HERE/m13_board.tar.gz}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

D="$STAGE/m13"
mkdir -p "$D"
# stage 1-2 (software เท่านั้น — ไม่ต้องมี xclbin)
cp "$HERE/preproc_lib.py" "$HERE/bench_preproc.py" \
   "$HERE/bench_latency.py" "$HERE/video_detect.py" "$HERE/yolo_dpu_detect.py" "$D/"
cp "$PL/golden/preproc_golden.py" "$PL/host/preproc_tables.py" "$D/"
# stage 3 (ใช้ตอนมี xclbin แล้ว)
cp "$PL/host/preproc_accel.py" "$PL/host/preproc_xrt.cpp" "$PL/hls/preproc_accel.h" "$D/"
cp "$HERE/M13_BOARD_TEST.md" "$D/" 2>/dev/null || true

cat > "$D/build_hw_lib.sh" <<'EOF'
#!/bin/sh
# build libpreproc_xrt.so บนบอร์ด (stage 3 — ต้องมี xclbin แล้วเท่านั้น)
set -e
g++ -O2 -std=c++17 -fPIC -Wall -I/usr/include/xrt -I. -shared \
    -o libpreproc_xrt.so preproc_xrt.cpp -lxrt_coreutil -lpthread
echo "ok: $(ls -l libpreproc_xrt.so)"
EOF
chmod +x "$D/build_hw_lib.sh"
# preproc_xrt.cpp include "../hls/preproc_accel.h" -> flat layout ต้องแก้เป็นไฟล์ข้างกัน
sed -i 's|#include "../hls/preproc_accel.h"|#include "preproc_accel.h"|' "$D/preproc_xrt.cpp"

tar czf "$OUT" -C "$STAGE" m13
echo "[bundle] $OUT"
tar tzf "$OUT" | sed 's/^/  /'
