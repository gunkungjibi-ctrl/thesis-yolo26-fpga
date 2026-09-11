#!/bin/sh
# ============================================================================
# m13_quickstart.sh — เทส M13 Stage A0 + B0 บนบอร์ด KV260 ในคำสั่งเดียว
# ----------------------------------------------------------------------------
# ทำให้ครบตั้งแต่ต้น: เช็ค env → เช็คเน็ต → โหลดสคริปต์จาก GitHub → รันเทส
# ไม่ต้องมีไฟล์รูป/คลิป/xmodel บนบอร์ดเลย (Stage C ค่อยว่ากันตอนมี PC)
#
# วิธีใช้ (เลือกทางใดทางหนึ่ง):
#   A) โหลดมาดูก่อนแล้วค่อยรัน  ← แนะนำ
#      wget -O m13.sh https://raw.githubusercontent.com/gunkungjibi-ctrl/thesis-yolo26-fpga/claude/accelerator-test-apuc1z/04-deploy/board/m13_quickstart.sh
#      sh m13.sh
#   B) บรรทัดเดียวจบ (พิมพ์บนมือถือง่ายกว่า)
#      wget -qO- <URL ข้างบน> | sh
#
# เขียนด้วย POSIX sh ล้วน (busybox ash ของ PetaLinux รันได้)
# ============================================================================

BRANCH=claude/accelerator-test-apuc1z
RAW=https://raw.githubusercontent.com/gunkungjibi-ctrl/thesis-yolo26-fpga/$BRANCH
WORKDIR=${M13_DIR:-/tmp/m13}
WGET_OPTS=""

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }
ok()  { printf '  [ok] %s\n' "$1"; }
bad() { printf '  [!!] %s\n' "$1"; }

# ---------------------------------------------------------------- 1. env
say "1. สภาพแวดล้อมบอร์ด"
printf '  arch    : %s\n' "$(uname -m)"
printf '  kernel  : %s\n' "$(uname -r)"
printf '  date    : %s\n' "$(date)"
python3 - <<'PY' 2>/dev/null || bad "python3 / numpy / cv2 ไม่ครบ — เทสนี้ต้องใช้ทั้งสามตัว"
import sys, numpy, cv2
print("  python  : %s" % sys.version.split()[0])
print("  numpy   : %s" % numpy.__version__)
print("  cv2     : %s" % cv2.__version__)
PY

# นาฬิกาบอร์ดไม่มีแบต RTC — ถ้าเป็นปี 1970 การตรวจ certificate จะล้มทุกครั้ง
YEAR=$(date +%Y)
if [ "$YEAR" -lt 2020 ] 2>/dev/null; then
    bad "วันที่บอร์ดเป็นปี $YEAR — HTTPS จะล้มเพราะ cert ยัง 'not yet valid'"
    bad "บอร์ดไม่มีแบต RTC เลยลืมเวลาทุกครั้งที่ดับไฟ ตั้งเวลาก่อนแล้วรันใหม่:"
    bad "    date -s \"2026-09-11 12:00:00\"   (ใส่วันที่วันนี้จริง ๆ)"
    echo
fi

# ---------------------------------------------------------------- 2. net
say "2. เช็คอินเทอร์เน็ต"
PROBE="$RAW/04-deploy/board/preproc_lib.py"
if wget -q -O /dev/null "$PROBE" 2>/dev/null; then
    ok "ต่อ GitHub ได้ (HTTPS ปกติ)"
elif wget -q --no-check-certificate -O /dev/null "$PROBE" 2>/dev/null; then
    WGET_OPTS="--no-check-certificate"
    ok "ต่อได้ แต่ต้องข้ามการตรวจ certificate (CA bundle เก่า/นาฬิกาเพี้ยน)"
else
    bad "ต่อ GitHub ไม่ได้ — ไล่เช็คตามนี้:"
    echo "     ip addr show | grep 'inet '        # มี IP ไหม"
    echo "     ip route | grep default            # มี default route ไหม"
    echo "     ping -c2 8.8.8.8                   # ออกเน็ตได้ไหม"
    echo "     ping -c2 raw.githubusercontent.com # DNS ทำงานไหม"
    echo "     cat /etc/resolv.conf               # ว่างอยู่ไหม"
    echo "  ถ้า DNS ไม่มี:  echo 'nameserver 8.8.8.8' > /etc/resolv.conf"
    echo "  ถ้าเน็ตออกไม่ได้จริง ให้ใช้ทาง 0A (PC ตั้ง http.server) ใน M13_BOARD_TEST.md"
    exit 1
fi

# ---------------------------------------------------------------- 3. download
say "3. โหลดสคริปต์เทส -> $WORKDIR"
mkdir -p "$WORKDIR" || exit 1
cd "$WORKDIR" || exit 1
FAIL=0
for f in 04-deploy/pl-preproc/host/preproc_tables.py \
         04-deploy/pl-preproc/golden/preproc_golden.py \
         04-deploy/board/preproc_lib.py \
         04-deploy/board/bench_preproc.py \
         04-deploy/board/bench_latency.py \
         04-deploy/board/video_detect.py ; do
    n=$(basename "$f")
    if wget -q $WGET_OPTS -O "$n" "$RAW/$f"; then
        ok "$n ($(wc -c < "$n") bytes)"
    else
        bad "$n โหลดไม่สำเร็จ"; FAIL=1
    fi
done
[ "$FAIL" = 0 ] || { bad "โหลดไฟล์ไม่ครบ หยุดก่อน"; exit 1; }

# ---------------------------------------------------------------- 4. stage A0
say "4. Stage A0 — golden model vs cv2 ของบอร์ด (ARM/NEON)"
echo "  เทียบสูตร fixed-point ที่จะเอาไปทำ HW กับ cv2 ตัวจริงบนบอร์ด"
echo "  ต้องได้ worst = 0 ทุกขนาด ถึงจะลงทุน synth ได้"
python3 preproc_golden.py check --sizes 640x360,640x640,1280x720,1920x1080 --n 2
A0=$?

# ---------------------------------------------------------------- 5. stage B0
say "5. Stage B0 — วัด preprocessing จริงบน Cortex-A53"
echo "  numpy = โค้ดเดิมที่ M12 วัดได้ 49.7 ms · lut = ทางที่ optimize แล้ว"
python3 bench_preproc.py --synth 640x640 --modes numpy,lut --verify --iters 50
B0=$?
echo
echo "  -- ขนาดเฟรมที่ deploy จริง (คลิปนับกล่อง 640x360) --"
python3 bench_preproc.py --synth 640x360 --modes numpy,lut --verify --iters 50

# ---------------------------------------------------------------- 6. summary
say "6. สรุป"
[ "$A0" = 0 ] && ok "Stage A0 ผ่าน — สูตร HW ตรงกับ cv2 ของบอร์ด" \
              || bad "Stage A0 ไม่ผ่าน — ส่ง output ทั้งหมดมาให้ Claude ปรับสูตร (ยังไม่ได้ synth เลยแก้ได้ฟรี)"
[ "$B0" = 0 ] && ok "Stage B0 ผ่าน — lut ให้ผลตรงกับโค้ดเดิมทุกไบต์" \
              || bad "Stage B0 ไม่ผ่าน — ห้ามใช้โหมด lut ต่อ ส่ง output มาก่อน"
echo
echo "  ก๊อป output ทั้งหมดตั้งแต่หัวข้อ 1 ส่งให้ Claude เพื่อบันทึกลง timeline + บท Results"
echo "  Stage C (e2e + COUNT 215) ต้องมี .xmodel กับคลิป — โอนจาก PC ตาม M13_BOARD_TEST.md"
