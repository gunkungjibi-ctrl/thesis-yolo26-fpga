#!/usr/bin/env bash
# ============================================================================
# build.sh — compile-check M4 host code.
#   host (compile-check เท่านั้น): รันใน Vitis AI 3.0 container (x86) เพื่อยืนยัน
#     ว่าโค้ด compile ผ่าน + ลิงก์ VART/XIR ครบ. รันจริงไม่ได้ (ไม่มี DPU).
#   board (Phase 1): cross-compile หรือ build บน KV260 (Ubuntu-on-Kria) ตรง ๆ.
#
# ใช้ CMake ถ้ามี, ไม่งั้น fallback g++ บรรทัดเดียว.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

if command -v cmake >/dev/null 2>&1; then
  echo "[build] cmake path"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j"$(nproc)"
  echo "[build] OK -> build/yolo_dpu_infer"
else
  echo "[build] cmake not found -> g++ fallback"
  g++ -std=c++17 -O2 yolo_dpu_infer.cpp -o yolo_dpu_infer \
    $(pkg-config --cflags --libs opencv4 2>/dev/null || echo "-lopencv_core -lopencv_imgproc -lopencv_imgcodecs") \
    -lvart-runner -lxir -lglog -pthread
  echo "[build] OK -> ./yolo_dpu_infer"
fi
