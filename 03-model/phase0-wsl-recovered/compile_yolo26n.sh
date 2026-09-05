#!/usr/bin/env bash
# compile_yolo26n.sh  —  run INSIDE the Vitis AI 3.0 Docker container
# ===================================================================
# Compiles the quantized XIR xmodel into a DPU-deployable xmodel for the
# Kria KV260 (DPUCZDX8G_ISA1_B3136), and prints the subgraph partition that
# is the Phase 0 exit gate.
#
# PREREQUISITE: quantize_yolo26n.py was run with enable_dpu=True and produced
#               yolo26n_int.xmodel (XIR format). The ONNX/VOE flow does NOT
#               produce a vai_c_xir-compatible input.
set -euo pipefail

INT_XMODEL="quantize_result/BackboneHead_int.xmodel"
OUT_NAME="yolo26n_kv260"
NET_NAME="yolo26n_kv260"

# ---- arch.json location -------------------------------------------------
# arch.json ships INSIDE the Docker image, NOT in any git repo. For MPSoC the
# folder is /opt/vitis_ai/compiler/arch/DPUCZDX8G/<Target>/arch.json .
# For KV260 the directory is "KV260".
ARCH="/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"

# ---- sanity checks ------------------------------------------------------
[ -f "$INT_XMODEL" ] || { echo "ERROR: $INT_XMODEL not found. Run quantize_yolo26n.py first."; exit 1; }
[ -f "$ARCH" ]       || { echo "ERROR: $ARCH not found. Is this the Vitis AI 3.0 image? Check ls /opt/vitis_ai/compiler/arch/DPUCZDX8G/"; exit 1; }

echo "=== arch.json being used ==="
cat "$ARCH"
echo
echo "=== compiling with vai_c_xir ==="

vai_c_xir \
  -x "$INT_XMODEL" \
  -a "$ARCH" \
  -o . \
  -n "$NET_NAME" \
  2>&1 | tee vai_c_xir_${OUT_NAME}.log

echo
echo "================ PHASE 0 GATE — read the log above ================"
echo "PASS  = compiler reports exactly 1 DPU subgraph (kernel count: DPU=1)."
echo "        Anything that lands on CPU shows as extra non-DPU subgraphs."
echo "INSPECT the produced xmodel's partition explicitly:"
echo "        xdputil xmodel ${NET_NAME}.xmodel -l"
echo "   -> every op should map to the DPU subgraph; CPU ops = fallback = bad."
echo
echo "!! FINGERPRINT WARNING !!"
echo "  The KV260 arch.json in the Docker image carries a default fingerprint."
echo "  The DPU actually loaded on YOUR board depends on the firmware/overlay"
echo "  you load with 'xmutil loadapp'. If the xmodel fingerprint != the"
echo "  board's live DPU fingerprint, the board throws at runtime:"
echo "    'CHECK fingerprint fail! ... Please re-compile xmodel with"
echo "     dpu_fingerprint 0x...'  (NOT a Phase-0 issue — surfaces in Phase 1)."
echo "  When the board arrives, run 'xdputil query' on it, read the live"
echo "  DPUCZDX8G_ISA1_B3136 fingerprint, and confirm it matches this arch.json"
echo "  before trusting on-board inference."
