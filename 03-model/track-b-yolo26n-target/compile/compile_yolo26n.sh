#!/usr/bin/env bash
# compile_yolo26n.sh  —  Track B  —  run INSIDE the Vitis AI 3.0 Docker container
# ==============================================================================
# Compiles the quantized XIR xmodel (from quantize_yolo26n_pytorch.py, the
# vai_q_pytorch flow) into a DPU-deployable xmodel for the Kria KV260, and
# prints the subgraph partition that is the Track B gate.
#
# PREREQUISITE: quantize_yolo26n_dpu.py was run with --quant_mode test
#               --deploy, producing quantize_result/YOLO26nBackboneHead_int.xmodel
#               (the wrapper class name determines the filename — check it).
set -euo pipefail

# ---- input -----------------------------------------------------------------
# The exported filename comes from the nn.Module class name in the quantize
# script. Track B uses YOLO26nBackboneHead; Track A used BackboneHead.
# If this path is wrong, `ls quantize_result/` and fix it — do not guess.
INT_XMODEL="${INT_XMODEL:-quantize_result/YOLO26nBackboneHead_int.xmodel}"
NET_NAME="yolo26n_kv260"

# ---- DPU target ------------------------------------------------------------
# RESOLVED in P1 (07-notes/P1_arch_and_identity_resolution.md): the real arch is
# B4096, confirmed from three sources (the phase-0 vai_c_xir log, meta.json, and
# the KV260 arch.json) and later by `xdputil query` on the board itself
# (fingerprint 0x101000056010407). The B3136 in the old phase-0 scripts was a
# stale comment from an inspect dry-run.
# This MUST match TARGET in quantize_yolo26n_dpu.py — the quantizer is
# hardware-aware, so a mismatch is not fixable at compile time.
DPU_TARGET="${DPU_TARGET:-KV260}"
ARCH="${ARCH:-/opt/vitis_ai/compiler/arch/DPUCZDX8G/${DPU_TARGET}/arch.json}"

# ---- sanity checks ---------------------------------------------------------
[ -f "$INT_XMODEL" ] || { echo "ERROR: $INT_XMODEL not found. Run quantize_yolo26n_pytorch.py --deploy first, then 'ls quantize_result/'."; exit 1; }
[ -f "$ARCH" ]       || { echo "ERROR: $ARCH not found. Available targets:"; ls /opt/vitis_ai/compiler/arch/DPUCZDX8G/; exit 1; }

echo "=== arch.json being used ==="
cat "$ARCH"
echo
echo "=== compiling with vai_c_xir ==="

vai_c_xir \
  -x "$INT_XMODEL" \
  -a "$ARCH" \
  -o . \
  -n "$NET_NAME" \
  2>&1 | tee "vai_c_xir_${NET_NAME}.log"

echo
echo "================ TRACK B GATE — read the log above ================"

# Pull the verdict out of the log rather than leaving it to the eye. The log
# line is "DPU subgraph number N"; N is the whole answer for M2-B3.
LOG="vai_c_xir_${NET_NAME}.log"
SUBGRAPHS="$(grep -oE 'DPU subgraph number[[:space:]]+[0-9]+' "$LOG" | grep -oE '[0-9]+$' | tail -1 || true)"
if [ -n "$SUBGRAPHS" ]; then
  echo "PARSED: DPU subgraph number = ${SUBGRAPHS}"
  if [ "$SUBGRAPHS" = "1" ]; then
    echo "VERDICT: PASS — single DPU subgraph."
  else
    echo "VERDICT: PARTITIONED into ${SUBGRAPHS} DPU subgraphs."
    echo "  Expected if the attention sites (model.10 C2PSA, model.22 C3k2 attn=True)"
    echo "  did not map — that is the M2-B1 prediction, now measured rather than inferred."
  fi
else
  echo "VERDICT: compiler produced no subgraph count — read the log for the error."
fi

echo
echo "PASS = compiler reports exactly 1 DPU subgraph (kernel count: DPU=1)."
echo "       Anything landing on CPU shows as extra non-DPU subgraphs."
echo
echo "Inspect the partition explicitly (this, not the log summary, is truth):"
echo "       xdputil xmodel ${NET_NAME}.xmodel -l"
echo "  -> any op on device USER/CPU = fell back to the ARM PS."
echo
echo "IF IT FAILS: that is the expected risk, not wasted work."
echo "  Record WHICH ops fell back and why -> that is the Track B contribution."
echo "  Then continue with Track A (YOLOv8n), which already passed Phase 0."
echo
echo "!! FINGERPRINT WARNING (surfaces in Phase 1, not here) !!"
echo "  arch.json in the Docker image carries a default fingerprint. The DPU"
echo "  actually loaded on the board depends on the overlay from 'xmutil loadapp'."
echo "  On the real board run 'xdputil query' and compare before trusting"
echo "  on-board inference. Mismatch => re-compile with the board's arch.json."
