# Phase 0 — quantize + compile YOLO26n for KV260 (B3136), Vitis AI 3.0
# Copy-paste runbook. Everything below runs on a HOST that can pull Docker
# (this Claude window cannot — its network blocks Docker Hub / xilinx registry).

# ----------------------------------------------------------------------------
# 0. Files you carry into the container
#    - yolo26n_o2m_leakyrelu.onnx   (from export_yolo26n_for_dpu.py, with REAL
#      trained+fine-tuned weights — the YAML-only graph from the Claude window
#      has no trained weights and is for op-graph inspection only)
#    - quantize_yolo26n.py
#    - compile_yolo26n.sh
#    - calib_images/   (100-1000 real conveyor frames; NOT random data)
# ----------------------------------------------------------------------------

# 1. Pull the PINNED 3.0 image. Do NOT use :latest (it is 3.5; KV260 prebuilt
#    image + board_setup/mpsoc only exist on the 3.0 branch).
docker pull xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106

# 2. Launch the container with your work mounted.
docker run -it --rm \
  -v "$PWD":/workspace \
  -w /workspace \
  xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106 bash

# ---- everything below runs INSIDE the container ----

# 3. Activate the env that has vai_q_onnx + vai_c_xir.
conda activate vitis-ai-pytorch

# 4. Confirm the toolchain is the 3.0 one and the KV260 arch.json exists.
python -c "import vai_q_onnx, sys; print('vai_q_onnx OK', sys.executable)"
ls /opt/vitis_ai/compiler/arch/DPUCZDX8G/        # expect KV260/ among targets
cat /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json

# 5. Quantize: float ONNX -> INT8 XIR xmodel (enable_dpu=True path).
python quantize_yolo26n.py

# 6. Compile: XIR xmodel -> DPU xmodel + the Phase 0 gate log.
bash compile_yolo26n.sh

# 7. Phase 0 PASS criterion (the whole point of Phase 0):
#    vai_c_xir_*.log shows exactly 1 DPU subgraph, no ops kicked to CPU.
#    Inspect explicitly:
xdputil xmodel yolo26n_kv260.xmodel -l
#    -> if any op shows device USER/CPU instead of DPU, that op fell back.
#       Note which op, decide: replace it host-side, or fall back to a
#       Model-Zoo-proven net (YOLOv5/v4) per the project's fallback plan.

# ----------------------------------------------------------------------------
# IF YOLO26n DOES NOT REACH 1 DPU SUBGRAPH
#   This is the expected risk, not a failure of the work. The fallback chain
#   (YOLOv5 -> YOLOv8 -> YOLOv3/v4 -> YOLOX) exists precisely for this. The
#   deliverable is a working object-counting system, not YOLO26 specifically.
# ----------------------------------------------------------------------------
