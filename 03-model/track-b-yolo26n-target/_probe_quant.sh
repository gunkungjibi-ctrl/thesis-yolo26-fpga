#!/usr/bin/env bash
source /opt/vitis_ai/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate vitis-ai-pytorch 2>/dev/null
echo "=== in vitis-ai-pytorch env ==="
which vai_c_xir || echo "vai_c_xir: NOT in PATH"
which xcompiler 2>/dev/null
python -c "import vai_q_onnx; print('vai_q_onnx', vai_q_onnx.__version__)" 2>&1 | tail -1
echo "=== arch.json for B4096 present? ==="
ls -la /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json 2>&1 | tail -1
find / -name "arch.json" -path "*KV260*" 2>/dev/null | head -3
