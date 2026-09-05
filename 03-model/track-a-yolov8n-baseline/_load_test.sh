#!/usr/bin/env bash
# ทดสอบว่า checkpoint (train บน ultralytics 8.4.71) โหลดในคอนเทนเนอร์ (py3.7, ultralytics<8.1) ได้ไหม
set +e
source /opt/vitis_ai/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate vitis-ai-pytorch 2>/dev/null

echo "=== ensure ultralytics<8.1 (py3.7) ==="
python -c "import ultralytics; print('already', ultralytics.__version__)" 2>/dev/null || pip install --no-deps -q "ultralytics<8.1" 2>&1 | tail -3

W=/workspace/03-model/track-a-yolov8n-baseline/finetune/yolov8n_leaky_pkg_ft.pt
echo "=== load test: $W ==="
python - <<PY
import torch, torch.nn as nn
try:
    from ultralytics import YOLO
    print("ultralytics:", __import__("ultralytics").__version__)
    y = YOLO("$W")
    print("LOAD OK")
    inner = y.model
    silu = sum(isinstance(m, nn.SiLU) for m in inner.modules())
    leaky = sum(isinstance(m, nn.LeakyReLU) for m in inner.modules())
    print(f"activations: SiLU={silu} LeakyReLU={leaky}")
    det = inner.model[-1]
    print("head type:", type(det).__name__, "| nc:", getattr(det,'nc','?'))
except Exception as e:
    import traceback; traceback.print_exc()
    print("LOAD FAIL:", type(e).__name__, str(e)[:300])
PY
