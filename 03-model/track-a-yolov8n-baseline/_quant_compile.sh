#!/usr/bin/env bash
# Track A — quantize (PTQ) + compile yolov8n_leaky_pkg → xmodel สำหรับ KV260 (B4096)
set +e
source /opt/vitis_ai/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate vitis-ai-pytorch 2>/dev/null
python -c "import ultralytics" 2>/dev/null || pip install --no-deps -q "ultralytics<8.1"

Q=/workspace/03-model/track-a-yolov8n-baseline/quantize
ART=/workspace/03-model/track-a-yolov8n-baseline/artifacts/yolov8n_pkg_B4096
W=/workspace/03-model/track-a-yolov8n-baseline/finetune/yolov8n_leaky_pkg_ft.pt
CALIB=/workspace/02-dataset/calib/images
ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
mkdir -p "$ART"
cd "$Q"
rm -rf quantize_result

echo "########## PASS 1: calibrate (32 imgs — low-RAM host, avoid OOM) ##########"
python quantize_yolov8n_pytorch.py --weights "$W" --calib_dir "$CALIB" \
    --quant_mode calib --subset_len 32 --batch_size 4 2>&1 | tail -15

echo "########## PASS 2: test + export xmodel ##########"
python quantize_yolov8n_pytorch.py --weights "$W" --calib_dir "$CALIB" \
    --quant_mode test --subset_len 1 --batch_size 1 --deploy 2>&1 | tail -15

echo "########## quantize_result ##########"
ls -la quantize_result/

INT_XMODEL=$(ls quantize_result/*_int.xmodel 2>/dev/null | head -1)
if [ -z "$INT_XMODEL" ]; then
    echo "########## ABORT: no *_int.xmodel produced by PASS 2 — check XIR convert errors above. Skipping compile. ##########"
    exit 1
fi
echo "using INT_XMODEL=$INT_XMODEL"

echo "########## COMPILE (vai_c_xir, B4096) ##########"
vai_c_xir -x "$INT_XMODEL" -a "$ARCH" \
    -o "$ART" -n yolov8n_pkg_kv260 2>&1 | tee "$ART/vai_c_xir_yolov8n_pkg.log"

echo "########## PARTITION CHECK ##########"
xdputil xmodel "$ART/yolov8n_pkg_kv260.xmodel" -l 2>&1 | grep -E '"(name|device|fingerprint|DPU Arch)"' | head -40
echo "########## md5 ##########"
md5sum "$ART/yolov8n_pkg_kv260.xmodel"
echo "DONE"
