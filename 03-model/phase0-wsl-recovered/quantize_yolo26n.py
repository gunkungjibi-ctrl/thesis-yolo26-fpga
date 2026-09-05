#!/usr/bin/env python3
"""
quantize_yolo26n.py  —  run INSIDE the Vitis AI 3.0 Docker container
====================================================================
Docker image (pinned, do NOT use :latest which pulls 3.5):
    xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106

Conda env inside container that has vai_q_onnx:
    conda activate vitis-ai-pytorch     # vai_q_onnx lives here in 3.0

WHICH FLOW THIS IS — read before running
----------------------------------------
There are TWO different quantize outputs in Vitis AI 3.0, and they do NOT
both feed vai_c_xir:

  (A) ONNX-Runtime / VOE flow  -> quantize_static() with enable_dpu left False
      produces a quantized *.onnx*. Subgraph partition happens at RUNTIME via
      VOE on the board. There is NO vai_c_xir step and NO standalone
      "1 DPU subgraph" compiler log. NOT what we want for the Phase 0 gate.

  (B) XIR flow  -> quantize_static() with enable_dpu=True produces a
      *_int.xmodel* (XIR). THIS is the input vai_c_xir expects, and vai_c_xir
      is what prints the DPU-subgraph partition log that is our Phase 0 exit
      gate (target: 1 DPU subgraph).

This script uses flow (B). If you instead want the runtime/VOE path, set
ENABLE_DPU=False and skip compile_yolo26n.sh entirely.

CALIBRATION
-----------
PowerOfTwoMethod.MinMSE is the calibration method that matches the DPUCZDX8G
fixed-point (power-of-two scale) hardware. Per AMD docs, when using
PowerOfTwo calibration you MUST also set, in extra_options:
    ActivationSymmetric = True
    WeightSymmetric     = True
    AddQDQPairToWeight  = True
    DedicatedQDQPair    = True
Use 100–1000 real calibration images from your conveyor dataset (NOT random
data) or the INT8 accuracy will be meaningless.

NOTE on accuracy: the host export already swapped SiLU->LeakyReLU, which
shifts accuracy. Quantization shifts it again. Phase 0 only proves the model
COMPILES to 1 DPU subgraph; accuracy/fine-tuning is a later phase.
"""

import os, glob
import numpy as np
import cv2
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader
import vai_q_onnx

# ---------------- config ----------------
FLOAT_ONNX   = "yolo26n_o2m_leakyrelu.onnx"      # from the host export step
OUT_XMODEL   = "yolo26n_int.xmodel"              # XIR output for vai_c_xir
CALIB_DIR    = "calib_images"                    # 100-1000 real conveyor frames
INPUT_NAME   = "images"
IMGSZ        = 640
CALIB_LEN    = 200
ENABLE_DPU   = True                              # True = XIR flow (-> vai_c_xir)


class ConveyorCalibReader(CalibrationDataReader):
    """Feeds real preprocessed frames. Preprocessing MUST match training:
       BGR->RGB, /255, NCHW, float32. Adjust if your pipeline differs."""
    def __init__(self, image_dir, input_name, imgsz, limit):
        files = sorted(glob.glob(os.path.join(image_dir, "*")))[:limit]
        if not files:
            raise FileNotFoundError(
                f"No calibration images in '{image_dir}'. "
                "Phase-0 with random data is invalid — use real frames.")
        self.input_name = input_name
        self.imgsz = imgsz
        self.files = files
        self._it = iter(files)

    def _load(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]      # 1x3xHxW
        return np.ascontiguousarray(img)

    def get_next(self):
        try:
            return {self.input_name: self._load(next(self._it))}
        except StopIteration:
            return None

    def rewind(self):
        self._it = iter(self.files)


def main():
    reader = ConveyorCalibReader(CALIB_DIR, INPUT_NAME, IMGSZ, CALIB_LEN)

    vai_q_onnx.quantize_static(
        model_input=FLOAT_ONNX,
        model_output=OUT_XMODEL,
        calibration_data_reader=reader,
        quant_format=vai_q_onnx.QuantFormat.QDQ,
        calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
        activation_type=vai_q_onnx.QuantType.QInt8,
        weight_type=vai_q_onnx.QuantType.QInt8,
        enable_dpu=ENABLE_DPU,            # True -> emits XIR *_int.xmodel
        extra_options={
            "ActivationSymmetric": True,  # required for PowerOfTwo
            "WeightSymmetric":     True,  # required for PowerOfTwo
            "AddQDQPairToWeight":  True,  # required for PowerOfTwo
            "DedicatedQDQPair":    True,  # required for PowerOfTwo
        },
    )
    print(f"[ok] wrote {OUT_XMODEL}")
    print("NEXT: bash compile_yolo26n.sh   (runs vai_c_xir -> Phase 0 gate)")


if __name__ == "__main__":
    main()
