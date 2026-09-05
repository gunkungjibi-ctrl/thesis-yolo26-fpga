#!/usr/bin/env python3
"""
quantize_yolo26n_pytorch.py  —  run INSIDE the Vitis AI 3.0 Docker container
============================================================================
Docker image (pinned): xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106
Conda env (has vai_q_pytorch):  conda activate vitis-ai-pytorch

This replaces the earlier vai_q_onnx approach. The 3.0 PyTorch docker ships
vai_q_pytorch (pytorch_nndct), NOT vai_q_onnx. This path quantizes the PyTorch
model directly -> XIR *_int.xmodel -> vai_c_xir, which matches the Phase 0
exit gate (1 DPU subgraph from the vai_c_xir log).

TWO HARD REQUIREMENTS from the vai_q_pytorch docs, both handled below:

  (1) "The model to be quantized should include forward method only. All other
       functions (pre/post-processing) must be moved outside."
       -> YOLO26n's Detect head runs decode/postprocess inside forward(). We
          wrap the model so forward() returns the THREE raw head feature maps
          only. Counting/NMS/decoding happen later on the ARM PS at deploy
          time, NOT inside the quantized graph. This is also exactly what we
          want architecturally: only the conv backbone+neck+head go on the DPU.

  (2) "The float model should pass the jit.trace test."
       -> We run torch.jit.trace as a pre-flight check and STOP if it fails,
          rather than letting the quantizer fail deep in calibration.

USAGE (two passes, as the API requires):
  # pass 1 - calibrate using real conveyor frames
  python quantize_yolo26n_pytorch.py --weights yolo26n_finetuned.pt \
         --calib_dir calib_images --quant_mode calib --subset_len 200

  # pass 2 - test + export the deployable XIR xmodel (batch=1, 1 iteration)
  python quantize_yolo26n_pytorch.py --weights yolo26n_finetuned.pt \
         --calib_dir calib_images --quant_mode test --subset_len 1 \
         --batch_size 1 --deploy

Output: quantize_result/<name>_int.xmodel  -> feed to compile_yolo26n.sh

PRE-FLIGHT (run once, optional but recommended) - hardware-aware inspection:
  python quantize_yolo26n_pytorch.py --weights ... --inspect
  -> Inspector reports which ops will NOT map to the DPU (i.e. ARM/PS fallback)
     BEFORE you spend time compiling. This is advisory; the real gate is still
     the vai_c_xir log.

NOTE ON WEIGHTS: this needs a REAL trained .pt. The YAML-only graph has no
trained weights and would calibrate to garbage. Fine-tune AFTER the
SiLU->LeakyReLU swap, since that swap shifts accuracy.
"""

import os, glob, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn

# ---- DPU target for KV260 (confirmed: DPUCZDX8G_ISA1_B3136) ----
TARGET = "DPUCZDX8G_ISA1_B3136"
IMGSZ = 640


# ----------------------------------------------------------------------------
# forward-only wrapper: expose ONLY the raw detection-head outputs.
# Everything decode/NMS/count related stays OFF the quantized graph.
# ----------------------------------------------------------------------------
class YOLO26nBackboneHead(nn.Module):
    """Forward-only wrapper that satisfies the vai_q_pytorch requirement.

    VERIFIED with the real yolo26n.pt (ultralytics 8.4.71):
      - end2end=False + export=True + format='onnx'  -> forward() returns a
        single DECODED tensor of shape (1, 84, 8400)  [4 bbox + 80 cls, anchors]
      - jit.trace PASSES on this output (no control flow left in forward)
      - TopK/GatherElements/ReduceMax/Mod (the NMS-free postprocess ops) are
        skipped because end2end=False.

    NOTE: this output is decoded (includes anchor concat/reshape). Whether those
    decode ops map to the DPU or fall back to the ARM PS is NOT assumed here -
    run `--inspect` first. If the Inspector shows decode ops going to CPU and it
    hurts the subgraph count, switch to a raw per-stride head output then. Do
    NOT pre-optimize that before the Inspector says it's needed.
    """
    def __init__(self, ultra_model):
        super().__init__()
        self.model = ultra_model
        det = self.model.model[-1]
        det.end2end = False        # skip one2one branch + postprocess() (TopK/Gather)
        det.export = True
        det.format = "onnx"

    def forward(self, x):
        y = self.model(x)
        if isinstance(y, (list, tuple)):
            return tuple(y)
        return y


def replace_silu_with_leakyrelu(module, slope=0.1015625):
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.LeakyReLU(slope, inplace=True))
        else:
            replace_silu_with_leakyrelu(child, slope)


def build_float_model(weights, keep_silu=False):
    from ultralytics import YOLO
    try:
        y = YOLO(weights)
        print(f"[ok] loaded weights: {weights}")
    except Exception as e:
        print(f"[warn] could not load '{weights}' ({e.__class__.__name__}); "
              f"building from yolo26n.yaml (NO trained weights - graph only, "
              f"calibration will be meaningless)")
        y = YOLO("yolo26n.yaml")
    inner = y.model
    if not keep_silu:
        replace_silu_with_leakyrelu(inner)
        print("[ok] SiLU -> LeakyReLU(0.1015625) done")
    model = YOLO26nBackboneHead(inner)
    model.eval()
    return model


# ----------------------------------------------------------------------------
# calibration data: real conveyor frames, preprocessed to match training.
# ----------------------------------------------------------------------------
def load_calib_batch(calib_dir, subset_len, imgsz):
    files = sorted(glob.glob(os.path.join(calib_dir, "*")))[:subset_len]
    if not files:
        raise FileNotFoundError(
            f"No images in '{calib_dir}'. Phase 0 with random data is invalid.")
    imgs = []
    for p in files:
        im = cv2.imread(p)
        if im is None:
            continue
        im = cv2.resize(im, (imgsz, imgsz))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        imgs.append(np.transpose(im, (2, 0, 1)))
    if not imgs:
        raise RuntimeError(f"No readable images in '{calib_dir}'.")
    return torch.from_numpy(np.ascontiguousarray(np.stack(imgs)))


def jit_trace_preflight(model, example):
    """Docs require the float model to pass jit.trace before quantizing."""
    try:
        with torch.no_grad():
            torch.jit.trace(model, example, check_trace=False)
        print("[ok] jit.trace pre-flight passed")
    except Exception as e:
        raise SystemExit(
            f"[FATAL] jit.trace failed: {e}\n"
            "The model still has non-traceable control flow in forward(). "
            "Move remaining post-processing out of the forward path before "
            "quantizing (vai_q_pytorch requirement).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--calib_dir", default="calib_images")
    ap.add_argument("--quant_mode", choices=["float", "calib", "test"], default="calib")
    ap.add_argument("--subset_len", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--deploy", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="run hardware-aware Inspector and exit")
    ap.add_argument("--keep_silu", action="store_true")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    args = ap.parse_args()

    device = torch.device("cpu")
    model = build_float_model(args.weights, keep_silu=args.keep_silu).to(device)
    example = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    # ---- Inspector path: advisory hardware-aware op assignment ----
    if args.inspect:
        from pytorch_nndct.apis import Inspector
        insp = Inspector(TARGET)
        insp.inspect(model, (example,), device=device)
        print("\n[info] Inspector done. Ops marked CPU/USER will fall back to "
              "the ARM PS. The real Phase 0 gate is still the vai_c_xir log.")
        return

    from pytorch_nndct.apis import torch_quantizer

    # export xmodel requires batch=1 and exactly 1 iteration
    if args.deploy:
        if args.batch_size != 1 or args.subset_len != 1:
            print("[warn] deploy requires batch_size=1 & subset_len=1; forcing.")
            args.batch_size, args.subset_len = 1, 1

    jit_trace_preflight(model, (example,))

    quantizer = torch_quantizer(
        quant_mode=args.quant_mode,
        module=model,
        input_args=(example,),
        device=device,
        target=TARGET,            # hardware-aware quantization for B3136
    )
    quant_model = quantizer.quant_model

    # ---- forward pass(es) over calibration data ----
    data = load_calib_batch(args.calib_dir, args.subset_len, args.imgsz)
    print(f"[ok] calibration tensor: {tuple(data.shape)}")
    with torch.no_grad():
        bs = args.batch_size
        for i in range(0, data.shape[0], bs):
            batch = data[i:i + bs].to(device)
            quant_model(batch)
            print(f"  forwarded {min(i + bs, data.shape[0])}/{data.shape[0]}")

    # ---- export results ----
    if args.quant_mode == "calib":
        quantizer.export_quant_config()
        print("[ok] wrote quantize_result/quant_info.json "
              "(now run --quant_mode test --deploy)")
    elif args.quant_mode == "test" and args.deploy:
        quantizer.export_xmodel(deploy_check=False)
        print("[ok] wrote quantize_result/*_int.xmodel")
        print("NEXT: edit compile_yolo26n.sh INT_XMODEL to point at that file, "
              "then bash compile_yolo26n.sh")


if __name__ == "__main__":
    main()
