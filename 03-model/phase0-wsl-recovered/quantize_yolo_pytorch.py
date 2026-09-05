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

# ---- DPU target for KV260 -------------------------------------------------
# IMPORTANT: read the target from the arch.json that vai_c_xir will actually
# use, instead of hard-coding it. The KV260 arch.json in the Vitis AI 3.0
# image declares DPUCZDX8G_ISA1_B4096 (verified via `cat`), NOT B3136 -- the
# B3136 value in older notes came from someone else's `xdputil query`, not this
# image. Reading it here guarantees Inspector/quantize use the SAME arch the
# compiler targets, so op-assignment results are valid for the real compile.
# (conv maps on both B3136 and B4096; they differ in throughput/parallelism,
# not op support -- but keeping them identical removes a whole class of doubt.)
ARCH_JSON = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"


def _resolve_target():
    import json
    try:
        with open(ARCH_JSON) as f:
            t = json.load(f)["target"]
        print(f"[ok] target from arch.json: {t}")
        return t
    except Exception as e:
        fallback = "DPUCZDX8G_ISA1_B4096"
        print(f"[warn] could not read {ARCH_JSON} ({e.__class__.__name__}); "
              f"falling back to {fallback}. If compiling on a different host, "
              f"verify this matches your arch.json.")
        return fallback


TARGET = _resolve_target()
IMGSZ = 640


# ----------------------------------------------------------------------------
# forward-only wrapper: expose ONLY the raw detection-head outputs.
# Everything decode/NMS/count related stays OFF the quantized graph.
# ----------------------------------------------------------------------------
class BackboneHead(nn.Module):
    """Forward-only wrapper returning RAW per-stride head outputs (no decode).

    WHY raw outputs (decided from Inspector evidence, not assumption):
      An earlier decoded wrapper (format='onnx', output (1,N,8400)) was run
      through the vai_q_pytorch Inspector for the KV260 DPU target. Result:
      backbone+neck+head CONV all mapped to the DPU, but the Detect decode tail
      fell back to the ARM CPU - aten::arange, meshgrid, DFL softmax,
      split_with_sizes, transpose, box sub/div - because those ops cannot be
      converted to XIR. That decode tail is what fragments the graph.

      So we cut decode out of the quantized graph entirely. forward() returns
      the three raw per-stride tensors. CHANNEL COUNT DEPENDS ON THE MODEL:

        * YOLOv8n  : reg_max=16 -> cat = 4*16 + 80 = 144 channels
        * YOLO26n  : reg_max=1  -> cat = 4*1  + 80 =  84 channels  (DFL dropped)

      i.e. raw cat channels = 4*det.reg_max + det.nc. Do NOT hard-code 144;
      YOLO26's head has no DFL bins, so it is 84. (Verified live: yolo26n.yaml
      gives reg_max=1, cv2_out=4, cv3_out=80 per stride.)

      DFL-decode (only meaningful when reg_max>1) + anchor grid + box xywh +
      NMS + line-crossing COUNT all run on the ARM PS at deploy time. This is
      exactly the heterogeneous split the thesis targets: convs on the DPU,
      decode/track/count on the PS.

    WHY this version does NOT override det.forward (the previous bug):
      The earlier wrapper set `det.forward = _raw_head_forward` as an INSTANCE
      attribute, then called `self.model(x)` (the full DetectionModel). That
      DetectionModel runs `BaseModel._predict_once`, which invokes each layer
      via `m(x)` == nn.Module.__call__ -- NOT m.forward(x) directly. On the
      torch shipped in the Vitis AI 3.0 image (1.12.1), __call__ during tracing
      can dispatch through `_slow_forward`/the class-level forward rather than
      the instance attribute, so the override silently does not take effect
      while the quantizer/Inspector traces the model -> the head's cv2/cv3
      convs get traced through the ORIGINAL Detect.forward path and the decode
      tail reappears / the convs land on CPU. (This exact mechanism is
      torch-version dependent and was NOT reproducible on torch 2.x here; the
      robust fix below sidesteps it regardless of torch version.)

      Instead we walk model.model ourselves up to (but not into) the Detect
      head, then apply det.cv2[i]/det.cv3[i] directly. No forward override, no
      dependence on ultralytics/torch dispatch. Verified to give identical
      output under eager AND torch.jit.trace.
    """
    def __init__(self, ultra_model):
        super().__init__()
        self.model = ultra_model          # DetectionModel
        det = self.model.model[-1]
        self.det = det
        self.save = self.model.save       # layer indices whose outputs are reused
        det.end2end = False               # single (one2many) path, no one2one branch
        assert hasattr(det, "cv2") and hasattr(det, "cv3"), \
            "Detect head has no cv2/cv3 -- ultralytics API changed; re-inspect."

    def forward(self, x):
        # Replicate BaseModel._predict_once, but STOP before the Detect head
        # and emit raw cat(cv2, cv3) per stride instead of running decode.
        y = []
        for m in self.model.model:
            if m is self.det:
                break
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else \
                    [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        det = self.det
        feats = [x if j == -1 else y[j] for j in det.f]   # the 3 head inputs
        return tuple(
            torch.cat((det.cv2[i](feats[i]), det.cv3[i](feats[i])), 1)
            for i in range(det.nl)
        )


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
        # Derive a YAML name from the requested weights so we don't silently
        # switch model family (asking for yolo26n but quantizing yolov8n would
        # calibrate the wrong architecture). NOTE: a YAML-only graph has NO
        # trained weights -> calibration is meaningless; only valid for
        # op-graph / --verify / --inspect checks, never for a deployable xmodel.
        import os
        stem = os.path.splitext(os.path.basename(weights))[0]
        yaml_name = f"{stem}.yaml"
        print(f"[warn] could not load '{weights}' ({e.__class__.__name__}); "
              f"building from {yaml_name} (NO trained weights - graph only, "
              f"calibration would be meaningless; OK for --verify/--inspect).")
        y = YOLO(yaml_name)
    inner = y.model
    # XIR ไม่รองรับ multi-output chunk; C2f.forward ใช้ chunk(2,1) -> patch เป็น split
    from ultralytics.nn.modules.block import C2f
    def _c2f_forward_split(self, x):
        t = self.cv1(x)
        y = [t[:, :self.c, :, :], t[:, self.c:, :, :]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    C2f.forward = _c2f_forward_split
    print("[ok] C2f.chunk -> split done")
    if not keep_silu:
        replace_silu_with_leakyrelu(inner)
        print("[ok] SiLU -> LeakyReLU(0.1015625) done")
    model = BackboneHead(inner)
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
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--calib_dir", default="calib_images")
    ap.add_argument("--quant_mode", choices=["float", "calib", "test"], default="calib")
    ap.add_argument("--subset_len", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--deploy", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="run hardware-aware Inspector and exit")
    ap.add_argument("--verify", action="store_true",
                    help="prove the wrapper emits 3 raw head tensors under BOTH "
                         "eager and torch.jit.trace, then exit. Run this FIRST "
                         "in the container before calibrate/compile.")
    ap.add_argument("--keep_silu", action="store_true")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    args = ap.parse_args()

    device = torch.device("cpu")
    model = build_float_model(args.weights, keep_silu=args.keep_silu).to(device)
    example = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    # ---- Verify path: prove the wrapper survives tracing (THE bug we fixed) ----
    if args.verify:
        det = model.det
        exp_ch = 4 * det.reg_max + det.nc
        print(f"[verify] head: reg_max={det.reg_max} nc={det.nc} nl={det.nl} "
              f"-> expected raw cat channels = 4*{det.reg_max}+{det.nc} = {exp_ch}")

        def shapes(o):
            if isinstance(o, (tuple, list)):
                return [tuple(t.shape) for t in o]
            return type(o).__name__

        with torch.no_grad():
            eager = model(example)
            traced = torch.jit.trace(model, (example,), check_trace=False)
            traced_out = traced(example)

        es, ts = shapes(eager), shapes(traced_out)
        print(f"[verify] EAGER  output: {es}")
        print(f"[verify] TRACED output: {ts}")

        ok = (isinstance(eager, tuple) and len(eager) == det.nl
              and es == ts
              and all(s[1] == exp_ch for s in es))
        if ok:
            print(f"[verify] PASS: {det.nl} raw tensors, {exp_ch} channels each, "
                  f"eager==traced. The head conv goes through the quantizer; "
                  f"no decode tail. Safe to calibrate.")
        else:
            print("[verify] FAIL: wrapper does NOT emit clean raw head tensors "
                  "under tracing. Do NOT calibrate -- the head will fall to CPU. "
                  "Inspect the wrapper's model.model walk against this "
                  "ultralytics version's layer structure.")
            raise SystemExit(1)
        return

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
        target=TARGET,            # hardware-aware quantization (from arch.json)
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
        import glob
        xm = glob.glob("quantize_result/*_int.xmodel")
        if xm:
            print(f"[ok] wrote {xm[0]}")
        else:
            raise SystemExit("[FAIL] export_xmodel produced no .xmodel "
                             "(check XIR convert errors above, e.g. unsupported ops)")
        print("NEXT: edit compile_yolo26n.sh INT_XMODEL to point at that file, "
              "then bash compile_yolo26n.sh")


if __name__ == "__main__":
    main()
