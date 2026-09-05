#!/usr/bin/env python3
"""
export_yolo26n_for_dpu.py
-------------------------
Phase 0 (host-side) export of YOLO26n -> ONNX for the Vitis AI 3.0 quantize/compile
pipeline targeting the Kria KV260 (Zynq UltraScale+ MPSoC, DPUCZDX8G B3136).

VERIFIED against: ultralytics 8.4.71, torch onnx legacy exporter, opset 17.
This is NOT a guarantee of DPU support. The op counts below only show that the
one-to-many path removes the ops the DPUCZDX8G provably cannot map. The real
Phase 0 exit gate is the vai_c_xir compiler log (target: 1 DPU subgraph),
which must be run inside the Vitis AI 3.0 Docker image, not here.

What this script does, and WHY each step matters:

  1. end2end = False
     - In 8.4.71 the head attribute is `end2end` (NO underscore).
       Older notes referencing `det._end2end` are WRONG for this version and
       would silently no-op, leaving the NMS-free path active.
     - Setting end2end=False makes Detect.forward() skip the one2one branch and
       skip postprocess() entirely. postprocess() is where TopK (x2) and
       GatherElements (x2) are generated -- the ops the DPU cannot map.
     - You do NOT need to manually delete one2one_cv2/one2one_cv3; forward()
       already skips them when end2end is False.

  2. SiLU -> LeakyReLU
     - DPUCZDX8G does not support SiLU. Most SiLU lives *inside* ultralytics
       Conv blocks (not as standalone child modules), so a recursive replace is
       required. negative_slope 0.1015625 == 26/256, a DPU-friendly fixed-point value.
     - NOTE: this changes accuracy. After this swap you MUST fine-tune/retrain
       before trusting detection quality. For Phase 0 (compile feasibility) we
       only care about the op graph; accuracy work comes later.

  3. Export ONNX (opset 17, legacy exporter, static 1x3x640x640, constant-folded).

  4. Print a DPU-risky op histogram as a *smoke test*, not as evidence.
"""

import argparse, collections, warnings
import torch
import torch.nn as nn
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# Ops the DPUCZDX8G (Zynq US+) commonly cannot map -> would fall back to ARM/PS.
DPU_RISKY = ["TopK", "GatherElements", "GatherND", "ReduceMax", "Mod",
             "ScatterND", "NonMaxSuppression", "Range"]


def replace_silu_with_leakyrelu(module, slope=0.1015625):
    """Recursively swap every nn.SiLU (incl. those nested in Conv blocks)."""
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.LeakyReLU(slope, inplace=True))
        else:
            replace_silu_with_leakyrelu(child, slope)


def build_model(weights, imgsz):
    """Load weights if available; fall back to YAML architecture (no pretrained)."""
    try:
        m = YOLO(weights)            # e.g. 'yolo26n.pt' -> auto-downloads
        print(f"[ok] loaded weights: {weights}")
    except Exception as e:
        print(f"[warn] could not load '{weights}' ({e.__class__.__name__}); "
              f"building from yolo26n.yaml (NO pretrained weights -- graph only)")
        m = YOLO("yolo26n.yaml")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo26n.pt",
                    help="path to .pt (falls back to yolo26n.yaml if unreachable)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="yolo26n_o2m_leakyrelu.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--keep-silu", action="store_true",
                    help="skip SiLU->LeakyReLU (for A/B comparison only)")
    args = ap.parse_args()

    m = build_model(args.weights, args.imgsz)
    det = m.model.model[-1]

    # --- 1. force the safe one-to-many path ---
    assert hasattr(det, "end2end"), \
        "Detect head has no 'end2end' attr -- ultralytics API changed; re-inspect."
    det.end2end = False
    det.export = True
    det.format = "onnx"

    # --- 2. SiLU -> LeakyReLU (DPU requirement) ---
    if not args.keep_silu:
        before = sum(isinstance(mm, nn.SiLU) for _, mm in m.model.named_modules())
        replace_silu_with_leakyrelu(m.model)
        after_silu = sum(isinstance(mm, nn.SiLU) for _, mm in m.model.named_modules())
        after_lr = sum(isinstance(mm, nn.LeakyReLU) for _, mm in m.model.named_modules())
        print(f"[ok] SiLU replaced: {before} standalone modules detected; "
              f"now SiLU={after_silu}, LeakyReLU={after_lr}")

    m.model.eval()

    # --- 3. export ---
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    torch.onnx.export(
        m.model, dummy, args.out,
        opset_version=args.opset, dynamo=False,
        input_names=["images"], output_names=["output"],
        do_constant_folding=True,
    )
    print(f"[ok] wrote {args.out}")

    # --- 4. smoke-test op histogram (NOT proof of DPU support) ---
    import onnx
    g = onnx.load(args.out)
    ops = collections.Counter(n.op_type for n in g.graph.node)
    print(f"[info] total ONNX nodes: {sum(ops.values())}")
    hit = False
    for r in DPU_RISKY:
        if ops.get(r):
            print(f"  [RISKY] {r}: {ops[r]}  -- expect ARM/PS fallback")
            hit = True
    if not hit:
        print("  no watchlist ops present (TopK/GatherElements/ReduceMax/Mod/... all gone)")
    print("\nNEXT: run vai_q_onnx (quantize) then vai_c_xir (compile) inside the")
    print("Vitis AI 3.0 Docker image. The compiler log -- not this histogram --")
    print("is the Phase 0 exit gate (target: 1 DPU subgraph).")


if __name__ == "__main__":
    main()
