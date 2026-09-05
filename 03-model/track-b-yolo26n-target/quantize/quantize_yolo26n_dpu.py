#!/usr/bin/env python3
"""quantize_yolo26n_dpu.py — run INSIDE the Vitis AI 3.0 container. M2-B3.

    docker image: xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106
    conda env:    vitis-ai-pytorch   (Python 3.7.12, torch 1.12.1, pytorch_nndct)

This replaces `quantize_yolo26n_pytorch.py` for the compile gate. The old
script calls `from ultralytics import YOLO`, which cannot import in the only
env that has `pytorch_nndct` — that is the blocker recorded in
`07-notes/M2B1_inspector_python_blocker.md`. Here the graph comes from
`yolo26n_dpu.py` (plain PyTorch, no ultralytics) and the weights from a plain
state_dict exported on the host by `export_yolo26n_state_dict.py`, which also
proves the two graphs agree numerically before you get this far.

Two passes, as the vai_q_pytorch API requires:

    # pass 1 — calibrate on real conveyor frames
    python quantize_yolo26n_dpu.py --state_dict yolo26n_pkg_state_dict.pt \
        --calib_dir /workspace/calib/images --quant_mode calib --subset_len 32

    # pass 2 — export the deployable XIR xmodel (batch=1, one iteration)
    python quantize_yolo26n_dpu.py --state_dict yolo26n_pkg_state_dict.pt \
        --calib_dir /workspace/calib/images --quant_mode test \
        --subset_len 1 --batch_size 1 --deploy

Output: quantize_result/YOLO26nBackboneHead_int.xmodel
        -> feed to compile/compile_yolo26n.sh, which is the gate.

MEMORY: Track A died here with `unexpected EOF` / exit 125, which was the host
running out of RAM rather than a script bug (worklog 2026-08-16). Keep
--subset_len modest (32 worked) and make sure the pagefile can grow.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo26n_dpu import YOLO26nBackboneHead  # noqa: E402

# Confirmed B4096 in P1 (07-notes/P1_arch_and_identity_resolution.md); the
# board's own `xdputil query` reports fingerprint 0x101000056010407.
TARGET = "DPUCZDX8G_ISA1_B4096"
IMGSZ = 640


def load_calib_batch(calib_dir, subset_len, imgsz):
    """Real conveyor frames, preprocessed exactly as the board app does.

    Plain resize (not letterbox) + BGR->RGB + /255, matching
    04-deploy/board/yolo_dpu_detect.py. Calibrating on anything else — random
    data, or COCO images — makes the resulting INT8 model meaningless.
    """
    files = sorted(glob.glob(os.path.join(calib_dir, "*")))[:subset_len]
    if not files:
        raise SystemExit("No images in '%s'. Calibrating on random data is invalid." % calib_dir)
    try:
        import cv2
    except ImportError:
        raise SystemExit("cv2 not available in this env; needed to read calibration frames.")
    imgs = []
    for p in files:
        im = cv2.imread(p)
        if im is None:
            continue
        im = cv2.resize(im, (imgsz, imgsz))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        imgs.append(np.transpose(im, (2, 0, 1)))
    if not imgs:
        raise SystemExit("No readable images in '%s'." % calib_dir)
    return torch.from_numpy(np.ascontiguousarray(np.stack(imgs)))


def jit_trace_preflight(model, example):
    """vai_q_pytorch requires the float model to pass jit.trace first."""
    try:
        with torch.no_grad():
            torch.jit.trace(model, example, check_trace=False)
        print("[ok] jit.trace pre-flight passed")
    except Exception as e:
        raise SystemExit("[FATAL] jit.trace failed: %s" % e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_dict", required=True,
                    help="plain state_dict from export_yolo26n_state_dict.py")
    ap.add_argument("--nc", type=int, default=1)
    ap.add_argument("--calib_dir", default="calib_images")
    ap.add_argument("--quant_mode", choices=["float", "calib", "test"], default="calib")
    ap.add_argument("--subset_len", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--deploy", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="run the hardware-aware Inspector and exit (advisory only)")
    ap.add_argument("--use_one2one", action="store_true")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    args = ap.parse_args()

    device = torch.device("cpu")
    model = YOLO26nBackboneHead(nc=args.nc, end2end=True, use_one2one=args.use_one2one)
    sd = torch.load(args.state_dict, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    print("[ok] loaded %s into YOLO26nBackboneHead(nc=%d)" % (args.state_dict, args.nc))
    print("[info] head is DFL-free (reg_max=1) -> %d channels per stride" % (4 + args.nc))

    example = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    if args.inspect:
        # Advisory: tells you which ops will land on the ARM PS. The gate is
        # still the vai_c_xir log, not this.
        from pytorch_nndct.apis import Inspector
        Inspector(TARGET).inspect(model, (example,), device=device)
        print("\n[info] Inspector done. Ops marked CPU/USER fall back to the PS.")
        print("[info] Expect the attention sites (model.10 C2PSA, model.22 C3k2 attn) here.")
        return

    from pytorch_nndct.apis import torch_quantizer

    if args.deploy and (args.batch_size != 1 or args.subset_len != 1):
        print("[warn] deploy requires batch_size=1 and subset_len=1; forcing.")
        args.batch_size, args.subset_len = 1, 1

    jit_trace_preflight(model, (example,))

    quantizer = torch_quantizer(
        quant_mode=args.quant_mode,
        module=model,
        input_args=(example,),
        device=device,
        target=TARGET,
    )
    quant_model = quantizer.quant_model

    data = load_calib_batch(args.calib_dir, args.subset_len, args.imgsz)
    print("[ok] calibration tensor: %s" % (tuple(data.shape),))
    with torch.no_grad():
        bs = args.batch_size
        for i in range(0, data.shape[0], bs):
            quant_model(data[i:i + bs].to(device))
            print("  forwarded %d/%d" % (min(i + bs, data.shape[0]), data.shape[0]))

    if args.quant_mode == "calib":
        quantizer.export_quant_config()
        print("[ok] wrote quantize_result/quant_info.json")
        print("NEXT: rerun with --quant_mode test --subset_len 1 --batch_size 1 --deploy")
    elif args.quant_mode == "test" and args.deploy:
        quantizer.export_xmodel(deploy_check=False)
        out = "quantize_result/YOLO26nBackboneHead_int.xmodel"
        print("[ok] exported %s" % out)
        if not os.path.exists(out):
            raise SystemExit("[FATAL] expected %s but it is missing — check quantize_result/" % out)
        print("NEXT: bash ../compile/compile_yolo26n.sh   <-- this is the Track B gate")


if __name__ == "__main__":
    main()
