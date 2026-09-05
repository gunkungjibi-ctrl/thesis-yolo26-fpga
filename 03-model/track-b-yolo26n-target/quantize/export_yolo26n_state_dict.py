#!/usr/bin/env python3
"""export_yolo26n_state_dict.py — run on a MODERN host, not in the container.

Requires: Python >= 3.8 with `ultralytics==8.4.71` (the version that knows
YOLO26). This is the only step that needs ultralytics; everything downstream
runs against `yolo26n_dpu.py`, which has no ultralytics dependency and imports
cleanly in the container's Python 3.7 `vitis-ai-pytorch` env.

What it does
------------
1. Loads a trained YOLO26n checkpoint (`yolo26n_leaky_pkg_ft.pt`).
2. Swaps every `nn.SiLU` for `nn.LeakyReLU(26/256)` — the DPU-representable
   slope. Activations carry no parameters, so this does not change the
   state_dict; it is done here only so the verification below compares the
   graph we will actually quantize.
3. Verifies, numerically, that `yolo26n_dpu.YOLO26nBackboneHead` loaded with
   this state_dict produces the same three raw head tensors as ultralytics.
   That check is the whole point of the file: it is what licenses using a
   hand-written graph as a stand-in for the real one.
4. Saves the plain state_dict (`torch.save` of a dict of tensors only — no
   pickled ultralytics classes, so torch 1.12 in the container can read it).

Usage
-----
    python export_yolo26n_state_dict.py \
        --weights yolo26n_leaky_pkg_ft.pt \
        --nc 1 \
        --out yolo26n_pkg_state_dict.pt
"""

import argparse
import sys

import torch
import torch.nn as nn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from yolo26n_dpu import DPU_LEAKY_SLOPE, YOLO26nBackboneHead  # noqa: E402


def replace_silu_with_leakyrelu(module, slope=DPU_LEAKY_SLOPE):
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.LeakyReLU(slope, inplace=True))
            n += 1
        else:
            n += replace_silu_with_leakyrelu(child, slope)
    return n


def ultralytics_raw_heads(inner, x, use_one2one=False):
    """Run ultralytics' own layers and return the raw per-stride head maps.

    Mirrors YOLO26nBackboneHead.forward so the two are compared on identical
    semantics rather than on ultralytics' decoded output.
    """
    det = inner.model[-1]
    y = []
    for m in inner.model[:-1]:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x)
    feats = [y[j] for j in det.f]
    cv2 = det.one2one_cv2 if use_one2one else det.cv2
    cv3 = det.one2one_cv3 if use_one2one else det.cv3
    return tuple(torch.cat((cv2[i](feats[i]), cv3[i](feats[i])), 1) for i in range(det.nl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--nc", type=int, default=1, help="classes in the checkpoint (package = 1)")
    ap.add_argument("--out", default="yolo26n_pkg_state_dict.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--use_one2one", action="store_true",
                    help="export the NMS-free one2one branch instead of the o2m branch")
    ap.add_argument("--tol", type=float, default=1e-4, help="max abs diff allowed in verification")
    args = ap.parse_args()

    from ultralytics import YOLO

    y = YOLO(args.weights)
    inner = y.model.float().eval()
    n_swapped = replace_silu_with_leakyrelu(inner)
    print("[ok] loaded %s | SiLU -> LeakyReLU(%s): %d modules" % (args.weights, DPU_LEAKY_SLOPE, n_swapped))

    det = inner.model[-1]
    print("[info] head: nc=%d reg_max=%d (reg_max==1 => no DFL, %d ch/stride)"
          % (det.nc, det.reg_max, 4 * det.reg_max + det.nc))
    if det.nc != args.nc:
        print("[warn] checkpoint nc=%d but --nc=%d; using the checkpoint's value" % (det.nc, args.nc))

    sd = inner.state_dict()

    # ---- rebuild the standalone graph and load the weights strictly ----
    model = YOLO26nBackboneHead(nc=det.nc, end2end=hasattr(det, "one2one_cv2"),
                                use_one2one=args.use_one2one).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("[FATAL] state_dict does not match the standalone graph.")
        for k in list(missing)[:20]:
            print("   missing:   ", k)
        for k in list(unexpected)[:20]:
            print("   unexpected:", k)
        raise SystemExit(1)
    print("[ok] state_dict loaded 1:1 into yolo26n_dpu.YOLO26nBackboneHead (%d tensors)" % len(sd))

    # ---- non-parameter settings, which a state_dict cannot carry ----
    # ultralytics' initialize_weights() overrides BatchNorm eps/momentum and
    # LeakyReLU slope. None of them are parameters, so a strict load succeeds
    # while the model computes different numbers. Check them explicitly: the
    # numerical test below would catch it too, but this names the cause.
    def bn_settings(mod):
        return set((round(m.eps, 12), round(m.momentum, 12))
                   for m in mod.modules() if isinstance(m, nn.BatchNorm2d))

    def act_slopes(mod):
        return set(round(m.negative_slope, 12)
                   for m in mod.modules() if isinstance(m, nn.LeakyReLU))

    ref_bn, got_bn = bn_settings(inner), bn_settings(model)
    if ref_bn != got_bn:
        raise SystemExit("[FATAL] BatchNorm (eps, momentum) mismatch: checkpoint %s vs standalone %s\n"
                         "        Fix BN_EPS / BN_MOMENTUM in yolo26n_dpu.py." % (sorted(ref_bn), sorted(got_bn)))
    print("[ok] BatchNorm (eps, momentum) matches: %s" % sorted(ref_bn))

    ref_act, got_act = act_slopes(inner), act_slopes(model)
    if ref_act and ref_act != got_act:
        raise SystemExit("[FATAL] LeakyReLU slope mismatch: checkpoint %s vs standalone %s\n"
                         "        Fix DPU_LEAKY_SLOPE in yolo26n_dpu.py." % (sorted(ref_act), sorted(got_act)))
    print("[ok] LeakyReLU slope matches: %s" % sorted(ref_act))

    # ---- numerical equivalence check ----
    torch.manual_seed(0)
    x = torch.randn(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        ref = ultralytics_raw_heads(inner, x, use_one2one=args.use_one2one)
        got = model(x)
    worst = 0.0
    for i, (a, b) in enumerate(zip(ref, got)):
        if a.shape != b.shape:
            raise SystemExit("[FATAL] head %d shape %s vs %s" % (i, tuple(a.shape), tuple(b.shape)))
        d = (a - b).abs().max().item()
        worst = max(worst, d)
        print("   head %d %-22s max|diff| = %.3e" % (i, str(tuple(a.shape)), d))
    if worst > args.tol:
        raise SystemExit("[FATAL] graphs disagree (max %.3e > tol %.3e)" % (worst, args.tol))
    print("[ok] standalone graph matches ultralytics, max|diff| = %.3e" % worst)

    torch.save(sd, args.out)
    print("[ok] wrote %s" % args.out)
    print("NEXT: copy %s and yolo26n_dpu.py into the Vitis AI container." % args.out)


if __name__ == "__main__":
    main()
