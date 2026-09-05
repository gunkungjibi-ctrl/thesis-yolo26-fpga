"""yolo26n_dpu.py — standalone YOLO26n graph for the Vitis AI 3.0 container.

WHY THIS FILE EXISTS
====================
`07-notes/M2B1_inspector_python_blocker.md` documents the blocker that stopped
M2-B3: `pytorch_nndct` (the quantizer that produces `*_int.xmodel`) only exists
in the container's `vitis-ai-pytorch` conda env, which is **Python 3.7.12 +
torch 1.12.1**, while `ultralytics==8.4.71` (the only release that knows about
YOLO26) requires **Python >= 3.8**. No env in the image has both.

This module removes ultralytics from the container entirely. It reimplements
the YOLO26n graph in plain PyTorch, using only ops available in torch 1.12 and
only Python 3.7 syntax, so it imports cleanly inside `vitis-ai-pytorch`.

Weights come in as a plain `state_dict` produced on a modern host by
`export_yolo26n_state_dict.py`. Module attribute names here mirror ultralytics
exactly (`model.<i>.cv1.conv.weight`, ...), so the state_dict loads 1:1 with
`strict=True` — that strictness is the guarantee that nothing silently drifted.

TWO DELIBERATE DEVIATIONS FROM ULTRALYTICS, both XIR-driven
===========================================================
1. **No `chunk` / `split`.** `C2f.forward` (which `C3k2` inherits),
   `C2PSA.forward` and `Attention.forward` all slice tensors with
   `.chunk()` / `.split()`. Those lower to multi-output ops and are exactly
   what made Track A fail with `XIR don't support multi-outputs op` (see
   worklog 2026-08-16). Every one of them is written here as plain slicing,
   which is the same fix the proven Track A `quantize_yolo_pytorch.py` applies
   by monkeypatch — built in from the start instead.

2. **Raw head outputs only.** `forward()` returns the three per-stride feature
   maps `cat(cv2[i](x), cv3[i](x))` and nothing else. Anchor generation,
   box decode, sigmoid and NMS stay on the ARM PS at deploy time. This mirrors
   Track A's `BackboneHead` wrapper and keeps decode ops from splitting the
   subgraph and masking the real question (the attention blocks).

WHAT THIS MODEL IS FOR
======================
Answering the Track B gate: does YOLO26n compile to a single DPU subgraph on
DPUCZDX8G B4096? The two attention sites found in M2-B1 (`model.10` C2PSA and
`model.22` C3k2 with attn=True) are kept verbatim — they are the thing under
test, so nothing here tries to work around them.

NOTE ON THE HEAD: YOLO26 sets `reg_max=1`, i.e. it has **no DFL**. Output is
`4 * 1 + nc` channels per stride (5 for single-class `package`), not the 65
channels Track A's YOLOv8n produces. The deploy-side decoder must differ
accordingly.
"""

import math

import torch
import torch.nn as nn

# LeakyReLU slope 26/256 — representable exactly in the DPU's fixed-point
# format, which plain 0.1 is not. Same value Track A fine-tuned against.
DPU_LEAKY_SLOPE = 0.1015625

# ultralytics' initialize_weights() overrides the torch defaults on every
# BatchNorm2d, and these are NOT parameters, so they do not travel in a
# state_dict: a strict load succeeds and the model still computes different
# numbers. Getting this wrong shifted layer 0's output by 5.7 during bring-up
# of this file. Keep them in sync with ultralytics.utils.torch_utils.
BN_EPS = 1e-3
BN_MOMENTUM = 0.03


def autopad(k, p=None, d=1):
    """'same' padding, matching ultralytics.nn.modules.conv.autopad."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Conv2d + BatchNorm2d + activation, with ultralytics' attribute names."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=BN_EPS, momentum=BN_MOMENTUM)
        if act is True:
            self.act = nn.LeakyReLU(DPU_LEAKY_SLOPE, inplace=True)
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Depth-wise Conv, as ultralytics defines it (groups = gcd(c1, c2))."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super(DWConv, self).__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        if self.add:
            return x + self.cv2(self.cv1(x))
        return self.cv2(self.cv1(x))


class C3k(nn.Module):
    """C3 with configurable bottleneck kernel — ultralytics C3k(n=2, k=3)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super(C3k, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class Attention(nn.Module):
    """Position-sensitive self-attention — the op under test for Track B.

    Kept faithful to ultralytics apart from `split` -> slicing. The MatMul +
    Softmax pair here is what M2-B1 identified as unmappable on DPUCZDX8G.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x).view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N)
        kd = self.key_dim
        # slicing, not .split(...) — split is a multi-output op that XIR rejects
        q = qkv[:, :, 0:kd, :]
        k = qkv[:, :, kd:2 * kd, :]
        v = qkv[:, :, 2 * kd:2 * kd + self.head_dim, :]

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super(PSABlock, self).__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    """model.10 — the first attention site found in M2-B1."""

    def __init__(self, c1, c2, n=1, e=0.5):
        super(C2PSA, self).__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*[PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)])

    def forward(self, x):
        y = self.cv1(x)
        a = y[:, 0:self.c]            # was .split((c, c), dim=1)
        b = y[:, self.c:2 * self.c]
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C3k2(nn.Module):
    """C2f-style CSP block. `attn=True` at model.22 is the second attention site."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super(C3k2, self).__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        blocks = []
        for _ in range(n):
            if attn:
                blocks.append(
                    nn.Sequential(
                        Bottleneck(self.c, self.c, shortcut, g),
                        PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
                    )
                )
            elif c3k:
                blocks.append(C3k(self.c, self.c, 2, shortcut, g))
            else:
                blocks.append(Bottleneck(self.c, self.c, shortcut, g))
        self.m = nn.ModuleList(blocks)

    def forward(self, x):
        t = self.cv1(x)
        # was list(t.chunk(2, 1)) — chunk is the multi-output op that broke Track A
        y = [t[:, 0:self.c], t[:, self.c:2 * self.c]]
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """YOLO26's SPPF takes n pooling iterations and an optional residual."""

    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        super(SPPF, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        for _ in range(self.n):
            y.append(self.m(y[-1]))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Upsample(nn.Upsample):
    """nn.Upsample under a stable name so the layer table reads clearly."""

    def __init__(self):
        super(Upsample, self).__init__(scale_factor=2.0, mode="nearest")


class DetectHeads(nn.Module):
    """Detect, stripped to the raw per-stride outputs.

    Holds both the one-to-many (`cv2`/`cv3`) and one-to-one
    (`one2one_cv2`/`one2one_cv3`) branches so a full YOLO26 state_dict loads
    strictly, but forward() emits only the branch selected by `use_one2one`.

    Default is the o2m branch: it is what NMS is designed for, and NMS runs on
    the PS in this deployment exactly as it does for Track A. Set
    use_one2one=True to export the NMS-free branch instead.
    """

    def __init__(self, nc=80, reg_max=1, ch=(64, 128, 256), end2end=True, use_one2one=False):
        super(DetectHeads, self).__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.use_one2one = use_one2one
        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            [nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for x in ch]
        )
        self.cv3 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, nc, 1),
                )
                for x in ch
            ]
        )
        self.dfl = nn.Identity()  # reg_max == 1 -> YOLO26 has no DFL
        if end2end:
            import copy

            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, feats):
        cv2 = self.one2one_cv2 if self.use_one2one else self.cv2
        cv3 = self.one2one_cv3 if self.use_one2one else self.cv3
        out = []
        for i in range(self.nl):
            out.append(torch.cat((cv2[i](feats[i]), cv3[i](feats[i])), 1))
        return tuple(out)


# ---------------------------------------------------------------------------
# Resolved YOLO26n (scale 'n': depth 0.50, width 0.25, max_channels 1024)
# layer spec. Channel counts were read back off a real ultralytics-built
# yolo26n model rather than recomputed by hand, so they cannot drift from the
# scaling rules. `f` is the ultralytics "from" field.
# ---------------------------------------------------------------------------
def _build_layers(nc, end2end, use_one2one):
    return [
        # (from, module)
        (-1, Conv(3, 16, 3, 2)),                                    # 0  P1/2
        (-1, Conv(16, 32, 3, 2)),                                   # 1  P2/4
        (-1, C3k2(32, 64, n=1, c3k=False, e=0.25)),                 # 2
        (-1, Conv(64, 64, 3, 2)),                                   # 3  P3/8
        (-1, C3k2(64, 128, n=1, c3k=False, e=0.25)),                # 4
        (-1, Conv(128, 128, 3, 2)),                                 # 5  P4/16
        (-1, C3k2(128, 128, n=1, c3k=True)),                        # 6
        (-1, Conv(128, 256, 3, 2)),                                 # 7  P5/32
        (-1, C3k2(256, 256, n=1, c3k=True)),                        # 8
        (-1, SPPF(256, 256, k=5, n=3, shortcut=True)),              # 9
        (-1, C2PSA(256, 256, n=1)),                                 # 10 <- ATTENTION #1
        (-1, Upsample()),                                           # 11
        ([-1, 6], Concat(1)),                                       # 12
        (-1, C3k2(384, 128, n=1, c3k=True)),                        # 13
        (-1, Upsample()),                                           # 14
        ([-1, 4], Concat(1)),                                       # 15
        (-1, C3k2(256, 64, n=1, c3k=True)),                         # 16 P3/8  -> head
        (-1, Conv(64, 64, 3, 2)),                                   # 17
        ([-1, 13], Concat(1)),                                      # 18
        (-1, C3k2(192, 128, n=1, c3k=True)),                        # 19 P4/16 -> head
        (-1, Conv(128, 128, 3, 2)),                                 # 20
        ([-1, 10], Concat(1)),                                      # 21
        (-1, C3k2(384, 256, n=1, c3k=True, e=0.5, attn=True)),      # 22 <- ATTENTION #2
        ([16, 19, 22], DetectHeads(nc, 1, (64, 128, 256), end2end, use_one2one)),  # 23
    ]


class YOLO26nBackboneHead(nn.Module):
    """Forward-only YOLO26n for vai_q_pytorch.

    The class name matters: `quantizer.export_xmodel()` names the exported file
    after it, so this produces `quantize_result/YOLO26nBackboneHead_int.xmodel`,
    which is exactly what `compile/compile_yolo26n.sh` expects as INT_XMODEL.

    forward(x) -> tuple of 3 raw head tensors, each (1, 4 + nc, H, W).
    """

    def __init__(self, nc=1, end2end=True, use_one2one=False):
        super(YOLO26nBackboneHead, self).__init__()
        layers = _build_layers(nc, end2end, use_one2one)
        self.froms = [f for f, _ in layers]
        self.model = nn.ModuleList([m for _, m in layers])
        self.nc = nc

    def forward(self, x):
        y = []
        for i, m in enumerate(self.model):
            f = self.froms[i]
            if f != -1:
                if isinstance(f, int):
                    x = y[f]
                else:
                    x = [x if j == -1 else y[j] for j in f]
            x = m(x)
            y.append(x)
        return x


def load_state_dict_file(model, path):
    """Load a plain state_dict saved by export_yolo26n_state_dict.py.

    Uses strict=True on purpose: a silent key mismatch here would calibrate a
    subtly wrong graph and the compile result would mean nothing.
    """
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    return model
