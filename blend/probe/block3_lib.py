"""Shared code for probe 3 (task 147): block table access, scoring from Gram data, the block nets."""
import json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BLEND = os.environ.get("BLEND", "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend")
ROOT = f"{BLEND}/probe"
T3 = f"{ROOT}/block3"
GROUPS = ["forza", "cs2", "minecraft120", "battlefield"]
S2 = 255.0 ** 2
PAD = 22
PAIRS = json.load(open(f"{ROOT}/pairs.json"))
TRAIN = [k for k, p in enumerate(PAIRS) if p["split"] == "train"]
VAL = [k for k, p in enumerate(PAIRS) if p["split"] == "val"]


def load_pair(k, bs=1):
    """dict of float32 torch tensors; bs = 1 (B=8 table) or 2 (B=16: sums of 2x2 blocks; stats re-derived from the 8-blocks)."""
    z = np.load(f"{T3}/{k:02d}.npz", allow_pickle=False)
    d = {n: torch.from_numpy(z[n]) for n in z.files}
    if bs == 2:
        h, w = d["yy"].shape
        h2 = h // 2   # 135 is odd: drop the last block row
        for n in d:
            d[n] = d[n][:h2 * 2]
        sm = lambda a: a.reshape(h2, 2, w // 2, 2, *a.shape[2:]).sum((1, 3))
        for n in ("G", "b", "yy", "Ge", "be", "yye", "ne"):
            d[n] = sm(d[n])
        f = d["feat"].reshape(h2, 2, w // 2, 2, 136)
        mu, sd, mn, mx = f[..., :34], f[..., 34:68], f[..., 68:102], f[..., 102:]
        m = mu.mean((1, 3))
        var = (sd ** 2 + mu ** 2).mean((1, 3)) - m ** 2
        d["feat"] = torch.cat([m, var.clamp_min(0).sqrt(), mn.amin((1, 3)), mx.amax((1, 3))], -1)
        d["rule"] = d["rule"].reshape(h2, 2, w // 2, 2, 4).mean((1, 3))
    return d


def errs(w, d, dbl=False):
    """w [h,w,6] -> (err all, err edge) per block, in levels^2 summed over the block's pixels and RGB."""
    out = []
    for G, b, yy in (("G", "b", "yy"), ("Ge", "be", "yye")):
        g, bb, y = d[G], d[b], d[yy]
        ww = w
        if dbl:
            g, bb, y, ww = g.double(), bb.double(), y.double(), w.double()
        e = (ww * torch.einsum("...ij,...j->...i", g, ww)).sum(-1) - 2 * (bb * ww).sum(-1) + y
        out.append(e * S2)
    return out


def psnr(se, n):
    return 10 * math.log10(S2 / (se / (3 * n)))


def pix(d, bs=1):
    return 64 * bs * bs * d["yy"].numel(), d["ne"].sum().item()


class Acc:
    """pooled se / n per group over pairs"""
    def __init__(self):
        self.a = {}

    def add(self, k, se, se_e, n, n_e):
        g = PAIRS[k]["group"]
        for key in (g, "all"):
            t = self.a.setdefault(key, [0.0, 0, 0.0, 0])
            t[0] += se; t[1] += n; t[2] += se_e; t[3] += n_e

    def res(self):
        return {g: (psnr(t[0], t[1]), psnr(t[2], t[3])) for g, t in self.a.items()}


def score(wfun, pairs, data, bs=1):
    """wfun(k, d) -> w [h,w,6]; returns {group: (whole, edge)} PSNR"""
    A = Acc()
    for k in pairs:
        d = data[k]
        with torch.no_grad():
            w = wfun(k, d)
        e, ee = errs(w, d, True)
        n, ne = pix(d, bs)
        A.add(k, e.sum().item(), ee.sum().item(), n, ne)
    return A.res()


def rule6(d):
    h, w = d["yy"].shape
    return torch.cat([d["rule"], torch.zeros(h, w, 2)], -1)


# ---------- nets ----------
def init_net(m):
    for l in m.modules():
        if isinstance(l, nn.Conv2d):
            nn.init.normal_(l.weight, 0, 0.3 / math.sqrt(l.weight[0].numel()))
            nn.init.zeros_(l.bias)


class BNet(nn.Module):
    """layers = list of (kernel, dilation) for the hidden stack (valid convs); MLP = [(1,1),(1,1)] (1x1 convs)."""
    def __init__(self, cin, layers, ch=64, drop=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        c = cin
        for k, dl in layers:
            self.convs.append(nn.Conv2d(c, ch, k, dilation=dl))
            c = ch
        self.out = nn.Conv2d(c, 6, 1)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        init_net(self)
        nn.init.zeros_(self.out.weight)

    def forward(self, x, rule):
        for cv in self.convs:
            x = self.drop(F.leaky_relu(cv(x), 0.1))
        z = self.out(x)
        base = torch.cat([torch.log(rule.clamp_min(1e-3)), torch.full_like(rule[:, :2], -5.0)], 1)
        return torch.softmax(base + z, 1)


VARIANTS = {   # name: (stats, ctx, frame, layers)
    "b-mean": ("mean", False, False, [(1, 1), (1, 1)]),
    "b-stats": ("all", False, False, [(1, 1), (1, 1)]),
    "b-ctx": ("all", True, False, [(1, 1), (1, 1)]),
    "b-ctx-frame": ("all", True, True, [(1, 1), (1, 1)]),
    "b-conv": ("all", False, False, [(3, 1), (3, 1), (3, 1)]),
    "b-conv-big": ("all", False, True, [(3, 1), (3, 2), (3, 4), (3, 8)]),
}


def frame_stats(d):
    f = d["feat"]
    m = f[..., :34].mean((0, 1))
    v = (f[..., 34:68] ** 2 + f[..., :34] ** 2).mean((0, 1)) - m ** 2
    return torch.cat([m, v.clamp_min(0).sqrt()])


class Prep:
    """builds net inputs from a pair dict; normalisation from the training pairs"""
    def __init__(self, name, train_data):
        self.name = name
        self.stats, self.ctx, self.frame, self.layers = VARIANTS[name]
        fs = torch.cat([train_data[k]["feat"].reshape(-1, 136)[::7] for k in train_data])
        self.mu, self.sd = fs.mean(0), fs.std(0) + 1e-4
        if self.frame:
            F_ = torch.stack([frame_stats(train_data[k]) for k in train_data])
            self.fmu, self.fsd = F_.mean(0), F_.std(0) + 1e-4
        self.nin = (34 if self.stats == "mean" else 136) + (102 if self.ctx else 0) + (68 if self.frame else 0)
        self.rad = sum((k // 2) * dl for k, dl in self.layers)

    def pad(self, d):
        if "_pf" not in d:
            f = ((d["feat"] - self.mu) / self.sd).clamp(-10, 10).permute(2, 0, 1)[None]
            d["_pf"] = F.pad(f, (PAD,) * 4, mode="replicate")[0].half()
            d["_pr"] = F.pad(d["rule"].permute(2, 0, 1)[None], (PAD,) * 4, mode="replicate")[0]
            if self.frame:
                d["_fr"] = ((frame_stats(d) - self.fmu) / self.fsd).clamp(-10, 10)
            del d["feat"]   # memory: only the padded half-precision copy is kept
        return d["_pf"], d["_pr"], d.get("_fr")

    def make(self, d, y0, x0, h, w):
        """block rows y0..y0+h, cols x0..x0+w -> (x [1,C,h+2r,w+2r], rule [1,4,h,w]); r = conv radius; exact context
        (replicate-padded block grid)."""
        pf, pr, fr = self.pad(d)
        r = self.rad
        c = 7 if self.ctx else 0
        m = r + c
        reg = pf[:, PAD + y0 - m:PAD + y0 + h + m, PAD + x0 - m:PAD + x0 + w + m][None].float()
        sl = lambda t, cut: t[..., cut:t.shape[-2] - cut, cut:t.shape[-1] - cut] if cut else t
        base = reg if self.stats == "all" else reg[:, :34]
        parts = [sl(base, c)]
        if self.ctx:
            for k in (3, 7, 15):
                parts.append(sl(F.avg_pool2d(reg[:, :34], k, 1), c - k // 2))
        x = torch.cat(parts, 1)
        if self.frame:
            x = torch.cat([x, fr[None, :, None, None].expand(1, -1, x.shape[2], x.shape[3])], 1)
        rl = pr[None, :, PAD + y0:PAD + y0 + h, PAD + x0:PAD + x0 + w]
        return x, rl
