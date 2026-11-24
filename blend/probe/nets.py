"""The six networks of the probe and the shared data access.  Input X [B,34,h,w] (feat.features, f16 stored), the rule's weights R [B,4,h,w].
Output: six softmax weights (logits = log(max(rule,1e-3)) with -5, -5 for prev(x) / curr(x), plus the net's z), like blend/train/train.py (last layer zero at the start)."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H, W = 1080, 1920
NAMES = ["px8", "px32", "n3-all", "n5-all", "n3-frames", "n3-taps"]
MARGIN = 2   # a net sees at most 2 pixels away (n5-all)


class Net(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        a = lambda ci, co, k: nn.Conv2d(ci, co, k, padding=k // 2)
        if name in ("px8", "px32"):
            hd = 8 if name == "px8" else 32
            self.c1, self.c2 = a(34, hd, 1), a(hd, 6, 1)
        elif name == "n3-all":
            self.c1, self.c2 = a(34, 16, 3), a(16, 6, 1)
        elif name == "n5-all":
            self.c1, self.c1b, self.c2 = a(34, 16, 3), a(16, 16, 3), a(16, 6, 1)
        elif name == "n3-frames":   # the centre pixel's 34 inputs (1x1) + a 3x3 window of prev(x), curr(x) (6 channels)
            self.c1, self.cn, self.c2 = a(34, 16, 1), nn.Conv2d(6, 16, 3, padding=1, bias=False), a(16, 6, 1)
        elif name == "n3-taps":     # ... + a 3x3 window of the four taps' colours (12 channels)
            self.c1, self.cn, self.c2 = a(34, 16, 1), nn.Conv2d(12, 16, 3, padding=1, bias=False), a(16, 6, 1)
        else:
            raise ValueError(name)
        for m in self.modules():   # train.py: normal(0, 0.3/sqrt(fan_in)), biases 0, last layer 0
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.3 / math.sqrt(m.weight[0].numel()))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.c2.weight)

    def logits(self, X):
        lk = lambda z: torch.where(z > 0, z, 0.1 * z)
        n = self.name
        if n in ("px8", "px32", "n3-all"):
            h = lk(self.c1(X))
        elif n == "n5-all":
            h = lk(self.c1b(lk(self.c1(X))))
        elif n == "n3-frames":
            h = lk(self.c1(X) + self.cn(X[:, 25:31]))
        else:
            h = lk(self.c1(X) + self.cn(X[:, 0:12]))
        return self.c2(h)

    def forward(self, X, R):
        lg = torch.log(torch.clamp(R, min=1e-3))
        lg = torch.cat([lg, torch.full_like(lg[:, :2], -5.0)], 1)
        return torch.softmax(lg + self.logits(X), 1)


def mix(w, X):
    """weights [B,6,h,w], X [B,34,h,w] (/255 scale) -> colour [B,3,h,w] (/255 scale)."""
    pts = torch.cat([X[:, 0:12], X[:, 25:31]], 1).view(X.shape[0], 6, 3, *X.shape[2:])
    return (w[:, :, None] * pts).sum(1)


class Pair:
    """Memory-mapped arrays of one pair."""
    def __init__(self, root, k):
        d = f"{root}/pairs/{k:02d}"
        self.X = np.load(f"{d}/X.npy", mmap_mode="r", allow_pickle=False)
        self.R = np.load(f"{d}/rule.npy", mmap_mode="r", allow_pickle=False)
        self.Y = np.load(f"{d}/y.npy", mmap_mode="r", allow_pickle=False)

    def window(self, y0, y1, x0, x1):
        """rows y0..y1, cols x0..x1 (may stick out of the frame: zero padded) -> X [34,h,w] f32 (/1), R [4,h,w], Y [3,h,w] (/255)."""
        cy0, cy1, cx0, cx1 = max(y0, 0), min(y1, H), max(x0, 0), min(x1, W)
        out = []
        for a, c in ((self.X, 34), (self.R, 4), (self.Y, 3)):
            t = np.zeros((y1 - y0, x1 - x0, c), np.float32)
            t[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = a[cy0:cy1, cx0:cx1]
            out.append(t)
        out[2] /= 255.0
        return [np.ascontiguousarray(t.transpose(2, 0, 1)) for t in out]


@torch.no_grad()
def predict(net, pair, band=216):
    """The whole frame in row bands (with MARGIN rows / cols of context, zero padding at the frame border) -> colour [H,W,3] float32 levels."""
    out = np.empty((H, W, 3), np.float32)
    for y in range(0, H, band):
        X, R, _ = pair.window(y - MARGIN, min(y + band, H) + MARGIN, -MARGIN, W + MARGIN)
        w = net(torch.from_numpy(X)[None], torch.from_numpy(R)[None])
        c = mix(w, torch.from_numpy(X)[None])[0, :, MARGIN:-MARGIN, MARGIN:-MARGIN] * 255.0
        out[y:y + c.shape[1]] = c.permute(1, 2, 0).numpy()
    return out
