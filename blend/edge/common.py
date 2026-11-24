"""Shared pieces of blend/edge: the library nets' forward pass in torch (ref.py / train.py maths), crop loading, Sobel, whole-frame scoring."""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "probe"))
BLEND = os.environ.get("BLEND", "/home/mo/DieGanzeSau/AI_Video_Training_Data/blend")
PROBE = f"{BLEND}/probe"
WORK = f"{BLEND}/edge"
WDIR = os.path.join(HERE, "..", "weights")
H, W = 1080, 1920
KX = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None]


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    return [torch.from_numpy(np.asarray(z[k], np.float32).copy()) for k in ("W1", "b1", "W2", "b2")]


def weights(p, X, R):
    """p = [W1 (8,NI), b1, W2 (K,8), b2]; X [B,34,h,w] (the first NI channels are used), R [B,4,h,w] the rule's weights -> softmax weights [B,K,h,w]."""
    W1, b1, W2, b2 = p
    K, NI = W2.shape[0], W1.shape[1]
    z1 = torch.einsum("oc,bchw->bohw", W1, X[:, :NI]) + b1[None, :, None, None]
    h = torch.where(z1 > 0, z1, 0.1 * z1)
    z2 = torch.einsum("ko,bohw->bkhw", W2, h) + b2[None, :, None, None]
    lg = torch.log(torch.clamp(R, min=1e-3))
    if K == 6:
        lg = torch.cat([lg, torch.full_like(lg[:, :2], -5.0)], 1)
    return torch.softmax(lg + z2, 1)


def colour(w, X):
    """weights [B,K,h,w] -> colour [B,3,h,w] (scale of X: levels/255 for X's colour channels)."""
    K = w.shape[1]
    pts = torch.cat([X[:, 0:12], X[:, 25:31]], 1)[:, :3 * K].reshape(X.shape[0], K, 3, *X.shape[2:])
    return (w[:, :, None] * pts).sum(1)


def luma(c):
    return 0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2]


def sobel(L, pad=False, eps=0.0):
    """L [B,h,w] -> Sobel magnitude [B,h-2,w-2] (interior), or edge-replicated [B,h,w] with pad=True.  eps > 0 (training): sqrt(gx^2 + gy^2 + eps), finite gradient on flat areas."""
    L = L[:, None]
    if pad:
        L = F.pad(L, (1, 1, 1, 1), mode="replicate")
    return torch.sqrt(F.conv2d(L, KX) ** 2 + F.conv2d(L, KX.transpose(2, 3)) ** 2 + eps)[:, 0]


class Frame:
    """Whole pair (memory-mapped) for scoring."""
    def __init__(self, k):
        d = f"{PROBE}/pairs/{k:02d}"
        self.ld = lambda f: np.load(f"{d}/{f}.npy", mmap_mode="r", allow_pickle=False)


@torch.no_grad()
def predict_w(p, fr, band=216):
    """whole frame -> (weights [H,W,K] f32 np, colour [H,W,3] f32 levels)."""
    X, R = fr.ld("X"), fr.ld("rule")
    ws, cs = [], []
    for y in range(0, H, band):
        x = torch.from_numpy(np.asarray(X[y:y + band]).astype(np.float32)).permute(2, 0, 1)[None]
        r = torch.from_numpy(np.asarray(R[y:y + band]).astype(np.float32)).permute(2, 0, 1)[None]
        w = weights(p, x, r)
        ws.append(w[0].permute(1, 2, 0).numpy())
        cs.append((colour(w, x)[0] * 255.0).permute(1, 2, 0).numpy())
    return np.concatenate(ws), np.concatenate(cs)
