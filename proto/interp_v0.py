"""Image-only frame interpolation: CPU prototype of the planned GPU version.

Each step maps onto one compute pass:
  luma -> pyramid -> per level: bilateral SAD block search, 3x3 median -> warp both frames to t=0.5, blend.

Bilateral: for a block at position q of the *in-between* frame we search a half-vector h
minimising SAD(a[q - h], c[q + h]). Vectors land directly on the output grid, so no
splatting, no holes, no atomics on the GPU.
"""
import cv2
import numpy as np

BLOCK = 8          # block size at every pyramid level
RADIUS = 2         # search radius per level, in half-vector pixels of that level
FLOW_SCALE = 2     # flow runs at 1/FLOW_SCALE of the frame resolution
MIN_SIDE = 48      # stop the pyramid once the short side would drop below this
LAMBDA = 0.5       # per-pixel SAD penalty per step away from the seed (keeps flat areas still)
CUT_SAD = 30.0     # mean per-pixel SAD above this = scene cut / untrackable -> show frame a


def luma(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def pad_to_block(y):
    h, w = y.shape
    return cv2.copyMakeBorder(y, 0, -h % BLOCK, 0, -w % BLOCK, cv2.BORDER_REPLICATE)


def pyramid(y):
    levels = [y]
    while min(levels[-1].shape) // 2 >= MIN_SIDE:
        levels.append(cv2.resize(levels[-1], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    return levels


def search(a, c, seed):
    """One level: test every offset around each block's seed, keep the lowest-cost one."""
    h, w = a.shape
    bh, bw = h // BLOCK, w // BLOCK
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    seed_px = np.repeat(np.repeat(seed, BLOCK, 0), BLOCK, 1)
    best = np.full((bh, bw), np.inf, np.float32)
    best_v = seed.copy()
    best_sad = best.copy()
    for dy in range(-RADIUS, RADIUS + 1):
        for dx in range(-RADIUS, RADIUS + 1):
            hx, hy = seed_px[..., 0] + dx, seed_px[..., 1] + dy
            wa = cv2.remap(a, gx - hx, gy - hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            wc = cv2.remap(c, gx + hx, gy + hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            sad = np.abs(wa - wc).reshape(bh, BLOCK, bw, BLOCK).mean((1, 3))
            cost = sad + LAMBDA * (abs(dx) + abs(dy))
            better = cost < best
            best[better] = cost[better]
            best_sad[better] = sad[better]
            best_v[better] = seed[better] + (dx, dy)
    return best_v, best_sad


def median3(v):
    return np.dstack([cv2.medianBlur(np.ascontiguousarray(v[..., i]), 3) for i in range(2)])


def flow(a, c):
    """Half-vectors per BLOCK x BLOCK block at flow resolution, plus mean per-pixel SAD."""
    s = 1 / FLOW_SCALE
    pa = pyramid(cv2.resize(luma(a), None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    pc = pyramid(cv2.resize(luma(c), None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    v = None
    for la, lc in zip(reversed(pa), reversed(pc)):
        la, lc = pad_to_block(la), pad_to_block(lc)
        bh, bw = la.shape[0] // BLOCK, la.shape[1] // BLOCK
        if v is None:
            v = np.zeros((bh, bw, 2), np.float32)
        else:  # seed from the coarser level: 2x the vectors, 2x the block grid
            v = np.repeat(np.repeat(v * 2, 2, 0), 2, 1)
            v = np.pad(v, ((0, max(0, bh - v.shape[0])), (0, max(0, bw - v.shape[1])), (0, 0)), mode="edge")[:bh, :bw]
        v, sad = search(la, lc, v)
        v = median3(v)
    return v, float(sad.mean())


def interpolate(a, c):
    """Frame halfway between uint8 BGR frames a and c."""
    v, sad = flow(a, c)
    if sad > CUT_SAD:
        return a.copy()
    H, W = a.shape[:2]
    bh, bw = v.shape[:2]
    k = BLOCK * FLOW_SCALE
    # block vectors -> smooth per-pixel field at full resolution
    hv = cv2.resize(v, (bw * k, bh * k), interpolation=cv2.INTER_LINEAR)[:H, :W] * FLOW_SCALE
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    wa = cv2.remap(a, gx - hv[..., 0], gy - hv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wc = cv2.remap(c, gx + hv[..., 0], gy + hv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(wa, 0.5, wc, 0.5, 0)


if __name__ == "__main__":
    # self-check: a textured image panning 24 px right / 12 px up between a and c
    rng = np.random.default_rng(0)
    noise = lambda sigma: cv2.GaussianBlur(rng.standard_normal((400, 640, 3)).astype(np.float32), (0, 0), sigma)
    tex = sum(noise(s) / noise(s).std() for s in (1, 4, 16))  # detail at fine and coarse scales, like real images
    tex = np.clip(128 + 30 * tex, 0, 255).astype(np.uint8)
    shift = lambda img, dx, dy: np.roll(img, (dy, dx), (0, 1))
    a, b, c = tex, shift(tex, 12, -6), shift(tex, 24, -12)
    m = (slice(40, -40), slice(40, -40))  # ignore wrapped borders
    p_blend = cv2.PSNR(cv2.addWeighted(a, 0.5, c, 0.5, 0)[m], b[m])
    p_interp = cv2.PSNR(interpolate(a, c)[m], b[m])
    print(f"blend {p_blend:.1f} dB, interp {p_interp:.1f} dB")
    assert p_interp > p_blend + 10, "interpolation should beat a plain blend on a clean pan"
    print("ok")
