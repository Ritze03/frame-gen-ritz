"""Image-only frame interpolation: CPU prototype of the planned GPU version.

Bilateral: for a block at position q of the *in-between* frame we search a half-vector h
minimising SAD(a[q - h], c[q + h]). Vectors land directly on the output grid, so no
splatting, no holes, no atomics on the GPU.

GPU passes (flow res = 1/FLOW_SCALE of the frame; L = pyramid levels, 4 at 1440p):
  1  luma + 2x downsample of a and c                     (1 pass, per pixel)
  2  pyramid: 2x2 average per level                      (L-1 tiny passes, or one SPD pass)
  per level, coarse -> fine:
  3  seed: 6 candidates per block (own parent x2, 3 nearest other parents x2, zero,
     global median), pick lowest block SAD              (1 pass, 1 workgroup per 8x8 block)
     [global median: a tiny reduction over the previous level's few hundred vectors]
  4  integer search +-RADIUS around the seed (+-TOP_RADIUS at the coarsest level),
     cost = SAD + LAMBDA * |step|                        (1 pass, block + margin in LDS)
  5  3x3 median of the block vectors                     (1 pass)
  finest level only:
  6  sub-pixel refine: 3x3 at 0.5 px, then 3x3 at 0.25 px, bilinear taps from LDS (2 passes or 1)
  7  global confidence: mean best SAD > CUT_SAD -> present a, skip the rest
  8  per-pixel vector choice at flow res: candidates = bilinear field, own block, 4 neighbour
     blocks, zero; cost = 5x5 box of |a(q-h) - c(q+h)| (luma), small bonus for bilinear and
     zero (static HUD lock)                              (1 pass, tile + apron in LDS)
  9  full res: bilinear-upsample the per-pixel field, sample a(q-h), c(q+h) in RGB,
     mismatch e = max-channel |wa - wc|                  (1 pass, writes e)
 10  7x7 box of e (separable or LDS), trust t = ramp(e, TRUST), count untrusted pixels
     (one atomic add per workgroup)                       (1 pass)
 11  out = mix(0.5*(wa+wc), a, t); if the untrusted share > GLOBAL_FRAC present a instead
     (the decision needs pass 10's count, so either a separate pass or apply it next frame)
"""
import cv2
import numpy as np

BLOCK = 8              # block size at every pyramid level
RADIUS = 2             # search radius per level, in half-vector pixels of that level
TOP_RADIUS = 6         # search radius at the coarsest level (tiny level, so a wide window is cheap)
FLOW_SCALE = 2         # flow runs at 1/FLOW_SCALE of the frame resolution
MIN_SIDE = 48          # stop the pyramid once the short side would drop below this
LAMBDA = 0.5           # per-pixel SAD penalty per step away from the seed (keeps flat areas still)
SUBPIX = (0.5, 0.25)   # sub-pixel refinement steps at the finest level
CUT_SAD = 30.0         # mean per-pixel SAD above this = scene cut / untrackable -> show frame a
PIX_WIN = 5            # window (flow-res px) of the per-pixel vector choice
SMOOTH_BONUS = 0.5     # per-pixel cost bonus for the bilinear field (prefer smooth where it is as good)
ZERO_BONUS = 1.0       # per-pixel cost bonus for the zero vector (static HUD stays put)
MISMATCH_BOX = 7       # box filter over |wa - wc| before it becomes a trust value
TRUST = (12.0, 40.0)   # mismatch (8-bit levels) where the fallback to frame a starts / is complete
GLOBAL_FRAC = 0.15     # more than this share of untrusted pixels -> present frame a


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


_grids = {}


def grid(h, w):
    if (h, w) not in _grids:
        _grids[(h, w)] = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return _grids[(h, w)]


def warp_pair(a, c, hx, hy):
    """Sample a at q - h and c at q + h (bilinear)."""
    gx, gy = grid(*a.shape[:2])
    wa = cv2.remap(a, gx - hx, gy - hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wc = cv2.remap(c, gx + hx, gy + hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return wa, wc


def block_sad(a, c, v):
    """Mean |a(q - h) - c(q + h)| per block for a per-block half-vector field v."""
    h, w = a.shape
    bh, bw = h // BLOCK, w // BLOCK
    vp = np.repeat(np.repeat(v, BLOCK, 0), BLOCK, 1)
    wa, wc = warp_pair(a, c, vp[..., 0], vp[..., 1])
    return np.abs(wa - wc).reshape(bh, BLOCK, bw, BLOCK).mean((1, 3))


def pick(cands, costs):
    """Per element, the candidate with the lowest cost (first one wins ties)."""
    best, out = costs[0], cands[0]
    for cv, cost in zip(cands[1:], costs[1:]):
        m = cost < best
        best = np.where(m, cost, best)
        out = np.where(m[..., None], cv, out)
    return out, best


def search(a, c, seed, radius, step):
    """Test every offset (in units of `step`) around each block's seed, keep the lowest-cost one."""
    offs = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    cands = [seed + np.float32((dx * step, dy * step)) for dx, dy in offs]
    sads = [block_sad(a, c, cv) for cv in cands]
    costs = [s + LAMBDA * step * (abs(dx) + abs(dy)) for s, (dx, dy) in zip(sads, offs)]
    v, _ = pick(cands, costs)
    best_sad, _ = pick([s[..., None] for s in sads], costs)
    return v, best_sad[..., 0]


def median3(v):
    return np.dstack([cv2.medianBlur(np.ascontiguousarray(v[..., i]), 3) for i in range(2)])


def seed_from(v, la, lc, bh, bw):
    """Coarse block field -> seed per fine block: 2x vectors on a 2x grid, chosen among the own
    parent, the 3 nearest other parents, zero and the global median by block SAD. The global
    median rescues regions where a repetitive pattern (stairs, fences) locked onto a wrong period."""
    ph, pw = v.shape[:2]
    iy, ix = np.meshgrid(np.arange(bh), np.arange(bw), indexing="ij")
    py, px = np.minimum(iy // 2, ph - 1), np.minimum(ix // 2, pw - 1)
    sy, sx = np.where(iy % 2 == 0, -1, 1), np.where(ix % 2 == 0, -1, 1)
    parent = lambda oy, ox: v[np.clip(py + oy, 0, ph - 1), np.clip(px + ox, 0, pw - 1)] * 2
    own = parent(0, 0)
    cands = [own, parent(sy, 0), parent(0, sx), parent(sy, sx), np.zeros_like(own),
             np.broadcast_to(np.median(own.reshape(-1, 2), 0), own.shape).astype(np.float32)]
    costs = [block_sad(la, lc, cv) + (0 if i == 0 else LAMBDA) for i, cv in enumerate(cands)]
    return pick(cands, costs)[0]


def flow(a, c):
    """Half-vectors per BLOCK x BLOCK block at flow resolution, mean per-pixel SAD, flow-res lumas."""
    s = 1 / FLOW_SCALE
    ya = cv2.resize(luma(a), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    yc = cv2.resize(luma(c), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    pa, pc = pyramid(ya), pyramid(yc)
    v = None
    for la, lc in zip(reversed(pa), reversed(pc)):
        la, lc = pad_to_block(la), pad_to_block(lc)
        bh, bw = la.shape[0] // BLOCK, la.shape[1] // BLOCK
        if v is None:
            v, sad = search(la, lc, np.zeros((bh, bw, 2), np.float32), TOP_RADIUS, 1.0)
        else:
            v, sad = search(la, lc, seed_from(v, la, lc, bh, bw), RADIUS, 1.0)
        v = median3(v)
    for step in SUBPIX:  # la, lc are the finest level here
        v, sad = search(la, lc, v, 1, step)
    return v, float(sad.mean()), ya, yc


def pixel_field(v, ya, yc):
    """Per-pixel half-vectors at flow resolution: each pixel picks among the bilinear field, its own
    block, the 4 neighbour blocks and zero by a small-window match. Sharpens motion edges."""
    h, w = ya.shape
    bh, bw = v.shape[:2]
    up = lambda f: np.repeat(np.repeat(f, BLOCK, 0), BLOCK, 1)[:h, :w]
    vp = np.pad(v, ((1, 1), (1, 1), (0, 0)), mode="edge")
    cands = [cv2.resize(v, (bw * BLOCK, bh * BLOCK), interpolation=cv2.INTER_LINEAR)[:h, :w], up(v)]
    cands += [up(vp[1 + oy:1 + oy + bh, 1 + ox:1 + ox + bw]) for oy, ox in ((-1, 0), (1, 0), (0, -1), (0, 1))]
    cands.append(np.zeros_like(cands[0]))
    costs = []
    for cv in cands:
        wa, wc = warp_pair(ya, yc, cv[..., 0], cv[..., 1])
        costs.append(cv2.blur(np.abs(wa - wc), (PIX_WIN, PIX_WIN), borderType=cv2.BORDER_REPLICATE))
    costs[0] -= SMOOTH_BONUS
    costs[-1] -= ZERO_BONUS
    return pick(cands, costs)[0]


def interpolate(a, c):
    """Frame halfway between uint8 BGR frames a and c."""
    v, sad, ya, yc = flow(a, c)
    if sad > CUT_SAD:
        return a.copy()
    H, W = a.shape[:2]
    pf = pixel_field(v, ya, yc)
    hv = cv2.resize(pf, (pf.shape[1] * FLOW_SCALE, pf.shape[0] * FLOW_SCALE), interpolation=cv2.INTER_LINEAR)
    hv = np.pad(hv[:H, :W], ((0, max(0, H - hv.shape[0])), (0, max(0, W - hv.shape[1])), (0, 0)), mode="edge")
    hv *= FLOW_SCALE
    wa, wc = warp_pair(a, c, hv[..., 0], hv[..., 1])
    wa, wc = wa.astype(np.float32), wc.astype(np.float32)
    # where the two warped samples disagree (occlusion, wrong vector, changing HUD) the region falls
    # back to frame a: briefly holding still is invisible, a ghost or a smear is not
    e = cv2.blur(np.abs(wa - wc).max(2), (MISMATCH_BOX, MISMATCH_BOX), borderType=cv2.BORDER_REPLICATE)
    t = np.clip((e - TRUST[0]) / (TRUST[1] - TRUST[0]), 0, 1)
    if (t > 0.5).mean() > GLOBAL_FRAC:  # too much of the frame is untrusted (fast flick): no guessing
        return a.copy()
    t = t[..., None]
    out = 0.5 * (wa + wc) * (1 - t) + a.astype(np.float32) * t
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


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
    # static HUD over the same pan: the HUD must stay sharp and in place
    hud = (slice(300, 340), slice(60, 260))
    for f in (a, b, c):
        f[hud] = 255
        f[310:330, 80:240] = 20
    o = interpolate(a, c)
    p_hud = cv2.PSNR(o[hud], b[hud])
    print(f"static HUD {p_hud:.1f} dB")
    assert p_hud > 30, "a static overlay should not be dragged along with the background"
    print("ok")
