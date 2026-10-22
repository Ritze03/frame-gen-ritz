#!/usr/bin/env python3
"""Golden reference generator: runs frame-gen-ritz proto/interp.py (main @ b77d058) on synthetic test pairs and
writes frames, outputs, decisions and per-stage intermediates (raw little-endian .bin + manifest.json).

Reference location: env FGR_REF (default: the lead's scratchpad clone). interp.py is imported from there
(never copied). Array names equal the engine's debugRead names (extension/engine/LAYOUT.md 6.1), so the
browser side compares by name. Layout of every array is row-major (H, W, C), row 0 = top, channels interleaved,
exactly like debugRead's `data`. RGB images are RGBA (converted from interp.py's BGR), alpha 255.

Usage: ref_gen.py [--out DIR] [--ts 0.5,0.25,0.75] [--size WxH] [--pairs a,b,...] [--list]
"""
import argparse
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import cv2  # noqa: E402

HERE_ = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REF = os.path.dirname(os.path.dirname(HERE_))   # this repository (its proto/interp.py is the pinned reference)
WORK = os.environ.get("FG_WORK", os.path.join(DEFAULT_REF, ".local-work", "webgl"))
PINNED_COMMIT = "b77d058"
INTERP_SHA256 = "b5f5c7000221002e31a1f4c2b2883878eeab4ffd83760c4080facad527904315"  # sha256 of proto/interp.py at the pinned commit (guards against the unfinished framegen-v2 branch)

HERE = os.path.dirname(os.path.abspath(__file__))


def die(msg):
    print(f"ref_gen: {msg}", file=sys.stderr)
    sys.exit(2)


def load_interp():
    ref = os.environ.get("FGR_REF", DEFAULT_REF)
    path = os.path.join(ref, "proto", "interp.py")
    if not os.path.isfile(path):
        die(f"reference not found: {path}\n"
            f"  Set FGR_REF to a checkout of frame-gen-ritz `main` (NOT the framegen-v2 branch). Recreate with either:\n"
            f"    git clone --branch main https://github.com/Ritze03/frame-gen-ritz <dir>\n"
            f"    mkdir -p <dir> && git -C /home/mo/Schinken/Programming/GitHubProjects/frame-gen-ritz archive main proto | tar -x -C <dir>\n"
            f"  then: export FGR_REF=<dir>")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if sha != INTERP_SHA256 and not os.environ.get("FGR_ALLOW_ANY"):
        die(f"{path} sha256 {sha[:12]} != pinned {INTERP_SHA256[:12]} (main@{PINNED_COMMIT}).\n"
            f"  Wrong branch/commit? (framegen-v2 must not be used). Set FGR_ALLOW_ANY=1 to override.")
    sys.path.insert(0, os.path.join(ref, "proto"))
    import interp  # noqa: E402
    return interp, ref, sha


# ----------------------------------------------------------------------------------------------- footage
def make_tex(h, w, seed, amp=30):
    """Texture generator from interp.py's own __main__ self-check (detail at fine and coarse scales)."""
    rng = np.random.default_rng(seed)
    noise = lambda sigma: cv2.GaussianBlur(rng.standard_normal((h, w, 3)).astype(np.float32), (0, 0), sigma)
    tex = sum(noise(s) / noise(s).std() for s in (1, 4, 16))
    return np.clip(128 + amp * tex, 0, 255).astype(np.uint8)


def shift(img, dx, dy):
    return np.roll(img, (dy, dx), (0, 1))


def add_hud(f):
    f[300:340, 60:260] = 255          # same HUD as the interp.py self-check
    f[310:330, 80:240] = 20


def scene_pan(h, w):
    t = make_tex(h, w, 0)
    return t, shift(t, 24, -12), "textured pan 24 px right / 12 px up (integer half-vector 12,-6 at full res)"


def scene_hud_odd(h, w):
    t = make_tex(h, w, 1)
    a, c = t.copy(), shift(t, 13, -7)
    for f in (a, c):
        add_hud(f)
    return a, c, "pan 13 right / 7 up (half-pixel vectors, exercises sub-pixel) + static HUD box"


def scene_objects(h, w):
    bg = make_tex(h, w, 2)
    fg = make_tex(120, 160, 3)
    a, c = bg.copy(), shift(bg, 6, -3)
    a[120:240, 200:360] = fg
    c[128:248, 220:380] = fg          # foreground moves (+20,+8) against a panning background: occlusions
    return a, c, "background pan (6,-3) + 160x120 foreground patch moving (20,8): occlusion / trust ramp"


def scene_noisy(h, w):
    t = make_tex(h, w, 6)
    rng = np.random.default_rng(7)
    c = shift(t, 8, 4).astype(np.float32) + 22 * rng.standard_normal(t.shape).astype(np.float32)
    return t, np.clip(c, 0, 255).astype(np.uint8), "pan (8,4) + heavy per-pixel noise on c: mismatch high everywhere (untrusted / flick candidate)"


def scene_cut(h, w):
    return make_tex(h, w, 4, 50), make_tex(h, w, 5, 50), "two unrelated high-contrast textures: scene cut (mean block SAD > 30)"


SCENES = {"pan": scene_pan, "hud_odd": scene_hud_odd, "objects": scene_objects, "noisy": scene_noisy, "cut": scene_cut}


# ----------------------------------------------------------------------------------------------- tracing
def flow_trace(I, a, c):
    """Re-run interp.flow() capturing per-level intermediates (LAYOUT 6.1)."""
    s = 1 / I.FLOW_SCALE
    ya = cv2.resize(I.luma(a), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    yc = cv2.resize(I.luma(c), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    pa, pc = I.pyramid(ya), I.pyramid(yc)
    L = len(pa)
    fields, seeds, medians = {}, {}, {}
    v = None
    for idx, (la, lc) in enumerate(zip(reversed(pa), reversed(pc))):
        i = L - 1 - idx
        la, lc = I.pad_to_block(la), I.pad_to_block(lc)
        bh, bw = la.shape[0] // I.BLOCK, la.shape[1] // I.BLOCK
        if v is None:
            v, sad = I.search(la, lc, np.zeros((bh, bw, 2), np.float32), I.TOP_RADIUS, 1.0)
        else:
            ph, pw = v.shape[:2]
            iy, ix = np.meshgrid(np.arange(bh), np.arange(bw), indexing="ij")
            own = v[np.minimum(iy // 2, ph - 1), np.minimum(ix // 2, pw - 1)] * 2   # as seed_from's `own`
            medians[i] = np.median(own.reshape(-1, 2), 0).astype(np.float32)
            seed = I.seed_from(v, la, lc, bh, bw)
            seeds[i] = seed.copy()
            v, sad = I.search(la, lc, seed, I.RADIUS, 1.0)
        v = I.median3(v)
        fields[i] = v.copy()
    sub1 = None
    for k, step in enumerate(I.SUBPIX):
        v, sad = I.search(la, lc, v, 1, step)
        if k == 0:
            sub1 = (v.copy(), sad.copy())
    return dict(ya=ya, yc=yc, pa=pa, pc=pc, fields=fields, seeds=seeds, medians=medians, sub1=sub1, v=v, sad=sad)


def synth_trace(I, a, c, pf, mean_sad, t):
    """interp.synth() step by step; returns intermediates + decision. Computes everything even on a cut
    (the engine does too; interp.synth would return early)."""
    H, W = a.shape[:2]
    pft = I.field_at(pf, t)
    hv = cv2.resize(pft, (pft.shape[1] * I.FLOW_SCALE, pft.shape[0] * I.FLOW_SCALE), interpolation=cv2.INTER_LINEAR)
    hv = np.pad(hv[:H, :W], ((0, max(0, H - hv.shape[0])), (0, max(0, W - hv.shape[1])), (0, 0)), mode="edge")
    hv *= I.FLOW_SCALE
    wa, wc = I.warp_t(a, c, hv[..., 0], hv[..., 1], t)
    mism = np.abs(wa.astype(np.float32) - wc.astype(np.float32)).max(2)
    e = cv2.blur(mism, (I.MISMATCH_BOX, I.MISMATCH_BOX), borderType=cv2.BORDER_REPLICATE)
    u = np.clip((e - I.TRUST[0]) / (I.TRUST[1] - I.TRUST[0]), 0, 1)
    count = int((u > 0.5).sum())
    share = float((u > 0.5).mean())
    cut = mean_sad > I.CUT_SAD
    flick = share > I.GLOBAL_FRAC
    why_expected = "cut" if cut else ("untrusted" if flick else None)
    out, why = I.synth(a, c, {"sad": mean_sad, "pf": pf}, t, why=True)
    assert why == why_expected, f"trace decision {why_expected} != interp.synth {why}"
    return dict(pft=pft, wa=wa, wc=wc, mism=mism, trust=u.astype(np.float32), count=count, share=share,
                cut=bool(cut), flick=bool(flick), why=why, out=out)


# ----------------------------------------------------------------------------------------------- writing
class Writer:
    def __init__(self, root):
        self.root = root

    def put(self, rel, arr, dtype, entries, key):
        arr = np.ascontiguousarray(arr)
        if arr.ndim == 2:
            arr = arr[..., None]
        arr = arr.astype(dtype)
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arr.astype(arr.dtype.newbyteorder("<")).tofile(path)
        entries[key] = {"file": rel, "dtype": np.dtype(dtype).str.lstrip("<|"), "shape": list(arr.shape)}


def rgba(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(WORK, "ref"))
    ap.add_argument("--ts", default="0.5,0.25,0.75")
    ap.add_argument("--size", default="640x384", help="WxH; must be divisible by 4*flowScale with even pyramid levels (LAYOUT 5.8)")
    ap.add_argument("--pairs", default=",".join(SCENES))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("\n".join(SCENES))
        return
    W, H = map(int, args.size.lower().split("x"))
    ts = [float(x) for x in args.ts.split(",")]
    I, ref, sha = load_interp()
    cv2.setNumThreads(1)
    wr = Writer(args.out)
    manifest = {"version": 1, "reference": {"dir": ref, "pinned_commit": PINNED_COMMIT, "interp_sha256": sha},
                "width": W, "height": H, "flow_scale": I.FLOW_SCALE, "ts": ts,
                "params": {"CUT_SAD": I.CUT_SAD, "GLOBAL_FRAC": I.GLOBAL_FRAC, "TRUST": list(I.TRUST),
                           "FLOW_SCALE": I.FLOW_SCALE, "BLOCK": I.BLOCK},
                "pairs": []}
    for name in args.pairs.split(","):
        if name not in SCENES:
            die(f"unknown pair {name!r}; known: {', '.join(SCENES)}")
        a, c, desc = SCENES[name](H, W)
        assert a.shape == (H, W, 3) and c.shape == a.shape
        tr = flow_trace(I, a, c)
        # consistency of my trace with the reference's own flow()/estimate()
        v_ref, sad_ref, ya_ref, _ = I.flow(a, c)
        assert np.array_equal(v_ref, tr["v"]) and sad_ref == float(tr["sad"].mean()), "flow trace diverged from interp.flow"
        mean_sad = float(tr["sad"].mean())
        pf = I.pixel_field(tr["v"], tr["ya"], tr["yc"])        # always computed (engine has no GPU-side skip)
        est = I.estimate(a, c)
        assert est["sad"] == mean_sad and (est["pf"] is None) == (mean_sad > I.CUT_SAD)
        if est["pf"] is not None:
            assert np.array_equal(est["pf"], pf)

        arr = {}
        sub = f"{name}"
        wr.put(f"{sub}/frameA.bin", rgba(a), "u1", arr, "frame.prev")
        wr.put(f"{sub}/frameC.bin", rgba(c), "u1", arr, "frame.curr")
        L = len(tr["pa"])
        levels = []
        for i in range(L):
            h_, w_ = tr["pa"][i].shape
            levels.append({"w": w_, "h": h_, "bw": -(-w_ // 8), "bh": -(-h_ // 8)})
            wr.put(f"{sub}/luma.prev.{i}.bin", tr["pa"][i], "f4", arr, f"luma.prev.{i}")
            wr.put(f"{sub}/luma.curr.{i}.bin", tr["pc"][i], "f4", arr, f"luma.curr.{i}")
            wr.put(f"{sub}/field.{i}.bin", tr["fields"][i], "f4", arr, f"field.{i}")
            if i in tr["seeds"]:
                wr.put(f"{sub}/seed.{i}.bin", tr["seeds"][i], "f4", arr, f"seed.{i}")
                wr.put(f"{sub}/median.{i}.bin", tr["medians"][i].reshape(1, 1, 2), "f4", arr, f"median.{i}")
        s1v, s1s = tr["sub1"]
        z = np.zeros_like(s1s)
        wr.put(f"{sub}/sub1.bin", np.dstack([s1v, s1s, z]), "f4", arr, "sub1")
        wr.put(f"{sub}/sub2.bin", np.dstack([tr["v"], tr["sad"], np.zeros_like(tr["sad"])]), "f4", arr, "sub2")
        wr.put(f"{sub}/sadSum.bin", np.array([float(tr["sad"].sum(dtype=np.float64))]).reshape(1, 1, 1), "f4", arr, "sadSum")
        wr.put(f"{sub}/pixfield.bin", pf, "f4", arr, "pixfield")

        tdata = {}
        for t in ts:
            st = synth_trace(I, a, c, pf, mean_sad, t)
            ta = {}
            d = f"{sub}/t{t:g}"
            if t != 0.5:
                wr.put(f"{d}/fieldT.bin", st["pft"], "f4", ta, "fieldT")
            wr.put(f"{d}/warpA.bin", rgba(st["wa"]), "u1", ta, "warpA")
            wr.put(f"{d}/warpC.bin", rgba(st["wc"]), "u1", ta, "warpC")
            wr.put(f"{d}/mismatch.bin", st["mism"], "u1", ta, "mismatch")
            wr.put(f"{d}/trust.bin", st["trust"], "f4", ta, "trust")
            wr.put(f"{d}/untrusted.bin", np.array([float(st["count"])]).reshape(1, 1, 1), "f4", ta, "untrusted")
            wr.put(f"{d}/out.bin", rgba(st["out"]), "u1", ta, "out")
            tdata[f"{t:g}"] = {"fallback": st["why"], "cut": st["cut"], "flick": st["flick"],
                               "untrusted": st["count"], "untrusted_share": st["share"], "mean_sad": mean_sad,
                               "arrays": ta}
        manifest["pairs"].append({"name": name, "desc": desc, "levels": levels, "mean_sad": mean_sad,
                                  "arrays": arr, "t": tdata})
        fb = {k: v["fallback"] for k, v in tdata.items()}
        print(f"  {name:8} L={L} meanSad={mean_sad:7.3f} untrusted_share(t=.5)="
              f"{tdata.get('0.5', next(iter(tdata.values())))['untrusted_share']:.3f} fallback={fb}  ({desc})")
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"ref_gen: wrote {len(manifest['pairs'])} pairs to {args.out} (reference {ref} interp.py sha {sha[:12]})")


if __name__ == "__main__":
    main()
