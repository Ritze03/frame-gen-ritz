#!/usr/bin/env python3
"""Test pairs of the cross-backend check (webgl/tests/xbackend.mjs, vkref.mjs): synthetic scenes that reach the special paths of every
motion version, real footage, and several frame sizes (odd, tiny, 1080p, 4K). Writes $FG_WORK/pairs/<name>/a.rgba, c.rgba (RGBA8,
row 0 = top, W*H*4 bytes) and $FG_WORK/pairs/manifest.json:
  {"pairs": [{"name", "desc", "width", "height", "flow_scales": [..], "kind": "synthetic"|"footage"}], "ts": [..]}

Usage: pairs.py [--out DIR] [--only a,b] [--no-footage] [--list]
Footage: $FG_FOOTAGE (default <repo>/footage, the bench footage of README "Footage"): battlefield.mp4 and two minecraft120 cuts; pairs
whose file is missing are skipped with a note. Needs numpy, cv2 and (for footage) ffmpeg.
"""
import argparse
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import cv2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
WORK = os.environ.get("FG_WORK", os.path.join(REPO, ".local-work", "webgl"))
FOOTAGE = os.environ.get("FG_FOOTAGE", os.path.join(REPO, "footage"))
sys.path.insert(0, HERE)
import gen_pairs as G  # noqa: E402  (scenes: HUD text, panel, crosshair, rotation, fast motion, odd size, cut)
import ref_gen as R  # noqa: E402  (the V1 golden scenes: pan, hud_odd, objects, noisy, cut)


def scene_square(h, w):
    """A moving square over a static texture (uitest's moving object on static scenery)."""
    bg = G.make_tex(h, w, 31)
    sq = G.make_tex(64, 64, 32, 60)
    a, c = bg.copy(), bg.copy()
    a[140:204, 200:264] = sq
    c[140:204, 212:276] = sq
    return a, c, "static textured scene + a 64 px square moving 12 px right (V2 static mask must not cross-fade it)"


def scene_hudpan(h, w):
    bg = G.make_tex(h, w, 33)
    a, c = bg.copy(), G.shift(bg, 14, 6)
    for f in (a, c):
        G.hud(f)
        G.text(f, "AMMO 30 | 120", (w - 230, h - 24))
    return a, c, "pan (14,6) under static HUD text (score, HP, ammo) and a HUD box"


def scene_ui(h, w):
    """uitest check's scene: a textured background panning 14 / 6 px, 16 independently moving objects, a static opaque crosshair
    (plus-shaped arms, centre dot, 1 px black outline) and a static opaque HUD box in a corner."""
    rng = np.random.default_rng(41)
    bg = G.make_tex(h + 64, w + 64, 42)
    a, c = bg[32:32 + h, 32:32 + w].copy(), bg[26:26 + h, 18:18 + w].copy()   # c = a moved by (14, 6)
    for k in range(16):
        s = int(rng.integers(20, 60))
        x, y = int(rng.integers(0, w - s - 40)), int(rng.integers(0, h - s - 40))
        dx, dy = int(rng.integers(-16, 17)), int(rng.integers(-16, 17))
        x2, y2 = min(max(x + dx, 0), w - s), min(max(y + dy, 0), h - s)
        col = rng.integers(0, 256, 3).astype(np.uint8)
        obj = np.clip(G.make_tex(s, s, 100 + k, 40).astype(np.int32) // 2 + col // 2, 0, 255).astype(np.uint8)
        a[y:y + s, x:x + s] = obj
        c[y2:y2 + s, x2:x2 + s] = obj
    for f in (a, c):
        cx, cy = w // 2, h // 2
        for (x0, y0, x1, y1, col) in ((cx - 17, cy - 3, cx + 18, cy + 4, 0), (cx - 3, cy - 17, cx + 4, cy + 18, 0),
                                      (cx - 16, cy - 2, cx + 17, cy + 3, 255), (cx - 2, cy - 16, cx + 3, cy + 17, 255)):
            f[y0:y1, x0:x1] = col
        f[cy - 1:cy + 2, cx - 1:cx + 2] = (255, 40, 40)
        f[16:76, 16:196] = (30, 30, 30)
        f[22:70, 22:190] = (200, 180, 40)
    return a, c, "uitest-like: pan (14,6) + 16 moving objects + static opaque crosshair + static HUD box"


def scene_tiny(h, w):
    t = G.make_tex(h, w, 51)
    return t, G.shift(t, 3, -2), "tiny frame: pan (3,-2)"


def bgr2rgb(fn):
    def f(h, w):
        a, c, d = fn(h, w)
        return a[..., ::-1].copy(), c[..., ::-1].copy(), d
    return f


def footage(file, t, crop=None, scale=None):
    def f(h, w):
        path = os.path.join(FOOTAGE, file)
        if not os.path.isfile(path):
            return None
        cmd = ["ffmpeg", "-v", "error", "-ss", str(t), "-i", path, "-frames:v", "2"]
        if scale:
            cmd += ["-vf", f"scale={scale[0]}:{scale[1]}:flags=lanczos"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(cmd, check=True, capture_output=True).stdout
        sw, sh = scale if scale else (1920, 1080)
        fr = np.frombuffer(raw, np.uint8).reshape(2, sh, sw, 3)
        a, c = fr[0], fr[1]
        if crop:
            x, y = crop
            a, c = a[y:y + h, x:x + w], c[y:y + h, x:x + w]
        return a.copy(), c.copy(), f"{file} @ {t} s" + (f", crop {w}x{h}+{crop[0]}+{crop[1]}" if crop else "") + \
            (f", lanczos to {scale[0]}x{scale[1]}" if scale else "")
    return f


FS = [1, 2, 4]
# name, generator, (w, h), flow scales
PAIRS = [
    ("pan", bgr2rgb(R.scene_pan), (640, 384), FS),
    ("hud_odd", bgr2rgb(R.scene_hud_odd), (640, 384), FS),
    ("objects", bgr2rgb(R.scene_objects), (640, 384), FS),
    ("noisy", bgr2rgb(R.scene_noisy), (640, 384), FS),
    ("cut", bgr2rgb(R.scene_cut), (640, 384), FS),
    ("panel", G.scene_panel, (640, 384), FS),
    ("cross", G.scene_cross, (640, 384), FS),
    ("zoom", G.scene_zoom, (640, 384), FS),
    ("fast", G.scene_fast, (640, 384), FS),
    ("square", scene_square, (640, 384), FS),
    ("hudpan", scene_hudpan, (640, 384), FS),
    ("odd", G.scene_odd, (645, 383), FS),
    ("tiny", scene_tiny, (97, 61), FS),
    ("ui", scene_ui, (1280, 720), [2]),
    ("battlefield", footage("battlefield.mp4", 10.0), (1920, 1080), FS),
    ("mc_furnace", footage("cuts/minecraft120_03_furnace_hud.mp4", 4.0), (1920, 1080), FS),
    ("mc_sprint", footage("cuts/minecraft120_10_forest_sprint.mp4", 8.0), (1920, 1080), FS),
    ("bf_odd", footage("battlefield.mp4", 14.0, crop=(301, 157)), (1279, 719), [2]),
    ("bf_4k", footage("battlefield.mp4", 10.0, scale=(3840, 2160)), (3840, 2160), [2, 4]),
    ("bf_8k", footage("battlefield.mp4", 10.0, scale=(7680, 4320)), (7680, 4320), [4]),   # (xbackend.mjs --gpu: SwiftShader needs GBs)
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(WORK, "pairs"))
    ap.add_argument("--only", default="")
    ap.add_argument("--no-footage", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for name, _, (w, h), fs in PAIRS:
            print(f"{name:12s} {w}x{h} flowScale {fs}")
        return
    only = set(filter(None, args.only.split(",")))
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    mpath = os.path.join(out, "manifest.json")
    man = json.load(open(mpath)) if os.path.exists(mpath) else {"pairs": []}
    have = {p["name"]: p for p in man["pairs"]}
    for name, fn, (w, h), fs in PAIRS:
        if only and name not in only:
            continue
        kind = "footage" if fn.__name__ == "f" and fn.__qualname__.startswith("footage") else "synthetic"
        if kind == "footage" and args.no_footage:
            continue
        r = fn(h, w)
        if r is None:
            print(f"pairs: {name}: footage file missing under {FOOTAGE}, skipped")
            continue
        a, c, desc = r
        assert a.shape == (h, w, 3) and c.shape == (h, w, 3), (name, a.shape)
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        for k, img in (("a", a), ("c", c)):
            G.rgba(np.ascontiguousarray(img)).tofile(os.path.join(d, k + ".rgba"))
        have[name] = {"name": name, "desc": desc, "width": w, "height": h, "flow_scales": fs, "kind": kind}
        print(f"pairs: {name:12s} {w}x{h} flowScale {fs}: {desc}")
    order = [p[0] for p in PAIRS]
    man = {"ts": [0.5, 0.25], "pairs": sorted(have.values(), key=lambda p: order.index(p["name"]) if p["name"] in order else 99)}
    json.dump(man, open(mpath, "w"), indent=1)
    print(f"pairs: {len(man['pairs'])} pairs in {out}")


if __name__ == "__main__":
    main()
