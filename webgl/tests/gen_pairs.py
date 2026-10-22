#!/usr/bin/env python3
"""Extra synthetic pairs that exercise V2 Performance's special paths (the five golden pairs hardly reach them): static overlays over a
moving scene (HUD text, a semi-transparent panel, a crosshair: the static mask, its enclosure and the zero-vector blend), rotation / zoom
and large motion (block-vector candidates, taps outside the frame), frame sizes that are no multiple of 4 / 8 (partial 4x4 groups, partial
tiles and cells) and the three flow scales. Writes tools/golden/out/v2pairs/<name>/frameA.bin, frameC.bin (RGBA8, row 0 = top) and
manifest.json (the golden manifest's shape, with width / height / flow_scale per pair). Needs numpy + cv2 (as ref_gen.py).

Usage: gen_pairs.py [--out DIR]
"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import cv2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def make_tex(h, w, seed, amp=30):
    rng = np.random.default_rng(seed)
    noise = lambda sigma: cv2.GaussianBlur(rng.standard_normal((h, w, 3)).astype(np.float32), (0, 0), sigma)
    tex = sum(noise(s) / noise(s).std() for s in (1, 4, 16))
    return np.clip(128 + amp * tex, 0, 255).astype(np.uint8)


def shift(img, dx, dy):
    return np.roll(img, (dy, dx), (0, 1))


def text(img, s, org, scale=0.7, col=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def hud(f):
    text(f, "SCORE 004217", (24, 40))
    text(f, "HP 83/100", (24, f.shape[0] - 24))
    cv2.rectangle(f, (f.shape[1] - 160, 16), (f.shape[1] - 24, 40), (255, 255, 255), 2)


def panel(f):
    h, w = f.shape[:2]
    x0, y0, x1, y1 = w // 2 - 130, h - 150, w // 2 + 130, h - 30
    roi = f[y0:y1, x0:x1].astype(np.float32)
    f[y0:y1, x0:x1] = np.clip(roi * 0.45 + np.array([20, 30, 60], np.float32) * 0.55, 0, 255).astype(np.uint8)
    cv2.rectangle(f, (x0, y0), (x1, y1), (230, 230, 230), 2)
    text(f, "Player joined", (x0 + 12, y0 + 36), 0.6)
    text(f, "Quest updated", (x0 + 12, y0 + 70), 0.6)
    text(f, "[E] Interact", (x0 + 12, y0 + 104), 0.6)


def crosshair(f):
    h, w = f.shape[:2]
    cx, cy = w // 2, h // 2
    roi = f[cy - 12:cy + 13, cx - 12:cx + 13].astype(np.int32)
    f[cy - 12:cy + 13, cx - 12:cx + 13] = np.clip(255 - roi, 0, 255).astype(np.uint8)   # inverted square
    cv2.line(f, (cx - 14, cy), (cx + 14, cy), (255, 255, 255), 1)
    cv2.line(f, (cx, cy - 14), (cx, cy + 14), (255, 255, 255), 1)


def affine(img, angle, scale, dx, dy):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += (dx, dy)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def scene_panel(h, w):
    bg = make_tex(h, w, 11)
    fg = make_tex(70, 90, 12, 50)
    a, c = bg.copy(), shift(bg, 10, -4)
    a[100:170, 120:210] = fg
    c[104:174, 134:224] = fg
    for f in (a, c):
        hud(f)
        panel(f)
    return a, c, "pan (10,-4) + moving patch + static HUD text and a semi-transparent text panel: static mask, enclosure, zero vector"


def scene_cross(h, w):
    bg = make_tex(h, w, 13)
    a, c = bg.copy(), shift(bg, 9, 5)
    for f in (a, c):
        crosshair(f)
        hud(f)
    return a, c, "pan (9,5) + inverted-colour crosshair + HUD text"


def scene_zoom(h, w):
    t = make_tex(h, w, 14)
    return t, affine(t, 2.5, 1.03, 5, -3), "rotation 2.5 deg + zoom 1.03 + shift: vectors differ per block (block-vector candidates)"


def scene_fast(h, w):
    t = make_tex(h, w, 15)
    fg = make_tex(80, 80, 16, 60)
    a, c = t.copy(), shift(t, 44, 16)
    a[60:140, 40:120] = fg
    c[90:170, 120:200] = fg
    return a, c, "large pan (44,16) + fast patch: search window edge, taps outside the frame"


def scene_odd(h, w):
    t = make_tex(h, w, 17)
    fg = make_tex(60, 70, 18, 50)
    a, c = t.copy(), shift(t, 7, -5)
    a[70:130, 90:160] = fg
    c[76:136, 102:172] = fg
    for f in (a, c):
        hud(f)
    return a, c, "odd frame size (no multiple of 4 / 8): pan (7,-5) + patch + HUD"


def scene_cut2(h, w):
    return make_tex(h, w, 21, 100), make_tex(h, w, 22, 100), "two unrelated very high-contrast textures: the scene-cut decision (mean re-picked SAD > 30)"


# name, scene, (w, h), flowScale
PAIRS = [("panel", scene_panel, (640, 384), 2), ("cross", scene_cross, (640, 384), 2), ("zoom", scene_zoom, (640, 384), 2),
         ("fast", scene_fast, (640, 384), 2), ("odd", scene_odd, (646, 382), 2), ("fs1", scene_panel, (640, 384), 1),
         ("fs4", scene_panel, (640, 384), 4), ("cut2", scene_cut2, (640, 384), 2)]


def rgba(img):
    h, w = img.shape[:2]
    out = np.empty((h, w, 4), np.uint8)
    out[..., :3] = img
    out[..., 3] = 255
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "out", "v2pairs"))
    out = os.path.abspath(ap.parse_args().out)
    man = []
    for name, fn, (w, h), fs in PAIRS:
        a, c, desc = fn(h, w)
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        rgba(a).tofile(os.path.join(d, "frameA.bin"))
        rgba(c).tofile(os.path.join(d, "frameC.bin"))
        man.append({"name": name, "desc": desc, "width": w, "height": h, "flow_scale": fs,
                    "arrays": {"frame.prev": {"file": f"{name}/frameA.bin"}, "frame.curr": {"file": f"{name}/frameC.bin"}}})
        print(f"gen_pairs: {name} {w}x{h} flowScale {fs}: {desc}")
    json.dump({"width": 640, "height": 384, "flow_scale": 2, "ts": [0.5, 0.25], "pairs": man},
              open(os.path.join(out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
