#!/usr/bin/env python3
"""Motion maps and contact sheets of a motion estimate (see motion/README.md).

  maps.py --fgtest PATH --set 'algo=v3' --out DIR --pairs pairs.json [--pairs more.json] [--only substr,...] [--label TEXT]

Per pair writes DIR/sheets/<name>.png (one contact sheet, 1800 px wide, same layout every time) and DIR/png/<name>_{motion,truth,error,steps}.png
at the flow grid's resolution.  Colours: motion = hue is the direction, saturation the magnitude (white = no motion), full saturation at the
pair's fixed scale (its `scale` in the list: 8 / 24 / 64 px for the synthetic strengths), the wheel in the corner; error = end-point error against
the truth in px, inferno, 0 .. scale/8, dark grey = not valid (hidden in a frame / invented border / cut-out outline); steps = magnitude of the
estimated field's Jacobian, grey = (g / 2 px per flow px)^0.6, black = smooth, white = a step of 2 px or more between neighbours.
Layout, synthetic pairs: row 1 true middle frame | true motion | estimated motion, row 2 steps of the estimate | error | steps of the truth.
Real pairs: row 1 prev frame | estimated motion | steps, row 2 real in-between frame (or curr) | generated t = 0.5 frame | |generated - real| x 4.

Before / after sheets (task 157): `--ref-fgtest PATH [--ref-set 'algo=v3']` runs a second build / setting on the same pairs and writes DIR/compare/<name>.png instead
(the same 1800 px width), one fixed error scale and one fixed wheel scale for both sides.
Synthetic: true motion | REF estimate | estimate  /  middle frame | REF error | error  /  the falsely still pixels (estimate under 0.5 px where the truth is above 2 px) of
REF | of the estimate | relative error |e| / max(|true|, 1 px) of the estimate (0..1).
Real: prev | REF estimate | estimate  /  real in-between (curr when there is none) | REF generated-vs-real x4 | generated-vs-real x4; without a real in-between frame the
panels show the warp residual |prev - curr warped back by the field| x 4 instead.  Each side's numbers are in the caption.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import mkpairs  # noqa: E402

PW, PH = 600, 338   # panel size: 3 x 600 = 1800 px wide
TITLE = 64


def sheet(r, setstr, label, outdir):
    p = r["pair"]
    name, scale = p["name"], float(p["scale"])
    fl = C.fields(r, setstr)
    if "main" not in fl:
        raise RuntimeError("%s: this estimator's dump holds no field (set: %s)" % (name, setstr))
    f = fl[fl["main"]]
    F, fs = f["F"], f["fs"]
    size = (PW, PH)
    os.makedirs(os.path.join(outdir, "png"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "sheets"), exist_ok=True)
    est = C.motion_panel(F, scale, size, "estimated motion (%s)" % fl["main"])
    stp = C.steps_panel(F, size, "steps of the estimate")
    png = lambda s, img: cv2.imwrite(os.path.join(outdir, "png", "%s_%s.png" % (name, s)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    stats = ""
    if p["truth"]:
        import json
        spec = json.load(open(os.path.join(p["dir"], "spec.json")))
        ft, valid = mkpairs.truth(spec, p["dir"], fs, f["grid"])
        e, ok = C.epe(F, ft, valid)
        vmax = max(scale / 8, 1.0)
        m = C.pair_metrics(r, setstr)
        fm = m["fields"][fl["main"]]
        stats = "EPE mean %.2f p95 %.2f px | blk16 %.3f (truth %.3f, block-const truth %.3f)" % (
            fm["epe_mean"], fm["epe_p95"], fm["blk16"], m["truth_blk16"], m["truthbc_blk16"])
        if "psnr" in m:
            stats += " | PSNR %.2f / edge %.2f dB" % (m["psnr"], m["psnr_edge"])
        panels = [
            [C.image_panel(C.read_rgba(p["mid"], p["W"], p["H"]), size, "true middle frame"),
             C.motion_panel(ft, scale, size, "true motion"), est],
            [stp, C.heat_panel(e, ok, vmax, size, "error 0..%g px (grey: not valid)" % vmax),
             C.steps_panel(ft, size, "steps of the truth")]]
        png("truth", C.motion_panel(ft, scale, (ft.shape[1], ft.shape[0]), "true motion", legend=False))
        png("error", C.heat_panel(e, ok, vmax, (ft.shape[1], ft.shape[0]), "error 0..%g px" % vmax))
    else:
        m = C.pair_metrics(r, setstr)
        fm = m["fields"][fl["main"]]
        stats = "blk16 %.3f" % fm["blk16"]
        prev = C.read_rgba(p["prev"], p["W"], p["H"])
        gen = r["out"]
        if p.get("mid"):
            real = C.read_rgba(p["mid"], p["W"], p["H"])
            d = np.clip(np.abs(gen.astype(np.int16) - real.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
            c3 = C.image_panel(d, size, "|generated - real| x 4")
            stats += " | PSNR %.2f / edge %.2f dB" % (m["psnr"], m["psnr_edge"])
            second = C.image_panel(real, size, "real in-between frame")
        else:
            c3 = np.full((PH, PW, 3), 30, np.uint8)
            C.put(c3, "no real in-between frame", (8, 22), 0.6)
            second = C.image_panel(C.read_rgba(p["curr"], p["W"], p["H"]), size, "curr frame")
        panels = [[C.image_panel(prev, size, "prev frame"), est, stp],
                  [second, C.image_panel(gen, size, "generated t = 0.5"), c3]]
    png("motion", C.motion_panel(F, scale, (F.shape[1], F.shape[0]), "", legend=False)[:, :])
    png("steps", C.steps_panel(F, (F.shape[1], F.shape[0]), ""))
    body = np.concatenate([np.concatenate(row, 1) for row in panels], 0)
    head = np.full((TITLE, body.shape[1], 3), 24, np.uint8)
    cv2.putText(head, "%s    [%s]" % (name, label), (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(head, stats, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    img = np.concatenate([head, body], 0)
    path = os.path.join(outdir, "sheets", name + ".png")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


def _warp_residual(prev, curr, F):
    """|prev - curr sampled at p + F(p)| x 4 (F full-res px on the flow grid, resized to the frame)"""
    h, w = prev.shape[:2]
    Fu = cv2.resize(F.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    wc = cv2.remap(np.ascontiguousarray(curr), gx + Fu[..., 0], gy + Fu[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.clip(np.abs(prev.astype(np.int16) - wc.astype(np.int16)) * 4, 0, 255).astype(np.uint8)


def _caption(m, fm, synthetic):
    if synthetic:
        s = "EPE %.2f / p95 %.2f px, false still %.1f%%, blk16 %.3f" % (fm["epe_mean"], fm["epe_p95"], 100 * fm["false_still"], fm["blk16"])
    else:
        s = "blk16 %.3f" % fm["blk16"]
    if "psnr" in m:
        s += ", PSNR %.2f / edge %.2f" % (m["psnr"], m["psnr_edge"])
    return s


def _signed_row(real, g3, g35, ref_label, label, size, tile=(320, 180)):
    """third row of a real before / after sheet: where the new estimate is worse (red) / better (blue) than the reference, |gen_new - real| - |gen_ref - real| (luma-ish: mean
    over the channels, smoothed over 8 px, fixed scale +-12 levels), and the 2x crop of the tile where it loses most (or, with no loss, where the error is largest)."""
    ea = np.abs(g3.astype(np.float32) - real.astype(np.float32)).mean(-1)
    eb = np.abs(g35.astype(np.float32) - real.astype(np.float32)).mean(-1)
    sd = cv2.blur(eb - ea, (8, 8))
    img = np.zeros(sd.shape + (3,), np.uint8)
    img[..., 0] = np.clip(sd * (255 / 12.0), 0, 255)
    img[..., 2] = np.clip(-sd * (255 / 12.0), 0, 255)
    img[..., 1] = 0
    # worst tile: the largest mean signed difference over tiles stepped by half a tile
    th, tw = tile[1], tile[0]
    H, W = sd.shape
    integ = cv2.integral(sd)
    best, pos = -1e9, (0, 0)
    for y in range(0, H - th + 1, th // 2):
        for x in range(0, W - tw + 1, tw // 2):
            v = (integ[y + th, x + tw] - integ[y, x + tw] - integ[y + th, x] + integ[y, x]) / (th * tw)
            if v > best: best, pos = v, (x, y)
    if best <= 0.3:   # no loss anywhere: the tile with the largest error of the new estimate
        ie = cv2.integral(eb)
        best, pos = -1e9, (0, 0)
        for y in range(0, H - th + 1, th // 2):
            for x in range(0, W - tw + 1, tw // 2):
                v = (ie[y + th, x + tw] - ie[y, x + tw] - ie[y + th, x] + ie[y, x]) / (th * tw)
                if v > best: best, pos = v, (x, y)
        why = "largest error"
    else:
        why = "worst loss"
    x, y = pos
    crop = lambda e: cv2.resize(np.clip(e[y:y + th, x:x + tw] * 4, 0, 255).astype(np.uint8), (size[0], size[1]), interpolation=cv2.INTER_NEAREST)
    to3 = lambda e: np.repeat(e[..., None], 3, 2) if e.ndim == 2 else e
    c1 = C.image_panel(to3(crop(ea)), size, "%s error x4, 2x crop of the %s tile at (%d, %d)" % (ref_label, why, x, y))
    c2 = C.image_panel(to3(crop(eb)), size, "%s error x4, same tile" % label)
    c0 = C.image_panel(img, size, "%s worse (red) / better (blue), +-12 levels" % label)
    return [c0, c1, c2]


def compare_sheet(rr, rn, ref_set, setstr, ref_label, label, outdir):
    """before / after sheet of one pair: rr = the reference run (ref_set), rn = the run under test (setstr)"""
    import json
    p = rn["pair"]
    name, scale = p["name"], float(p["scale"])
    fr, fn = C.fields(rr, ref_set), C.fields(rn, setstr)
    Fr, Fn, fs = fr[fr["main"]]["F"], fn[fn["main"]]["F"], fn[fn["main"]]["fs"]
    mr, mn = C.pair_metrics(rr, ref_set), C.pair_metrics(rn, setstr)
    fmr, fmn = mr["fields"][fr["main"]], mn["fields"][fn["main"]]
    size = (PW, PH)
    os.makedirs(os.path.join(outdir, "compare"), exist_ok=True)
    if p["truth"]:
        spec = json.load(open(os.path.join(p["dir"], "spec.json")))
        ft, valid = mkpairs.truth(spec, p["dir"], fs, fn[fn["main"]]["grid"])
        er, okr = C.epe(Fr, ft, valid)
        en, okn = C.epe(Fn, ft, valid)
        vmax = max(scale / 8, 1.0)
        tm = np.hypot(ft[..., 0], ft[..., 1])
        def still(F):
            m = (np.hypot(F[..., 0], F[..., 1]) < 0.5) & (tm > 2.0) & valid
            img = np.full(F.shape[:2] + (3,), 30, np.uint8)
            img[valid] = (70, 70, 80)
            img[m] = (255, 60, 60)
            return img
        rel = np.clip(en / np.maximum(tm, 1.0), 0, 1)
        panels = [
            [C.motion_panel(ft, scale, size, "true motion"), C.motion_panel(Fr, scale, size, "%s estimate" % ref_label), C.motion_panel(Fn, scale, size, "%s estimate" % label)],
            [C.image_panel(C.read_rgba(p["mid"], p["W"], p["H"]), size, "true middle frame"), C.heat_panel(er, okr, vmax, size, "%s error 0..%g px" % (ref_label, vmax)),
             C.heat_panel(en, okn, vmax, size, "%s error 0..%g px" % (label, vmax))],
            [C.image_panel(still(Fr), size, "%s falsely still (est < 0.5 px, true > 2 px) %.1f%%" % (ref_label, 100 * fmr["false_still"])),
             C.image_panel(still(Fn), size, "%s falsely still %.1f%%" % (label, 100 * fmn["false_still"])),
             C.heat_panel(rel, okn, 1.0, size, "%s relative error |e| / max(|true|, 1) 0..1" % label)]]
        sub = ["%s: %s" % (ref_label, _caption(mr, fmr, True)), "%s: %s" % (label, _caption(mn, fmn, True))]
    else:
        prev, curr = C.read_rgba(p["prev"], p["W"], p["H"]), C.read_rgba(p["curr"], p["W"], p["H"])
        if p.get("mid"):
            real = C.read_rgba(p["mid"], p["W"], p["H"])
            d = lambda g: np.clip(np.abs(g.astype(np.int16) - real.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
            second = [C.image_panel(real, size, "real in-between frame"), C.image_panel(d(rr["out"]), size, "%s generated vs real x4" % ref_label),
                      C.image_panel(d(rn["out"]), size, "%s generated vs real x4" % label)]
        else:
            second = [C.image_panel(curr, size, "curr frame (no real in-between)"),
                      C.image_panel(_warp_residual(prev, curr, Fr), size, "%s warp residual x4" % ref_label),
                      C.image_panel(_warp_residual(prev, curr, Fn), size, "%s warp residual x4" % label)]
        panels = [[C.image_panel(prev, size, "prev frame"), C.motion_panel(Fr, scale, size, "%s estimate" % ref_label), C.motion_panel(Fn, scale, size, "%s estimate" % label)], second]
        if p.get("mid"):
            panels.append(_signed_row(real, rr["out"], rn["out"], ref_label, label, size))
        sub = ["%s: %s" % (ref_label, _caption(mr, fmr, False)), "%s: %s" % (label, _caption(mn, fmn, False))]
    body = np.concatenate([np.concatenate(row, 1) for row in panels], 0)
    head = np.full((TITLE + 24, body.shape[1], 3), 24, np.uint8)
    cv2.putText(head, name, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(head, sub[0], (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(head, sub[1], (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 120), 1, cv2.LINE_AA)
    path = os.path.join(outdir, "compare", name + ".png")
    cv2.imwrite(path, cv2.cvtColor(np.concatenate([head, body], 0), cv2.COLOR_RGB2BGR))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fgtest", required=True)
    ap.add_argument("--set", default="", dest="setstr", help="fgtest --set string")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", action="append", required=True)
    ap.add_argument("--only")
    ap.add_argument("--label", help="text on the sheet (default: the --set string)")
    ap.add_argument("--ref-fgtest", help="before / after sheets: the reference build (runs the same pairs twice; writes DIR/compare/<name>.png)")
    ap.add_argument("--ref-set", default="algo=v3", help="the reference's --set string")
    ap.add_argument("--ref-label", default="V3")
    a = ap.parse_args()
    pairs = []
    for p in a.pairs:
        pairs += C.load_pairs(p, a.only)
    label = a.label or a.setstr or "defaults"
    if a.ref_fgtest:
        refs = C.run_fgtest(a.ref_fgtest, a.ref_set, pairs, os.path.join(a.out, "work_ref"))
        news = C.run_fgtest(a.fgtest, a.setstr, pairs, os.path.join(a.out, "work"))
        for i, (rr, rn) in enumerate(zip(refs, news)):
            print("%3d/%d" % (i + 1, len(pairs)), compare_sheet(rr, rn, a.ref_set, a.setstr, a.ref_label, a.label or "V3.5", a.out), flush=True)
        return
    for i, (r) in enumerate(C.run_fgtest(a.fgtest, a.setstr, pairs, os.path.join(a.out, "work"))):
        print("%3d/%d" % (i + 1, len(pairs)), sheet(r, a.setstr, label, a.out), flush=True)


if __name__ == "__main__":
    main()
