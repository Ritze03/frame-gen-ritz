#!/usr/bin/env python3
"""Python side of learned.mjs (the learned motion estimator vs the ml-fg test vectors). Reads $FG_MLFG_EXPORT (default
~/Schinken/Programming/Local/ml-fg/export, read only), writes under $FG_WORK/learned.

    python3 -I learned_ref.py prep           every <net>/testvec/<pair> (net: each webgl/nets/<net>.js): a.rgba, c.rgba (the PNGs decoded by cv2 to 8-bit BGR, as
                                             RGBA, row 0 = top), b.bgr (the real middle frame from $FG_FOOTAGE, default <repo>/footage,
                                             when the pair comes from one clip and the footage reproduces a.png) and manifest.json
    python3 -I learned_ref.py synth          end to end through proto/interp.py's synth (export/SPEC.md section 8 step 6): for every
                                             pair x precision with the engine's dumps (<prec>/pf.f32 + est.json, out_t<T>.rgba), the PSNR
                                             of synth(engine pf) vs synth(expected pf) at t = 0.5 and 0.25, both vs the real middle frame
                                             (t = 0.5), and the engine's own output (V1 synth on its learned field) vs synth(expected pf).
                                             -> synth.json. With the zero-motion test (<prec>/zt/: the engine's
                                             final field, est.json, out_t<T>.rgba): the same against synth(pf_zt, sad_zt) -> key .../zt
    python3 -I learned_ref.py sguard         the static guard (learned.mjs dumps under sguard/<case>/: a, c, the unguarded output,
                                             the field at t, the guarded output) against ml-fg's scripts/static_guard.py on the same
                                             inputs ($FG_MLFG_SCRIPTS, default <export>/../scripts) -> sguard.json
"""
import json
import os
import re
import sys

import cv2
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', '..', 'proto'))
EXPORT = os.environ.get('FG_MLFG_EXPORT', os.path.expanduser('~/Schinken/Programming/Local/ml-fg/export'))
WORK = os.environ.get('FG_WORK', os.path.join(here, '..', '..', '.local-work', 'webgl'))
OUT = os.path.join(WORK, 'learned')
FOOTAGE = os.environ.get('FG_FOOTAGE', os.path.join(here, '..', '..', 'footage'))
TS = (0.5, 0.25)
SCRIPTS = os.environ.get('FG_MLFG_SCRIPTS', os.path.join(os.path.dirname(os.path.abspath(EXPORT)), 'scripts'))   # static_guard.py


def nets():
    """The nets the engine can load (webgl/nets/<net>.js), in name order."""
    d = os.path.join(here, '..', 'nets')
    return sorted(f[:-3] for f in os.listdir(d) if f.endswith('.js'))


EXPORT_DIR = {'c2smoke': 'c2smoke-colour', 'c2feat': 'c2smoke-features'}   # nets/<net>.js whose export directory has another name (make_nets.py NAME=DIR)


def pairs():
    """Every (net, pair, dir) of the loadable nets: the pairs are the export's testvec/<pair>/ directories with a vec.json."""
    for net in nets():
        root = os.path.join(EXPORT, EXPORT_DIR.get(net, net), 'testvec')
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if os.path.isfile(os.path.join(d, 'vec.json')):
                yield net, name, d


def psnr(x, y):
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    return None if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))   # None (JSON null): identical


def middle(v, a_bgr):
    """The real frame halfway between a and c from FGR's footage (read the way the vectors were made: cv2 seek by frame index, then
    INTER_AREA to the pair's size), when a and c come from one clip and the footage reproduces a.png exactly; else None."""
    m = re.match(r'(\S+) frame (\d+) -> (\S+) frame (\d+)$', v.get('source', ''))
    if not m or m.group(1) != m.group(3):
        return None, 'none (a and c from different clips)'
    clip = os.path.join(FOOTAGE, m.group(1))
    if not os.path.exists(clip):
        return None, 'none (no ' + clip + ')'
    ia, ic = int(m.group(2)), int(m.group(4))
    cap = cv2.VideoCapture(clip)
    W, H = v['size']
    got = {}
    for i in (ia, (ia + ic) // 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            return None, f'none (cannot read frame {i})'
        got[i] = f if (f.shape[1], f.shape[0]) == (W, H) else cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
    if not np.array_equal(got[ia], a_bgr):
        return None, 'none (footage frame %d differs from a.png)' % ia
    return got[(ia + ic) // 2], 'frame %d of %s' % ((ia + ic) // 2, m.group(1))


def prep():
    man = []
    for net, name, d in pairs():
        with open(os.path.join(d, 'vec.json')) as f:
            v = json.load(f)
        o = os.path.join(OUT, net, name)
        os.makedirs(o, exist_ok=True)
        imgs = {}
        for k in ('a', 'c'):
            img = imgs[k] = cv2.imread(os.path.join(d, v[k]), cv2.IMREAD_COLOR)
            assert img is not None and img.dtype == np.uint8, (d, k)
            rgba = np.dstack([img[..., 2], img[..., 1], img[..., 0], np.full(img.shape[:2], 255, np.uint8)])
            rgba.tofile(os.path.join(o, k + '.rgba'))
        H, W = img.shape[:2]
        assert [W, H] == v['size'], (d, v['size'])
        mid, mid_why = os.path.join(o, 'b.bgr'), os.path.join(o, 'b.txt')
        if not os.path.exists(mid_why):
            b, why = middle(v, imgs['a'])
            if b is not None:
                b.tofile(mid)
            with open(mid_why, 'w') as f:
                f.write(why)
        with open(mid_why) as f:
            why = f.read()
        man.append(dict(net=net, pair=name, dir=d, W=W, H=H, flow=v['flow_size'], padded=v['padded_flow_size'], sad=v['sad'], cut=v['cut'], colour=int(v.get('colour', 0)), features=int(v.get('features', 0)),
                        sad_zt=v.get('sad_zt'), cut_zt=v.get('cut_zt'), zeroed_zt=v.get('zeroed_zt'),
                        middle=os.path.exists(mid), middleWhy=why,
                        files={k: dict(file=os.path.join(d, x['file']), shape=x['shape']) for k, x in v['files'].items()}))
        print(f'prep: {net} {name} {W}x{H}, middle frame: {why}')
    with open(os.path.join(OUT, 'manifest.json'), 'w') as f:
        json.dump(dict(export=EXPORT, pairs=man), f, indent=1)


def synth():
    import interp
    res = {}
    with open(os.path.join(OUT, 'manifest.json')) as f:
        man = json.load(f)['pairs']
    db = lambda x: 'identical' if x is None else f'{x:.1f} dB'
    for p in man:
        base = os.path.join(OUT, p['net'], p['pair'])
        W, H = p['W'], p['H']
        fw, fh = p['flow']
        rd = lambda k: np.fromfile(os.path.join(base, k + '.rgba'), np.uint8).reshape(H, W, 4)[..., 2::-1].copy()   # -> BGR
        a, c = rd('a'), rd('c')
        b = np.fromfile(os.path.join(base, 'b.bgr'), np.uint8).reshape(H, W, 3) if p['middle'] else None
        pf_x = np.fromfile(p['files']['pf']['file'], '<f4').reshape(fh, fw, 2)
        for prec in ('fp32', 'fp16', 'hdr'):   # (hdr: fp16 with RGBA16F input, learned.mjs)
            o = os.path.join(base, prec)
            if not os.path.exists(os.path.join(o, 'est.json')):
                continue
            with open(os.path.join(o, 'est.json')) as f:
                est = json.load(f)
            pf_e = np.fromfile(os.path.join(o, 'pf.f32'), '<f4').reshape(fh, fw, 2)
            r = {}
            for t in TS:
                s_e = interp.synth(a, c, {'sad': est['sad'], 'pf': pf_e}, t)
                s_x = interp.synth(a, c, {'sad': p['sad'], 'pf': pf_x}, t)
                row = dict(synthDb=psnr(s_e, s_x))
                if b is not None and t == 0.5:
                    row.update(midDb=psnr(s_e, b), midDbExp=psnr(s_x, b))
                    row['midDelta'] = row['midDb'] - row['midDbExp']
                of = os.path.join(o, f'out_t{t}.rgba')
                if os.path.exists(of):
                    out = np.fromfile(of, np.uint8).reshape(H, W, 4)[..., 2::-1]
                    d = np.abs(out.astype(np.int16) - s_x)
                    row.update(engineMae=float(d.mean()), engineMax=int(d.max()), engineDb=psnr(out, s_x))
                r[str(t)] = row
            res[f"{p['net']}/{p['pair']}/{prec}"] = r
            print(f"synth: {p['net']} {p['pair']} {prec}: " + ', '.join(
                f"t={t} synth {db(v['synthDb'])}" +
                (f" (vs the real middle frame {v['midDb']:.3f} dB, expected pf {v['midDbExp']:.3f} dB)" if 'midDb' in v else '') +
                (f", engine out MAE {v['engineMae']:.3f} max {v['engineMax']}" if 'engineMae' in v else '') for t, v in r.items()))
            oz = os.path.join(o, 'zt')   # the zero-motion test: the engine's final field vs the expected pf_zt (with sad_zt)
            if os.path.exists(os.path.join(oz, 'est.json')) and 'pf_zt' in p['files']:
                with open(os.path.join(oz, 'est.json')) as f:
                    estz = json.load(f)
                pz_e = np.fromfile(os.path.join(oz, 'pf.f32'), '<f4').reshape(fh, fw, 2)
                pz_x = np.fromfile(p['files']['pf_zt']['file'], '<f4').reshape(fh, fw, 2)
                rz = {}
                for t in TS:
                    s_e = interp.synth(a, c, {'sad': estz['sad'], 'pf': pz_e}, t)
                    s_x = interp.synth(a, c, {'sad': p['sad_zt'], 'pf': pz_x}, t)
                    row = dict(synthDb=psnr(s_e, s_x))
                    if b is not None and t == 0.5:
                        row.update(midDb=psnr(s_e, b), midDbExp=psnr(s_x, b))
                        row['midDelta'] = row['midDb'] - row['midDbExp']
                    of = os.path.join(oz, f'out_t{t}.rgba')
                    if os.path.exists(of):
                        out = np.fromfile(of, np.uint8).reshape(H, W, 4)[..., 2::-1]
                        d = np.abs(out.astype(np.int16) - s_x)
                        row.update(engineMae=float(d.mean()), engineMax=int(d.max()), engineDb=psnr(out, s_x))
                    rz[str(t)] = row
                res[f"{p['net']}/{p['pair']}/{prec}/zt"] = rz
                print(f"synth: {p['net']} {p['pair']} {prec} zero test: " + ', '.join(
                    f"t={t} synth {db(v['synthDb'])}" +
                    (f" (vs the real middle frame {v['midDb']:.3f} dB, expected pf_zt {v['midDbExp']:.3f} dB)" if 'midDb' in v else '') +
                    (f", engine out MAE {v['engineMae']:.3f} max {v['engineMax']}" if 'engineMae' in v else '') for t, v in rz.items()))
    with open(os.path.join(OUT, 'synth.json'), 'w') as f:
        json.dump(res, f, indent=1)


def sguard():
    """learned.mjs's static-guard cases (sguard/<case>/meta.json + raw dumps) against static_guard.py: guard() for RGBA8 output; for
    RGBA16F output (no 8-bit rounding) its weight() and the same mix, compared in levels (value x 255)."""
    sys.path.insert(0, SCRIPTS)
    import static_guard as sg
    root, res = os.path.join(OUT, 'sguard'), {}
    for case in sorted(os.listdir(root)):
        d = os.path.join(root, case)
        with open(os.path.join(d, 'meta.json')) as f:
            m = json.load(f)
        W, H, fw, fh, t = m['W'], m['H'], m['fw'], m['fh'], m['t']
        rgb = lambda k: np.fromfile(os.path.join(d, k + '.rgba'), np.uint8).reshape(H, W, 4)[..., :3].copy()   # (the guard is per channel)
        a, c = rgb('a'), rgb('c')
        field = np.fromfile(os.path.join(d, 'field.f32'), '<f4').reshape(fh, fw, 2)
        params = {'flow_scale': m['flowScale']}
        w = sg.weight(a, c, field, t, params)
        eps = 0.25 if m["f16"] else 0.5
        if m['f16']:
            fl = lambda k: np.fromfile(os.path.join(d, k + '.f32'), '<f4').reshape(H, W, 4)[..., :3] * np.float32(255)
            out0, outg = fl('out0'), fl('outg')
            st = w > 0.001
            exp = out0.copy()
            still = (1 - t) * a.astype(np.float32) + t * c.astype(np.float32)
            exp[st] = out0[st] * (1 - w[st][:, None]) + still[st] * w[st][:, None]
        else:
            out0, outg = rgb('out0'), rgb('outg')
            exp = sg.guard(a, c, out0, field, t, params)
        if m.get('fallback'):   # a cut / untrusted frame: the guard must leave it untouched (static_guard.wrap)
            exp = out0
        dd = np.abs(outg.astype(np.float64) - exp.astype(np.float64))
        changed = lambda x: int((np.abs(x.astype(np.float64) - out0.astype(np.float64)).max(2) > eps).sum())
        r = dict(maxDiff=float(dd.max()), meanDiff=float(dd.mean()), nDiff=int((dd > eps).any(2).sum()), changed=changed(outg),
                 changedExp=changed(exp), wStatic50=float((w > 0.5).mean()), px=W * H)
        res[case] = r
        print(f"sguard: {case} ({W}x{H}, t={t}, {'RGBA16F' if m['f16'] else 'RGBA8'}): guarded output vs static_guard.py max |d| {r['maxDiff']:.3f} "
              f"levels (mean {r['meanDiff']:.2e}, {r['nDiff']} px differ); the guard changed {r['changed']} px (static_guard.py: "
              f"{r['changedExp']}), wStatic > 0.5 on {100 * r['wStatic50']:.1f}%")
    with open(os.path.join(OUT, 'sguard.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    {'prep': prep, 'synth': synth, 'sguard': sguard}[sys.argv[1]]()
