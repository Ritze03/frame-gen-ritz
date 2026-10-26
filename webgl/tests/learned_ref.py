#!/usr/bin/env python3
"""Python side of learned.mjs (the learned motion estimator vs the ml-fg test vectors). Reads $FG_MLFG_EXPORT (default
~/Schinken/Programming/Local/ml-fg/export, read only), writes under $FG_WORK/learned.

    python3 -I learned_ref.py prep           every <net>/testvec/<pair>: a.rgba, c.rgba (the PNGs decoded by cv2 to 8-bit BGR, as
                                             RGBA, row 0 = top) and manifest.json (sizes, expected sad / cut, the vector files)
    python3 -I learned_ref.py synth          end to end through proto/interp.py's synth (export/SPEC.md section 8 step 6): for every
                                             pair with the engine's dumps (pf.f32 + est.json, out_t<T>.rgba), the PSNR of synth(engine pf)
                                             vs synth(expected pf) at t = 0.5 and 0.25, and the engine's own output (V1 synth on its
                                             learned field) vs synth(expected pf). -> synth.json
"""
import json
import os
import sys

import cv2
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', '..', 'proto'))
EXPORT = os.environ.get('FG_MLFG_EXPORT', os.path.expanduser('~/Schinken/Programming/Local/ml-fg/export'))
WORK = os.environ.get('FG_WORK', os.path.join(here, '..', '..', '.local-work', 'webgl'))
OUT = os.path.join(WORK, 'learned')
TS = (0.5, 0.25)


def pairs():
    for net in ('e2v3', 'e1v3'):
        root = os.path.join(EXPORT, net, 'testvec')
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if os.path.isfile(os.path.join(d, 'vec.json')):
                yield net, name, d


def psnr(x, y):
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    return None if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))   # None (JSON null): identical


def prep():
    man = []
    for net, name, d in pairs():
        with open(os.path.join(d, 'vec.json')) as f:
            v = json.load(f)
        o = os.path.join(OUT, net, name)
        os.makedirs(o, exist_ok=True)
        for k in ('a', 'c'):
            img = cv2.imread(os.path.join(d, v[k]), cv2.IMREAD_COLOR)
            assert img is not None and img.dtype == np.uint8, (d, k)
            rgba = np.dstack([img[..., 2], img[..., 1], img[..., 0], np.full(img.shape[:2], 255, np.uint8)])
            rgba.tofile(os.path.join(o, k + '.rgba'))
        H, W = img.shape[:2]
        assert [W, H] == v['size'], (d, v['size'])
        man.append(dict(net=net, pair=name, dir=d, W=W, H=H, flow=v['flow_size'], padded=v['padded_flow_size'], sad=v['sad'], cut=v['cut'],
                        files={k: dict(file=os.path.join(d, x['file']), shape=x['shape']) for k, x in v['files'].items()}))
        print(f'prep: {net} {name} {W}x{H}')
    with open(os.path.join(OUT, 'manifest.json'), 'w') as f:
        json.dump(dict(export=EXPORT, pairs=man), f, indent=1)


def synth():
    import interp
    res = {}
    with open(os.path.join(OUT, 'manifest.json')) as f:
        man = json.load(f)['pairs']
    for p in man:
        o = os.path.join(OUT, p['net'], p['pair'])
        if not os.path.exists(os.path.join(o, 'est.json')):
            continue
        with open(os.path.join(o, 'est.json')) as f:
            est = json.load(f)
        W, H = p['W'], p['H']
        fw, fh = p['flow']
        rd = lambda k: np.fromfile(os.path.join(o, k + '.rgba'), np.uint8).reshape(H, W, 4)[..., 2::-1].copy()   # -> BGR
        a, c = rd('a'), rd('c')
        pf_e = np.fromfile(os.path.join(o, 'pf.f32'), '<f4').reshape(fh, fw, 2)
        pf_x = np.fromfile(p['files']['pf']['file'], '<f4').reshape(fh, fw, 2)
        r = {}
        for t in TS:
            s_e = interp.synth(a, c, {'sad': est['sad'], 'pf': pf_e}, t)
            s_x = interp.synth(a, c, {'sad': p['sad'], 'pf': pf_x}, t)
            row = dict(synthDb=psnr(s_e, s_x))
            of = os.path.join(o, f'out_t{t}.rgba')
            if os.path.exists(of):
                out = np.fromfile(of, np.uint8).reshape(H, W, 4)[..., 2::-1]
                d = np.abs(out.astype(np.int16) - s_x)
                row.update(engineMae=float(d.mean()), engineMax=int(d.max()), engineDb=psnr(out, s_x))
            r[str(t)] = row
        res[p['net'] + '/' + p['pair']] = r
        db = lambda x: 'identical' if x is None else f'{x:.1f} dB'
        print(f"synth: {p['net']} {p['pair']}: " + ', '.join(f"t={t} synth {db(v['synthDb'])}" + (f", engine out MAE {v['engineMae']:.3f} max {v['engineMax']}" if 'engineMae' in v else '') for t, v in r.items()))
    with open(os.path.join(OUT, 'synth.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    {'prep': prep, 'synth': synth}[sys.argv[1]]()
