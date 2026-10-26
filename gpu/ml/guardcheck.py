"""guardcheck.py: Settings::learnedStaticGuard (shaders/ml_guard.comp) against ml-fg's scripts/static_guard.py (authoritative, numpy).

    fgtest --set algo=learned,learnedStaticGuard=0 --t T --dump DIR W H a.rgba c.rgba off.rgba    (the frame without the guard + its field)
    fgtest --set algo=learned,learnedStaticGuard=1 --t T W H a.rgba c.rgba on.rgba                 (the same pair with the guard)
    python3 -I gpu/ml/guardcheck.py W H a.rgba c.rgba off.rgba on.rgba DIR/pair_0000.fgmaps [--t T] [--mlfg ML-FG]

static_guard.guard(a, c, off, interp.field_at(pf, t), t) with pf = the library's midpoint field from the dump (fgtest --dump, header
gpu/fgmaps.py) is the expectation for `on`: the script prints the max abs difference (gate: <= 1 level), how many pixels differ,
how many the guard changed (library / script) and the share with wStatic > 0.5. Files are tightly packed RGBA8 (fgtest's), alpha
ignored. Exit status 1 if the gate fails. Needs numpy and OpenCV (static_guard.py's cv2.blur / erode / resize).
"""
import argparse
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'proto'))
import interp  # noqa: E402  (FGR's field_at, unchanged)


def rgba(path, w, h):
    return np.fromfile(path, np.uint8).reshape(h, w, 4)[..., :3].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('W', type=int)
    ap.add_argument('H', type=int)
    ap.add_argument('a')
    ap.add_argument('c')
    ap.add_argument('off')
    ap.add_argument('on')
    ap.add_argument('maps')
    ap.add_argument('--t', type=float, default=0.5)
    ap.add_argument('--mlfg', default=os.environ.get('MLFG_ROOT', '/home/mo/Schinken/Programming/Local/ml-fg'))
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.mlfg, 'scripts'))
    import static_guard as sg  # noqa: E402

    W, H, t = args.W, args.H, args.t
    a, c, off, on = (rgba(p, W, H) for p in (args.a, args.c, args.off, args.on))
    raw = open(args.maps, 'rb').read()
    hdr = struct.unpack('<10I4f2I', raw[:64])
    assert hdr[0] == 0x504D4746 or raw[:4] == b'FGMP', 'not an fgmaps file'
    mw, mh, fw, fh, bw, bh, fscale = hdr[2:9]
    assert (mw, mh) == (W, H), (mw, mh)
    pix = 64 + bw * bh * 8
    pf = np.frombuffer(raw, '<f4', fw * fh * 2, pix).reshape(fh, fw, 2).copy()
    g = interp.field_at(pf, t)
    exp = sg.guard(a, c, off, g, t, {'flow_scale': fscale})
    w = sg.weight(a, c, g, t, {'flow_scale': fscale})
    d = np.abs(exp.astype(int) - on.astype(int))
    ch_lib = int((on != off).any(2).sum())
    ch_ref = int((exp != off).any(2).sum())
    ok = int(d.max()) <= 1
    print(f"{'PASS' if ok else 'FAIL'} {W}x{H} t={t}: library vs static_guard.py max |d| {int(d.max())} level(s), "
          f"{int((d > 0).any(2).sum())} px differ ({int((d > 1).any(2).sum())} by > 1); changed by the guard: library "
          f"{ch_lib} px, script {ch_ref} px ({100.0 * ch_ref / (W * H):.2f} %); wStatic > 0.5 on {100.0 * float((w > 0.5).mean()):.2f} %")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
