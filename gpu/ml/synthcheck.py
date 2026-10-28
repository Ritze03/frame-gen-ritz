"""synthcheck.py: the end-to-end check of MotionVersion::Learned (ml-fg export/SPEC.md section 8, step 6) on gpu/mltest's dumps.

    MLTEST_DUMP=DIR gpu/build/mltest
    python3 -I gpu/ml/synthcheck.py DIR [--export ML-FG/export] [--footage FGR/footage]

For every dumped pf (`<run>_<pair>_<kind>.f32` + `.json`: kind `ref` = fed the reference's luma, `fgr` = the library's own 14-bit
luma, `refh` / `fgrh` the same with learnedHalf) it runs FGR's synthesis (proto/interp.py synth, t = 0.5) with that pf and sad and
with the test vector's expected pf.f32 and sad, and prints the PSNR between the two frames (SPEC.md: >= 80 dB for an exact float32
port, >= 45 dB with shortcuts; the gate here is 45) and, where the pair's source has a real middle frame (frames 2 apart in one clip),
both frames' PSNR against it (within 0.05 dB). The zero-motion test's kinds (Settings::learnedZeroTest, mltest zeroChecks: `stagezt`
= the stage alone on pf.f32, `refzt`, `fgrzt`, `refhzt`, `fgrhzt`) are compared with the synthesis of the expected pf_zt.f32 and
sad_zt instead, and their cut with cut_zt (SPEC.md section 8 step 7c: >= 70 dB for an exact float32 port, the gate of `stagezt` /
`refzt`; 45 dB for the others). Exit status 1 if a criterion fails. Needs numpy and OpenCV (the PNGs, the footage).
"""
import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'proto'))
import interp  # noqa: E402  (FGR's synthesis, unchanged)


EXPORT_DIR = {}


def psnr(x, y):
    d = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    return float('inf') if d == 0 else 10 * np.log10(255.0 ** 2 / d)


def middle(footage, src, size):
    """The real frame between a and c if the source is 'clip frame i -> clip frame i + 2', scaled like make_testvec (INTER_AREA);
    a and c are re-read the same way so the caller can check the decode matches the PNGs."""
    m = re.match(r'(\S+) frame (\d+) -> (\S+) frame (\d+)', src)
    if not m or m.group(1) != m.group(3) or int(m.group(4)) - int(m.group(2)) != 2:
        return None
    cap = cv2.VideoCapture(os.path.join(footage, m.group(1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(m.group(2)))
    fr = [cap.read()[1] for _ in range(3)]
    if any(f is None for f in fr):
        return None
    if (fr[0].shape[1], fr[0].shape[0]) != size:
        fr = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in fr]
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump')
    ap.add_argument('--export', default=os.environ.get('MLFG_EXPORT', '/home/mo/Schinken/Programming/Local/ml-fg/export'))
    ap.add_argument('--footage', default=os.path.join(HERE, '..', '..', 'footage'))
    args = ap.parse_args()
    fails = 0
    for path in sorted(glob.glob(os.path.join(args.dump, '*_*_*.f32'))):
        run, pair, kind = os.path.basename(path)[:-4].split('_')
        d = os.path.join(args.export, EXPORT_DIR.get(run, run), 'testvec', pair)
        with open(os.path.join(d, 'vec.json')) as f:
            v = json.load(f)
        with open(path[:-4] + '.json') as f:
            mine = json.load(f)
        a, c = cv2.imread(os.path.join(d, 'a.png')), cv2.imread(os.path.join(d, 'c.png'))
        fh, fw = v['flow_size'][1], v['flow_size'][0]
        zt = kind.endswith('zt')   # the zero test's kinds: against pf_zt / sad_zt / cut_zt
        exp = np.fromfile(os.path.join(d, 'pf_zt.f32' if zt else 'pf.f32'), '<f4').reshape(fh, fw, 2)
        sad, cut = (v['sad_zt'], v['cut_zt']) if zt else (v['sad'], v['cut'])
        got = np.fromfile(path, '<f4').reshape(fh, fw, 2)
        s_exp = interp.synth(a, c, {'sad': sad, 'pf': exp}, 0.5)
        s_got = interp.synth(a, c, {'sad': mine['sad'], 'pf': got}, 0.5)
        db = psnr(s_got, s_exp)
        need = 70.0 if kind in ('stagezt', 'refzt') else 45.0
        ok = db >= need and mine['cut'] == cut
        line = (f'{run} {pair:7} {kind:7}: synth vs synth(expected {"pf_zt" if zt else "pf"}) {db:6.2f} dB (>= {need:.0f}), '
                f'cut {mine["cut"]} vs {cut}')
        fr = middle(args.footage, v['source'], (a.shape[1], a.shape[0]))
        if fr is not None and np.array_equal(fr[0], a) and np.array_equal(fr[2], c):
            pe, pg = psnr(s_exp, fr[1]), psnr(s_got, fr[1])
            ok = ok and abs(pg - pe) <= 0.05
            line += f', vs the real middle frame {pg:.3f} dB (expected pf {pe:.3f}, diff {pg - pe:+.3f})'
        elif fr is not None:
            line += ', (middle frame: the decode differs from the PNGs, not compared)'
        print(('PASS ' if ok else 'FAIL ') + line)
        fails += not ok
    print(f'{fails} failed')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
