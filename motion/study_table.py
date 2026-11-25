#!/usr/bin/env python3
"""Aggregate the .npz / .pkl of `study_match.py` into the study table (task 162).

    python study_table.py study_none.pkl [--md out.md] [--json out.json]
Groups: ground = lower half of the driving pairs (Forza, `road` set); drive-up = upper half of the driving pairs; tp = the third-person pairs; game = the other game pairs
(cs2, minecraft, battlefield, first person, drone); rest = everything but ground.  Per variant and level: patches in range, findable share (cost minimum within 1 level px of
the rounded truth) over all patches and over the moving ones (true vector at least 1 level px), and over the moving ones the share where the true vector beats zero and the
mean cost margin (cost at zero minus cost at the truth) / (mean cost over the search area)."""
import argparse
import json
import pickle
from collections import defaultdict

import numpy as np

LEVELS = (5, 4, 3, 2, 1)


def group_of(name, y):
    drive = name.startswith(('racing_forza', 'road_', 'forza_'))
    if drive:
        return 'ground' if y >= 540 else 'drive-up'
    return 'tp' if name.startswith('tp_') else 'game'


def aggregate(data):
    acc = defaultdict(lambda: np.zeros(7))   # n, find, nmov, findmov, beatmov, marginmov, (unused)
    for name, d in data.items():
        t = d['truth']
        for (pi, l, vn, fin, marg, mov, mag, ct, c0, sc) in d['rows']:
            g = group_of(name, t[pi][1])
            for gg in (g, 'rest' if g != 'ground' else None):
                if gg is None:
                    continue
                a = acc[(gg, vn, l)]
                a[0] += 1
                a[1] += fin
                if mov:
                    a[2] += 1
                    a[3] += fin
                    a[4] += marg > 0
                    a[5] += marg
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl')
    ap.add_argument('--md')
    ap.add_argument('--json')
    ap.add_argument('--groups', default='ground,rest')
    a = ap.parse_args()
    acc = aggregate(pickle.load(open(a.pkl, 'rb')))
    groups = a.groups.split(',')
    variants = sorted({k[1] for k in acc}, key=lambda s: (s.split('_')[0] != 'v3', s))
    lines = []
    for g in groups:
        lines.append(f'### {g}: findable share % (all patches / moving patches), then margin of the moving ones; n = patches / moving patches in range\n')
        lines.append('| variant | ' + ' | '.join(f'L{l}' for l in LEVELS) + ' |')
        lines.append('|---|' + '---|' * len(LEVELS))
        ns = [acc.get((g, variants[0], l), np.zeros(7)) for l in LEVELS]
        lines.append('| n | ' + ' | '.join(f'{int(x[0])} / {int(x[2])}' for x in ns) + ' |')
        for v in variants:
            cells = []
            for l in LEVELS:
                x = acc.get((g, v, l))
                if x is None or x[0] == 0:
                    cells.append('-')
                    continue
                m = f'{100 * x[3] / x[2]:.0f}' if x[2] else '-'
                mg = f'{x[5] / x[2]:+.2f}' if x[2] else '-'
                cells.append(f'{100 * x[1] / x[0]:.0f} / {m} / {mg}')
            lines.append(f'| {v} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    txt = '\n'.join(lines)
    print(txt)
    if a.md:
        open(a.md, 'w').write(txt + '\n')
    if a.json:
        json.dump({f'{k[0]}|{k[1]}|{k[2]}': list(map(float, v)) for k, v in acc.items()}, open(a.json, 'w'))


if __name__ == '__main__':
    main()
