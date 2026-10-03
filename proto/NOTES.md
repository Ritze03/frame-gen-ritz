# interp.py experiment log

Metrics as in harness.py (dup / blend / interp PSNR dB, worse% = triplets with interp < dup, harm% = pixels off from both a and c by >24).
Fast iteration used a scratchpad copy of the harness metrics over cached 5-frame windows at scale 0.5 (s1 = stride 1, s2 = stride 2).

## Finding: the harness's duplicate filter lets real duplicates through on the OBS clips
harness.py skips a triplet if any consecutive frame diff <= 0.2 mean abs levels. On the HEVC OBS clips, a
re-encoded duplicate frame differs by 0.20-0.53, so many "triplets" contain b == a (or b == c) with ~50 dB dup PSNR.
Those dominate `worse%` on obs1/obs2 stride 1 (v0: 13-18%), and they also make blend < dup on obs2.
Several worst-case PNGs are also uneven capture timing (b sits next to a or c, not halfway) during flicks.
For algorithm work I used a "clean" cache with every consecutive diff > 0.6 (no near-duplicates) in addition to
the harness's own filter. Both numbers are reported.

## Baseline v0 (official harness, scale 1.0; obs1 --every 240, obs2 --every 190, battlefield --every 17)
| clip | stride | n | dup | blend | interp | worse% | harm% |
|---|---|---|---|---|---|---|---|
| obs1 | 1 | 149 | 28.28 | 28.70 | 29.84 | 13.4 | 0.39 |
| obs2 | 1 | 155 | 28.77 | 27.48 | 28.33 | 18.1 | 0.51 |
| battlefield | 1 | 200 | 24.36 | 26.90 | 30.60 | 4.5 | 0.28 |
| obs1 | 2 | 101 | 24.59 | 26.33 | 26.71 | 21.8 | 0.56 |
| obs2 | 2 | 118 | 21.25 | 23.39 | 24.29 | 1.7 | 0.66 |
| battlefield | 2 | 197 | 21.78 | 24.08 | 26.52 | 2.5 | 0.54 |

## Experiments (clean cache, scale 0.5; interp dB / worse% / harm% per clip: obs1, obs2, battlefield)
Columns: obs1 s1 | obs2 s1 | battlefield s1 | obs1 s2 | obs2 s2 | battlefield s2 (n = 38 / 82 / 80 windows).

- v0 → 31.71/0/0.26 | 30.06/1/0.25 | 30.84/2/0.30 | 28.41/3/0.43 | 26.62/0/0.40 | 26.92/1/0.49 (mean gain over dup 4.94 dB, mean harm 0.355)
- v1 = v0 + seed candidates at 2x upscale + subpixel (0.5, 0.25) + per-pixel vector choice + mismatch→blend(a,c) fallback
  → 32.89/0/0.06 | 30.77/1/0.06 | 32.28/1/0.07 | 29.18/3/0.13 | 26.97/0/0.12 | 27.49/1/0.15 (5.78 dB, harm 0.100)
- ablations of v1 (mean gain dB / mean harm):
  - no seed candidates 5.77/0.105 (kept anyway, cheap, a bit less harm; becomes useful with the global-median candidate)
  - no subpixel 5.02/0.064; (0.5,) only 5.66/0.092; (0.5,0.25,0.125) 5.83/0.101 → keep (0.5, 0.25)
  - no per-pixel choice 5.56/0.098 → keep
  - no mismatch fallback 5.82/0.171 → keep fallback
  - zero bonus 0 → no change in numbers; kept for static-HUD safety (self-check HUD test: v0 21.9 dB, new 37.3 dB)
  - LAMBDA 0.25: 5.85/0.104, 1.0: 5.61/0.090; RADIUS 1: 5.67/0.089, 3: 5.82/0.107; PIX_WIN 3/9: 5.74/5.76;
    MIN_SIDE 96: 5.22/0.074; FLOW_SCALE 1: 5.82/0.104 (4x cost); BLOCK 16: 5.74/0.088 → all flat, kept v0 values
- TOP_RADIUS (coarsest-level search radius) 2 → 4/5/6/8: 5.90/5.95/5.96/5.97 dB, harm 0.126-0.137.
  Kept 6. The extra "harm" is sharp, correctly tracked fast pans that land a few px off the real b
  (looked at the PNGs: with radius 2 the same frames were a full-frame ghost, which harm% cannot see,
  because a 50/50 blend of a and c can never be "far from both").
- fallback target in untrusted pixels (TOP_RADIUS 2): blend(a,c) 5.78 dB / 0.100 harm, frame a 5.14 / 0.100, warped a 0.28 harm on battlefield (bad),
  "whichever warped side is nearer the blend" 5.45 / 0.152.
  Chose **frame a** despite ~0.5 dB PSNR: blend makes double images of the weapon/arm during viewmodel animations,
  frame a holds them still for one frame (user's rule: holding still is invisible, a smear is not). Pure fallback pixels can never be harm.
- mismatch box 3 → 7: +0.12 dB at same harm → kept 7. TAA-style clamp distance instead of |wa-wc|: same PSNR/harm frontier → kept simple |wa-wc|.
- trust ramp (8,32)/(12,40)/(16,48): 5.06/5.39/5.57 dB, harm 0.110/0.134/0.148 → kept (12,40).
- global-median vector as an extra seed candidate per level: fixes a stairs triplet where a band of blocks locked onto the wrong
  stair period (25.3 → 27.6 dB on that frame), overall +0.02 dB. Bias toward it (1.0/2.0) costs PSNR → no bias.
- global fallback (untrusted share > X → present a): 1.0 (off) 5.41/0.137, 0.2 5.39/0.134, 0.15 5.51/0.134, 0.1 5.44/0.126.
  Kept 0.15: the frames it catches are fast flicks where the output is a torn patchwork of warped and held regions.
- final (= interp.py) → 32.46/0/0.08 | 30.69/1/0.08 | 32.03/1/0.07 | 28.94/3/0.22 | 26.62/0/0.16 | 27.20/1/0.19 (5.51 dB, harm 0.134)

## Final vs v0 at scale 1.0
Official harness (same sampling as the baseline; includes the near-duplicate triplets, see top):
| clip | stride | dup | blend | v0 interp | final interp | v0 worse% | final worse% | v0 harm% | final harm% | ms v0 → final (CPU, loaded box) |
|---|---|---|---|---|---|---|---|---|---|---|
| obs1 | 1 | 28.28 | 28.70 | 29.84 | 30.04 | 13.4 | 12.1 | 0.39 | 0.21 | 876 → 992 |
| obs2 | 1 | 28.77 | 27.48 | 28.33 | 28.31 | 18.1 | 18.1 | 0.51 | 0.36 | 853 → 1054 |
| battlefield | 1 | 24.36 | 26.90 | 30.60 | 31.33 | 4.5 | 4.5 | 0.28 | 0.08 | 799 → 1291 |
| obs1 | 2 | 24.59 | 26.33 | 26.71 | 26.68 | 21.8 | 16.8 | 0.56 | 0.31 | 836 → 1053 |
| obs2 | 2 | 21.25 | 23.39 | 24.29 | 23.98 | 1.7 | 1.7 | 0.66 | 0.36 | 920 → 1089 |
| battlefield | 2 | 21.78 | 24.08 | 26.52 | 26.80 | 2.5 | 3.6 | 0.54 | 0.20 | 782 → 1282 |

Clean full-res windows (no near-duplicates; 38/40/40 windows), interp dB / worse% / harm%:
| clip | stride | dup | blend | v0 | final |
|---|---|---|---|---|---|
| obs1 | 1 | 25.79 | 28.55 | 31.31 / 0 / 0.29 | 31.47 / 0 / 0.12 |
| obs2 | 1 | 23.96 | 26.12 | 29.52 / 0 / 0.30 | 29.61 / 0 / 0.12 |
| battlefield | 1 | 23.79 | 26.29 | 29.79 / 2.5 / 0.36 | 30.96 / 0 / 0.10 |
| obs1 | 2 | 24.11 | 26.22 | 27.91 / 2.6 / 0.49 | 28.23 / 2.6 / 0.26 |
| obs2 | 2 | 21.91 | 23.83 | 25.46 / 0 / 0.50 | 25.29 / 0 / 0.20 |
| battlefield | 2 | 21.27 | 23.61 | 26.16 / 0 / 0.54 | 26.21 / 0 / 0.22 |

Full-res ablations of final (mean gain over dup / mean harm; final = 5.16 / 0.169, v0 = 4.89 / 0.411):
fallback to blend(a,c) instead of a 5.51 / 0.168 (the PSNR price of "hold instead of ghost" is ~0.35 dB),
GLOBAL_FRAC off 5.22 / 0.179, MISMATCH_BOX 13 5.22 / 0.178, TRUST (16,48) 5.29 / 0.188, SUBPIX +0.125 5.20 / 0.174.

## Remaining known failure classes
- Repetitive structure + large motion (battlefield stairs at stride 2): blocks lock onto the wrong period; both warps agree, so the
  mismatch test cannot see it. Global-median seed fixes the stride-1 case, not the stride-2 one.
- Very fast flicks: below GLOBAL_FRAC the output is a patchwork of warped and held regions with visible seams.
- Changing HUD text (timers, ammo) shows the old value (fallback to a): harmless.
