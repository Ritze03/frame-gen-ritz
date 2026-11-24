# Task 144 tables: block-constant ceiling, O6-target nets, seed noise


**Ceiling by block size, six candidates (4 taps + prev + curr)** (whole picture / edge pixels, PSNR dB)

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 30.86 / 26.23 | 28.79 / 25.23 | 40.05 / 35.26 | 32.36 / 25.93 | 31.24 / 26.75 |
| base C | 30.77 / 26.06 | 29.18 / 25.41 | 40.23 / 35.42 | 32.55 / 25.98 | 31.33 / 26.72 |
| forza-rx9070xt C | 30.71 / 25.96 | 29.26 / 25.51 | 40.40 / 35.61 | 32.59 / 26.01 | 31.33 / 26.69 |
| px8 | 30.51 / 25.88 | 29.06 / 25.33 | 40.75 / 36.02 | 32.41 / 25.88 | 31.14 / 26.59 |
| block 32 | 32.26 / 27.42 | 30.51 / 26.65 | 44.10 / 39.71 | 34.43 / 27.86 | 32.85 / 28.16 |
| block 16 | 32.80 / 27.90 | 31.10 / 27.15 | 44.44 / 40.00 | 35.02 / 28.40 | 33.41 / 28.65 |
| block 8 | 33.55 / 28.62 | 31.87 / 27.83 | 44.99 / 40.52 | 35.91 / 29.25 | 34.18 / 29.37 |
| block 4 | 34.50 / 29.65 | 32.75 / 28.70 | 45.84 / 41.38 | 37.15 / 30.47 | 35.13 / 30.37 |
| block 2 | 35.77 / 31.20 | 33.68 / 29.64 | 47.23 / 42.87 | 38.84 / 32.18 | 36.32 / 31.74 |
| whole frame (1 weight vector) | 31.30 / 26.60 | 29.42 / 25.84 | 42.84 / 38.40 | 32.84 / 26.38 | 31.79 / 27.26 |
| per-pixel O6 | 37.37 / 33.45 | 34.35 / 30.31 | 49.45 / 45.49 | 40.98 / 34.40 | 37.65 / 33.41 |

**Ceiling by block size, four taps only** (whole picture / edge pixels, PSNR dB)

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 30.86 / 26.23 | 28.79 / 25.23 | 40.05 / 35.26 | 32.36 / 25.93 | 31.24 / 26.75 |
| block 32 | 31.86 / 27.01 | 29.83 / 25.92 | 41.17 / 36.38 | 33.27 / 26.71 | 32.25 / 27.51 |
| block 16 | 32.22 / 27.32 | 30.16 / 26.18 | 41.32 / 36.51 | 33.54 / 26.96 | 32.59 / 27.80 |
| block 8 | 32.71 / 27.78 | 30.52 / 26.49 | 41.55 / 36.71 | 33.90 / 27.29 | 33.02 / 28.19 |
| block 4 | 33.31 / 28.42 | 30.89 / 26.84 | 41.82 / 36.95 | 34.35 / 27.73 | 33.52 / 28.70 |
| block 2 | 34.04 / 29.26 | 31.25 / 27.20 | 42.17 / 37.28 | 34.83 / 28.20 | 34.09 / 29.33 |
| whole frame (1 weight vector) | 31.29 / 26.59 | 29.10 / 25.47 | 40.76 / 36.01 | 32.59 / 26.14 | 31.63 / 27.07 |
| per-pixel O4 | 34.89 / 30.34 | 31.54 / 27.49 | 42.53 / 37.64 | 35.27 / 28.65 | 34.69 / 30.02 |

**Checkerboard split**: the block mix is fitted on the even pixels ((y+x) even) of each block and scored on the odd pixels only; 'plain' = fitted on all pixels, scored on the same odd pixels; rule on the same odd pixels for reference.

**Checkerboard, 6 candidates, scored on odd pixels** (whole picture / edge pixels, PSNR dB)

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule (odd pixels) | 30.86 / 26.23 | 28.79 / 25.23 | 40.05 / 35.25 | 32.36 / 25.93 | 31.24 / 26.75 |
| block 16 plain fit | 32.80 / 27.90 | 31.10 / 27.14 | 44.44 / 40.00 | 35.02 / 28.41 | 33.41 / 28.65 |
| block 16 fitted on even pixels | 32.76 / 27.84 | 31.10 / 27.14 | 44.38 / 39.94 | 34.94 / 28.32 | 33.37 / 28.60 |
| block 8 plain fit | 33.55 / 28.62 | 31.87 / 27.83 | 45.00 / 40.53 | 35.92 / 29.26 | 34.18 / 29.37 |
| block 8 fitted on even pixels | 33.41 / 28.44 | 31.84 / 27.80 | 44.80 / 40.32 | 35.72 / 29.05 | 34.06 / 29.23 |
| block 4 plain fit | 34.50 / 29.65 | 32.76 / 28.70 | 45.84 / 41.39 | 37.17 / 30.49 | 35.13 / 30.37 |
| block 4 fitted on even pixels | 33.99 / 29.01 | 32.66 / 28.61 | 45.21 / 40.71 | 36.53 / 29.83 | 34.73 / 29.89 |

**Checkerboard, 4 candidates, scored on odd pixels** (whole picture / edge pixels, PSNR dB)

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule (odd pixels) | 30.86 / 26.23 | 28.79 / 25.23 | 40.05 / 35.25 | 32.36 / 25.93 | 31.24 / 26.75 |
| block 16 plain fit | 32.22 / 27.32 | 30.16 / 26.17 | 41.34 / 36.53 | 33.54 / 26.96 | 32.59 / 27.80 |
| block 16 fitted on even pixels | 32.19 / 27.28 | 30.15 / 26.17 | 41.30 / 36.49 | 33.51 / 26.92 | 32.56 / 27.76 |
| block 8 plain fit | 32.71 / 27.78 | 30.52 / 26.49 | 41.56 / 36.72 | 33.90 / 27.29 | 33.02 / 28.19 |
| block 8 fitted on even pixels | 32.61 / 27.66 | 30.50 / 26.47 | 41.48 / 36.64 | 33.83 / 27.21 | 32.95 / 28.10 |
| block 4 plain fit | 33.31 / 28.41 | 30.88 / 26.84 | 41.83 / 36.96 | 34.36 / 27.73 | 33.52 / 28.70 |
| block 4 fitted on even pixels | 33.03 / 28.07 | 30.82 / 26.77 | 41.62 / 36.76 | 34.18 / 27.55 | 33.32 / 28.47 |

**Where the gain over the rule sits (squared-error reduction vs the rule, all 23 pairs pooled)**

| variant | total reduction (% of rule's squared error) | share at edge pixels | share elsewhere | edge pixels' own reduction (% of rule's se_e) | non-edge pixels' own reduction |
|---|---|---|---|---|---|
| 6 candidates, B=8 | 49.1 % | 66.8 % | 33.2 % | 45.2 % | 59.4 % |
| 6 candidates, B=4 | 59.1 % | 69.4 % | 30.6 % | 56.5 % | 66.0 % |
| 6 candidates, B=2 | 69.0 % | 71.9 % | 28.1 % | 68.3 % | 70.7 % |
| 4 candidates, B=8 | 33.5 % | 60.9 % | 39.1 % | 28.2 % | 47.8 % |
| 4 candidates, B=4 | 40.8 % | 64.3 % | 35.7 % | 36.2 % | 53.2 % |
| O6 per pixel | 77.1 % | 73.8 % | 26.2 % | 78.4 % | 73.8 % |

Edge pixels are 25.8 % of all pixels.

## Nets (parts 2 and 3)

**Nets against the real frame on the 23 held-out pairs** (whole picture / edge pixels, PSNR dB)

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 30.86 / 26.23 | 28.79 / 25.23 | 40.05 / 35.26 | 32.36 / 25.93 | 31.24 / 26.75 |
| px8 (task 143, random crops) | 30.51 / 25.88 | 29.06 / 25.33 | 40.75 / 36.02 | 32.41 / 25.88 | 31.14 / 26.59 |
| n3-all (task 143) | 30.46 / 25.79 | 29.08 / 25.32 | 40.71 / 35.97 | 32.43 / 25.89 | 31.11 / 26.54 |
| n5-all (task 143) | 30.45 / 25.77 | 29.02 / 25.28 | 40.80 / 36.06 | 32.43 / 25.88 | 31.09 / 26.52 |
| px8-fix | 30.69 / 26.02 | 29.13 / 25.40 | 40.64 / 35.89 | 32.52 / 25.94 | 31.28 / 26.70 |
| n3-all-fix | 30.63 / 25.87 | 29.05 / 25.33 | 40.55 / 35.78 | 32.41 / 25.87 | 31.21 / 26.58 |
| px8-o6 | 30.65 / 25.98 | 29.15 / 25.41 | 40.66 / 35.91 | 32.52 / 25.95 | 31.26 / 26.68 |
| n3-all-o6 | 30.55 / 25.79 | 29.07 / 25.35 | 40.61 / 35.85 | 32.40 / 25.85 | 31.16 / 26.54 |
| px8-s1 | 30.53 / 25.92 | 29.04 / 25.32 | 40.74 / 36.00 | 32.42 / 25.88 | 31.15 / 26.61 |
| block 8 (6 cand.) | 33.55 / 28.62 | 31.87 / 27.83 | 44.99 / 40.52 | 35.91 / 29.25 | 34.18 / 29.37 |
| per-pixel O6 | 37.37 / 33.45 | 34.35 / 30.31 | 49.45 / 45.49 | 40.98 / 34.40 | 37.65 / 33.41 |
