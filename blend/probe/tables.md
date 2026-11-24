
**PSNR dB, whole picture**

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 30.86 | 28.79 | 40.05 | 32.36 | 31.24 |
| baseC | 30.77 | 29.18 | 40.23 | 32.55 | 31.33 |
| forzaC | 30.71 | 29.26 | 40.40 | 32.59 | 31.33 |
| px8 | 30.51 | 29.06 | 40.75 | 32.41 | 31.14 |
| px32 | 30.52 | 29.05 | 40.69 | 32.43 | 31.15 |
| n3-all | 30.46 | 29.08 | 40.71 | 32.43 | 31.11 |
| n5-all | 30.45 | 29.02 | 40.80 | 32.43 | 31.09 |
| n3-frames | 30.48 | 29.06 | 40.67 | 32.38 | 31.11 |
| n3-taps | 30.43 | 29.08 | 40.77 | 32.36 | 31.09 |
| O4 | 34.89 | 31.54 | 42.53 | 35.27 | 34.69 |
| O6 | 37.37 | 34.35 | 49.45 | 40.98 | 37.65 |

**PSNR dB, edge pixels only (90th-percentile Sobel mask, 5x5 dilated)**

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 26.23 | 25.23 | 35.26 | 25.93 | 26.75 |
| baseC | 26.06 | 25.41 | 35.42 | 25.98 | 26.72 |
| forzaC | 25.96 | 25.51 | 35.61 | 26.01 | 26.69 |
| px8 | 25.88 | 25.33 | 36.02 | 25.88 | 26.59 |
| px32 | 25.87 | 25.32 | 35.94 | 25.89 | 26.58 |
| n3-all | 25.79 | 25.32 | 35.97 | 25.89 | 26.54 |
| n5-all | 25.77 | 25.28 | 36.06 | 25.88 | 26.52 |
| n3-frames | 25.82 | 25.34 | 35.92 | 25.86 | 26.56 |
| n3-taps | 25.76 | 25.34 | 36.04 | 25.83 | 26.52 |
| O4 | 30.34 | 27.49 | 37.64 | 28.65 | 30.02 |
| O6 | 33.45 | 30.31 | 45.49 | 34.40 | 33.41 |

**share of pixels with an error above 30 levels in any channel, whole picture (%)**

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 1.20 | 3.26 | 0.16 | 0.82 | 1.33 |
| baseC | 1.10 | 2.81 | 0.16 | 0.78 | 1.19 |
| forzaC | 1.14 | 2.77 | 0.15 | 0.78 | 1.21 |
| px8 | 1.13 | 2.92 | 0.15 | 0.79 | 1.23 |
| px32 | 1.13 | 2.91 | 0.16 | 0.78 | 1.22 |
| n3-all | 1.13 | 2.88 | 0.16 | 0.78 | 1.22 |
| n5-all | 1.14 | 2.89 | 0.16 | 0.80 | 1.23 |
| n3-frames | 1.14 | 2.93 | 0.15 | 0.79 | 1.23 |
| n3-taps | 1.14 | 2.89 | 0.15 | 0.79 | 1.23 |
| O4 | 0.42 | 1.48 | 0.10 | 0.39 | 0.55 |
| O6 | 0.24 | 0.75 | 0.02 | 0.10 | 0.27 |

**share of edge pixels with an error above 30 levels in any channel (%)**

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 3.75 | 7.77 | 0.51 | 3.85 | 3.94 |
| baseC | 3.65 | 7.23 | 0.48 | 3.73 | 3.76 |
| forzaC | 3.79 | 7.13 | 0.47 | 3.72 | 3.80 |
| px8 | 3.74 | 7.40 | 0.47 | 3.73 | 3.84 |
| px32 | 3.73 | 7.40 | 0.48 | 3.72 | 3.83 |
| n3-all | 3.77 | 7.39 | 0.49 | 3.74 | 3.85 |
| n5-all | 3.78 | 7.44 | 0.49 | 3.78 | 3.87 |
| n3-frames | 3.76 | 7.41 | 0.48 | 3.74 | 3.85 |
| n3-taps | 3.81 | 7.37 | 0.48 | 3.77 | 3.87 |
| O4 | 1.38 | 4.24 | 0.30 | 1.88 | 1.80 |
| O6 | 0.72 | 2.17 | 0.04 | 0.50 | 0.86 |

**gain over px8 in dB, whole**

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| n3-all | -0.05 | +0.01 | -0.04 | +0.02 | -0.03 |
| n5-all | -0.06 | -0.04 | +0.04 | +0.02 | -0.05 |
| n3-frames | -0.04 | -0.01 | -0.08 | -0.02 | -0.03 |
| n3-taps | -0.08 | +0.01 | +0.02 | -0.04 | -0.05 |

**gain over px8 in dB, edge**

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| n3-all | -0.09 | -0.01 | -0.05 | +0.01 | -0.06 |
| n5-all | -0.11 | -0.05 | +0.04 | -0.00 | -0.07 |
| n3-frames | -0.06 | +0.01 | -0.10 | -0.02 | -0.04 |
| n3-taps | -0.12 | +0.01 | +0.02 | -0.05 | -0.07 |

**gain over px32 in dB, whole**

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| n3-all | -0.06 | +0.02 | +0.02 | +0.00 | -0.03 |
| n5-all | -0.07 | -0.03 | +0.11 | -0.00 | -0.05 |
| n3-frames | -0.05 | +0.01 | -0.02 | -0.05 | -0.03 |
| n3-taps | -0.09 | +0.02 | +0.08 | -0.07 | -0.05 |

**gain over px32 in dB, edge**

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| n3-all | -0.08 | +0.00 | +0.02 | -0.00 | -0.05 |
| n5-all | -0.09 | -0.04 | +0.12 | -0.01 | -0.06 |
| n3-frames | -0.04 | +0.02 | -0.02 | -0.03 | -0.03 |
| n3-taps | -0.11 | +0.02 | +0.09 | -0.06 | -0.06 |

**share of the px8 -> O6 gap (dB) each net closes, whole (%)**

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| px32 | 0.1 | -0.2 | -0.7 | 0.3 | 0.1 |
| n3-all | -0.8 | 0.2 | -0.5 | 0.3 | -0.4 |
| n5-all | -0.9 | -0.8 | 0.5 | 0.2 | -0.7 |
| n3-frames | -0.6 | -0.1 | -1.0 | -0.3 | -0.4 |
| n3-taps | -1.2 | 0.2 | 0.2 | -0.5 | -0.8 |

**share of the px8 -> O6 gap (dB) each net closes, edge (%)**

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| px32 | -0.2 | -0.2 | -0.8 | 0.1 | -0.2 |
| n3-all | -1.2 | -0.2 | -0.6 | 0.1 | -0.8 |
| n5-all | -1.4 | -1.0 | 0.4 | -0.0 | -1.1 |
| n3-frames | -0.7 | 0.1 | -1.1 | -0.3 | -0.5 |
| n3-taps | -1.6 | 0.2 | 0.2 | -0.6 | -1.1 |

**gain of n3-frames / n3-taps as a share of n3-all's gain over px8, whole (%; meaningless when n3-all's gain is near zero)**

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| n3-frames | 71 | -53 | 192 | -100 | 102 |
| n3-taps | 154 | 116 | -37 | -180 | 181 |

**gain of n3-frames / n3-taps as a share of n3-all's gain over px8, edge (%; meaningless when n3-all's gain is near zero)**

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| n3-frames | 61 | -71 | 189 | -312 | 63 |
| n3-taps | 133 | -122 | -30 | -697 | 131 |
