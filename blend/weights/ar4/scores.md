# ar4 scores (task 146)

Held-out: the 23 pairs of blend/probe/pairs.json (split val), weight stage (sum of weights x candidates), measures of blend/edge/score_edge.py. ar4 = forza-rx9070xt continued 2000 steps with Charbonnier + lam * spurious-edge penalty, lam = 1.0 (net C choice: strongest lam of the sweep within 0.10 dB (all) and 0.25 dB (per group) of forza-rx9070xt C; lam = 1.763 fails with -0.15 dB all). Rows lamX are net C sweep runs; "lam1.0_ar4_C" is the chosen net.

## Net C, sweep

PSNR dB, whole picture

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 30.86 | 28.79 | 40.05 | 32.36 | 31.24 |
| baseC | 30.77 | 29.18 | 40.23 | 32.55 | 31.33 |
| forzaC | 30.71 | 29.26 | 40.40 | 32.59 | 31.33 |
| lam0 | 30.62 | 29.10 | 40.71 | 32.53 | 31.23 |
| lam0.1306 | 30.64 | 29.16 | 40.66 | 32.56 | 31.26 |
| lam0.5037 | 30.62 | 29.24 | 40.60 | 32.59 | 31.27 |
| lam1.0_ar4_C | 30.56 | 29.26 | 40.57 | 32.60 | 31.24 |
| lam1.763 | 30.46 | 29.26 | 40.52 | 32.59 | 31.18 |

PSNR dB, edge pixels

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 26.23 | 25.23 | 35.26 | 25.93 | 26.75 |
| baseC | 26.06 | 25.41 | 35.42 | 25.98 | 26.72 |
| forzaC | 25.96 | 25.51 | 35.61 | 26.01 | 26.69 |
| lam0 | 25.99 | 25.36 | 35.96 | 25.96 | 26.67 |
| lam0.1306 | 25.98 | 25.43 | 35.90 | 25.98 | 26.69 |
| lam0.5037 | 25.90 | 25.53 | 35.84 | 26.01 | 26.67 |
| lam1.0_ar4_C | 25.79 | 25.58 | 35.80 | 26.01 | 26.61 |
| lam1.763 | 25.65 | 25.60 | 35.74 | 26.00 | 26.53 |

spurious edge energy, whole picture (mean max(0, G_out - G_real), levels)

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 2.795 | 4.734 | 1.340 | 2.800 | 2.880 |
| baseC | 2.401 | 3.900 | 1.180 | 2.506 | 2.463 |
| forzaC | 2.286 | 3.721 | 1.144 | 2.423 | 2.355 |
| lam0 | 2.415 | 4.022 | 1.170 | 2.510 | 2.490 |
| lam0.1306 | 2.360 | 3.840 | 1.154 | 2.469 | 2.422 |
| lam0.5037 | 2.271 | 3.534 | 1.120 | 2.392 | 2.306 |
| lam1.0_ar4_C | 2.226 | 3.355 | 1.098 | 2.341 | 2.242 |
| lam1.763 | 2.205 | 3.237 | 1.083 | 2.304 | 2.202 |

missing edge energy, whole picture (mean max(0, G_real - G_out))

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 6.424 | 7.064 | 2.000 | 4.128 | 5.466 |
| baseC | 6.826 | 7.588 | 2.127 | 4.250 | 5.805 |
| forzaC | 7.123 | 7.701 | 2.168 | 4.346 | 6.000 |
| lam0 | 6.888 | 7.619 | 2.122 | 4.268 | 5.844 |
| lam0.1306 | 6.995 | 7.745 | 2.142 | 4.309 | 5.931 |
| lam0.5037 | 7.271 | 7.988 | 2.187 | 4.404 | 6.138 |
| lam1.0_ar4_C | 7.524 | 8.161 | 2.219 | 4.482 | 6.315 |
| lam1.763 | 7.758 | 8.290 | 2.245 | 4.550 | 6.474 |

spurious edge energy inside the edge mask

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 5.426 | 7.247 | 2.835 | 9.496 | 5.701 |
| baseC | 4.821 | 6.337 | 2.483 | 8.782 | 5.073 |
| forzaC | 4.484 | 5.997 | 2.380 | 8.436 | 4.781 |
| lam0 | 4.779 | 6.507 | 2.435 | 8.744 | 5.073 |
| lam0.1306 | 4.646 | 6.214 | 2.396 | 8.600 | 4.926 |
| lam0.5037 | 4.406 | 5.700 | 2.309 | 8.298 | 4.653 |
| lam1.0_ar4_C | 4.276 | 5.379 | 2.248 | 8.080 | 4.489 |
| lam1.763 | 4.209 | 5.153 | 2.204 | 7.907 | 4.384 |

missing edge energy inside the edge mask

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 17.793 | 14.656 | 4.629 | 15.481 | 14.312 |
| baseC | 18.706 | 15.443 | 4.914 | 15.775 | 15.012 |
| forzaC | 19.713 | 15.684 | 5.020 | 16.215 | 15.626 |
| lam0 | 18.981 | 15.532 | 4.908 | 15.876 | 15.175 |
| lam0.1306 | 19.303 | 15.741 | 4.959 | 16.047 | 15.405 |
| lam0.5037 | 20.200 | 16.151 | 5.073 | 16.461 | 15.998 |
| lam1.0_ar4_C | 21.064 | 16.457 | 5.158 | 16.819 | 16.543 |
| lam1.763 | 21.883 | 16.698 | 5.228 | 17.141 | 17.045 |

mean weight on prev + curr at edge pixels

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseC | 0.1846 | 0.2285 | 0.2123 | 0.1790 | 0.1984 |
| forzaC | 0.2111 | 0.2325 | 0.2531 | 0.2018 | 0.2228 |
| lam0 | 0.1807 | 0.1997 | 0.1984 | 0.1653 | 0.1865 |
| lam0.1306 | 0.2039 | 0.2314 | 0.2257 | 0.1899 | 0.2124 |
| lam0.5037 | 0.2486 | 0.2829 | 0.2825 | 0.2360 | 0.2610 |
| lam1.0_ar4_C | 0.2787 | 0.3155 | 0.3200 | 0.2663 | 0.2931 |
| lam1.763 | 0.3037 | 0.3406 | 0.3485 | 0.2907 | 0.3187 |

## Nets A and B (four choices, taps only; the prev + curr weight is 0 by construction)

PSNR dB, whole picture

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 30.86 | 28.79 | 40.05 | 32.36 | 31.24 |
| forza-rx9070xt_A | 30.86 | 28.92 | 40.54 | 32.52 | 31.31 |
| ar4_A | 30.89 | 28.93 | 40.64 | 32.51 | 31.33 |
| forza-rx9070xt_B | 30.90 | 28.94 | 40.48 | 32.51 | 31.34 |
| ar4_B | 30.91 | 28.94 | 40.58 | 32.50 | 31.35 |

PSNR dB, edge pixels

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 26.23 | 25.23 | 35.26 | 25.93 | 26.75 |
| forza-rx9070xt_A | 26.18 | 25.23 | 35.78 | 25.99 | 26.74 |
| ar4_A | 26.19 | 25.24 | 35.88 | 25.98 | 26.75 |
| forza-rx9070xt_B | 26.19 | 25.25 | 35.70 | 25.99 | 26.75 |
| ar4_B | 26.19 | 25.25 | 35.82 | 25.97 | 26.76 |

spurious edge energy, whole picture (mean max(0, G_out - G_real), levels)

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 2.795 | 4.734 | 1.340 | 2.800 | 2.880 |
| forza-rx9070xt_A | 2.623 | 4.504 | 1.298 | 2.661 | 2.724 |
| ar4_A | 2.608 | 4.492 | 1.293 | 2.664 | 2.714 |
| forza-rx9070xt_B | 2.627 | 4.509 | 1.301 | 2.665 | 2.729 |
| ar4_B | 2.613 | 4.499 | 1.296 | 2.669 | 2.719 |

missing edge energy, whole picture (mean max(0, G_real - G_out))

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 6.424 | 7.064 | 2.000 | 4.128 | 5.466 |
| forza-rx9070xt_A | 6.527 | 7.164 | 2.005 | 4.120 | 5.537 |
| ar4_A | 6.538 | 7.175 | 2.005 | 4.121 | 5.545 |
| forza-rx9070xt_B | 6.512 | 7.157 | 2.003 | 4.112 | 5.527 |
| ar4_B | 6.523 | 7.170 | 2.005 | 4.114 | 5.536 |

spurious edge energy inside the edge mask

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 5.426 | 7.247 | 2.835 | 9.496 | 5.701 |
| forza-rx9070xt_A | 5.108 | 7.051 | 2.723 | 9.077 | 5.438 |
| ar4_A | 5.060 | 7.027 | 2.708 | 9.089 | 5.408 |
| forza-rx9070xt_B | 5.113 | 7.057 | 2.735 | 9.097 | 5.447 |
| ar4_B | 5.071 | 7.036 | 2.716 | 9.113 | 5.419 |

missing edge energy inside the edge mask

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 17.793 | 14.656 | 4.629 | 15.481 | 14.312 |
| forza-rx9070xt_A | 18.093 | 14.837 | 4.639 | 15.382 | 14.490 |
| ar4_A | 18.137 | 14.866 | 4.641 | 15.389 | 14.518 |
| forza-rx9070xt_B | 18.046 | 14.825 | 4.636 | 15.347 | 14.459 |
| ar4_B | 18.086 | 14.856 | 4.640 | 15.355 | 14.487 |

mean weight on prev + curr at edge pixels

| method | forza | cs2 | minecraft120 | battlefield | all |
|---|---|---|---|---|---|
| rule | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| forza-rx9070xt_A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ar4_A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| forza-rx9070xt_B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ar4_B | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
