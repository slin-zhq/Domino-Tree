# Pairwise delta with 95% paired bootstrap CI

Delta is `100 * (mean(DominoTree (16) metric) / mean(baseline metric) - 1)`. Bootstrap resamples paired prompt rows.

| Temp | Dataset/Rollup | Comparison                   | Metric                                     |   N | Delta % | 95% CI          |
| ---- | -------------- | ---------------------------- | ------------------------------------------ | --: | ------: | --------------- |
| 0.0  | GSM8K          | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |   -0.94 | [-3.84, 2.21]   |
| 0.0  | GSM8K          | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |   44.49 | [39.04, 49.97]  |
| 0.0  | GSM8K          | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |   44.51 | [39.25, 50.03]  |
| 0.0  | GSM8K          | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |   50.68 | [44.58, 57.16]  |
| 0.0  | MATH-500       | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |    2.10 | [-0.92, 5.41]   |
| 0.0  | MATH-500       | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |   20.83 | [16.26, 25.54]  |
| 0.0  | MATH-500       | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |   20.81 | [16.01, 25.76]  |
| 0.0  | MATH-500       | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |   24.40 | [18.32, 30.97]  |
| 0.0  | AIME25         | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  29 |    4.81 | [1.42, 8.41]    |
| 0.0  | AIME25         | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  30 |   17.13 | [11.95, 22.49]  |
| 0.0  | AIME25         | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  30 |   16.80 | [11.65, 22.10]  |
| 0.0  | AIME25         | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  30 |   13.31 | [9.04, 17.65]   |
| 0.0  | HumanEval      | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |    1.74 | [-1.71, 5.56]   |
| 0.0  | HumanEval      | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |   15.79 | [12.10, 19.74]  |
| 0.0  | HumanEval      | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |   15.85 | [12.24, 19.55]  |
| 0.0  | HumanEval      | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |   21.62 | [16.70, 26.91]  |
| 0.0  | MBPP           | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |    4.55 | [0.98, 8.45]    |
| 0.0  | MBPP           | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |   22.86 | [18.25, 27.51]  |
| 0.0  | MBPP           | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |   22.42 | [18.00, 26.79]  |
| 0.0  | MBPP           | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |   25.38 | [19.60, 31.37]  |
| 0.0  | LCB            | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |    3.65 | [-0.08, 8.11]   |
| 0.0  | LCB            | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |   10.82 | [5.36, 16.98]   |
| 0.0  | LCB            | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |   10.41 | [4.78, 16.54]   |
| 0.0  | LCB            | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |   15.83 | [8.43, 24.08]   |
| 0.0  | MT-Bench       | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  98 |   10.17 | [7.27, 13.23]   |
| 0.0  | MT-Bench       | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        | 100 |   29.98 | [26.29, 33.73]  |
| 0.0  | MT-Bench       | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        | 100 |   29.51 | [25.81, 33.42]  |
| 0.0  | MT-Bench       | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        | 100 |   40.70 | [35.85, 46.03]  |
| 0.0  | Alpaca         | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |   12.66 | [7.89, 17.52]   |
| 0.0  | Alpaca         | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |   29.53 | [21.93, 37.72]  |
| 0.0  | Alpaca         | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |   29.50 | [21.59, 37.77]  |
| 0.0  | Alpaca         | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |   52.64 | [43.00, 62.41]  |
| 0.0  | Math           | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) | 127 |    1.33 | [-0.56, 3.30]   |
| 0.0  | Math           | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        | 130 |   28.57 | [24.98, 32.33]  |
| 0.0  | Math           | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        | 130 |   28.49 | [24.82, 32.29]  |
| 0.0  | Math           | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        | 130 |   31.13 | [26.86, 35.61]  |
| 0.0  | Code           | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) | 147 |    3.28 | [1.09, 5.59]    |
| 0.0  | Code           | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        | 150 |   16.11 | [13.32, 19.12]  |
| 0.0  | Code           | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        | 150 |   15.85 | [13.11, 18.69]  |
| 0.0  | Code           | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        | 150 |   20.68 | [16.83, 24.69]  |
| 0.0  | Chat           | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) | 147 |   10.87 | [8.40, 13.50]   |
| 0.0  | Chat           | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        | 150 |   29.85 | [26.39, 33.39]  |
| 0.0  | Chat           | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        | 150 |   29.51 | [25.94, 33.05]  |
| 0.0  | Chat           | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        | 150 |   43.86 | [39.31, 48.72]  |
| 0.0  | Overall        | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) | 421 |    4.32 | [3.03, 5.67]    |
| 0.0  | Overall        | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        | 430 |   23.96 | [21.93, 26.04]  |
| 0.0  | Overall        | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        | 430 |   23.74 | [21.73, 25.82]  |
| 0.0  | Overall        | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        | 430 |   29.87 | [27.27, 32.53]  |
| 0.5  | GSM8K          | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |    4.03 | [-0.76, 9.23]   |
| 0.5  | GSM8K          | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |   22.69 | [17.34, 28.38]  |
| 0.5  | GSM8K          | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |   22.35 | [16.64, 28.22]  |
| 0.5  | GSM8K          | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |   35.25 | [29.79, 41.39]  |
| 0.5  | MATH-500       | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |    1.46 | [-2.22, 5.37]   |
| 0.5  | MATH-500       | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |    1.39 | [-3.57, 6.37]   |
| 0.5  | MATH-500       | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |    1.80 | [-3.37, 7.04]   |
| 0.5  | MATH-500       | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |    6.73 | [1.32, 12.42]   |
| 0.5  | AIME25         | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  29 |    8.04 | [2.41, 14.20]   |
| 0.5  | AIME25         | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  30 |   -4.42 | [-9.06, 0.31]   |
| 0.5  | AIME25         | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  30 |   -5.05 | [-9.03, -0.61]  |
| 0.5  | AIME25         | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  30 |    2.96 | [-1.80, 7.89]   |
| 0.5  | HumanEval      | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |    4.48 | [0.23, 9.03]    |
| 0.5  | HumanEval      | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |  -10.00 | [-13.88, -6.18] |
| 0.5  | HumanEval      | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |   -9.23 | [-12.77, -5.37] |
| 0.5  | HumanEval      | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |    2.70 | [-1.69, 7.47]   |
| 0.5  | MBPP           | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |    7.86 | [2.81, 12.97]   |
| 0.5  | MBPP           | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |   -0.25 | [-4.83, 4.24]   |
| 0.5  | MBPP           | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |    0.40 | [-4.20, 5.02]   |
| 0.5  | MBPP           | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |   13.99 | [8.40, 19.27]   |
| 0.5  | LCB            | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |   -1.70 | [-6.82, 3.91]   |
| 0.5  | LCB            | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |  -11.26 | [-16.56, -5.48] |
| 0.5  | LCB            | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |  -13.36 | [-18.82, -7.90] |
| 0.5  | LCB            | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |   -6.43 | [-12.55, 0.21]  |
| 0.5  | MT-Bench       | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  98 |   14.21 | [10.62, 18.36]  |
| 0.5  | MT-Bench       | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        | 100 |    4.77 | [1.36, 8.32]    |
| 0.5  | MT-Bench       | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        | 100 |    3.13 | [-0.17, 6.70]   |
| 0.5  | MT-Bench       | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        | 100 |   19.83 | [15.90, 24.17]  |
| 0.5  | Alpaca         | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |   13.24 | [6.89, 19.67]   |
| 0.5  | Alpaca         | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |    5.31 | [-1.51, 12.83]  |
| 0.5  | Alpaca         | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |    5.56 | [-1.14, 12.63]  |
| 0.5  | Alpaca         | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |   29.72 | [20.14, 40.24]  |
| 0.5  | Math           | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) | 127 |    3.73 | [1.01, 6.60]    |
| 0.5  | Math           | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        | 130 |    7.96 | [4.32, 11.69]   |
| 0.5  | Math           | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        | 130 |    7.88 | [4.22, 11.71]   |
| 0.5  | Math           | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        | 130 |   16.14 | [12.07, 20.48]  |
| 0.5  | Code           | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) | 147 |    3.35 | [0.42, 6.40]    |
| 0.5  | Code           | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        | 150 |   -7.50 | [-10.35, -4.70] |
| 0.5  | Code           | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        | 150 |   -7.86 | [-10.68, -5.02] |
| 0.5  | Code           | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        | 150 |    2.56 | [-1.07, 6.24]   |
| 0.5  | Chat           | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) | 147 |   13.94 | [10.75, 17.40]  |
| 0.5  | Chat           | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        | 150 |    4.92 | [1.76, 8.16]    |
| 0.5  | Chat           | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        | 150 |    3.81 | [0.75, 6.99]    |
| 0.5  | Chat           | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        | 150 |   22.46 | [18.49, 26.85]  |
| 0.5  | Overall        | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) | 421 |    5.98 | [4.24, 7.76]    |
| 0.5  | Overall        | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        | 430 |    0.99 | [-1.03, 3.04]   |
| 0.5  | Overall        | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        | 430 |    0.55 | [-1.50, 2.58]   |
| 0.5  | Overall        | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        | 430 |   12.13 | [9.69, 14.60]   |
| 1.0  | GSM8K          | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |    0.93 | [-5.19, 7.63]   |
| 1.0  | GSM8K          | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |   15.07 | [9.52, 21.16]   |
| 1.0  | GSM8K          | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |   16.66 | [10.62, 22.91]  |
| 1.0  | GSM8K          | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |   24.36 | [17.97, 31.39]  |
| 1.0  | MATH-500       | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |   -2.05 | [-7.17, 3.64]   |
| 1.0  | MATH-500       | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |   -5.93 | [-10.09, -1.54] |
| 1.0  | MATH-500       | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |   -6.28 | [-10.38, -1.93] |
| 1.0  | MATH-500       | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |    2.55 | [-2.55, 7.96]   |
| 1.0  | AIME25         | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  29 |   16.65 | [10.99, 22.81]  |
| 1.0  | AIME25         | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  30 |  -13.02 | [-17.15, -8.59] |
| 1.0  | AIME25         | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  30 |  -14.09 | [-18.18, -9.97] |
| 1.0  | AIME25         | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  30 |    3.34 | [-1.04, 8.22]   |
| 1.0  | HumanEval      | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |    8.72 | [3.20, 14.31]   |
| 1.0  | HumanEval      | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |   -6.33 | [-10.92, -1.49] |
| 1.0  | HumanEval      | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |   -3.45 | [-8.38, 1.49]   |
| 1.0  | HumanEval      | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |    6.17 | [0.80, 11.52]   |
| 1.0  | MBPP           | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |    4.73 | [-0.76, 10.64]  |
| 1.0  | MBPP           | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |   -7.79 | [-11.45, -3.90] |
| 1.0  | MBPP           | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |   -6.96 | [-10.19, -3.60] |
| 1.0  | MBPP           | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |    4.12 | [-0.78, 9.02]   |
| 1.0  | LCB            | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |    2.43 | [-2.65, 7.70]   |
| 1.0  | LCB            | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |  -11.88 | [-16.38, -6.95] |
| 1.0  | LCB            | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |  -12.28 | [-17.40, -7.07] |
| 1.0  | LCB            | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |   -2.01 | [-7.52, 3.93]   |
| 1.0  | MT-Bench       | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  98 |   12.48 | [7.14, 18.23]   |
| 1.0  | MT-Bench       | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        | 100 |   -0.63 | [-4.50, 3.30]   |
| 1.0  | MT-Bench       | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        | 100 |   -0.03 | [-3.65, 3.52]   |
| 1.0  | MT-Bench       | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        | 100 |   12.94 | [7.04, 19.20]   |
| 1.0  | Alpaca         | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) |  49 |   15.42 | [9.04, 22.40]   |
| 1.0  | Alpaca         | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        |  50 |    0.13 | [-4.78, 5.27]   |
| 1.0  | Alpaca         | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        |  50 |    1.07 | [-4.48, 7.09]   |
| 1.0  | Alpaca         | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        |  50 |   16.75 | [8.52, 25.80]   |
| 1.0  | Math           | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) | 127 |    2.05 | [-1.74, 5.97]   |
| 1.0  | Math           | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        | 130 |    0.59 | [-2.89, 4.13]   |
| 1.0  | Math           | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        | 130 |    0.70 | [-2.85, 4.48]   |
| 1.0  | Math           | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        | 130 |   11.14 | [7.18, 15.34]   |
| 1.0  | Code           | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) | 147 |    5.15 | [2.02, 8.36]    |
| 1.0  | Code           | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        | 150 |   -8.87 | [-11.42, -6.23] |
| 1.0  | Code           | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        | 150 |   -7.90 | [-10.70, -5.07] |
| 1.0  | Code           | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        | 150 |    2.44 | [-0.76, 5.61]   |
| 1.0  | Chat           | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) | 147 |   13.32 | [9.02, 17.92]   |
| 1.0  | Chat           | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        | 150 |   -0.41 | [-3.47, 2.73]   |
| 1.0  | Chat           | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        | 150 |    0.29 | [-2.74, 3.41]   |
| 1.0  | Chat           | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        | 150 |   14.02 | [8.97, 19.14]   |
| 1.0  | Overall        | DominoTree (16) vs Domino    | raw per-prompt TPS (shared lean common AR) | 421 |    5.95 | [3.77, 8.18]    |
| 1.0  | Overall        | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness)        | 430 |   -3.50 | [-5.39, -1.64]  |
| 1.0  | Overall        | DominoTree (16) vs CaDDTree  | speedup-over-own-AR (cross harness)        | 430 |   -2.88 | [-4.83, -0.94]  |
| 1.0  | Overall        | DominoTree (16) vs DFlash    | speedup-over-own-AR (cross harness)        | 430 |    8.31 | [6.06, 10.67]   |
