# Pairwise delta with 95% paired bootstrap CI

Delta is `100 * (mean(DominoTree (16) metric) / mean(baseline metric) - 1)`. Bootstrap resamples paired prompt rows.

| Temp | Dataset/Rollup | Comparison | Metric | N | Delta % | 95% CI |
| --- | --- | --- | --- | ---: | ---: | --- |
| 0.0 | GSM8K | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 3.91 | [2.03, 5.96] |
| 0.0 | GSM8K | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 33.83 | [29.21, 38.73] |
| 0.0 | GSM8K | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 33.77 | [29.18, 38.64] |
| 0.0 | GSM8K | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 47.43 | [42.10, 53.13] |
| 0.0 | MATH-500 | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 5.90 | [3.56, 8.53] |
| 0.0 | MATH-500 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 4.83 | [-0.33, 10.37] |
| 0.0 | MATH-500 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 1.95 | [-5.42, 9.50] |
| 0.0 | MATH-500 | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 9.72 | [4.20, 15.88] |
| 0.0 | AIME25 | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 29 | 8.12 | [2.81, 13.54] |
| 0.0 | AIME25 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 29 | -3.98 | [-8.11, 0.30] |
| 0.0 | AIME25 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 29 | -3.97 | [-8.24, 0.37] |
| 0.0 | AIME25 | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 29 | 5.25 | [0.69, 10.32] |
| 0.0 | HumanEval | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 10.93 | [8.87, 13.05] |
| 0.0 | HumanEval | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -2.17 | [-4.12, -0.08] |
| 0.0 | HumanEval | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -2.16 | [-4.12, 0.01] |
| 0.0 | HumanEval | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 8.49 | [5.95, 11.08] |
| 0.0 | MBPP | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 9.01 | [5.97, 12.10] |
| 0.0 | MBPP | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 9.16 | [6.03, 12.34] |
| 0.0 | MBPP | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 9.26 | [6.21, 12.40] |
| 0.0 | MBPP | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 19.26 | [14.75, 23.99] |
| 0.0 | LCB | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 8.55 | [3.22, 15.15] |
| 0.0 | LCB | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -10.05 | [-14.59, -4.74] |
| 0.0 | LCB | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -9.94 | [-14.52, -4.82] |
| 0.0 | LCB | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | -3.28 | [-8.06, 2.03] |
| 0.0 | MT-Bench | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 99 | 13.72 | [10.16, 17.93] |
| 0.0 | MT-Bench | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 99 | 9.73 | [6.21, 13.36] |
| 0.0 | MT-Bench | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 99 | 9.78 | [6.20, 13.52] |
| 0.0 | MT-Bench | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 99 | 24.22 | [19.05, 29.70] |
| 0.0 | Alpaca | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 17.78 | [12.35, 23.50] |
| 0.0 | Alpaca | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 23.62 | [16.49, 31.11] |
| 0.0 | Alpaca | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 23.80 | [16.67, 31.51] |
| 0.0 | Alpaca | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 48.93 | [41.33, 57.36] |
| 0.0 | Math | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 127 | 5.40 | [3.87, 7.08] |
| 0.0 | Math | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 127 | 13.62 | [9.59, 17.75] |
| 0.0 | Math | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 127 | 12.28 | [7.38, 17.18] |
| 0.0 | Math | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 127 | 22.35 | [17.59, 27.24] |
| 0.0 | Code | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 147 | 9.51 | [7.31, 11.88] |
| 0.0 | Code | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 147 | -1.30 | [-3.72, 1.27] |
| 0.0 | Code | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 147 | -1.22 | [-3.67, 1.32] |
| 0.0 | Code | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 147 | 7.78 | [4.98, 10.73] |
| 0.0 | Chat | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 148 | 14.85 | [11.88, 18.27] |
| 0.0 | Chat | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 148 | 13.36 | [9.90, 17.07] |
| 0.0 | Chat | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 148 | 13.45 | [10.03, 17.10] |
| 0.0 | Chat | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 148 | 30.39 | [25.50, 35.68] |
| 0.0 | Overall | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 422 | 9.21 | [7.95, 10.57] |
| 0.0 | Overall | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 422 | 7.67 | [5.63, 9.80] |
| 0.0 | Overall | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 422 | 7.26 | [4.93, 9.58] |
| 0.0 | Overall | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 422 | 18.45 | [15.90, 20.96] |
| 0.5 | GSM8K | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 3.50 | [-0.10, 7.15] |
| 0.5 | GSM8K | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 29.13 | [23.52, 35.02] |
| 0.5 | GSM8K | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 29.58 | [23.79, 35.61] |
| 0.5 | GSM8K | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 42.05 | [35.44, 48.83] |
| 0.5 | MATH-500 | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 3.59 | [-0.69, 7.91] |
| 0.5 | MATH-500 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 3.78 | [-0.16, 7.83] |
| 0.5 | MATH-500 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 0.85 | [-3.17, 4.72] |
| 0.5 | MATH-500 | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 10.82 | [5.67, 16.19] |
| 0.5 | AIME25 | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 29 | 16.20 | [11.18, 22.19] |
| 0.5 | AIME25 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 29 | -3.23 | [-6.99, 0.58] |
| 0.5 | AIME25 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 29 | -4.15 | [-8.20, 0.19] |
| 0.5 | AIME25 | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 29 | 6.52 | [1.09, 12.39] |
| 0.5 | HumanEval | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 8.74 | [4.39, 13.35] |
| 0.5 | HumanEval | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -3.72 | [-7.50, 0.10] |
| 0.5 | HumanEval | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -3.44 | [-7.10, 0.37] |
| 0.5 | HumanEval | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 8.80 | [4.50, 13.07] |
| 0.5 | MBPP | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 14.94 | [11.16, 18.67] |
| 0.5 | MBPP | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 5.10 | [1.44, 8.84] |
| 0.5 | MBPP | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 8.08 | [4.47, 11.85] |
| 0.5 | MBPP | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 21.18 | [16.50, 26.27] |
| 0.5 | LCB | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 6.00 | [0.70, 11.92] |
| 0.5 | LCB | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -12.19 | [-16.87, -7.30] |
| 0.5 | LCB | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -11.43 | [-16.45, -6.04] |
| 0.5 | LCB | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | -5.82 | [-11.81, 0.40] |
| 0.5 | MT-Bench | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 99 | 14.07 | [9.99, 18.51] |
| 0.5 | MT-Bench | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 99 | 5.55 | [1.70, 9.54] |
| 0.5 | MT-Bench | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 99 | 5.40 | [1.10, 10.01] |
| 0.5 | MT-Bench | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 99 | 23.86 | [18.64, 29.82] |
| 0.5 | Alpaca | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 23.43 | [17.72, 29.03] |
| 0.5 | Alpaca | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 23.07 | [15.17, 32.19] |
| 0.5 | Alpaca | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 22.55 | [14.69, 31.24] |
| 0.5 | Alpaca | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 50.76 | [42.56, 59.73] |
| 0.5 | Math | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 127 | 5.48 | [2.93, 8.13] |
| 0.5 | Math | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 127 | 11.96 | [8.36, 15.71] |
| 0.5 | Math | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 127 | 10.53 | [6.77, 14.47] |
| 0.5 | Math | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 127 | 21.62 | [17.17, 26.19] |
| 0.5 | Code | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 147 | 9.99 | [7.21, 12.79] |
| 0.5 | Code | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 147 | -3.76 | [-6.46, -1.08] |
| 0.5 | Code | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 147 | -2.51 | [-5.39, 0.41] |
| 0.5 | Code | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 147 | 7.49 | [3.87, 11.09] |
| 0.5 | Chat | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 148 | 16.64 | [13.12, 20.37] |
| 0.5 | Chat | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 148 | 10.06 | [6.30, 14.17] |
| 0.5 | Chat | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 148 | 9.83 | [5.66, 14.21] |
| 0.5 | Chat | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 148 | 30.58 | [25.25, 36.30] |
| 0.5 | Overall | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 422 | 9.89 | [8.21, 11.61] |
| 0.5 | Overall | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 422 | 5.18 | [3.07, 7.29] |
| 0.5 | Overall | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 422 | 5.19 | [3.08, 7.45] |
| 0.5 | Overall | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 422 | 18.05 | [15.48, 20.67] |
| 1.0 | GSM8K | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 4.55 | [-0.29, 9.87] |
| 1.0 | GSM8K | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 22.24 | [16.80, 27.91] |
| 1.0 | GSM8K | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 23.86 | [17.79, 29.99] |
| 1.0 | GSM8K | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 36.06 | [29.42, 42.79] |
| 1.0 | MATH-500 | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 8.57 | [3.58, 14.15] |
| 1.0 | MATH-500 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 0.99 | [-3.60, 5.89] |
| 1.0 | MATH-500 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 0.21 | [-4.06, 4.82] |
| 1.0 | MATH-500 | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 12.17 | [6.70, 18.35] |
| 1.0 | AIME25 | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 29 | 9.52 | [0.61, 19.74] |
| 1.0 | AIME25 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 29 | -9.61 | [-14.65, -4.21] |
| 1.0 | AIME25 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 29 | -10.14 | [-15.17, -5.24] |
| 1.0 | AIME25 | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 29 | 6.02 | [-2.08, 14.34] |
| 1.0 | HumanEval | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 14.00 | [8.45, 19.98] |
| 1.0 | HumanEval | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -3.92 | [-7.23, -0.55] |
| 1.0 | HumanEval | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -0.95 | [-5.18, 3.42] |
| 1.0 | HumanEval | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 10.06 | [5.56, 14.95] |
| 1.0 | MBPP | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 15.45 | [10.61, 20.82] |
| 1.0 | MBPP | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 5.99 | [2.09, 10.13] |
| 1.0 | MBPP | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 5.65 | [2.01, 9.57] |
| 1.0 | MBPP | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 18.62 | [14.28, 23.03] |
| 1.0 | LCB | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 2.83 | [-1.73, 7.77] |
| 1.0 | LCB | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -15.49 | [-19.54, -11.34] |
| 1.0 | LCB | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -16.55 | [-20.04, -12.91] |
| 1.0 | LCB | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | -2.98 | [-7.90, 2.42] |
| 1.0 | MT-Bench | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 99 | 13.91 | [9.31, 19.05] |
| 1.0 | MT-Bench | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 99 | 4.16 | [0.49, 7.99] |
| 1.0 | MT-Bench | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 99 | 4.81 | [0.91, 8.69] |
| 1.0 | MT-Bench | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 99 | 22.15 | [17.84, 27.02] |
| 1.0 | Alpaca | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 49 | 16.79 | [6.65, 27.00] |
| 1.0 | Alpaca | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 17.19 | [10.37, 24.20] |
| 1.0 | Alpaca | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 15.18 | [8.38, 22.72] |
| 1.0 | Alpaca | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 41.79 | [33.33, 50.39] |
| 1.0 | Math | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 127 | 6.84 | [3.46, 10.36] |
| 1.0 | Math | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 127 | 7.62 | [3.78, 11.55] |
| 1.0 | Math | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 127 | 7.71 | [3.81, 11.67] |
| 1.0 | Math | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 127 | 20.76 | [16.26, 25.50] |
| 1.0 | Code | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 147 | 10.86 | [7.79, 14.04] |
| 1.0 | Code | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 147 | -4.74 | [-7.40, -2.06] |
| 1.0 | Code | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 147 | -4.28 | [-6.97, -1.57] |
| 1.0 | Code | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 147 | 8.39 | [5.30, 11.52] |
| 1.0 | Chat | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 148 | 14.73 | [10.34, 19.40] |
| 1.0 | Chat | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 148 | 7.62 | [4.23, 11.17] |
| 1.0 | Chat | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 148 | 7.61 | [4.11, 11.19] |
| 1.0 | Chat | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 148 | 27.24 | [22.92, 31.90] |
| 1.0 | Overall | DominoTree (16) vs Domino | raw per-prompt TPS (shared lean common AR) | 422 | 10.39 | [8.38, 12.56] |
| 1.0 | Overall | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 422 | 2.55 | [0.55, 4.59] |
| 1.0 | Overall | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 422 | 2.78 | [0.76, 4.82] |
| 1.0 | Overall | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 422 | 17.26 | [14.87, 19.71] |
