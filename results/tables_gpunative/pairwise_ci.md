# Pairwise delta with 95% paired bootstrap CI

Delta is `100 * (mean(DominoTree (16) metric) / mean(baseline metric) - 1)`. Bootstrap resamples paired prompt rows.

| Temp | Dataset/Rollup | Comparison | Metric | N | Delta % | 95% CI |
| --- | --- | --- | --- | ---: | ---: | --- |
| 0.0 | GSM8K | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 5.30 | [3.45, 7.23] |
| 0.0 | GSM8K | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 33.83 | [29.22, 38.66] |
| 0.0 | GSM8K | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 33.77 | [29.12, 38.69] |
| 0.0 | GSM8K | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 47.43 | [42.10, 53.13] |
| 0.0 | MATH-500 | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 8.35 | [6.06, 10.85] |
| 0.0 | MATH-500 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 4.83 | [-0.13, 10.19] |
| 0.0 | MATH-500 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 1.95 | [-5.56, 9.56] |
| 0.0 | MATH-500 | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 9.72 | [4.20, 15.88] |
| 0.0 | AIME25 | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 29 | 13.14 | [7.70, 18.82] |
| 0.0 | AIME25 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 29 | -3.98 | [-8.12, 0.35] |
| 0.0 | AIME25 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 29 | -3.97 | [-8.05, 0.39] |
| 0.0 | AIME25 | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 29 | 5.25 | [0.69, 10.32] |
| 0.0 | HumanEval | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 14.80 | [12.56, 17.04] |
| 0.0 | HumanEval | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -2.17 | [-4.20, -0.06] |
| 0.0 | HumanEval | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -2.16 | [-4.15, -0.01] |
| 0.0 | HumanEval | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 8.49 | [5.95, 11.08] |
| 0.0 | MBPP | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 13.05 | [10.44, 15.78] |
| 0.0 | MBPP | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 9.16 | [5.99, 12.31] |
| 0.0 | MBPP | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 9.26 | [6.09, 12.35] |
| 0.0 | MBPP | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 19.26 | [14.75, 23.99] |
| 0.0 | LCB | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 12.38 | [6.86, 19.30] |
| 0.0 | LCB | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -10.05 | [-14.58, -4.75] |
| 0.0 | LCB | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -9.94 | [-14.53, -4.63] |
| 0.0 | LCB | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | -3.28 | [-8.06, 2.03] |
| 0.0 | MT-Bench | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 99 | 16.48 | [13.07, 20.68] |
| 0.0 | MT-Bench | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 99 | 9.73 | [6.23, 13.43] |
| 0.0 | MT-Bench | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 99 | 9.78 | [6.19, 13.45] |
| 0.0 | MT-Bench | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 99 | 24.22 | [19.05, 29.70] |
| 0.0 | Alpaca | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 24.10 | [18.84, 29.59] |
| 0.0 | Alpaca | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 23.62 | [16.46, 31.47] |
| 0.0 | Alpaca | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 23.80 | [16.73, 31.76] |
| 0.0 | Alpaca | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 48.93 | [41.33, 57.36] |
| 0.0 | Math | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 127 | 7.80 | [6.21, 9.47] |
| 0.0 | Math | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 127 | 13.62 | [9.56, 17.86] |
| 0.0 | Math | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 127 | 12.28 | [7.35, 17.23] |
| 0.0 | Math | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 127 | 22.35 | [17.59, 27.24] |
| 0.0 | Code | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 147 | 13.42 | [11.27, 15.81] |
| 0.0 | Code | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 147 | -1.30 | [-3.71, 1.21] |
| 0.0 | Code | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 147 | -1.22 | [-3.69, 1.27] |
| 0.0 | Code | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 147 | 7.78 | [4.98, 10.73] |
| 0.0 | Chat | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 148 | 18.56 | [15.55, 21.99] |
| 0.0 | Chat | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 148 | 13.36 | [9.82, 17.04] |
| 0.0 | Chat | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 148 | 13.45 | [10.01, 17.07] |
| 0.0 | Chat | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 148 | 30.39 | [25.50, 35.68] |
| 0.0 | Overall | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 422 | 12.46 | [11.15, 13.85] |
| 0.0 | Overall | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 422 | 7.67 | [5.58, 9.78] |
| 0.0 | Overall | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 422 | 7.26 | [4.97, 9.50] |
| 0.0 | Overall | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 422 | 18.45 | [15.90, 20.96] |
| 0.5 | GSM8K | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 5.48 | [2.10, 9.34] |
| 0.5 | GSM8K | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 29.13 | [23.68, 35.14] |
| 0.5 | GSM8K | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 29.58 | [23.79, 35.70] |
| 0.5 | GSM8K | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 42.05 | [35.44, 48.83] |
| 0.5 | MATH-500 | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 10.35 | [5.91, 15.08] |
| 0.5 | MATH-500 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 3.78 | [-0.24, 7.88] |
| 0.5 | MATH-500 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 0.85 | [-3.07, 4.84] |
| 0.5 | MATH-500 | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 10.82 | [5.67, 16.19] |
| 0.5 | AIME25 | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 29 | 15.98 | [12.23, 20.11] |
| 0.5 | AIME25 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 29 | -3.23 | [-6.96, 0.60] |
| 0.5 | AIME25 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 29 | -4.15 | [-8.42, 0.04] |
| 0.5 | AIME25 | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 29 | 6.52 | [1.09, 12.39] |
| 0.5 | HumanEval | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 14.49 | [10.29, 18.85] |
| 0.5 | HumanEval | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -3.72 | [-7.64, 0.22] |
| 0.5 | HumanEval | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -3.44 | [-7.14, 0.41] |
| 0.5 | HumanEval | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 8.80 | [4.50, 13.07] |
| 0.5 | MBPP | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 12.92 | [8.72, 17.41] |
| 0.5 | MBPP | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 5.10 | [1.41, 8.90] |
| 0.5 | MBPP | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 8.08 | [4.58, 11.86] |
| 0.5 | MBPP | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 21.18 | [16.50, 26.27] |
| 0.5 | LCB | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 10.80 | [6.09, 15.65] |
| 0.5 | LCB | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -12.19 | [-16.81, -7.21] |
| 0.5 | LCB | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -11.43 | [-16.52, -6.08] |
| 0.5 | LCB | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | -5.82 | [-11.81, 0.40] |
| 0.5 | MT-Bench | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 99 | 17.45 | [13.70, 21.50] |
| 0.5 | MT-Bench | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 99 | 5.55 | [1.68, 9.57] |
| 0.5 | MT-Bench | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 99 | 5.40 | [1.02, 9.89] |
| 0.5 | MT-Bench | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 99 | 23.86 | [18.64, 29.82] |
| 0.5 | Alpaca | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 22.64 | [17.52, 28.16] |
| 0.5 | Alpaca | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 23.07 | [15.22, 32.19] |
| 0.5 | Alpaca | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 22.55 | [14.62, 30.88] |
| 0.5 | Alpaca | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 50.76 | [42.56, 59.73] |
| 0.5 | Math | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 127 | 9.02 | [6.53, 11.73] |
| 0.5 | Math | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 127 | 11.96 | [8.20, 15.75] |
| 0.5 | Math | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 127 | 10.53 | [6.74, 14.46] |
| 0.5 | Math | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 127 | 21.62 | [17.17, 26.19] |
| 0.5 | Code | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 147 | 12.79 | [10.22, 15.40] |
| 0.5 | Code | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 147 | -3.76 | [-6.41, -1.03] |
| 0.5 | Code | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 147 | -2.51 | [-5.40, 0.37] |
| 0.5 | Code | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 147 | 7.49 | [3.87, 11.09] |
| 0.5 | Chat | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 148 | 18.91 | [15.85, 22.30] |
| 0.5 | Chat | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 148 | 10.06 | [6.32, 14.15] |
| 0.5 | Chat | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 148 | 9.83 | [5.62, 14.19] |
| 0.5 | Chat | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 148 | 30.58 | [25.25, 36.30] |
| 0.5 | Overall | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 422 | 12.86 | [11.27, 14.50] |
| 0.5 | Overall | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 422 | 5.18 | [3.13, 7.33] |
| 0.5 | Overall | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 422 | 5.19 | [3.07, 7.37] |
| 0.5 | Overall | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 422 | 18.05 | [15.48, 20.67] |
| 1.0 | GSM8K | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 10.89 | [6.69, 15.36] |
| 1.0 | GSM8K | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 22.24 | [16.74, 27.81] |
| 1.0 | GSM8K | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 23.86 | [17.76, 29.84] |
| 1.0 | GSM8K | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 36.06 | [29.42, 42.79] |
| 1.0 | MATH-500 | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 11.84 | [7.51, 17.00] |
| 1.0 | MATH-500 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 0.99 | [-3.61, 6.08] |
| 1.0 | MATH-500 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 0.21 | [-4.10, 4.85] |
| 1.0 | MATH-500 | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 12.17 | [6.70, 18.35] |
| 1.0 | AIME25 | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 29 | 15.44 | [6.34, 24.62] |
| 1.0 | AIME25 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 29 | -9.61 | [-14.77, -4.30] |
| 1.0 | AIME25 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 29 | -10.14 | [-15.00, -5.16] |
| 1.0 | AIME25 | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 29 | 6.02 | [-2.08, 14.34] |
| 1.0 | HumanEval | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 15.96 | [11.76, 20.30] |
| 1.0 | HumanEval | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -3.92 | [-7.18, -0.46] |
| 1.0 | HumanEval | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -0.95 | [-5.11, 3.49] |
| 1.0 | HumanEval | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 10.06 | [5.56, 14.95] |
| 1.0 | MBPP | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 12.27 | [7.62, 17.48] |
| 1.0 | MBPP | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 5.99 | [2.19, 10.14] |
| 1.0 | MBPP | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 5.65 | [1.87, 9.59] |
| 1.0 | MBPP | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 18.62 | [14.28, 23.03] |
| 1.0 | LCB | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 11.28 | [7.26, 15.82] |
| 1.0 | LCB | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -15.49 | [-19.58, -11.33] |
| 1.0 | LCB | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -16.55 | [-19.98, -12.99] |
| 1.0 | LCB | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | -2.98 | [-7.90, 2.42] |
| 1.0 | MT-Bench | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 99 | 13.24 | [8.28, 17.92] |
| 1.0 | MT-Bench | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 99 | 4.16 | [0.57, 7.98] |
| 1.0 | MT-Bench | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 99 | 4.81 | [0.95, 8.62] |
| 1.0 | MT-Bench | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 99 | 22.15 | [17.84, 27.02] |
| 1.0 | Alpaca | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 20.44 | [15.31, 25.87] |
| 1.0 | Alpaca | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 17.19 | [10.41, 24.49] |
| 1.0 | Alpaca | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 15.18 | [8.51, 22.46] |
| 1.0 | Alpaca | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 49 | 41.79 | [33.33, 50.39] |
| 1.0 | Math | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 127 | 11.91 | [8.99, 15.09] |
| 1.0 | Math | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 127 | 7.62 | [3.72, 11.43] |
| 1.0 | Math | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 127 | 7.71 | [3.84, 11.67] |
| 1.0 | Math | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 127 | 20.76 | [16.26, 25.50] |
| 1.0 | Code | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 147 | 13.20 | [10.55, 15.91] |
| 1.0 | Code | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 147 | -4.74 | [-7.45, -2.06] |
| 1.0 | Code | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 147 | -4.28 | [-6.97, -1.57] |
| 1.0 | Code | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 147 | 8.39 | [5.30, 11.52] |
| 1.0 | Chat | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 148 | 15.23 | [11.42, 19.02] |
| 1.0 | Chat | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 148 | 7.62 | [4.31, 11.15] |
| 1.0 | Chat | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 148 | 7.61 | [4.07, 11.27] |
| 1.0 | Chat | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 148 | 27.24 | [22.92, 31.90] |
| 1.0 | Overall | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 422 | 13.27 | [11.46, 15.05] |
| 1.0 | Overall | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 422 | 2.55 | [0.59, 4.57] |
| 1.0 | Overall | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 422 | 2.78 | [0.76, 4.85] |
| 1.0 | Overall | DominoTree (16) vs DFlash | speedup-over-own-AR (cross harness) | 422 | 17.26 | [14.87, 19.71] |
