# Pairwise delta with 95% paired bootstrap CI

Delta is `100 * (mean(DominoTree (16) metric) / mean(baseline metric) - 1)`. Bootstrap resamples paired prompt rows.

| Temp | Dataset/Rollup | Comparison | Metric | N | Delta % | 95% CI |
| --- | --- | --- | --- | ---: | ---: | --- |
| 0.0 | GSM8K | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | -1.85 | [-3.60, -0.01] |
| 0.0 | GSM8K | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 26.86 | [22.57, 31.36] |
| 0.0 | GSM8K | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 26.81 | [22.48, 31.34] |
| 0.0 | MATH-500 | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 1.47 | [-0.72, 3.84] |
| 0.0 | MATH-500 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -0.73 | [-5.41, 4.34] |
| 0.0 | MATH-500 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -3.45 | [-10.59, 3.76] |
| 0.0 | AIME25 | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 29 | 6.63 | [1.55, 11.91] |
| 0.0 | AIME25 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 29 | -7.86 | [-11.83, -3.70] |
| 0.0 | AIME25 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 29 | -7.84 | [-11.75, -3.67] |
| 0.0 | HumanEval | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 7.15 | [5.12, 9.22] |
| 0.0 | HumanEval | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -7.58 | [-9.48, -5.59] |
| 0.0 | HumanEval | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -7.56 | [-9.43, -5.56] |
| 0.0 | MBPP | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 5.20 | [2.77, 7.75] |
| 0.0 | MBPP | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 2.73 | [-0.23, 5.68] |
| 0.0 | MBPP | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 2.83 | [-0.15, 5.68] |
| 0.0 | LCB | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 5.38 | [0.23, 11.90] |
| 0.0 | LCB | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -14.67 | [-19.02, -9.59] |
| 0.0 | LCB | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -14.56 | [-18.94, -9.47] |
| 0.0 | MT-Bench | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 99 | 8.91 | [5.70, 12.83] |
| 0.0 | MT-Bench | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 99 | 3.99 | [0.66, 7.51] |
| 0.0 | MT-Bench | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 99 | 4.05 | [0.62, 7.49] |
| 0.0 | Alpaca | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 15.78 | [10.97, 20.85] |
| 0.0 | Alpaca | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 17.03 | [10.24, 24.46] |
| 0.0 | Alpaca | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 17.19 | [10.44, 24.70] |
| 0.0 | Math | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 127 | 0.85 | [-0.66, 2.47] |
| 0.0 | Math | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 127 | 7.90 | [4.09, 11.89] |
| 0.0 | Math | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 127 | 6.63 | [1.97, 11.29] |
| 0.0 | Code | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 147 | 5.90 | [3.89, 8.11] |
| 0.0 | Code | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 147 | -6.76 | [-9.02, -4.41] |
| 0.0 | Code | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 147 | -6.68 | [-9.00, -4.34] |
| 0.0 | Chat | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 148 | 10.80 | [8.00, 13.98] |
| 0.0 | Chat | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 148 | 7.40 | [4.07, 10.85] |
| 0.0 | Chat | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 148 | 7.48 | [4.22, 10.90] |
| 0.0 | Overall | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 422 | 5.10 | [3.87, 6.40] |
| 0.0 | Overall | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 422 | 1.99 | [0.02, 3.99] |
| 0.0 | Overall | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 422 | 1.61 | [-0.55, 3.73] |
| 0.5 | GSM8K | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | -1.45 | [-4.61, 2.13] |
| 0.5 | GSM8K | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 21.91 | [16.80, 27.55] |
| 0.5 | GSM8K | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 22.33 | [16.86, 28.10] |
| 0.5 | MATH-500 | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 3.27 | [-0.89, 7.71] |
| 0.5 | MATH-500 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -1.21 | [-4.96, 2.66] |
| 0.5 | MATH-500 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -4.00 | [-7.69, -0.24] |
| 0.5 | AIME25 | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 29 | 8.99 | [5.45, 12.88] |
| 0.5 | AIME25 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 29 | -8.48 | [-12.00, -4.90] |
| 0.5 | AIME25 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 29 | -9.36 | [-13.35, -5.42] |
| 0.5 | HumanEval | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 7.11 | [3.16, 11.17] |
| 0.5 | HumanEval | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -8.88 | [-12.57, -5.18] |
| 0.5 | HumanEval | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -8.62 | [-12.15, -4.96] |
| 0.5 | MBPP | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 5.27 | [1.33, 9.48] |
| 0.5 | MBPP | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -1.07 | [-4.51, 2.46] |
| 0.5 | MBPP | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 1.74 | [-1.54, 5.30] |
| 0.5 | LCB | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 4.28 | [-0.18, 8.91] |
| 0.5 | LCB | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -16.19 | [-20.65, -11.39] |
| 0.5 | LCB | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -15.46 | [-20.37, -10.32] |
| 0.5 | MT-Bench | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 99 | 10.28 | [6.77, 14.09] |
| 0.5 | MT-Bench | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 99 | 0.52 | [-3.16, 4.35] |
| 0.5 | MT-Bench | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 99 | 0.38 | [-3.78, 4.66] |
| 0.5 | Alpaca | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 14.47 | [9.66, 19.69] |
| 0.5 | Alpaca | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 16.51 | [9.11, 25.14] |
| 0.5 | Alpaca | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 16.02 | [8.54, 23.92] |
| 0.5 | Math | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 127 | 2.03 | [-0.31, 4.60] |
| 0.5 | Math | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 127 | 6.07 | [2.56, 9.61] |
| 0.5 | Math | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 127 | 4.71 | [1.18, 8.37] |
| 0.5 | Code | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 147 | 5.57 | [3.18, 8.03] |
| 0.5 | Code | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 147 | -8.85 | [-11.34, -6.32] |
| 0.5 | Code | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 147 | -7.67 | [-10.37, -4.96] |
| 0.5 | Chat | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 148 | 11.46 | [8.59, 14.63] |
| 0.5 | Chat | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 148 | 4.64 | [1.09, 8.51] |
| 0.5 | Chat | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 148 | 4.42 | [0.43, 8.48] |
| 0.5 | Overall | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 422 | 5.69 | [4.20, 7.23] |
| 0.5 | Overall | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 422 | -0.27 | [-2.20, 1.75] |
| 0.5 | Overall | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 422 | -0.26 | [-2.27, 1.78] |
| 1.0 | GSM8K | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 3.35 | [-0.61, 7.52] |
| 1.0 | GSM8K | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 15.79 | [10.56, 21.02] |
| 1.0 | GSM8K | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 17.32 | [11.57, 22.99] |
| 1.0 | MATH-500 | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 4.67 | [0.66, 9.44] |
| 1.0 | MATH-500 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -4.63 | [-8.94, 0.13] |
| 1.0 | MATH-500 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -5.36 | [-9.37, -1.03] |
| 1.0 | AIME25 | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 29 | 8.68 | [0.21, 17.23] |
| 1.0 | AIME25 | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 29 | -13.32 | [-18.23, -8.27] |
| 1.0 | AIME25 | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 29 | -13.83 | [-18.41, -9.18] |
| 1.0 | HumanEval | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 8.49 | [4.54, 12.59] |
| 1.0 | HumanEval | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -8.80 | [-11.88, -5.48] |
| 1.0 | HumanEval | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -5.97 | [-9.97, -1.72] |
| 1.0 | MBPP | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 4.69 | [0.37, 9.51] |
| 1.0 | MBPP | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 0.17 | [-3.47, 4.11] |
| 1.0 | MBPP | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -0.15 | [-3.72, 3.61] |
| 1.0 | LCB | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 4.75 | [0.96, 9.04] |
| 1.0 | LCB | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | -19.15 | [-23.14, -15.12] |
| 1.0 | LCB | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | -20.17 | [-23.52, -16.74] |
| 1.0 | MT-Bench | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 99 | 5.99 | [1.30, 10.41] |
| 1.0 | MT-Bench | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 99 | -1.30 | [-4.74, 2.35] |
| 1.0 | MT-Bench | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 99 | -0.68 | [-4.35, 2.95] |
| 1.0 | Alpaca | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 49 | 12.69 | [7.88, 17.81] |
| 1.0 | Alpaca | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 49 | 11.38 | [4.92, 18.37] |
| 1.0 | Alpaca | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 49 | 9.47 | [3.11, 16.42] |
| 1.0 | Math | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 127 | 4.62 | [1.91, 7.60] |
| 1.0 | Math | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 127 | 2.00 | [-1.66, 5.60] |
| 1.0 | Math | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 127 | 2.09 | [-1.54, 5.84] |
| 1.0 | Code | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 147 | 5.97 | [3.49, 8.52] |
| 1.0 | Code | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 147 | -9.50 | [-12.06, -6.98] |
| 1.0 | Code | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 147 | -9.07 | [-11.59, -6.51] |
| 1.0 | Chat | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 148 | 7.87 | [4.27, 11.43] |
| 1.0 | Chat | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 148 | 2.07 | [-1.10, 5.46] |
| 1.0 | Chat | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 148 | 2.06 | [-1.31, 5.55] |
| 1.0 | Overall | DominoTree (16) vs Domino-chain | raw per-prompt TPS (same harness) | 422 | 5.99 | [4.28, 7.66] |
| 1.0 | Overall | DominoTree (16) vs DDTree@16 | speedup-over-own-AR (cross harness) | 422 | -2.69 | [-4.54, -0.79] |
| 1.0 | Overall | DominoTree (16) vs CaDDTree | speedup-over-own-AR (cross harness) | 422 | -2.48 | [-4.40, -0.53] |
