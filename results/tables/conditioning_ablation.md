# Conditioning Ablation: DominoTree (16) vs marginal tree (DDTree-analogue)@16

Matched-budget T=0.0 comparison using public JSONL records. DominoTree rows come from `raw/dominotree/*_T0.0.jsonl`; marginal-tree rows come from `raw/conditioning_ablation/*_T0.0.jsonl`.

Speedup is relative to AR rows from the same DominoTree file. Delta is `DominoTree speedup / marginal-tree speedup - 1`; 95% CIs are paired bootstraps.

| Dataset / Rollup | DominoTree speedup | DominoTree tau | marginal tree (DDTree-analogue) speedup | marginal tree (DDTree-analogue) tau | Delta DominoTree vs marginal tree (95% CI) | n pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 6.31 | 10.72 | 5.84 | 9.35 | +8.1% [+6.0, +10.4] | 50 |
| MATH-500 | 5.59 | 9.77 | 5.14 | 8.68 | +8.8% [+6.3, +11.5] | 50 |
| AIME25 | 4.50 | 8.48 | 4.05 | 7.26 | +11.0% [+6.1, +16.7] | 30 |
| HumanEval | 4.62 | 8.00 | 4.22 | 6.91 | +9.4% [+7.5, +11.4] | 50 |
| MBPP | 4.82 | 8.16 | 4.18 | 6.69 | +15.4% [+12.1, +18.7] | 50 |
| LCB | 4.29 | 7.82 | 4.15 | 7.14 | +3.4% [-0.2, +6.5] | 50 |
| MT-Bench | 3.48 | 6.14 | 3.21 | 5.30 | +8.2% [+5.6, +11.3] | 100 |
| Alpaca | 2.79 | 4.78 | 2.54 | 4.06 | +10.2% [+7.0, +13.3] | 50 |
| Math Avg | 5.47 | 9.66 | 5.01 | 8.43 | +9.1% [+7.3, +11.1] | 130 |
| Code Avg | 4.58 | 7.99 | 4.19 | 6.91 | +9.4% [+7.6, +11.1] | 150 |
| Chat Avg | 3.14 | 5.46 | 2.88 | 4.68 | +9.1% [+7.0, +11.2] | 150 |
| Overall Avg | 4.55 | 7.98 | 4.17 | 6.92 | +9.2% [+8.1, +10.3] | 430 |

## Readout

Rollups with CI entirely above 0: Math Avg, Code Avg, Chat Avg, Overall Avg.
