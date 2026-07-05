# Table 2: Per-round stage time

Stage times are mean milliseconds per decoding round from our harness after warmup-row exclusion.

| Temp | Dataset | Method | draft ms | build ms | verify ms | commit ms | chain ms | n |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | GSM8K | Domino-chain | 3.48 | 0.00 | 0.00 | 0.00 | 20.51 | 49 |
| 0.0 | GSM8K | DominoTree cond@16 | 3.49 | 3.65 | 17.71 | 0.63 | 0.00 | 49 |
| 0.0 | MATH-500 | Domino-chain | 3.50 | 0.00 | 0.00 | 0.00 | 21.46 | 49 |
| 0.0 | MATH-500 | DominoTree cond@16 | 3.52 | 3.60 | 18.67 | 0.70 | 0.00 | 49 |
| 0.0 | AIME25 | Domino-chain | 3.60 | 0.00 | 0.00 | 0.00 | 23.89 | 29 |
| 0.0 | AIME25 | DominoTree cond@16 | 3.61 | 3.67 | 21.04 | 0.95 | 0.00 | 29 |
| 0.0 | HumanEval | Domino-chain | 3.50 | 0.00 | 0.00 | 0.00 | 21.16 | 49 |
| 0.0 | HumanEval | DominoTree cond@16 | 3.51 | 3.67 | 18.31 | 0.64 | 0.00 | 49 |
| 0.0 | MBPP | Domino-chain | 3.47 | 0.00 | 0.00 | 0.00 | 20.33 | 49 |
| 0.0 | MBPP | DominoTree cond@16 | 3.48 | 3.66 | 17.50 | 0.61 | 0.00 | 49 |
| 0.0 | LCB | Domino-chain | 3.58 | 0.00 | 0.00 | 0.00 | 22.69 | 49 |
| 0.0 | LCB | DominoTree cond@16 | 3.60 | 3.68 | 19.89 | 0.80 | 0.00 | 49 |
| 0.0 | MT-Bench | Domino-chain | 3.53 | 0.00 | 0.00 | 0.00 | 21.83 | 99 |
| 0.0 | MT-Bench | DominoTree cond@16 | 3.55 | 3.68 | 19.02 | 0.73 | 0.00 | 99 |
| 0.0 | Alpaca | Domino-chain | 3.47 | 0.00 | 0.00 | 0.00 | 20.47 | 49 |
| 0.0 | Alpaca | DominoTree cond@16 | 3.48 | 3.69 | 17.61 | 0.60 | 0.00 | 49 |
| 0.0 | Overall | Domino-chain | 3.51 | 0.00 | 0.00 | 0.00 | 21.47 | 422 |
| 0.0 | Overall | DominoTree cond@16 | 3.53 | 3.67 | 18.65 | 0.70 | 0.00 | 422 |
| 0.5 | GSM8K | Domino-chain | 3.49 | 0.00 | 0.00 | 0.00 | 20.69 | 49 |
| 0.5 | GSM8K | DominoTree cond@16 | 3.49 | 3.64 | 17.70 | 0.78 | 0.00 | 49 |
| 0.5 | MATH-500 | Domino-chain | 3.51 | 0.00 | 0.00 | 0.00 | 21.56 | 49 |
| 0.5 | MATH-500 | DominoTree cond@16 | 3.52 | 3.64 | 18.65 | 0.85 | 0.00 | 49 |
| 0.5 | AIME25 | Domino-chain | 3.62 | 0.00 | 0.00 | 0.00 | 23.88 | 29 |
| 0.5 | AIME25 | DominoTree cond@16 | 3.64 | 3.71 | 21.01 | 1.09 | 0.00 | 29 |
| 0.5 | HumanEval | Domino-chain | 3.51 | 0.00 | 0.00 | 0.00 | 21.27 | 49 |
| 0.5 | HumanEval | DominoTree cond@16 | 3.52 | 3.68 | 18.34 | 0.80 | 0.00 | 49 |
| 0.5 | MBPP | Domino-chain | 3.47 | 0.00 | 0.00 | 0.00 | 20.46 | 49 |
| 0.5 | MBPP | DominoTree cond@16 | 3.47 | 3.63 | 17.51 | 0.76 | 0.00 | 49 |
| 0.5 | LCB | Domino-chain | 3.59 | 0.00 | 0.00 | 0.00 | 22.78 | 49 |
| 0.5 | LCB | DominoTree cond@16 | 3.59 | 3.65 | 19.78 | 0.94 | 0.00 | 49 |
| 0.5 | MT-Bench | Domino-chain | 3.53 | 0.00 | 0.00 | 0.00 | 21.90 | 99 |
| 0.5 | MT-Bench | DominoTree cond@16 | 3.54 | 3.65 | 19.04 | 0.88 | 0.00 | 99 |
| 0.5 | Alpaca | Domino-chain | 3.47 | 0.00 | 0.00 | 0.00 | 20.59 | 49 |
| 0.5 | Alpaca | DominoTree cond@16 | 3.48 | 3.68 | 17.64 | 0.75 | 0.00 | 49 |
| 0.5 | Overall | Domino-chain | 3.52 | 0.00 | 0.00 | 0.00 | 21.57 | 422 |
| 0.5 | Overall | DominoTree cond@16 | 3.53 | 3.66 | 18.64 | 0.85 | 0.00 | 422 |
| 1.0 | GSM8K | Domino-chain | 3.48 | 0.00 | 0.00 | 0.00 | 20.63 | 49 |
| 1.0 | GSM8K | DominoTree cond@16 | 3.49 | 3.65 | 17.73 | 0.73 | 0.00 | 49 |
| 1.0 | MATH-500 | Domino-chain | 3.52 | 0.00 | 0.00 | 0.00 | 21.52 | 49 |
| 1.0 | MATH-500 | DominoTree cond@16 | 3.53 | 3.62 | 18.74 | 0.81 | 0.00 | 49 |
| 1.0 | AIME25 | Domino-chain | 3.62 | 0.00 | 0.00 | 0.00 | 24.15 | 29 |
| 1.0 | AIME25 | DominoTree cond@16 | 3.62 | 3.70 | 21.18 | 1.05 | 0.00 | 29 |
| 1.0 | HumanEval | Domino-chain | 3.50 | 0.00 | 0.00 | 0.00 | 21.24 | 49 |
| 1.0 | HumanEval | DominoTree cond@16 | 3.51 | 3.67 | 18.32 | 0.75 | 0.00 | 49 |
| 1.0 | MBPP | Domino-chain | 3.47 | 0.00 | 0.00 | 0.00 | 20.42 | 49 |
| 1.0 | MBPP | DominoTree cond@16 | 3.48 | 3.67 | 17.50 | 0.71 | 0.00 | 49 |
| 1.0 | LCB | Domino-chain | 3.58 | 0.00 | 0.00 | 0.00 | 22.68 | 49 |
| 1.0 | LCB | DominoTree cond@16 | 3.59 | 3.63 | 19.76 | 0.88 | 0.00 | 49 |
| 1.0 | MT-Bench | Domino-chain | 3.53 | 0.00 | 0.00 | 0.00 | 21.91 | 99 |
| 1.0 | MT-Bench | DominoTree cond@16 | 3.53 | 3.64 | 19.00 | 0.82 | 0.00 | 99 |
| 1.0 | Alpaca | Domino-chain | 3.47 | 0.00 | 0.00 | 0.00 | 20.58 | 49 |
| 1.0 | Alpaca | DominoTree cond@16 | 3.47 | 3.63 | 17.61 | 0.69 | 0.00 | 49 |
| 1.0 | Overall | Domino-chain | 3.52 | 0.00 | 0.00 | 0.00 | 21.55 | 422 |
| 1.0 | Overall | DominoTree cond@16 | 3.53 | 3.65 | 18.65 | 0.80 | 0.00 | 422 |
