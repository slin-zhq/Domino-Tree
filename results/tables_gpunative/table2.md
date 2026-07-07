# Table 2: Per-round stage time

Mean milliseconds per decoding round, our harness, Overall across all eight datasets at T=0,
after warmup-row exclusion. Stage times are temperature-invariant (each stage varies <2% across
T in {0, 0.5, 1.0}), so we report T=0. Both methods use identical stage boundaries: `build` is
drafter-side candidate construction (the sequential GRU-correction pass for Domino-chain;
best-first tree construction plus attention-mask assembly for DominoTree), `verify` is the single
target forward, `commit` is the acceptance check plus KV/output write, and `total` is their sum.
Domino-chain's stage split is from a dedicated instrumented run (`results/raw/chain_stage_timing/`),
because the chain's original code path timed all post-draft work as one fused block.

| Method | draft ms | build ms | verify ms | commit ms | total/round ms | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Domino-chain | 3.55 | 2.76 | 19.04 | 0.19 | 25.54 | 422 |
| DominoTree (16) | 3.52 | 2.31 | 18.70 | 0.70 | 25.24 | 422 |
