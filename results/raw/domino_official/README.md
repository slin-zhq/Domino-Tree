# Official Domino baseline (raw)

Per-prompt JSONL from the **released Domino decoder's own benchmark** (`benchmark_noar.py`
= Domino's `benchmark.py` with the b=1 AR removed and a warmup prompt added), used for the
**Domino** row of Table 1 and the "vs Domino" column of the pairwise table.

## Layout

```
domino_official/
  qwen3-4b/T{0.0,0.5,1.0}/{graph,eager}_<dataset>.jsonl   # 48 files
  qwen3-8b/T{0.0,0.5,1.0}/{graph,eager}_<dataset>.jsonl   # 48 files
```

Each line = `{"question_id", "choices":[choice_b1(empty), choice_bk], ...}`; use `choices[1]`
(block=16). Per-prompt TPS = per-turn `new_tokens[i] / decode_times[i]`; τ = mean per-turn
`mean(acceptance_lengths[i])`.

## How it's used (surgical AR, best-of)

- Per dataset the **Domino** baseline = **best-of(graph, eager)** (its fastest config; graph ≥ eager
  on every cell). Speedup = Domino TPS ÷ the **lean common AR** (our-harness AR: 4B `dominotree` `ar`
  ≈66 tok/s; 8B `8b/collect_8b_2048_20260704/our` `ar` ≈16.8) — NOT Domino's own slow spec-loop AR.
- **Warmup**: 4B collected with an in-benchmark warmup → no drop; 8B without → drop the first prompt.

## Provenance

Collected 2026-07-09 single-GPU with the sibling GPU idle — 4B on a dual-5080 node (GPU0, GPU1
idle), 8B on an A6000 (single-GPU by construction): we saw run-to-run variance for Domino's
CUDA-graph runner on a shared node, so we removed the contention rather than average over it.
The derived rows regenerate from these files with `make_latex_table.py` (repository root).
