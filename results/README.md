# DominoTree results — raw data and regenerated tables

Evidence bundle backing the DominoTree paper. Raw per-prompt records live under
`raw/`; regenerate the paper tables with:

```
python make_latex_table.py --raw-dir results/raw --out-dir results/tables_gpunative
```

## Convention: the GPU-native builder is the default

DominoTree's default tree builder is the **GPU-native CUDA-graph builder**. The
Python reference builder is kept **only as a comparison** — it produces
bit-identical trees (same tokens, same accepted length), so it isolates *build
cost* alone (see the paper's "Python vs. GPU-native builder" section). Every
main result (Table 1, pairwise, stage-time) uses the GPU-native builder.

- `raw/dominotree/` — **DEFAULT.** DominoTree built with the GPU-native builder
  (records carry `gpu_native_build: true`).
- `raw/dominotree_python_builder/` — the Python reference builder (identical
  trees; `gpu_native_build` absent). Comparison only.

## `raw/` → paper tables

| dir | method / builder | backs |
|---|---|---|
| `dominotree/` | DominoTree, **GPU-native (default)** | Table 1 (Domino/DominoTree rows), pairwise, stage-time |
| `dominotree_python_builder/` | DominoTree, Python builder | Python-vs-GPU-native builder comparison; conditioning ablation (cond side) |
| `baseline_ddtree_caddtree/` | AR / DFlash / DDTree / CaDDTree — official reference harness on native DFlash | Table 1 baseline rows, pairwise |
| `chain_stage_timing/` | Domino-chain, instrumented stage split | stage-time table (chain row) |
| `conditioning_ablation/` | marginal-tree `marg@16`, Python builder | conditioning ablation (marg side) |
| `draft_sampling_ablation/` | DominoTree + Domino-chain, greedy vs. sampled draft, T∈{0.5,1.0} | draft-sampling ablation table |
| `candidate_width_saturation/` | DominoTree, `corr-topm` ∈ {16,32,64,128,256,0} | candidate-width saturation table |

Qwen3-4B, `max_new_tokens=2048`, `n=50` unless noted. All tables apply a
warmup-row exclusion (drop the first prompt per method), matching the reference
harness's warmup prompt.

Notes: the conditioning ablation deliberately reads `dominotree_python_builder/`
(not the default) so it holds the *builder* fixed across cond/marg and isolates
the *scorer* — `marg@16` has no GPU-native path. The
`candidate_width_saturation/` set is missing one cell (`gsm8k`, `corr-topm=32`);
the paper's saturation table uses M ∈ {16, 64, 128, 256, full}.

## Qwen3-8B

The 8B blocks (Table 1 8B at T=0 and the newer T>0 rows) are collected
separately on an RTX A6000 and are added here alongside the finalized 8B Table 1
blocks.

## Table output dirs

- `tables_gpunative/` — current tables regenerated from the GPU-native default.
- `tables/` — earlier (pre-GPU-native) output, kept for provenance.
