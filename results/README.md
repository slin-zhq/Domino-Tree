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

The 8B blocks (Table 1 8B at every temperature, and the pairwise CIs) are
collected separately on an RTX A6000 and regenerated from
`results/raw/8b/` by two standalone scripts at the repo root
(`build_8b_v2.py`, `build_8b_pairwise.py`) — kept separate from
`make_latex_table.py` because the 8B collection used a different (older,
per-directory-glob) harness layout than the unified 4B `raw/<dir>/*.jsonl`
convention. Same GPU-native DominoTree builder, same benchmark SOP.

### Data layout — `results/raw/8b/`

| dir | contents | backs |
|---|---|---|
| `recollect_gpunative_8b_t0_20260707/` | GPU-native `dominotree@16` + `chain`, T=0 | Table 1 8B (Domino/DominoTree rows, T=0), pairwise T=0 |
| `recollect_gpunative_8b_2048_20260707/` | GPU-native `dominotree@16` + `chain`, T=0.5/T=1.0 | Table 1 8B (Domino/DominoTree rows, T>0), pairwise T>0 |
| `collect_8b_2048_20260704/our/` | `ar` (+ Python-builder `cond@16`, comparison only), T=0 | AR baseline for our-side speedup normalization, all temps (AR is temperature-independent to <1%, reused from T=0) |
| `collect_8b_2048_20260704/baseline_official/` | `*.pt.summary.json` + `MANIFEST.json` for official-harness AR/DFlash/DDTree@16/CaDDTree, T=0 (no `*.pt` tensors — too large, and unneeded: the script only reads `rows`) | Table 1 8B baseline rows (DFlash/DDTree/CaDDTree), T=0 |
| `reference_8b_tgt0_20260707/` | same, T=0.5/T=1.0 | Table 1 8B baseline rows, T>0 |
| `ref8b_perprompt_jsonl/` | per-prompt `dflash`/`ddtree_tb16`/`caddtree`/`baseline` records, all temps | pairwise CIs vs. DFlash/DDTree@16/CaDDTree |

The `*.pt.summary.json` / `MANIFEST.json` files had their container-local
paths (e.g. `/mnt/zhiqi/...`) stripped/redacted to basenames on import — only
path fields were touched, all `rows` (the numeric results) are copied
byte-for-byte from the source collection.

### Generator scripts + run commands

```bash
python3 build_8b_v2.py        # Table 1 8B, all temps -> paste-ready LaTeX rows on stdout
python3 build_8b_pairwise.py  # pairwise 95% paired-bootstrap CIs, all temps -> stdout
```

Both scripts read `results/raw/8b/` by relative path, so run them from the
repo root. Regenerated output is saved at `tables_gpunative/table1_8b.md` and
`tables_gpunative/pairwise_8b.md`.

### Convention notes (differ from the 4B convention above)

- **No warmup-row trim.** The 8B collection runs on `benchmark.py`'s
  warmup-enabled harness (an in-loop "Warmup" prompt heats kernels/caches
  before any prompt is timed), so — unlike the 4B tables, which drop the
  first prompt per method — every measured 8B prompt is already warm and
  every script takes a plain mean over all of them. This matches the
  DFlash/DDTree/CaDDTree reference SOP and the reference rows (which are
  all-prompt aggregates).
- **AR reused from T=0.** AR throughput is builder-independent and
  temperature-independent to <1%, so the T=0 AR collection
  (`collect_8b_2048_20260704/our/*ar*` on our side; the `baseline` row in
  `*.pt.summary.json` / `ref8b_perprompt_jsonl` on the reference side) is
  reused as the speedup denominator at T=0.5 and T=1.0 too.
- **Table 1 reference rows come from `.pt.summary.json`** (official CaDDTree
  harness cache summaries), not from the per-prompt jsonl.
- **Pairwise CIs come from `ref8b_perprompt_jsonl`** (per-prompt records),
  since paired bootstrap needs prompt-indexed pairs, not aggregate summaries.
- The default tree builder is still the **GPU-native builder**
  (`recollect_gpunative_8b_*`); the Python-builder `cond@16` collection
  (`collect_8b_2048_20260704/our/`) is read only for its `ar` rows here — it
  is not used for the 8B Domino/DominoTree Table 1 rows.

### Validation (regenerated numbers match the paper to 2 d.p.)

Table 1, Overall column:

| Temp | DominoTree (16) speedup/tau | Domino-chain speedup/tau |
|---|---|---|
| 0.0 | 5.71 / 8.09 | 4.96 / 7.32 |
| 0.5 | 5.34 / 7.61 | — |
| 1.0 | 4.64 / 6.61 | — |

Pairwise, Overall column, DominoTree (16) vs.:

| Temp | vs DDTree@16 | vs Domino-chain |
|---|---|---|
| 0.0 | +23.53% | +14.96% |
| 0.5 | +0.77% | — |
| 1.0 | -3.79% | — |

## Table output dirs

- `tables_gpunative/` — current tables regenerated from the GPU-native default.
- `tables/` — earlier (pre-GPU-native) output, kept for provenance.
