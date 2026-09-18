# DominoTree results — raw data and regenerated tables

Evidence bundle backing the DominoTree paper. Raw per-prompt records live under
`raw/`; regenerate the paper tables with:

```
python make_latex_table.py --raw-dir results/raw --out-dir results/tables_gpunative
```

## Table 1's headline node budget changed in the IEEE Access revision

**The rest of this file describes the ORIGINAL (arXiv) Table 1**, headlined at node
budget 16 (both model sizes), reproduced by `make_latex_table.py` above. **The current
paper's Table 1** (`tab:main`) uses a wider headline budget at both sizes — **32 at
Qwen3-4B, 128 at Qwen3-8B** — following further tree-builder work after arXiv (the fused
GPU-native builder; see `docs/domino_tree_ieee_access/` in the research repo). Both
pipelines are kept: the arXiv one is still exact and citable for that version of the
paper, and the current one supersedes it for anything printed in the IEEE Access
manuscript. To reproduce the CURRENT Table 1:

```bash
python gen_table1.py --out-dir results/table1_audit
```

Raw sources (new, added alongside the arXiv-era `raw/` directories below):

| dir | contents | backs |
| --- | --- | --- |
| `raw/tab1f_4b/` | `ar` + `dominotree@{16,32,64}`, GPU-native fused builder, Qwen3-4B | current Table 1 4B (DominoTree row, headline @32), the budget-ablation appendix (@16/@32/@64) |
| `raw/tab1_8b/` | `ar` + `dominotree@128`, GPU-native builder, Qwen3-8B, RTX A6000 | current Table 1 8B (DominoTree row, headline @128) |
| `raw/ddtree_b32_4b/` | `baseline` + `ddtree_tb32`, portable export of the official CaDDTree harness cache, Qwen3-4B | current Table 1 4B DDTree row and paired CIs |
| `raw/ddtree_b128_8b/` | `baseline` + `ddtree_tb128`, portable export of the official CaDDTree harness cache, Qwen3-8B | current Table 1 8B DDTree row and paired CIs |

`gen_table1.py` requires **numpy only** and reads, besides the directories above:
`raw/baseline_ddtree_caddtree/` (4B AR/DFlash/CaDDTree),
`raw/8b/ref8b_perprompt_jsonl/` (8B AR/DFlash/CaDDTree), `raw/domino_official/qwen3-4b/`
and `raw/8b/domino_official/qwen3-8b/` (official Domino, both sizes), and
`raw/conditioning_ladder/matched/` (the conditioning-decomposition appendix) — all
already present in this repo.

The DDTree exports retain one JSONL row for each prompt/turn and each of the two arms,
including `time_per_output_token`, all per-round acceptance lengths, output length, and
the checked protocol fields. They intentionally omit output tensors, prompts, and
harness-local paths. Thus the command above is the complete one-command, dependency-free
reproduction of the current Table 1; it fails loudly on a missing, duplicate, short, or
protocol-mismatched cell. The private pickle fallback in `gen_table1.py` is opt-in only
and is not needed for public reproduction.

## Convention: the GPU-native builder is the default

DominoTree's default tree builder is the **GPU-native CUDA-graph builder**. The
Python reference builder is kept **only as a comparison** — it produces
bit-identical trees (same tokens, same accepted length), so it isolates _build
cost_ alone (see the paper's "Python vs. GPU-native builder" section). Every
main result (Table 1, pairwise, stage-time) uses the GPU-native builder.

- `raw/dominotree/` — **DEFAULT.** DominoTree built with the GPU-native builder
  (records carry `gpu_native_build: true`).
- `raw/dominotree_python_builder/` — the Python reference builder (identical
  trees; `gpu_native_build` absent). Comparison only.

## `raw/` → paper tables

| dir                           | method / builder                                                              | backs                                                                      |
| ----------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `dominotree/`                 | DominoTree, **GPU-native (default)**                                          | Table 1 (Domino/DominoTree rows), pairwise, stage-time                     |
| `dominotree_python_builder/`  | DominoTree, Python builder                                                    | Python-vs-GPU-native builder comparison; conditioning ablation (cond side) |
| `baseline_ddtree_caddtree/`   | AR / DFlash / DDTree / CaDDTree — official reference harness on native DFlash | Table 1 baseline rows, pairwise                                            |
| `chain_stage_timing/`         | Domino-chain, instrumented stage split                                        | stage-time table (chain row)                                               |
| `conditioning_ablation/`      | marginal-tree `marg@16`, Python builder                                       | conditioning ablation (marg side)                                          |
| `draft_sampling_ablation/`    | DominoTree + Domino-chain, greedy vs. sampled draft, T∈{0.5,1.0}              | draft-sampling ablation table                                              |
| `candidate_width_saturation/` | DominoTree, `corr-topm` ∈ {16,32,64,128,256,0}                                | candidate-width saturation table                                           |

Qwen3-4B, `max_new_tokens=2048`, `n=50` unless noted. All tables apply a
warmup-row exclusion (drop the first prompt per method), matching the reference
harness's warmup prompt.

Notes: the conditioning ablation deliberately reads `dominotree_python_builder/`
(not the default) so it holds the _builder_ fixed across cond/marg and isolates
the _scorer_ — `marg@16` has no GPU-native path. The
`candidate_width_saturation/` set is missing one cell (`gsm8k`, `corr-topm=32`);
the paper's saturation table uses M ∈ {16, 64, 128, 256, full}.

## Qwen3-8B

The 8B blocks (Table 1 8B at every temperature, and the pairwise CIs) were
collected separately on an RTX A6000 — same GPU-native DominoTree builder, same
benchmark SOP as 4B. They regenerate with the same `make_latex_table.py` as the
4B tables, pointed at `results/raw/8b/` (command below).

### Data layout — `results/raw/8b/`

The generator reads the three **unified** directories, which follow the same
`raw/<dir>/<dataset>_T<temp>.jsonl` convention as 4B:

| dir                          | contents                                                          | backs                                              |
| ---------------------------- | ----------------------------------------------------------------- | -------------------------------------------------- |
| `dominotree/`                | `ar` + `dominotree@16`, GPU-native builder, 8 datasets × 3 temps  | Table 1 8B (AR + DominoTree rows), pairwise        |
| `baseline_ddtree_caddtree/`  | per-prompt DFlash / DDTree@16 / CaDDTree / AR, reference harness   | Table 1 8B baseline rows, pairwise vs. baselines   |
| `domino_official/qwen3-8b/`  | official released Domino decoder, `graph_`/`eager_` per dataset    | Table 1 8B Domino row, pairwise vs. Domino         |

The remaining directories are the **original collections** those three were
exported from, kept as provenance:

| dir                                           | contents                                                                                                                                                                 | backs                                                                                                             |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `recollect_gpunative_8b_t0_20260707/`         | GPU-native `dominotree@16` + `chain`, T=0                                                                                                                                | Table 1 8B (Domino/DominoTree rows, T=0), pairwise T=0                                                            |
| `recollect_gpunative_8b_2048_20260707/`       | GPU-native `dominotree@16` + `chain`, T=0.5/T=1.0                                                                                                                        | Table 1 8B (Domino/DominoTree rows, T>0), pairwise T>0                                                            |
| `collect_8b_2048_20260704/our/`               | `ar` (+ Python-builder `cond@16`, comparison only), T=0                                                                                                                  | AR baseline for our-side speedup normalization, all temps (AR is temperature-independent to <1%, reused from T=0) |
| `collect_8b_2048_20260704/baseline_official/` | `*.pt.summary.json` + `MANIFEST.json` for official-harness AR/DFlash/DDTree@16/CaDDTree, T=0 (no `*.pt` tensors — too large, and unneeded: the script only reads `rows`) | Table 1 8B baseline rows (DFlash/DDTree/CaDDTree), T=0                                                            |
| `reference_8b_tgt0_20260707/`                 | same, T=0.5/T=1.0                                                                                                                                                        | Table 1 8B baseline rows, T>0                                                                                     |
| `ref8b_perprompt_jsonl/`                      | per-prompt `dflash`/`ddtree_tb16`/`caddtree`/`baseline` records, all temps                                                                                               | pairwise CIs vs. DFlash/DDTree@16/CaDDTree                                                                        |

The `*.pt.summary.json` / `MANIFEST.json` files had their container-local
paths (e.g. `/mnt/zhiqi/...`) stripped/redacted to basenames on import — only
path fields were touched, all `rows` (the numeric results) are copied
byte-for-byte from the source collection.

### Run command

From the repo root — one invocation writes both 8B tables:

```bash
python make_latex_table.py --raw-dir results/raw/8b --domino-model-dir qwen3-8b \
  --domino-no-warmup --no-warmup-drop --model-label Qwen3-8B --table-suffix _8b \
  --out-dir results/tables_gpunative
```

Outputs `tables_gpunative/table1_8b.md`, `table1_cells_8b.csv`,
`pairwise_ci_8b.md`, and `pairwise_ci_8b.csv`. The two warmup flags encode the 8B
conventions below: `--no-warmup-drop` because our 8B collection ran with an in-loop
warmup prompt, `--domino-no-warmup` because the official Domino 8B collection did
not. (Earlier revisions used two standalone scripts, `build_8b_v2.py` and
`build_8b_pairwise.py`; they were retired once the 8B data was re-exported into the
unified layout and `make_latex_table.py` absorbed the 8B path.)

### Convention notes (differ from the 4B convention above)

- **No warmup-row trim on our rows** (`--no-warmup-drop`). Our 8B collection ran
  on `benchmark.py`'s warmup-enabled harness (an in-loop "Warmup" prompt heats
  kernels/caches before any prompt is timed), so — unlike the 4B tables, which
  drop the first prompt per method — every measured 8B prompt is already warm and
  the generator takes a plain mean over all of them. This matches the
  DFlash/DDTree/CaDDTree reference SOP and the reference rows (which are
  all-prompt aggregates).
- **Warmup-row trim on the official-Domino row** (`--domino-no-warmup`). The
  official Domino 8B collection ran *without* a warmup prompt (the 4B one had
  it), so its first prompt is dropped instead. Opposite flag, same goal: every
  row is a mean over warm prompts.
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

The command above overwrites the four checked-in 8B table files. Re-running it reproduces
the two CSVs **byte for byte**, which is the sharpest check available — including the
paired-bootstrap CIs, whose resampling is seeded (`--seed`):

```bash
cmp results/tables_gpunative/table1_cells_8b.csv  <regenerated>/table1_cells_8b.csv
cmp results/tables_gpunative/pairwise_ci_8b.csv   <regenerated>/pairwise_ci_8b.csv
```

The two `.md` files reproduce identically **cell for cell**, but not byte for byte: the
checked-in copies were run through a markdown formatter that pads table columns to a common
width, so a plain `diff` shows whitespace noise. Compare the CSVs, or diff the `.md` files
with cells stripped.

Below is the Overall column of each, so a mismatch is visible without regenerating anything.

**Table 1 (8B), Overall Avg — `speedup / tau`.** The Domino row is the official released
decoder, `best-of(graph, eager)`, normalized by the lean common AR (see the repo
AR-normalization note); DominoTree is normalized by its own harness's AR.

| Temp | DominoTree (16) | Domino (official) |
| ---- | --------------- | ----------------- |
| 0.0  | 5.71 / 8.09     | 5.50 / 7.33       |
| 0.5  | 5.34 / 7.61     | 5.08 / 6.83       |
| 1.0  | 4.64 / 6.61     | 4.38 / 5.78       |

**Pairwise 95% paired-bootstrap CI, Overall — DominoTree (16) vs. baseline.** The two
comparisons use different metrics because only Domino shares our harness: vs. Domino is raw
per-prompt TPS under the shared lean common AR (N=421), vs. DDTree@16 is
speedup-over-own-AR across harnesses (N=430).

| Temp | vs Domino (official)   | vs DDTree@16             |
| ---- | ---------------------- | ------------------------ |
| 0.0  | +4.32% [3.03, 5.67]    | +23.96% [21.93, 26.04]   |
| 0.5  | +5.98% [4.24, 7.76]    | +0.99% [-1.03, 3.04]     |
| 1.0  | +5.95% [3.77, 8.18]    | -3.50% [-5.39, -1.64]    |

The T>0 DDTree@16 rows are the honest negative reported in the paper: at 8B the tree's
tau advantage holds at every temperature, but its throughput advantage over DDTree does not
survive sampling.

## Table output dirs

- `tables_gpunative/` — current tables regenerated from the GPU-native default.
- `tables/` — earlier (pre-GPU-native) output, kept for provenance.
