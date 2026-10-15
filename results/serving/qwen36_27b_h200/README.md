# Qwen3.6-27B on one H200 — raw serving data + table generators

Everything behind the paper's "scaling to a 27B hybrid-attention target" section: raw
per-prompt / per-cell measurement files, plus the scripts that re-derive every reported
number from them. This is a **separate bundle** from [`../`](../) (Qwen3-4B/8B, 2x RTX
5080 / 1x RTX 5090): different target architecture and device, with the same aggregation conventions
as the corresponding serving tables (including merging MT-Bench turns per prompt).
Speedups are normalized within each machine; absolute throughput is not a controlled
comparison across machines.

```
python3 results/serving/qwen36_27b_h200/verify_qwen36_27b.py
```

Expected output ends with **`ALL 581 CELLS REPRODUCE FROM RAW DATA.`** (exit 0). Needs
only the Python standard library — no GPU, no model weights, no LaTeX.

## Why this target needs code changes at all

Qwen3.6-27B is a **hybrid** model: 48 Gated-DeltaNet (linear-attention / recurrent)
layers plus 16 full-attention layers. DominoTree's tree verify was written against
plain-attention targets (Qwen3-4B/8B) and, unmodified, silently corrupts a hybrid
target's recurrent state: with `speculative_eagle_topk=1`, the Gated-DeltaNet layers
never see the tree's branching structure, so they verify the whole flattened tree as
one chain — every node's recurrent state absorbs the states of every earlier node in
verify order, **including rejected siblings**. This does not crash, and it does not
depress τ (see the losslessness gate below): the visible symptom is that the served
text quietly stops matching what plain autoregressive decoding (AR) would have
produced. **P9** (`sglang_dominotree/src/dominotree_sglang/{__init__,worker}.py`,
committed on this branch) fixes this by raising the GDN backend's own topk gate so its
Triton kernels walk parent pointers, switching to the dense per-node conv-window
layout, and committing each accepted node's recurrent state right after
`eagle_sample`. See `../../../sglang_dominotree/README.md`'s "Requirements the plugin
enforces" section for the full mechanism, and the P9 commit itself
(`sglang_dominotree/src/dominotree_sglang/{__init__,worker}.py` on this branch) for the
implementation.

**Every performance result in this bundle was collected with P9 enabled**, and only after the
losslessness gate (below) passed. τ alone does not certify this — see the gate result.

## Model pair

| Role | Checkpoint |
| --- | --- |
| Target | `Qwen/Qwen3.6-27B` |
| Draft (Domino chain + DominoTree) | `Huang2020/qwen3.6-27B-domino` |
| Draft (DFlash) | the model's own DFlash checkpoint (`Qwen3.6-27B-DFlash`) |
| Draft (MTP) | *(none — served from the target checkpoint's own MTP head)* |

Five methods: `ar` (τ=1 reference), `dflash`, `domino_chain`, `dominotree`, `mtp`. **No
`eagle3` arm** — no third-party EAGLE-3 head exists for this target, so `mtp` (the
checkpoint's own multi-token-prediction head, served with SGLang's NEXTN algorithm) is
the draft-head baseline instead.

## Layout

```
qwen36_27b_h200/
  PROVENANCE.txt              full collection record: dates, flags, config decisions
  PUBLISHED.json              values recomputed from raw, cross-checked against the paper
  MANIFEST.sha256             integrity digest of every raw data file below
  verify_qwen36_27b.py        recompute every cell, diff vs PUBLISHED.json (main entry point)
  gen_qwen36_27b_tables.py    regenerate the 27B-only tables from raw data (see below)

  bs1/T<temp>/<method>/<dataset>_T<temp>.jsonl        single-request serving
  helmet/<method>/helmet.jsonl (+ .prompts.jsonl)      long-context (HELMET Summarization)
  concurrency/<method>/{<method>.jsonl,.md,admission_cap.txt}
  helmet_budget/dominotree_b{15,32,64}/helmet.jsonl (+ .prompts.jsonl)
                                      HELMET tree-budget sweep, one session (2026-09-26)
  ablations/budget_ladder_m128/       DominoTree at tree budget {8,12,15,32,64}, all at
                                      node_topk 8 / corr_topm 128, one session (2026-09-26)
  ablations/budget_ladder/            the earlier budget runs (b15/b32/b64 at corr_topm 64,
                                      b8/b12 at 128); b15 is the width sweep's (k,M)=(8,64) cell
  ablations/node_topk_topm_grid/      node_topk x topM grid at budget=15
  ablations/timing/                   per-phase step timing at budget {8,15,32,64}
  losslessness_gate/                  the P9 gate run (chain vs. tree vs. P9-off control)
  PROVENANCE_RAW/                     machine-generated config/env records, verbatim
```

Server/driver logs are **not** shipped here (this repo's `.gitignore` excludes
`logs/`/`*.log` everywhere, and the Qwen3-4B/8B bundle ships none either); the flags
they'd document are already in `PROVENANCE_RAW/` and `PROVENANCE.txt`.

`<method>` ∈ {`ar`, `dflash`, `domino_chain`, `dominotree`, `mtp`}. DominoTree is always
**tree budget 15, node_topk 8, corr_topm 128**, GRU table on — this device's own measured
optimum (see `ablations/`), different from the Qwen3-4B/8B bundle's budget 16/32.

## Which file backs which table

| Paper table | Data | Command |
| --- | --- | --- |
| bs=1 serving, Qwen3.6-27B rows (`tab:sglang-bs1`) | `bs1/` | `../gen_sglang_bs1_table.py` |
| Concurrency goodput, Qwen3.6-27B rows (`tab:sglang-conc`) | `concurrency/` | `gen_qwen36_27b_tables.py conc_main` |
| Long context, Qwen3.6-27B columns (`tab:sglang-longctx`) | `helmet/` | `../gen_sglang_longctx_table.py` |
| Long-context tree budget, Qwen3.6-27B row (`tab:longctx-budget`) | `helmet_budget/` | `../gen_longctx_budget_table.py` |
| HELMET per task + paired CIs, Qwen3.6-27B block (`tab:app-helmet`) | `helmet/` | `../gen_appendix_5090.py` |
| Concurrency per dataset (`tab:app-conc-27b`) | `concurrency/` | `gen_qwen36_27b_tables.py conc` |
| Tree-budget sweep + per-step timing + width sweep (`tab:app-qwen27b-ablation`) | `ablations/` | `gen_qwen36_27b_tables.py ablation` |
| Numbers quoted in the 27B section's text | all of the above + `losslessness_gate/` | `gen_qwen36_27b_tables.py evidence` |

The concurrency tables report c = 2 to 32; the raw files also hold c = 1, which the paper
does not report. The tuning table's budget sweep uses `ablations/budget_ladder_m128/` (every
budget at the same width, one session); its width sweep and timing panels use the earlier
runs listed above.

The verifier independently recomputes every Qwen3.6-27B cell of these tables (including CI
endpoints, speedups, Overall values, tuning, timing and the gate numbers), checks them
against `PUBLISHED.json`, and parses each generator's output to check the printed cells.
To also compare the 27B cells of a copy of the paper's `tables/` directory (report only):

```
python3 verify_qwen36_27b.py --paper-dir /path/to/domino_tree
```

This is **one script with subcommands**, not five standalone files like `../gen_*.py`.
That deviation is deliberate: the loaders for bs=1 / HELMET / concurrency cells are
identical across all five tables here, and duplicating ~150 lines of loader logic five
times over one collection campaign risked the tables silently drifting apart. Every
table is still a single command that reads only raw JSONL and prints LaTeX — the
property that actually matters — and `verify_qwen36_27b.py` re-derives the same cells
independently (its own copies of the loader functions, not an import of this file), so
a bug in one script's convention would surface as a mismatch against the other on the
same raw data.

## Conventions, stated so you can check we describe what we do

This bundle sits next to the Qwen3-4B/8B serving tables in the same paper, so it uses
the **same estimands as those tables** — not a convention of its own. Every formula
below is checked, in code, against the script that defines the Qwen3-4B/8B version of
the same table (see the cross-references in `gen_qwen36_27b_tables.py`'s docstring).

- **bs=1.** PER-PROMPT, turns merged — exactly `../gen_sglang_bs1_table.py`'s `cell()` /
  `../verify_published_numbers.py`'s `bs1_cell()` (the estimand behind Tables 5/6). A
  prompt's TPS is its summed output tokens over its summed decode time (MT-Bench sums
  both turns before dividing — a turn is not its own sampling unit); its τ is MACRO
  within the prompt (turn-length-weighted). A dataset cell is the mean of that
  per-prompt value over prompts; Overall is the unweighted mean over the 8 datasets.
  Using a flat per-row mean instead (treating each MT-Bench turn as its own unit) would
  shift that one column by roughly 1–1.5% and put it on a different estimand from every
  other serving table in the paper — this bundle does not do that.
- **HELMET.** Point estimate matches `../verify_published_numbers.py`'s
  `helmet_cell_stats` (Tables 7/8): a cell's TPS is `sum(output_tokens) /
  sum(decode_time)` over its 50 prompts (aggregate, not mean-of-per-prompt); τ is the
  mean per-prompt accepted length; speedup is that cell's TPS over AR's TPS at the same
  (task, length_bin); same convention as `../gen_sglang_longctx_table.py`. The paired
  bootstrap CI (DominoTree vs. the chain) instead matches `../gen_appendix_5090.py`'s
  `paired_ci` / `../verify_r4_longctx.py`'s `paired()`: it resamples the per-prompt
  `tps`/`accept` fields directly (mean-of-per-prompt-ratios) — a deliberately different
  estimand from the point estimate, which both published scripts also use; not an
  inconsistency introduced here.
- **Concurrency.** Goodput at one (dataset, concurrency) cell is
  `sum(completion_tokens) / sum(wall_s)`; Overall is the **unweighted** mean over
  datasets, matching `../verify_published_numbers.py`'s `check_conc` (confirmed against
  `../aggregate_concurrency.py`, which reports both an unweighted and a prompt-
  count-weighted mean and states the unweighted one is what the paper uses). τ's
  Overall is the unweighted mean of `mean_accept` over all 18 (dataset, concurrency)
  cells — the MACRO estimand. `../verify_r3_conc.py` documents that this axis has no
  per-prompt sidecar, so the alternative (verify-count-weighted "micro" τ) cannot be
  computed and is an open question there, not resolved in either direction; the same
  posture applies here — macro is what's reported, not a claim that it dominates micro.
- **DominoTree's config is fixed across every axis in this bundle** (budget 15,
  node_topk 8, topM 128) — the ladder + grid search that chose it (`ablations/`) ran
  *before* the main bs=1/helmet/concurrency collection; see `PROVENANCE.txt` for the
  timestamped decision.

## Coverage caveats — read before concluding something is missing

1. **No `eagle3` arm anywhere in this bundle** — see "Model pair" above.
2. **`ar` has no `admission_cap.txt`-driven speculative acceptance field** —
   `mean_accept` is `null` in its concurrency rows and `1.0` is used wherever a τ is
   required (matching the Qwen3-4B/8B convention).
3. **The node_topk × topM grid is not a full 3×4 factorial** — 11 of 12 combinations
   were run; `node_topk=8, topM=64` is covered by the budget ladder's own "default" row
   instead of being duplicated in the grid (see `PROVENANCE.txt`).
4. **The budget ladder mixes two node_topk/topM configurations** — budgets {15,32,64}
   were run at the pre-grid default (topM=64); budgets {8,12} were added after the grid
   picked topM=128. Both are P9-gated PASS individually; do not read the ladder as a
   single controlled sweep over budget alone. `gen_qwen36_27b_tables.py ablation`
   prints each row's own topM so this is visible in the output, not just in prose.

## Losslessness gate (why this bundle can be trusted at all)

`losslessness_gate/20260923T171916Z_356641/` compares, token-by-token, plain AR against
the Domino chain (reference), DominoTree with P9 on (candidate), and DominoTree with P9
**off** (`dominotree_flat`, the negative control — the exact bug P9 fixes, diagnostic
only, never used for a throughput number). Result: reference median agreement with AR
= 333 tokens, candidate = 306, **control = 46** — while the control's τ (9.58) is
almost the same as the candidate's (9.59). This is the whole point:
**τ cannot certify hybrid-target correctness**; only a token-level AR diff can. Every
ablation cell in this bundle carries its own such gate (`.verdict.txt` files in
`ablations/`), and every one is `P9_GATE=PASS`. The diagnostic gate above ran on **two H100 GPUs
with tensor parallelism**, with the GRU input table off and top-M 64. Each H200
configuration subsequently passed the same pre-registered gate against H200
references, using its run's flat-control arm. Passing is a relative check against
the chain with a negative control, not proof of byte-identical outputs.

## Reproducing the raw data yourself

Clean, generic (non-cluster-specific) launch/run scripts for the five methods, the
losslessness gate, and HELMET prompt prep for this target's tokenizer are under
[`../../../sglang_dominotree/benchmarks/qwen36_27b/`](../../../sglang_dominotree/benchmarks/qwen36_27b/).
They are **not** the SLURM scripts that originally collected this data (those are
cluster-specific and stay in the working repo) — same serving flags and env vars,
rewritten for a generic single-GPU host with per-run ports and PID-scoped server kills.

## Known limitation, out of scope for this bundle

The GPU-native (frontier) tree builder's Python fallback path
(`_build_conditional_trees`) raises `IndexError` at tree budget > 16 when the frontier
builder is unavailable. Every measurement in this bundle used the frontier builder (the
default), so it is never hit here — noted for anyone who disables it and reruns at a
larger budget. Not fixed in this branch; tracked as a known limitation, not a
regression this bundle introduces.
