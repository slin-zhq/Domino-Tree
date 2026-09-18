# Raw serving results — every number in the paper's SGLang section

This directory holds the **raw measurement files** behind the DominoTree (IEEE Access)
paper's three serving axes, plus the scripts to re-derive every published cell from them.
Nothing here is a summary you have to trust: each table cell is recomputed from
per-prompt or per-cell JSONL, and one command checks all of them.

```
python3 results/serving/verify_published_numbers.py
```

Expected output: **`ALL 1109 CELLS REPRODUCE FROM RAW DATA.`** (exit 0). It needs only
the Python standard library — no GPU, no model weights, no LaTeX.

All measurements in this directory are from a **single RTX 5090 (32 GB), TP=1**, for
**both** Qwen3-4B and Qwen3-8B — no tensor parallelism, and one card for both model
sizes. This **replaces** an earlier bundle collected on 2x RTX 5080 (16 GB each, TP=2);
see "What changed from the 5080 bundle" below.

---

## Layout

```
results/serving/
  PUBLISHED.json                     the values recomputed from raw, cross-checked against the paper
  MANIFEST.sha256                    integrity digest of every data file below
  verify_published_numbers.py        recompute every cell, diff vs PUBLISHED.json (the main entry point)
  verify_r2_5090.py                  bs=1: full per-dataset TPS/tau + paired CIs, printed as markdown
  verify_r3_conc.py                  concurrency: completeness + macro/micro acceptance audit
  verify_r4_longctx.py               long context (default budget arms): aggregate-vs-raw audit + CIs
  ci_r4_8b_b32.py                    long context: Qwen3-8B budget-32 arm vs the chain (the table's arm)
  gen_appendix_5090.py               regenerate tab:app-conc / tab:app-helmet / tab:app-helmet-8b
  aggregate_helmet.py                long-context report + paired-bootstrap CIs (general purpose)
  aggregate_concurrency.py           concurrency "Overall" rollup (general purpose)

  bs1/<size>/<method>/<dataset>_T<temp>.jsonl
  concurrency/<size>/<method>/<method>.jsonl        + caps.env
  longcontext/<size>/<method>/helmet.jsonl          + helmet.prompts.jsonl
  longcontext/<size>/dominotree_b32/                DominoTree at tree budget 32 (sweep; 8B's MAIN-table arm)
  longcontext/8b/dominotree_b64/                    DominoTree at tree budget 64, Qwen3-8B only (sweep)
```

`<size>` ∈ {`4b`, `8b`} · `<method>` ∈ {`ar`, `eagle3`, `dflash`, `domino_chain`,
`dominotree`}. `dominotree` in `bs1/` and `concurrency/` is always tree budget 32 (the
serving headline at both sizes). In `longcontext/`, `dominotree` is budget 16 (the
directory holding the per-task appendix's "(16)" rows and 4B's main-table column); the
budget-32/64 sweep arms are separate directories, and Qwen3-8B's **main-table**
DominoTree column is `dominotree_b32/`, not `dominotree/` — see
`longcontext/PROVENANCE.txt` for exactly which arm backs which table cell.

**Per-cell vs per-prompt.** `concurrency/*.jsonl` holds one aggregate record per measured
cell (no per-prompt sidecar exists for this axis — see the coverage caveats below).
`bs1/*.jsonl` and `longcontext/*/*.prompts.jsonl` hold one record per individual prompt —
that is what the paired bootstrap resamples, and it is why the confidence intervals in
the paper are checkable rather than merely quoted. `longcontext/` ships both an aggregate
(`helmet.jsonl`) and a per-prompt (`helmet.prompts.jsonl`) file per arm.

## Which file backs which table

| Paper table (label)                                              | Data                             | Script                        |
| ------------------------------------------------------------------ | --------------------------------- | ------------------------------ |
| bs=1 grid + Overall (`tab:sglang-bs1`)                              | `bs1/{4b,8b}/`                    | `verify_published_numbers.py` / `verify_r2_5090.py` |
| bs=1 paired CIs (`tab:sglang-bs1-cis`)                              | `bs1/{4b,8b}/`                    | `verify_published_numbers.py` / `verify_r2_5090.py` |
| Concurrency Overall (`tab:sglang-conc`)                             | `concurrency/{4b,8b}/`            | `verify_published_numbers.py` / `aggregate_concurrency.py` |
| Concurrency per-dataset appendix (`tab:app-conc`)                   | `concurrency/{4b,8b}/`            | `gen_appendix_5090.py`        |
| Long context main table (`tab:sglang-longctx`)                     | `longcontext/{4b,8b}/`, `longcontext/8b/dominotree_b32/` | `verify_published_numbers.py` / `verify_r4_longctx.py` / `ci_r4_8b_b32.py` |
| Long-context tree-budget sweep (`tab:longctx-budget`)               | `longcontext/*/dominotree*`       | `verify_published_numbers.py` |
| HELMET per-task appendix (`tab:app-helmet`, `tab:app-helmet-8b`)    | `longcontext/{4b,8b}/`, `longcontext/8b/dominotree_b32/` | `gen_appendix_5090.py`        |

Regenerate the appendix tables (concurrency per-dataset + HELMET per-task detail) as
LaTeX, straight from raw:

```bash
python3 results/serving/gen_appendix_5090.py
```

`aggregate_helmet.py` and `aggregate_concurrency.py` are older, general-purpose reports
(not tied to this paper's exact table shape) kept for ad hoc inspection — point them at
any `<axis>/<size>/` directory:

```bash
python3 results/serving/aggregate_helmet.py --root results/serving/longcontext/8b --model qwen3-8b
python3 results/serving/aggregate_concurrency.py results/serving/concurrency/8b --label 8B
```

## Conventions, stated so you can check we describe what we do

- **bs=1.** Per-prompt TPS sums a prompt's turns (MT-Bench: both turns) before dividing;
  per-prompt tau is MACRO within the prompt (token-weighted across turns). A cell's
  reported value is the MACRO mean over prompts. Overall is the unweighted mean over the
  8 datasets. DominoTree runs at tree budget 32, verifying 33 tokens/round; the chain
  drafts 15 (16 verify slots).
- **Long context.** A table cell averages the two HELMET summarization tasks
  (∞Bench-Sum, Multi-LexSum). τ is the mean of the two task means. Speedup is the mean
  of the two **per-task ratios** — mean-of-ratios, not ratio-of-means. Qwen3-4B's
  main-table DominoTree column is tree budget 16; Qwen3-8B's is budget 32 (each model's
  measured long-context optimum up to 16K — see `tab:longctx-budget` and
  `longcontext/<size>/dominotree_b32/`).
- **Concurrency.** Goodput at offered concurrency `c` is the **unweighted** mean over the
  three datasets (gsm8k, mbpp, mt-bench). τ is the unweighted mean over datasets **and
  over the full measured sweep** `c = 2,4,8,16,32` (τ is flat in `c`, drifting < 0.1).
  There is no `c=1` cell on this axis (bs=1 covers that point already).
- **`c` is offered concurrency, not batch size.** Each server runs at its own
  `--max-running-requests` cap; on this 32 GB card **every method reaches the largest
  cap probed (32) at both model sizes** — see `concurrency/<size>/caps.env` — so every
  column compares methods at an identical number of in-flight requests. This is a
  change from the 16 GB bundle (below), where DominoTree's own cap was well under 32.

## Coverage caveats — read these before concluding something is missing

1. **8B AR in `bs1/` was measured at `T=0` only** and reused as the
   temperature-independent normalizer (AR carries no draft model, so its throughput is
   the same denominator at every temperature). Hence `bs1/8b/ar/` has 8 files, not 16.
2. **Concurrency has no per-prompt sidecar.** `verify_r3_conc.py` documents this
   explicitly: only aggregate rows per (dataset, concurrency) were written, so the
   macro/micro acceptance-estimand gap on this axis is *reported*, not adjudicated.
3. **All eight datasets are present at both model sizes on this card** (unlike the
   16 GB bundle, which omitted LiveCodeBench from 8B). No partial rows on this axis.
4. **Both model sizes reach 32K long context on this card** (unlike the 16 GB bundle,
   which capped Qwen3-4B at 16K).

## Provenance

`bs1/PROVENANCE.txt`, `concurrency/PROVENANCE.txt` (+ per-size `caps.env`), and
`longcontext/PROVENANCE.txt` record the collection timestamps, SGLang version, tree
budgets, and which long-context arm backs which table cell.

`MANIFEST.sha256` digests **every data file** in this directory (the scripts and
`PUBLISHED.json` are versioned in git instead), so you can confirm the measurements are
byte-for-byte what was published:

```bash
cd results/serving && shasum -a 256 -c MANIFEST.sha256 | grep -v ': OK$'   # silence = all match
```

## What changed from the 5080 bundle

The bundle this replaces was collected on 2x RTX 5080 (16 GB each), TP=2, at a smaller
bs=1 protocol (n=20, 512-token cap) that truncated 37% of generations. The current
bundle is a full recollection on one RTX 5090 (32 GB), TP=1, at the wider n=50/2048
protocol, which also removed two card-memory artifacts of the smaller GPU: 8B's
LiveCodeBench column (previously dropped for exceeding the 16 GB KV budget at TP=2) and
Qwen3-4B's 32K long-context bin (previously exceeding the 16 GB KV pool). It also
resolved DominoTree's concurrency admission cap, previously well below the other
methods' on the 16 GB card (a batch-cap artifact, not a property of the method) — every
method now reaches cap 32 on the 32 GB card. The old bundle's own budget-ablation
directories (`bs1/4b/dominotree_b32`, `bs1/4b/ar_b32session`,
`longcontext/4b/dominotree_b15_control`) are gone; the current paper's budget-16-vs-32
serving comparison is `tab:longctx-budget`, backed by the `dominotree_b32`/`dominotree_b64`
directories under `longcontext/` in this bundle.

---

_The served Domino chain is the official Domino drafter running through **our** plugin,
not the released Domino fork's own serving path — see the paper's Limitations. That makes
it the tightest available tree-vs-chain control (identical weights, engine, and flags; only
the algorithm differs) and we state it plainly rather than implying a fork comparison._
