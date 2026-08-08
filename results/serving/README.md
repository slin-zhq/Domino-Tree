# Raw serving results — every number in the paper's SGLang section

This directory holds the **raw measurement files** behind the DominoTree v2 paper's three
serving axes, plus the scripts to re-derive every published cell from them. Nothing here
is a summary you have to trust: each table cell is recomputed from per-prompt or per-cell
JSONL, and one command checks all 88 of them.

```
python3 results/serving/verify_published_numbers.py
```

Expected output: **`ALL CELLS REPRODUCE FROM RAW DATA.`** (exit 0). It needs only the
Python standard library — no GPU, no model weights, no LaTeX.

---

## Layout

```
results/serving/
  PUBLISHED.json                     the values exactly as printed in the paper
  MANIFEST.sha256                    integrity digest of every file below
  verify_published_numbers.py        recompute all 88 cells, diff vs PUBLISHED.json
  aggregate_helmet.py                long-context tables + paired-bootstrap CIs
  aggregate_concurrency.py           concurrency "Overall" rollup

  bs1/<size>/<method>/<dataset>_T<temp>.jsonl
  concurrency/<size>/<method>/<method>.jsonl        + status_<method>.done
  longcontext/<size>/<method>/helmet_<size>.jsonl   + .prompts.jsonl
                              PROVENANCE.txt
```

`<size>` ∈ {`4b`, `8b`} · `<method>` ∈ {`ar`, `eagle3`, `dflash`, `domino_chain`,
`dominotree`}.

**Per-cell vs per-prompt.** `*.jsonl` holds one aggregate record per measured cell.
`*.prompts.jsonl` holds one record per individual prompt — that is what the paired
bootstrap resamples, and it is why the confidence intervals in the paper are checkable
rather than merely quoted. Long context ships both.

## Which file backs which table

| Paper table                         | Data                   | Aggregator                                                               |
| ----------------------------------- | ---------------------- | ------------------------------------------------------------------------ |
| bs=1 Overall + per-dataset appendix | `bs1/{4b,8b}/`         | `benchmark.py` conventions; per-cell `tps` / `mean_accept` in each JSONL |
| Concurrency goodput + appendix      | `concurrency/{4b,8b}/` | `aggregate_concurrency.py <dir> --label 4B`                              |
| Long context + appendix             | `longcontext/{4b,8b}/` | `aggregate_helmet.py --root <dir> --model qwen3-4b`                      |

Reproduce the published long-context table:

```bash
python3 results/serving/aggregate_helmet.py \
  --root results/serving/longcontext/8b --model qwen3-8b
```

Reproduce the concurrency rollup (it prints **both** an unweighted and a
prompt-count-weighted dataset mean, so you can confirm which convention the paper uses —
it is the **unweighted** one):

```bash
python3 results/serving/aggregate_concurrency.py results/serving/concurrency/8b --label 8B
```

## Conventions, stated so you can check we describe what we do

- **Long context.** A table cell averages the two HELMET summarization tasks
  (∞Bench-Sum, Multi-LexSum). τ is the mean of the two task means. Speedup is the mean
  of the two **per-task ratios** — mean-of-ratios, not ratio-of-means.
- **Concurrency.** Goodput at offered concurrency `c` is the **unweighted** mean over the
  three datasets. τ is the unweighted mean over datasets **and over the full measured
  sweep** `c = 1,2,4,8,16,32` (τ is flat in `c`, drifting < 0.1).
- **`c` is offered concurrency, not batch size.** Each server runs at its own
  `--max-running-requests` cap and queues the rest, so a column compares like with like
  only where `c` ≤ **every** method's cap. The caps are recorded in
  `concurrency/<size>/<method>/status_<method>.done` and printed by the verifier:

  |     | AR  | EAGLE-3 | DFlash | Domino chain | DominoTree |
  | --- | --- | ------- | ------ | ------------ | ---------- |
  | 4B  | 32  | 32      | 32     | 16           | 12         |
  | 8B  | 32  | 16      | 16     | 16           | 8          |

  Matched-admission region = `c ≤ 8`. That is why the paper's main table stops there and
  the full sweep to `c = 32` sits in the appendix, labeled as an admission ceiling.

## Coverage caveats — read these before concluding something is missing

These are real gaps in the grid. All are disclosed in the paper; they are listed here so
the file counts make sense.

1. **8B AR was measured at `T=0` only** and reused as the temperature-independent
   normalizer. AR carries no draft model, so its throughput is the same denominator at
   every temperature. Hence `bs1/8b/ar/` has 8 files, not 24.
2. **LiveCodeBench is absent from 8B** for `domino_chain` and `dominotree` (21 files each,
   not 24): its long prompts exceed the 16 GB-card KV budget at TP=2. The 8B tables omit
   that column entirely rather than reporting a partial row.
3. **4B long context stops at 16K**, 8B at 32K. Card memory, not method — a 32K prompt
   already exceeds the 4B KV pool (21,635 tokens at the published config).

## Provenance

`longcontext/{4b,8b}/PROVENANCE.txt` and `concurrency/8b/PROVENANCE.txt` record the
collection timestamp, the SGLang version, and the serving configuration.

**These sweeps were recollected on 2026-07-29 with SGLang's stock prefill CUDA graph**
(`--mem-fraction-static 0.80`, no `--cuda-graph-backend-prefill` flag). An earlier pass had
disabled the prefill graph — an opt-out we needed because the Domino chain hit an OOM with
it enabled at `0.85`. That flag's measured effect is ~0% (mean paired residual −0.06%), and
it biased results _against_ us rather than for us, since throughput is measured over the
full request round trip so prefill time sits in both sides of every ratio. We recollected
anyway so that every published number is the **stock** configuration.

`MANIFEST.sha256` digests **every data file** in this directory (the scripts and
`PUBLISHED.json` are versioned in git instead), so you can confirm the measurements are
byte-for-byte what was published:

```bash
cd results/serving && shasum -a 256 -c MANIFEST.sha256 | grep -v ': OK$'   # silence = all match
```

## Directory-name normalization (disclosed, not silent)

The three axes were collected at different times with different naming. The layout above
is uniform; two 4B bs=1 directories were renamed on export:

| Original collection dir        | Published as          |
| ------------------------------ | --------------------- |
| `bs1/chain_cudagraph`          | `bs1/4b/domino_chain` |
| `bs1/dominotree_frontiergraph` | `bs1/4b/dominotree`   |

The suffixes recorded the CUDA-graph configuration at a time when it was still being
varied; both are the final published configuration, in which **every** method runs at its
own CUDA-graph best. File contents are unmodified — `MANIFEST.sha256` is computed over the
bytes as shipped.

---

_The served Domino chain is the official Domino drafter running through **our** plugin,
not the released Domino fork's own serving path — see the paper's Limitations. That makes
it the tightest available tree-vs-chain control (identical weights, engine, and flags; only
the algorithm differs) and we state it plainly rather than implying a fork comparison._
