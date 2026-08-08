<h1 align="center">DominoTree</h1>

<p align="center">
  Official research implementation of <strong>DominoTree</strong>, a training-free
  conditional draft tree for Domino block-diffusion drafters.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.08642">arXiv:2607.08642</a>
</p>

DominoTree scores DDTree's best-first heap with node scores a factorized formulation
cannot express, by re-running Domino's GRU correction along each candidate's specific
root-to-node path. It is **training-free** on the released Domino checkpoint and
**lossless by construction** — the target verifier is unchanged, so only tokens the
target itself would have emitted are ever committed.

The same method has two entry points: [`benchmark.py`](benchmark.py), the HF research
harness behind Table 1 (offline, one request at a time, 8 datasets × 3 temperatures,
Qwen3-4B and 8B), and [`sglang_dominotree/`](sglang_dominotree/), an SGLang plugin behind
the serving tables (bs=1, concurrency goodput, HELMET long context).

```
benchmark.py, dominotree.py, dominotree_gpu.py   the HF research harness
  run_benchmark.sh, run_pipeline.sh              …its drivers
  make_latex_table.py                            …raw JSONL -> the paper's tables
sglang_dominotree/                               the SGLang plugin
  src/dominotree_sglang/                         …algorithm registration + tree builder
  benchmarks/{bs1,concurrency,helmet}/           …the three serving benchmarks
  PROVENANCE.md, verify_vendored_head.py         …copied-code manifest + copy proof
results/raw/, results/tables_gpunative/          harness raw data + derived tables
results/serving/                                 serving raw data + the no-GPU audit
demo/                                            side-by-side record-then-replay demo
```

## Verify without a GPU

Every published number can be re-derived on a laptop — no GPU, no model weights, no access
to the paper's LaTeX source. Clone the repo and run these three, in any order:

```bash
# 1. Serving tables: recompute all 88 published cells from raw per-prompt JSONL
#    and diff them against the values printed in the paper.  (stdlib only, ~0.3 s)
python3 results/serving/verify_published_numbers.py
#    expected: ALL CELLS REPRODUCE FROM RAW DATA.

# 2. Copied code: prove the vendored Domino head is byte-identical to the official
#    source at a pinned commit, modulo declared patches.  (stdlib only, ~1 s)
python3 sglang_dominotree/verify_vendored_head.py
#    expected: PASS: the vendored Domino head is byte-identical ...

# 3. Offline tables: regenerate Table 1, the pairwise CIs, and the ablation from raw
#    JSONL.  (needs `pip install -r requirements.txt`; ~60 s each, bootstrap-bound)
python make_latex_table.py --raw-dir results/raw --out-dir /tmp/check4b
python make_latex_table.py --raw-dir results/raw/8b --domino-model-dir qwen3-8b \
  --domino-no-warmup --no-warmup-drop --model-label Qwen3-8B --table-suffix _8b \
  --out-dir /tmp/check8b
cmp /tmp/check4b/table1_cells.csv    results/tables_gpunative/table1_cells.csv
cmp /tmp/check4b/pairwise_ci.csv     results/tables_gpunative/pairwise_ci.csv
cmp /tmp/check8b/table1_cells_8b.csv results/tables_gpunative/table1_cells_8b.csv
cmp /tmp/check8b/pairwise_ci_8b.csv  results/tables_gpunative/pairwise_ci_8b.csv
#    expected: silence — the CSVs reproduce byte for byte, bootstrap CIs included
```

Compare the **CSVs**, not the `.md` tables: the checked-in Markdown was run through a
formatter that pads table columns, so it matches cell-for-cell but not byte-for-byte.

Reproducing the measurements themselves needs GPUs and is described under
[HF research harness](#hf-research-harness) and [SGLang serving](#sglang-serving).

## Setup

This repository contains no Domino code and no drafter weights. Clone the official
[Domino repository](https://github.com/jianuo-huang/Domino) and download the released
checkpoints:

```bash
git clone https://github.com/jianuo-huang/Domino
```

- Domino drafter (Domino and DominoTree): [`Huang2020/Qwen3-4B-Domino-b16`](https://huggingface.co/Huang2020/Qwen3-4B-Domino-b16)
- DFlash drafter (DDTree/CaDDTree baselines): [`z-lab/Qwen3-4B-DFlash-b16`](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16)
- Target: [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B)

The harness loads Domino's `code/` directory directly, so its behavior follows whichever
commit is checked out. The plugin does not — the Domino head it carries is copied at a
pinned commit, recorded in [`sglang_dominotree/PROVENANCE.md`](sglang_dominotree/PROVENANCE.md).

**Upstream projects.** Only the first is needed to run anything here; the rest are the
baselines, pinned at the commits their numbers were collected from.

| Project                                                  | Role here                       | Pinned at    |
| -------------------------------------------------------- | ------------------------------- | ------------ |
| [Domino](https://github.com/jianuo-huang/Domino)         | drafter + head; required to run | `e4aad4851`  |
| [CaDDTree](https://github.com/ZhangShuai1230/CaDDTree)   | AR / DFlash / DDTree / CaDDTree baselines | `a88f3f3` |
| [DFlash](https://github.com/z-lab/dflash)                | the DFlash drafter checkpoint   | —            |
| [DDTree](https://github.com/liranringel/ddtree)          | the tree baseline's origin      | —            |
| [SGLang](https://github.com/sgl-project/sglang)          | serving engine for the plugin   | `1adb53f14`  |

## HF research harness

Requires a CUDA-enabled PyTorch environment; install a CUDA-compatible PyTorch build first
if the environment does not already provide one.

```bash
pip install -r requirements.txt

export DOMINO_CODE=/path/to/Domino/code    # --domino-code for benchmark.py
export MODEL_PATH=/path/to/Qwen3-4B
export DRAFT_PATH=/path/to/Qwen3-4B-Domino-b16

SMOKE=1 MAX_SAMPLES=2 bash run_benchmark.sh   # functional check
bash run_benchmark.sh                          # full run
```

The headline public configuration is `dominotree@16` with `--corr-topm 64`. Runs write
JSONL under `runs/`: per-prompt TPS, accepted tokens per round, per-round stage timings,
and an output signature for losslessness checks. Every method runs a warmup prompt before
timing, matching the DDTree/CaDDTree benchmark convention.

A single-GPU collection of DominoTree against AR:

```bash
DATASET=gsm8k MAX_SAMPLES=50 MAX_NEW_TOKENS=2048 METHODS=ar,dominotree BUDGETS=16 CORR_TOPM=64 bash run_benchmark.sh
```

**Datasets need no local staging.** `DATASET` is handed straight to Domino's own
`load_and_process_dataset` (`$DOMINO_CODE/model/utils.py`), which downloads the split from
the HuggingFace Hub and applies Domino's prompt template — so both methods see byte-identical
prompts, and there is no dataset directory to populate. The eight names the paper reports:

| `DATASET`   | `gsm8k` | `math500` | `aime25` | `humaneval` | `mbpp` | `livecodebench` | `mt-bench` | `alpaca` |
| ----------- | ------- | --------- | -------- | ----------- | ------ | --------------- | ---------- | -------- |
| paper label | GSM8K   | MATH-500  | AIME25   | HumanEval   | MBPP   | LCB             | MT-Bench   | Alpaca   |

Set `HF_HOME` to control where the downloads are cached, and `HF_TOKEN` for gated splits.
The only paths you supply are the three exports above. Runs write to
`runs/<timestamp>/tps_<dataset>_T<temp>.jsonl` alongside a `run.log` (override the directory
with `OUT_DIR`).

The default builder is the GPU-native CUDA-graph tree builder (`dominotree_gpu.py`),
bit-identical to the pure-Python reference (`--python-builder`) at lower build cost.

## SGLang serving

DominoTree also runs inside SGLang as a speculative-decoding plugin: a separate package
that registers a `DOMINOTREE` algorithm through SGLang's plugin entry point, leaving
SGLang's own source unmodified — no fork, no patches. It reuses SGLang's EAGLE tree-verify
and carries a copy of Domino's official GRU-correction head; what was copied, and the
one-command proof that the copy is verbatim, are in
[`sglang_dominotree/PROVENANCE.md`](sglang_dominotree/PROVENANCE.md).

```bash
pip install -e sglang_dominotree

SGLANG_PLUGINS=dominotree \
python -m sglang.launch_server \
  --model-path /path/to/Qwen3-4B \
  --speculative-algorithm DOMINOTREE \
  --speculative-draft-model-path /path/to/Qwen3-4B-Domino-b16 \
  --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 16 \
  --tp-size 1 --trust-remote-code --mem-fraction-static 0.7 --port 30000
```

The server then serves DominoTree on SGLang's OpenAI-compatible and `/generate` endpoints
at any temperature. The defaults are the configuration the paper reports, so no
environment knobs are needed. For the plain Domino chain baseline, use
`--speculative-algorithm DOMINO`; for a larger target model, raise `--tp-size` (e.g., Qwen3-8B uses
`--tp-size 2` in our experiment on a 2xRTX-5080 machine.).

Supported SGLang versions, requirements, tuning knobs, and what the plugin overrides:
[`sglang_dominotree/README.md`](sglang_dominotree/README.md). The three serving benchmarks
and the admission cap that governs the concurrency comparison:
[`sglang_dominotree/benchmarks/README.md`](sglang_dominotree/benchmarks/README.md).

## Results

Frozen Qwen3-4B and Qwen3-8B artifacts ship under `results/`, and every table and figure
is rebuilt from the shipped raw JSONL.

- `results/tables_gpunative/` — the derived tables (`table1.md`/`table1_8b.md`,
  `pairwise_ci.md`/`pairwise_ci_8b.md`, `conditioning_ablation.md`, and `table1_cells.csv`,
  which drives Figure 1).
- `results/raw/dominotree/` — harness AR and `dominotree@16` per-prompt JSONL (Qwen3-4B);
  `results/raw/8b/` is the same layout for Qwen3-8B.
- `results/raw/baseline_ddtree_caddtree/` — AR / DFlash / DDTree@16 / CaDDTree, exported
  from the official CaDDTree-on-DFlash caches.
- `results/raw/domino_official/{qwen3-4b,qwen3-8b}/` — the Domino baseline, from the
  released Domino decoder's own benchmark, run twice per dataset: with its `--use-graph`
  flag (files prefixed `graph_`) and without (`eager_`). Its `README.md` documents the
  layout and the AR normalization.
- `results/raw/conditioning_ablation/`, `results/raw/dominotree_python_builder/` — the
  conditioning ablation (Cond@16 vs Marg@16, matched builder).
- `results/serving/` — the SGLang serving raw data, covering the three serving benchmarks
  (single request, goodput under concurrency, HELMET long context), both model sizes
  (Qwen3-4B at TP=1, Qwen3-8B at TP=2), and all five methods compared under identical
  serving flags (AR, DFlash, EAGLE-3, the Domino chain, DominoTree). Includes the
  per-prompt sidecars behind the paired-bootstrap CIs, a `MANIFEST.sha256`, and its own
  `README.md` mapping each file to the table it backs.

Both halves of the evidence — the offline tables above and the serving tables — can be
re-derived on a laptop: no GPU, no model weights, no access to the paper's LaTeX source.
For serving, this recomputes all 88 published cells from the raw per-prompt JSONL and diffs
them against the values printed in the paper (stdlib only, expected output
`ALL CELLS REPRODUCE FROM RAW DATA.`):

```bash
python3 results/serving/verify_published_numbers.py
```

The offline tables regenerate the same way:

```bash
# Qwen3-4B: Table 1, pairwise CIs, conditioning ablation
python make_latex_table.py --raw-dir results/raw --out-dir results/tables_gpunative

# Qwen3-8B. The two extra flags are the warmup conventions, explained just below.
python make_latex_table.py --raw-dir results/raw/8b --domino-model-dir qwen3-8b \
  --domino-no-warmup --no-warmup-drop --model-label Qwen3-8B --table-suffix _8b \
  --out-dir results/tables_gpunative
```

### Warmup: why the 8B command needs two extra flags

**The problem.** The first prompt a process ever runs is slow — CUDA kernels compile, caches
and the allocator are cold. Timing it would understate every method. The DFlash / DDTree /
CaDDTree reference benchmarks handle this by sending one throwaway "Warmup" prompt before
timing starts, and we follow that convention: **every number in every table is a mean over
prompts that were already warm.**

**Why it isn't uniform.** There are two ways to get there, and this project used both,
because neither benchmark had a warmup prompt when we started:

1. **Warm up in the run** — send a throwaway prompt first, then time all 50. Costs nothing.
2. **Drop the cold row** — time all 50, then discard each method's first prompt at
   table-build time, leaving 49. Equivalent for the mean, but one prompt smaller.

We added a warmup prompt to our `benchmark.py` partway through the project, and generated a
warmup-enabled variant of Domino's benchmark (`benchmark_noar.py`) for the 4B Domino
baseline. Collections taken **before** their script had that prompt use method 2 instead —
which is what the flags select. Nothing was re-run: the frozen JSONL is what it is, and the
flags describe it honestly rather than papering over it.

| Collection          | Script had a warmup prompt?                         | So the table builder    | n  |
| ------------------- | --------------------------------------------------- | ----------------------- | -- |
| Ours, 4B (frozen)   | no — predates the warmup we added to `benchmark.py` | drops row 1 (default)   | 49 |
| Ours, 8B            | yes                                                 | keeps all (`--no-warmup-drop`)   | 50 |
| Official Domino, 4B | yes — our `benchmark_noar.py` variant               | keeps all (default)     | 50 |
| Official Domino, 8B | no — Domino's stock `benchmark.py`                  | drops row 1 (`--domino-no-warmup`) | 49 |

So your reading of Domino is right: **the released Domino benchmark has no warmup prompt of
its own.** The 4B row is warm because we ran it through a variant with one inserted; the 8B
row was collected with the stock script, so its first prompt is dropped instead.

A fresh 4B run today needs neither flag — `benchmark.py` warms up on its own. The exclusion
stays in the published command so it reproduces the paper exactly instead of silently
drifting from it.

## Baselines (offline HF harness)

This section covers the Table 1 baselines only. The serving tables use entirely different
baselines, measured inside SGLang — see
[`sglang_dominotree/benchmarks/README.md`](sglang_dominotree/benchmarks/README.md).

The AR / DFlash / DDTree / CaDDTree numbers come from benchmarking with the official
[CaDDTree repository](https://github.com/ZhangShuai1230/CaDDTree), commit `a88f3f3`, on the
native DFlash drafter `Qwen3-4B-DFlash-b16`. Domino and DominoTree use the Domino drafter.
Each method must be paired with its own drafter: the two checkpoints are interchangeable in
shape, so running DDTree against the Domino draft silently ignores Domino's `shift_label`
correction and completes with a degenerate tau of about 1.04.

The Domino baseline is the released Domino decoder ([`jianuo-huang/Domino`](https://github.com/jianuo-huang/Domino),
commit `e4aad4851`), run through its own benchmark, `best-of(graph, eager)` per dataset. It
was collected single-GPU with the sibling GPU idle: we saw run-to-run variance for its
CUDA-graph runner on a shared node, so we removed the contention rather than average over it.

**Speedup normalization.** Speedups are throughput over AR, each method normalized by its
own harness's AR — with one exception, official Domino. Domino's released benchmark
measures AR as `spec_generate(block_size=1)`, its speculative loop with drafting disabled.
The forward is identical to a plain AR forward, but the loop still pays per-token
speculative bookkeeping (a `verify_ids` allocation, a trivially-zero acceptance length
computed through a GPU→CPU `.item()` sync, a KV-cache crop, a stop-tensor allocation).
That is per-round overhead, so it lands hardest on the one-token-per-round AR baseline and
amortizes away in the block and tree methods, leaving Domino's AR measurably slower than a
lean one. Two independently written lean harnesses — ours and the official CaDDTree
harness, whose AR rows ship in `results/raw/baseline_ddtree_caddtree/` — agree on the
faster AR floor to within ~2% dataset by dataset, which makes Domino's AR the outlier.
Normalizing Domino by its own AR would inflate its speedup by roughly 1.2×, so Domino is
reported over the common lean AR and every other method keeps its own. The default
`python make_latex_table.py --ar-norm surgical` reproduces this; the alternatives are
documented in `results/raw/domino_official/README.md`.

## Citation

```bibtex
@article{lin2026dominotree,
  title={DominoTree: Conditional Tree-Structured Drafting with Domino for Speculative Decoding},
  author={Lin, Saw S.},
  year={2026},
  journal={arXiv preprint arXiv:2607.08642},
  url={https://arxiv.org/abs/2607.08642}
}
```
