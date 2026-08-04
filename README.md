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

## Two ways to run it

The repository supports both settings the paper reports, as first-class paths. Pick the
one that matches what you want to measure.

| | **A. HF research harness** | **B. SGLang serving** |
|---|---|---|
| **Reproduces** | Table 1 — offline single-stream, 8 datasets × 3 temperatures, Qwen3-4B + 8B | The serving tables — bs=1, concurrency goodput, HELMET long context |
| **Compares against** | DFlash, DDTree, CaDDTree, released Domino decoder | AR, DFlash, EAGLE-3, Domino chain — all in one engine, identical flags |
| **Entry point** | [`run_benchmark.sh`](run_benchmark.sh) → [`benchmark.py`](benchmark.py) | [`sglang_dominotree/`](sglang_dominotree/) plugin → [`benchmarks/`](sglang_dominotree/benchmarks/) |
| **Engine** | plain PyTorch / HF, one request at a time | stock upstream SGLang, out-of-tree plugin |
| **Install** | `pip install -r requirements.txt` | `pip install -e sglang_dominotree` |
| **Use it when** | you want the algorithm measured without engine effects | you want production behaviour: batching, paged KV, CUDA graphs, long prompts |

Both run the **same method**. The harness isolates the algorithm; the serving path shows
what survives inside a heavily optimized engine (less, and we say so — a stronger
baseline shrinks the margin).

## Checking the published numbers without a GPU

You do not need hardware, model weights, or the paper's LaTeX to audit the results. Every
raw measurement behind the serving tables ships in [`results/serving/`](results/serving/),
with a one-command audit that recomputes **all 88 published serving cells** from that raw
data and diffs them against the values printed in the paper:

```bash
python3 results/serving/verify_published_numbers.py   # stdlib only — no GPU, no weights
```

Expected: `ALL CELLS REPRODUCE FROM RAW DATA.` It also prints each run's admission cap,
read out of that run's own server-status file, so the fairness of the concurrency
comparison is checkable rather than asserted. The offline tables regenerate the same way
from `results/raw/` — see [Results](#results).

## Repository layout

```
benchmark.py, dominotree.py, dominotree_gpu.py   path A: the HF research harness
  run_benchmark.sh, run_pipeline.sh              …its drivers
  make_latex_table.py                            …raw JSONL -> the paper's tables
sglang_dominotree/                               path B: the SGLang plugin
  src/dominotree_sglang/                         …algorithm registration + tree builder
  benchmarks/{bs1,concurrency,helmet}/           …the three serving benchmarks
  PROVENANCE.md, verify_vendored_head.py         …vendored-code manifest + copy proof
results/raw/, results/tables_gpunative/          path A raw data + derived tables
results/serving/                                 path B raw data + the no-GPU audit
demo/                                            side-by-side record-then-replay demo
```

## Prerequisites / Getting the Domino drafter

DominoTree vendors no Domino code and no drafter weights. Clone the official
[Domino repository](https://github.com/jianuo-huang/Domino) separately:

```bash
git clone https://github.com/jianuo-huang/Domino
```

Download the released checkpoints from Hugging Face:

- Domino drafter (for Domino and DominoTree): [`Huang2020/Qwen3-4B-Domino-b16`](https://huggingface.co/Huang2020/Qwen3-4B-Domino-b16)
- DFlash drafter (for the DDTree/CaDDTree baselines): [`z-lab/Qwen3-4B-DFlash-b16`](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16)
- Target model: [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B)

Pass the Domino code directory at runtime with `--domino-code /path/to/Domino/code`.
The helper script reads the same path from `DOMINO_CODE`.

## Path A — HF research harness

### Setup

This codebase is intended for a CUDA-enabled PyTorch environment. Install a CUDA-compatible
PyTorch build first if your environment does not already provide one.

```bash
pip install -r requirements.txt
```

Set the target model, Domino code, and Domino drafter paths:

```bash
export DOMINO_CODE=/path/to/Domino/code
export MODEL_PATH=/path/to/Qwen3-4B
export DRAFT_PATH=/path/to/Qwen3-4B-Domino-b16
```

### Run

The headline public configuration is `dominotree@16` with `--corr-topm 64`.

```bash
SMOKE=1 MAX_SAMPLES=2 bash run_benchmark.sh   # quick functional check
bash run_benchmark.sh                          # full run
```

This writes JSONL records under `runs/`. Each record contains per-prompt TPS, accepted
tokens per round, per-round stage timings, and a short output signature for losslessness
checks. Methods run through a warmup prompt before timing (matching the DDTree/CaDDTree
benchmark convention), so measured prompts are warm from the start.

## Path B — SGLang serving

DominoTree runs inside the **SGLang** production serving engine — not just the single-stream
research harness above — as an out-of-tree speculative-decoding plugin (the package under
`sglang_dominotree/`). It registers a `DOMINOTREE` speculative algorithm on **stock upstream
SGLang** and reuses SGLang's EAGLE tree-verify; it vendors Domino's official GRU-correction
head (copied verbatim, not reimplemented — see
[`sglang_dominotree/PROVENANCE.md`](sglang_dominotree/PROVENANCE.md) for the manifest and a
one-command proof that the copy is byte-identical to the official fork) and adds our
conditional best-first tree builder.

**The default config is the zero-sync frontier builder with CUDA-graph capture** — the
fastest configuration and the one the paper reports. No env vars are needed to get it: a bare
`--speculative-algorithm DOMINOTREE` uses it. It is lossless by construction (the target
verifier is unchanged) and runs under SGLang's decode CUDA graphs, at any temperature.

> **Status:** on the `sglang-integration` branch. GPU-validated on RTX 5080s at **T=0 and
> T>0** — lossless, and the highest accepted length of every method we evaluated, in every
> cell we measured. Tensor parallelism (`--tp-size > 1`) is supported and is what the
> Qwen3-8B results use.

**Reproducing the paper's serving results.** [`sglang_dominotree/benchmarks/`](sglang_dominotree/benchmarks/)
holds all three serving benchmarks — single-request (`bs1/`), goodput under concurrency
(`concurrency/`), and long context on HELMET (`helmet/`) — each comparing DominoTree with
AR, DFlash, EAGLE-3, and the Domino chain under identical serving flags. Start at
[`benchmarks/README.md`](sglang_dominotree/benchmarks/README.md); it also explains the one
thing that is easy to misread under concurrency, the per-method **admission cap**.

**Install** (needs a recent upstream SGLang with the `sglang.srt.plugins` entry point +
DFLASH-v2; validated against commit `1adb53f14`):

```bash
pip install -e sglang_dominotree      # registers the `dominotree` sglang.srt.plugins entry point
```

**Serve** DominoTree on the Domino drafter (the best config is the default — no env knobs needed):

```bash
SGLANG_PLUGINS=dominotree \
python -m sglang.launch_server \
  --model-path /path/to/Qwen3-4B \
  --speculative-algorithm DOMINOTREE \
  --speculative-draft-model-path /path/to/Qwen3-4B-Domino-b16 \
  --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 16 \
  --tp-size 1 --trust-remote-code --mem-fraction-static 0.7 --port 30000
```

`SGLANG_PLUGINS=dominotree` whitelists the plugin; the server then serves DominoTree on
SGLang's OpenAI-compatible / `/generate` endpoints, at any temperature. For the plain
**Domino chain** baseline on SGLang, use `--speculative-algorithm DOMINO`. For a larger
target, raise `--tp-size` (e.g. `--tp-size 2` for Qwen3-8B on two GPUs).

**Config knobs** — the defaults are the best config, so routine serving needs none; these are
for science / ablation / reproducibility:
`DOMINOTREE_BUILDER` (`frontier` default; `conditional` = the per-request best-first heap used
in the paper's Python-vs-GPU-native builder ablation, `toy` = a fixed caterpillar),
`DOMINOTREE_FRONTIER_GRAPH` (`1` default = capture the frontier builder itself in a CUDA
graph), `DOMINOTREE_GPU_BUILDER` (`1` default; the CUDA-graph node expander for the
`conditional` builder), `DOMINOTREE_NODE_TOPK` (8), `DOMINOTREE_CORR_TOPM` (64). The plugin
requires page-size 1 and a non-Mamba target; unsupported server configs fail fast rather than
run silently-wrong.

## Baselines

The paper's AR / DFlash / DDTree / CaDDTree baseline numbers come from the official
[CaDDTree repository](https://github.com/ZhangShuai1230/CaDDTree), commit `a88f3f3`, run on
the native DFlash drafter `Qwen3-4B-DFlash-b16`.

Do not run the DDTree baseline against the Domino draft checkpoint for paper comparison.
That path ignores Domino's `shift_label` correction and is degenerate (tau around 1.04).
Domino and DominoTree use the Domino drafter; DDTree/CaDDTree use the DFlash-native
drafter.

Upstream references:

- Domino: https://github.com/jianuo-huang/Domino
- DFlash: https://github.com/z-lab/dflash
- DDTree: https://github.com/liranringel/ddtree
- CaDDTree: https://github.com/ZhangShuai1230/CaDDTree

## Speedup normalization (AR baseline)

All speedups are throughput relative to autoregressive (AR) decoding, and each method
is normalized by its own harness's AR **with one exception: official Domino**.

Domino's released benchmark measures its AR baseline as `spec_generate(block_size=1)`
— its speculative decode loop with drafting disabled. At `block_size=1` the draft is
skipped and the single "verify" forward is byte-for-byte identical to a plain AR
forward, but the loop still runs per-token *speculative bookkeeping* that a
purpose-built AR loop does not: allocating a `verify_ids` tensor, computing a
(trivially-zero) acceptance length with a GPU→CPU `.item()` sync, cropping the KV
cache, and allocating a stop tensor — every token. Measured back-to-back on the same
GPU, this makes Domino's AR **~23% slower** than a lean AR (≈55 vs ≈66 tps), and it is
per-round overhead, so it hits the one-token-per-round AR baseline hard while
amortizing away in the block/tree methods.

Two independent lean harnesses corroborate the ≈66 tps floor: our AR and the **official
CaDDTree** harness's AR agree to within ~2% dataset-by-dataset. So ≈66 is the validated
lean-AR baseline and Domino's ≈55 is the outlier. Normalizing Domino by its own slow AR
would inflate its speedup by ~1.2× (e.g. 8.0× vs 6.7× at the same throughput), so we
report Domino's speedup over the lean common AR. DFlash/DDTree/CaDDTree and DominoTree
keep their own (already-lean ≈66) AR, unchanged.

Regenerate with the default `python make_latex_table.py --ar-norm surgical` (`own`
reproduces the pre-correction Domino-over-own-AR numbers; `common` normalizes every
method by the lean AR).

## Results

Frozen Qwen3-4B and Qwen3-8B paper artifacts are included under `results/`, and every
table/figure is rebuilt from the shipped raw JSONLs by `make_latex_table.py`.

- `results/tables_gpunative/`: the paper's derived tables (`table1.md`/`table1_8b.md`,
  `pairwise_ci.md`/`pairwise_ci_8b.md`, `conditioning_ablation.md`, and `table1_cells.csv`
  which drives Figure 1).
- `results/raw/dominotree/`: our-harness AR and DominoTree `dominotree@16` per-prompt JSONLs (Qwen3-4B).
- `results/raw/baseline_ddtree_caddtree/`: AR / DFlash / DDTree@16 / CaDDTree, exported from the
  official CaDDTree-on-DFlash caches.
- `results/raw/domino_official/{qwen3-4b,qwen3-8b}/`: the **Domino** baseline — the released Domino
  decoder's own benchmark (`graph` + `eager`, best-of per dataset). See its `README.md` for layout and
  the AR-normalization rationale.
- `results/raw/conditioning_ablation/` + `results/raw/dominotree_python_builder/`: the conditioning
  ablation (Cond@16 vs Marg@16, matched builder).
- `results/raw/8b/`: the Qwen3-8B raw data in the same layout (`dominotree/`, `baseline_ddtree_caddtree/`,
  `domino_official/qwen3-8b/`).
- **`results/serving/`**: the **SGLang serving** raw data — all three axes (single-request bs=1,
  concurrency goodput, HELMET long context), both model sizes, all five methods, including the
  per-prompt sidecars the paired-bootstrap CIs are computed from. It ships its own
  [`README.md`](results/serving/README.md) mapping each file to the table it backs, a
  `MANIFEST.sha256`, and a one-command audit that recomputes **all 88 published serving cells**
  from the raw JSONL and diffs them against the values printed in the paper:

  ```bash
  python3 results/serving/verify_published_numbers.py   # stdlib only; no GPU, no weights
  ```

  Expected: `ALL CELLS REPRODUCE FROM RAW DATA.`

Regenerate the tables:

```bash
# Qwen3-4B: Table 1, pairwise CIs, conditioning ablation
python make_latex_table.py --raw-dir results/raw --out-dir results/tables_gpunative

# Qwen3-8B: Table 1 + pairwise. Its official Domino was collected without an in-benchmark
# warmup, so drop the first prompt (--domino-no-warmup); the 8B our-harness data is warmup-enabled
# (--no-warmup-drop keeps all rows).
python make_latex_table.py --raw-dir results/raw/8b --domino-model-dir qwen3-8b \
  --domino-no-warmup --no-warmup-drop --model-label Qwen3-8B --table-suffix _8b \
  --out-dir results/tables_gpunative
```

Speedups use "surgical" AR normalization (each method over its own lean harness AR; the released
Domino over the lean common AR — see *Speedup normalization* above), and the **Domino** baseline is
`best-of(graph, eager)` per dataset.

## Reproduce a run

Collect DominoTree + AR on a single GPU:

```bash
DATASET=gsm8k MAX_SAMPLES=50 MAX_NEW_TOKENS=2048 METHODS=ar,dominotree BUDGETS=16 CORR_TOPM=64 bash run_benchmark.sh
```

For the **Domino** baseline, run the released Domino repository's own benchmark (both `eager` and
`--use-graph`); on a shared multi-GPU node collect it single-GPU (sibling idle), because its
CUDA-graph runner is sensitive to host-side contention. DominoTree's default builder is the GPU-native
CUDA-graph tree builder (`dominotree_gpu.py`), bit-identical to the pure-Python reference
(`--python-builder`) at lower build cost. Fixed/star topologies, wave batching, hybrid builders, and
adaptive conditional stopping are future work and not part of this release.

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
