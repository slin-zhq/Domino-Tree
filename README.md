<h1 align="center">DominoTree</h1>

<p align="center">
  Official research implementation of <strong>DominoTree</strong>, a training-free
  conditional draft tree for Domino block-diffusion drafters.
</p>

<p align="center">
  <em>Paper (arXiv) link coming soon.</em>
</p>

## Prerequisites / Getting the Domino drafter

DominoTree vendors no Domino code and no drafter weights. Clone the official
[Domino repository](https://github.com/jianuo-huang/Domino) separately:

```bash
git clone https://github.com/jianuo-huang/Domino
```

Download the released checkpoints from Hugging Face:

- Domino drafter (for Domino-chain and DominoTree): [`Huang2020/Qwen3-4B-Domino-b16`](https://huggingface.co/Huang2020/Qwen3-4B-Domino-b16)
- DFlash drafter (for the DDTree/CaDDTree baselines): [`z-lab/Qwen3-4B-DFlash-b16`](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16)
- Target model: [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B)

Pass the Domino code directory at runtime with `--domino-code /path/to/Domino/code`.
The helper script reads the same path from `DOMINO_CODE`.

## Setup

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

## Run

The headline public configuration is `dominotree@16` with `--corr-topm 64`.

```bash
SMOKE=1 MAX_SAMPLES=2 bash run_benchmark.sh   # quick functional check
bash run_benchmark.sh                          # full run
```

This writes JSONL records under `runs/`. Each record contains per-prompt TPS, accepted
tokens per round, per-round stage timings, and a short output signature for losslessness
checks. Methods run through a warmup prompt before timing (matching the DDTree/CaDDTree
benchmark convention), so measured prompts are warm from the start.

## Baselines

The paper's AR / DFlash / DDTree / CaDDTree baseline numbers come from the official
[CaDDTree repository](https://github.com/ZhangShuai1230/CaDDTree), commit `a88f3f3`, run on
the native DFlash drafter `Qwen3-4B-DFlash-b16`.

Do not run the DDTree baseline against the Domino draft checkpoint for paper comparison.
That path ignores Domino's `shift_label` correction and is degenerate (tau around 1.04).
Domino-chain and DominoTree use the Domino drafter; DDTree/CaDDTree use the DFlash-native
drafter.

Upstream references:

- Domino: https://github.com/jianuo-huang/Domino
- DFlash: https://github.com/z-lab/dflash
- DDTree: https://github.com/liranringel/ddtree
- CaDDTree: https://github.com/ZhangShuai1230/CaDDTree

## Results

Frozen Qwen3-4B paper artifacts are included under `results/`.

- `results/tables/`: derived Markdown/CSV tables, including the conditioning ablation.
- `results/raw/dominotree/`: Domino AR / Domino-chain / DominoTree `dominotree@16` JSONLs.
- `results/raw/dominotree_recollected/`: two timing-clean override cells recollected after GPU contention.
- `results/raw/baseline_ddtree_caddtree/`: exported JSONL summaries from official CaDDTree-on-DFlash caches.
- `results/raw/conditioning_ablation/`: matched-budget marginal-tree control records for the conditioning ablation.
- `results/raw/chain_stage_timing/`: dedicated instrumented Domino-chain run used for the per-round stage-time table.

The recollected files `alpaca_T0.5.jsonl` and `mt-bench_T1.0.jsonl` override the
corresponding files in `results/raw/dominotree/` during table regeneration.

Regenerate the main tables:

```bash
python make_latex_table.py --raw-dir results/raw --out-dir results/tables
```

## Reproduce

Run the benchmark on a single GPU:

```bash
DATASET=gsm8k MAX_SAMPLES=50 MAX_NEW_TOKENS=2048 METHODS=ar,chain,dominotree BUDGETS=16 CORR_TOPM=64 bash run_benchmark.sh
```

The v1 public code ships the Python tree backend only. GPU-native builders, fixed/star
topologies, wave batching, hybrid builders, and adaptive conditional stopping are future work
and are not part of this release.

## Citation

```bibtex
@article{lin2026dominotree,
  title={DominoTree: Conditional Tree-Structured Drafting with Domino for Speculative Decoding},
  author={Lin, Saw S.},
  year={2026},
  note={arXiv preprint; link to be added on release}
}
```
