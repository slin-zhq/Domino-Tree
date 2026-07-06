<h1 align="center">DominoTree</h1>

<p align="center">
  Official research implementation of <strong>DominoTree</strong>, a training-free
  conditional draft tree for Domino block-diffusion drafters.
</p>

<p align="center">
  <a href="<TODO: project page URL>">Project Page</a>
  &nbsp;|&nbsp;
  <a href="<TODO: arXiv URL>">Paper</a>
</p>

## Prerequisites / Getting the Domino drafter

DominoTree vendors no Domino code and no drafter weights. Clone the official
Domino repository separately: _(zhq: We don't need this level of babysitting. just point to the project URL: https://github.com/jianuo-huang/Domino)_

```bash
git clone <TODO: Domino repo URL>
```

Download the released drafter checkpoints:

- Domino drafter for Domino-chain and DominoTree: `<TODO: Qwen3-4B-Domino-b16 HF id>` _(zhq: you take care of this.)_
- DFlash drafter for DDTree/CaDDTree baselines: `<TODO: Qwen3-4B-DFlash-b16 HF id>` _(zhq: you take care of this.)_

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
SMOKE=1 MAX_SAMPLES=2 bash run_benchmark.sh
bash run_benchmark.sh
```

This writes JSONL records under `runs/`. Each record contains per-prompt TPS, accepted
tokens per round, stage timings, and a short output signature for losslessness checks.

## Baselines

The paper's AR / DFlash / DDTree / CaDDTree baseline numbers are from the official
CaDDTree repository (`<TODO: CaDDTree repo URL>`) _(zhq: you take care of this.)_, commit `a88f3f3`, run on the
native DFlash drafter `Qwen3-4B-DFlash-b16`.

Do not run the DDTree baseline against the Domino draft checkpoint for paper
comparison. That path ignores Domino's `shift_label` correction and is degenerate
(tau around 1.04). Domino-chain and DominoTree use the Domino drafter;
DDTree/CaDDTree use the DFlash-native drafter.

Upstream references:

- Domino: `<TODO: Domino repo URL>` _(zhq: you take care of this.)_
- DDTree: https://github.com/liranringel/ddtree
- CaDDTree: `<TODO: CaDDTree repo URL>` _(zhq: you take care of this.)_

## Results

Frozen Qwen3-4B paper artifacts are included under `results/`.

- `results/tables/`: derived Markdown/CSV tables, including the conditioning ablation.
- `results/raw/dominotree/`: Domino AR / Domino-chain / DominoTree `dominotree@16` JSONLs.
- `results/raw/dominotree_recollected/`: two timing-clean override cells recollected after GPU contention.
- `results/raw/baseline_ddtree_caddtree/`: exported JSONL summaries from official CaDDTree-on-DFlash caches.
- `results/raw/conditioning_ablation/`: matched-budget marginal-tree control records for the conditioning ablation.

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
@article{dominoTree2026,
  title={DominoTree: Conditional Draft Trees for Domino Block Diffusion Drafters},
  author={<TODO: authors>},
  journal={<TODO: arXiv preprint>},
  year={2026}
}
```
