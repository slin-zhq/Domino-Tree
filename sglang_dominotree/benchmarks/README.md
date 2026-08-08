# SGLang serving benchmarks

Reproduction scripts for the paper's SGLang serving results. Three benchmarks, five
methods each.

> **You do not need a GPU to check the published numbers.** The raw output of every run
> reported in the paper ships in [`results/serving/`](../../results/serving/), together
> with a one-command audit that recomputes all 88 published serving cells from that raw
> data and diffs them against the paper:
> `python3 results/serving/verify_published_numbers.py`.
> Use the scripts below when you want to **re-measure** on your own hardware; use
> `results/serving/` when you want to **check our arithmetic**.

| Directory                      | Measures                                                                          | Paper                       |
| ------------------------------ | --------------------------------------------------------------------------------- | --------------------------- |
| [`bs1/`](bs1/)                 | single-request (bs=1) throughput + acceptance length, 8 datasets × 3 temperatures | single-stream serving table |
| [`concurrency/`](concurrency/) | goodput as offered concurrency rises, on full datasets                            | concurrency table           |
| [`helmet/`](helmet/)           | acceptance + throughput as input context grows, on HELMET                         | long-context table          |

| Method         | What runs                                               |
| -------------- | ------------------------------------------------------- |
| `ar`           | target only, no speculation (τ = 1 floor)               |
| `dflash`       | DFlash block-diffusion chain drafter                    |
| `eagle3`       | EAGLE-3                                                 |
| `domino_chain` | the Domino drafter, chain verify                        |
| `dominotree`   | **DominoTree** — the conditional draft tree (this repo) |

## Prerequisites

- SGLang with the `dominotree` plugin installed — see
  [`../README.md`](../README.md) → "Install", which pins the SGLang commit and gives the
  exact environment. `dflash` and `eagle3` are native SGLang algorithms and need no plugin.
- `pip install requests transformers datasets`.
- Every checkpoint below, downloaded locally. The runner scripts take **filesystem paths**,
  not HF ids, so fetch them first (`huggingface-cli download <id> --local-dir <path>`).

### Checkpoints — every method, both sizes

| Method         | Draft checkpoint (`4B` / `8B`)                                                | Env var         |
| -------------- | ----------------------------------------------------------------------------- | --------------- |
| `ar`           | *(none — target only)*                                                        | —               |
| `dflash`       | [`z-lab/Qwen3-4B-DFlash-b16`](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16) / [`z-lab/Qwen3-8B-DFlash-b16`](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) | `DRAFT_DFLASH`  |
| `eagle3`       | [`AngelSlim/Qwen3-4B_eagle3`](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3) / [`AngelSlim/Qwen3-8B_eagle3`](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3) | `DRAFT_EAGLE3`  |
| `domino_chain` | [`Huang2020/Qwen3-4B-Domino-b16`](https://huggingface.co/Huang2020/Qwen3-4B-Domino-b16) / [`Huang2020/Qwen3-8B-Domino-b16`](https://huggingface.co/Huang2020/Qwen3-8B-Domino-b16) | `DRAFT_DOMINO`  |
| `dominotree`   | same as `domino_chain` — the tree is training-free on Domino's checkpoint     | `DRAFT_DOMINO`  |

Targets: [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) (TP=1) and
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) (TP=2), passed as `MODEL`.

The example invocations in the sub-benchmark READMEs use short local paths such as
`./Qwen3-4B_eagle3`; substitute wherever you downloaded each checkpoint.

## The fairness rule

Every method is launched by the **same** script ([`launch_server.sh`](launch_server.sh))
with **identical** serving flags — attention backend, page size, memory fraction,
CUDA graphs, draft budget (16 tokens). Only the speculative algorithm and the draft
model differ. Any throughput difference therefore reflects the speculative method,
not the launch configuration. If you must change a flag (e.g. `PREFILL_GRAPH=disabled`
because one method OOMs during prefill graph replay), change it for **all** methods.

`helmet/` uses its own launcher because long context needs a different KV/batch
recipe (a large KV pool for a 32K prompt, a tiny batch cap); within that benchmark
all five methods still share identical flags.

## One thing to get right: admission caps

`--max-running-requests` bounds how many requests the server runs **simultaneously**.
Requests offered beyond it queue. Each method sustains a different cap on a given
GPU, because a draft model costs weights plus its own KV, and tree verify
additionally materializes a `batch × draft × vocab` logits buffer — so on a
memory-constrained card DominoTree admits fewer concurrent requests than a chain.

Two consequences when you read concurrency results:

1. **Compare methods only at concurrencies within every compared method's cap.**
   Past its cap a method is running its cap, not the offered load; its goodput
   plateaus, and the gap you measure is admission capacity, not per-step cost.
2. **Report the cap alongside the number.** It is a property of the method _on that
   hardware_, and it moves as GPU memory changes.

See [`concurrency/README.md`](concurrency/README.md) for how to find each method's cap
and for the caps used in the paper.

## Hardware assumptions (read if your GPUs differ from ours)

Every number in the paper was measured on **two RTX 5080s (16 GB each)** — Qwen3-4B at
TP=1, Qwen3-8B at TP=2. Two settings in these scripts are tuned to that, and are the
first things to revisit on different hardware:

- **`MEMFRAC` (`--mem-fraction-static`), default 0.7.** It trades KV-pool size against
  transient workspace. On a 16 GB card, DominoTree's tree KV-move kernel needs headroom
  at high concurrency and OOMs above ~0.7; long context wants the opposite (a big pool
  for a 32K prompt), so `helmet/` uses 0.85 with a tiny batch cap, as did the 8B/TP=2
  concurrency runs where sharded weights free up card memory. **With plenty of VRAM you
  can raise it; if you OOM, lower it — always for all methods.**
- **The admission caps** in `concurrency/README.md` are 16 GB-card numbers. On an 80 GB
  card every method will admit far more, the caps may stop binding entirely within
  `c ≤ 32`, and the concurrency crossover we report moves right or disappears. Run
  `concurrency/find_caps.sh` and use _your_ caps; do not inherit ours.

Anything hardware-independent — the fairness rule, the τ ordering, losslessness, the
methodology — carries over unchanged.

## Reproducing

Each directory has its own README with the exact commands. All paths are environment
variables; nothing is site-specific. Long runs should be detached
(`setsid ... > log 2>&1 < /dev/null &`) so a dropped SSH connection does not kill them.
