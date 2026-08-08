# HELMET long-context benchmark (SGLang serving)

Measures **speculative-decoding acceptance length τ** (`meta_info.spec_accept_length`)
and single-request (bs=1) goodput as **input context length grows**, on
[HELMET](https://github.com/princeton-nlp/HELMET) (Yen et al., ICLR 2025,
[arXiv:2410.02694](https://arxiv.org/abs/2410.02694)), for five serving configs:

| method         | what runs                                               |
| -------------- | ------------------------------------------------------- |
| `ar`           | target only, no speculation (τ = 1 floor)               |
| `dflash`       | DFlash block-diffusion chain drafter                    |
| `eagle3`       | EAGLE-3                                                 |
| `domino_chain` | the Domino drafter, chain verify                        |
| `dominotree`   | **DominoTree** — the conditional draft tree (this repo) |

All five share identical serving flags, so any τ/throughput difference reflects the
speculative method, not the launch config.

### Why HELMET Summarization specifically

τ is a **decode-side** metric, so it needs tasks with a **long generated output**.
HELMET's **Summarization** tasks (`infbench_sum`, `multi_lexsum`; generation cap
1,200 tokens) give the longest decode phase in any long-context benchmark; its
**Cite** tasks (`alce_asqa`, `alce_qampari`; cap 300) are a shorter second workload.
Short-answer long-context tasks (single-token QA, multiple-choice, retrieval) decode
too few tokens to measure τ and are intentionally excluded. HELMET also natively bins
input length to {8K, 16K, 32K, 64K, 128K}, giving a clean controlled length sweep.

## Prerequisites

- SGLang with the `dominotree` plugin installed (see the repo root README →
  "Serving on SGLang"); `dflash` / `eagle3` are native SGLang algorithms.
- The target model and each method's draft model available locally.
- `pip install transformers huggingface_hub` for the prep step.

## Step 1 — materialise HELMET's own Summarization examples (`helmet_source_dump.py`)

**HELMET's Summarization tasks are not shipped as prebuilt jsonl.** Unlike its
Recall/RAG/Rerank/Cite families, `infbench_sum` and `multi_lexsum` are loaded _live_ by
HELMET's own `data.py` (`load_infbench`, `load_multi_lexsum`) from HF datasets and then
truncated to a length bin with HELMET's own `truncate_llama2`. Downloading the
`princeton-nlp/HELMET` dataset repo will **not** give you these files.

So we call HELMET's own loaders — we do not reimplement them:

```bash
git clone https://github.com/princeton-nlp/HELMET ~/HELMET
pip install datasets transformers

python helmet_source_dump.py --helmet-repo ~/HELMET \
  --out-dir ./helmet_src --tasks infbench_sum,multi_lexsum \
  --length-bins 8192,16384,32768 --limit 100
```

This imports HELMET's `data.py` unmodified and uses its own bin → dataset-name
convention from `configs/summ_short.yaml`. It patches around exactly two environment
breakages, both documented at the top of the script: HF `datasets` >= 4 dropped
loading-script support (needed by `allenai/multi_lexsum`), and the gated
`meta-llama/Llama-2-7b-hf` tokenizer is redirected to a byte-identical public mirror.
Nothing about the prompts themselves is ours.

Output: `./helmet_src/<task>_<bin>.jsonl` with a `helmet_prompt` field — HELMET's own
`user_template.format(**row)`, i.e. context + instruction, not yet chat-templated.

## Step 2 — apply your model's chat template (`helmet_prep.py`)

Step 1 produced HELMET's own prompt text. This step wraps each one with the **served
model's** chat template and writes the flat per-cell layout the driver reads,
`<out-dir>/<task>/<bin>.jsonl`. It re-implements nothing: with `--prompt-field` it takes
the prompt verbatim and only chat-templates it.

Use `--limit 100` so that an `n=50` run is a strict prefix of an `n=100` run — you can
extend one marginal cell later without re-running the rest.

```bash
python helmet_prep.py --out-dir ./prompts/Qwen3-8B \
  --tasks infbench_sum,multi_lexsum --length-bins 8192,16384,32768 --limit 100 \
  --model-path Qwen/Qwen3-8B \
  --helmet-data-glob "./helmet_src/{task}_{bin}.jsonl" \
  --prompt-field helmet_prompt
```

Run this once **per served model** — the chat template differs, so a 4B run needs its own
prompt set (`--model-path Qwen/Qwen3-4B --out-dir ./prompts/Qwen3-4B`).

**Eyeball one built prompt before scaling.** The `text` field must be HELMET's context +
instruction, wrapped in your model's chat template:

```bash
python -c "import json; r=json.loads(open('./prompts/Qwen3-8B/infbench_sum/8192.jsonl').readline()); print(r['text'][:600]); print('...'); print(r['text'][-400:])"
```

## Step 3 — run the sweep (`run_helmet_all.sh`, all 5 methods, drain-safe)

```bash
MODEL=Qwen/Qwen3-8B TP=2 \
  DRAFT_DOMINO=./Qwen3-8B-Domino-b16 \
  DRAFT_DFLASH=./Qwen3-8B-DFlash-b16 \
  DRAFT_EAGLE3=./Qwen3-8B_eagle3 \
  PROMPTS=./prompts/Qwen3-8B NPR=50 \
  setsid bash run_helmet_all.sh > orch.log 2>&1 < /dev/null &
tail -f orch.log
```

- Set `TP=2` for a large model on 2 GPUs; `TP=1` (single GPU) otherwise.
- **Fast pilot:** add `CLAMP=256 NPR=20` to cap generation and confirm the τ ordering
  and that the longest bin fits, before the full run.
- **If a method OOMs during prefill at long context** (activation spike in the prefill
  CUDA-graph replay, seen with the Domino chain on tight VRAM): add `PREFILL_GRAPH=disabled`.
  It skips only the prefill graph (decode graph + acceptance unaffected). Apply it to
  **all** methods in the comparison (identical flags = fair).
- Per-method fit + spec sanity land in `kvpool_<method>.txt`; progress in `orch.log`.

## Outputs

One JSONL row per (method, task, length_bin):

```
{method, model, task, length_bin, tps, mean_accept, n_prompts,
 input_tokens, output_tokens, gen_tokens, ...}
```

`mean_accept` is τ; `tps` is bs=1 goodput = sum(completion_tokens)/sum(decode_time).
Aggregate across methods (τ, throughput, Δ% vs each baseline) with your preferred
paired-bootstrap analysis over these rows.

## Notes / honest scope

- **Cite HELMET** — these are its Summarization (and optionally Cite) subtasks at its
  native length bins.
- **English only.** τ _magnitude_ is task-dependent; the robust result is the
  **paired relative** advantage (same drafter, same prompts).
- The maximum context you can run is bounded by your GPU memory (KV cache), not by
  this harness — a big model at long context needs large / multiple GPUs. State your
  ceiling honestly.
- HELMET's prompt data aggregates third-party datasets under their own licenses;
  check them before redistributing prompts.
