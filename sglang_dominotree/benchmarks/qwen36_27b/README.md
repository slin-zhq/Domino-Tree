# Qwen3.6-27B serving benchmarks (hybrid Gated-DeltaNet + attention target)

Reproduction scripts (single-GPU H200 by default) for the paper's Qwen3.6-27B serving section.
Same three benchmarks as [`../bs1/`](../bs1/), [`../concurrency/`](../concurrency/), and
[`../helmet/`](../helmet/) — this directory reuses their driver scripts unmodified — plus
a **losslessness gate** and the gate-qualified ablation workflow, explained below.

> **You do not need a GPU to check the published numbers.** Every number in the paper's
> Qwen3.6-27B section ships as raw per-prompt/per-cell JSONL in
> [`../../../results/serving/qwen36_27b_h200/`](../../../results/serving/qwen36_27b_h200/),
> together with a one-command audit
> (`python3 results/serving/qwen36_27b_h200/verify_qwen36_27b.py`, expects `ALL 480
> CELLS REPRODUCE FROM RAW DATA.`). Use the scripts here to **re-measure** on your own
> hardware.

| Method | What runs | Draft checkpoint |
| --- | --- | --- |
| `ar` | target only, no speculation (τ = 1 floor) | *(none)* |
| `dflash` | DFlash block-diffusion chain drafter | this target's own DFlash checkpoint |
| `domino_chain` | the Domino drafter, chain verify | [`Huang2020/qwen3.6-27B-domino`](https://huggingface.co/Huang2020/qwen3.6-27B-domino) |
| `dominotree` | **DominoTree** — the conditional draft tree (this repo) | same as `domino_chain` |
| `mtp` | the checkpoint's own multi-token-prediction head, served with SGLang's NEXTN algorithm | *(none — from the target's own weights)* |

Target: `Qwen/Qwen3.6-27B`, TP=1 (this target's opt-in GRU input table is **refused at
TP>1**; throughput and ablations use **one 141 GB H200**). The archived headline
losslessness gate used **two H100 GPUs with TP=2**, with the GRU table off. **No `eagle3`
arm** — no third-party EAGLE-3 head exists for this checkpoint, so `mtp` is the
draft-head baseline instead.

## Why this target needs P9

Qwen3.6-27B is **hybrid**: 48 Gated-DeltaNet (linear-attention/recurrent) layers plus 16
full-attention layers. DominoTree's tree verify was written against plain-attention
targets. Unmodified, it silently corrupts a hybrid target: with
`speculative_eagle_topk=1`, the GDN layers never see the tree's branching structure, so
they verify the flattened tree as one chain — every node's recurrent state absorbs the
states of every earlier node in verify order, **including rejected siblings**. This does
not crash, and it does not depress τ (see the gate result below) — the only visible
symptom is that the served text quietly stops matching what plain AR would have
produced. **P9** (`../../src/dominotree_sglang/{__init__,worker}.py`, this branch) fixes
this: it raises the GDN backend's own topk gate so its Triton kernels walk parent
pointers, switches to the dense per-node conv-window layout (the deduplicated layout is
chain-only), and commits each accepted node's recurrent state right after
`eagle_sample`, before the KV move/compaction. It activates automatically whenever
`SGLANG_PLUGINS=dominotree` and the target is detected as hybrid GDN — no separate flag
turns it on. Non-GDN recurrent targets are refused (fails closed), not silently
mishandled.

## Step 0 — the losslessness gate (`run_p9_gate_27b.sh`)

**Run this before trusting any benchmark number on a new hybrid target or after any
change to `dominotree_sglang`.** It is a token-by-token diff against plain AR, over four
arms: `ar` (the reference every arm is diffed against), `domino` (the chain — a
known-correct hybrid verify, since it doesn't need tree topology), `dominotree` (the
candidate, P9 on), and `dominotree_flat` (the **negative control** — P9 **off**, via
`DOMINOTREE_DIAG_FLAT_GDN=1` — the exact bug P9 fixes; diagnostic only, never publish a
throughput number from this arm).

```bash
MODEL=/path/Qwen3.6-27B DRAFT_DOMINO=/path/qwen3.6-27B-domino \
  bash run_p9_gate_27b.sh
```

Verdict is `P9_GATE=PASS` / `INCONCLUSIVE` / `FAIL` / `ERROR` (see `p9_gate_compare.py`'s
docstring for the exact decision rule, fixed before any data is seen: candidate median
agreement ≥ 0.5× reference's, candidate full-match rate ≥ reference's − 0.20, AND the
control is clearly worse — i.e. the test has power to catch the bug it's checking for).
The paper's gate result: reference median agreement with AR = 333 tokens, candidate =
306, **control = 46** — while the control's τ (9.58) is statistically indistinguishable
from the candidate's (9.59). τ alone would have called this fine; only the AR diff
catches it.

Those headline numbers are from **job 356641 on two H100 GPUs with tensor
parallelism (TP=2)**, admission cap 8, `--cuda-graph-max-bs-decode 8`,
`--disable-custom-all-reduce`, GRU table off, budget 15, node_topk 8, corr_topm 64.
Every archived H200 configuration passed the same gate against H200 AR/chain
references, using that archived gate run's flat-control arm. The new scripts collect
fresh AR, chain, and flat-control arms on the current hardware; do not mix reference
files across hardware or serving configurations when re-measuring.

To use the archived headline configuration inside a two-H100 allocation:

```bash
GATE_CONFIG=headline-h100 GPU=0,1 MODEL=/path/Qwen3.6-27B \
DRAFT_DOMINO=/path/qwen3.6-27B-domino bash run_p9_gate_27b.sh
```

The default `GATE_CONFIG=h200` uses TP=1, admission cap 2,
`--cuda-graph-max-bs 2`, GRU table on, and budget/k/M = 15/8/128.
The headline profile fixes its tree settings; the H200 profile accepts the tree
settings described below. Both use memory fraction 0.80. `GPU` selects the visible
GPU(s); the profile does not detect or reserve GPU hardware.
`OUT` is a **parent directory**: each invocation creates a fresh `run_<stamp>_<id>`
child, prints its location, and records its settings in `config.txt`. No old arm is
reused. Boot/client/identity failures abort immediately. The comparator requires all
four nonempty arms, unique prompt ids, and all expected ids in every arm (including
AR); invalid or incomplete inputs produce `P9_GATE=ERROR` and exit 2. FAIL and
INCONCLUSIVE exit 1; only PASS exits 0. `--per-dataset` on the comparator must match
`PER_DATASET` on the runner (default 5). The prompt construction and pre-registered
thresholds are unchanged.

## Step 1 — bs=1 (`run_bs1_27b.sh`)

```bash
MODEL=/path/Qwen3.6-27B \
DRAFT_DOMINO=/path/qwen3.6-27B-domino \
DRAFT_DFLASH=/path/Qwen3.6-27B-DFlash \
  setsid bash run_bs1_27b.sh > orch_bs1_27b.log 2>&1 < /dev/null &
tail -F orch_bs1_27b.log
```

8 datasets × {T=0.0, T=1.0}, n=50/dataset (AIME25 n=30, MT-Bench both turns),
max_new_tokens=2048. Reuses [`../bs1/bench_bs1.py`](../bs1/bench_bs1.py) unmodified.

## Step 2 — concurrency (`run_conc_27b.sh`)

```bash
MODEL=/path/Qwen3.6-27B \
DRAFT_DOMINO=/path/qwen3.6-27B-domino \
DRAFT_DFLASH=/path/Qwen3.6-27B-DFlash \
  setsid bash run_conc_27b.sh > orch_conc_27b.log 2>&1 < /dev/null &
```

gsm8k:128, mbpp:128, mt-bench:80 prompts × concurrency {1,2,4,8,16,32}, 512 tokens,
T=0.0. Each method boots at its own highest crash-free cap from `CAPS="32 16 8"`
(default); the cap is recorded in `admission_cap.txt` next to the data — see
[`../concurrency/README.md`](../concurrency/README.md) for why the cap matters when you
read the numbers. On the paper's 141 GB H200 every method reached cap 32.

## Step 3 — HELMET long context (`run_helmet_27b.sh`)

This target's chat template differs from Qwen3-4B/8B, so their prompt sets are **not**
reusable — build this model's own prompts first (see the script's own docstring, which
walks the same two-step process as [`../helmet/README.md`](../helmet/README.md)):

```bash
git clone https://github.com/princeton-nlp/HELMET ~/HELMET
python ../helmet/helmet_source_dump.py --helmet-repo ~/HELMET \
  --out-dir ./helmet_src --tasks infbench_sum,multi_lexsum \
  --length-bins 8192,16384,32768 --limit 100
python ../helmet/helmet_prep.py --out-dir ./prompts/Qwen3.6-27B \
  --tasks infbench_sum,multi_lexsum --length-bins 8192,16384,32768 --limit 100 \
  --model-path /path/Qwen3.6-27B \
  --helmet-data-glob "./helmet_src/{task}_{bin}.jsonl" --prompt-field helmet_prompt

MODEL=/path/Qwen3.6-27B \
DRAFT_DOMINO=/path/qwen3.6-27B-domino \
DRAFT_DFLASH=/path/Qwen3.6-27B-DFlash \
PROMPTS=./prompts/Qwen3.6-27B \
  setsid bash run_helmet_27b.sh > orch_helmet_27b.log 2>&1 < /dev/null &
```

`infbench_sum` + `multi_lexsum` × {8K, 16K, 32K}, n=50/cell, HELMET's own 1200-token
generation cap, bs=1, T=0.0. Reuses
[`../helmet/helmet_longctx.py`](../helmet/helmet_longctx.py) unmodified.

## Ablations (`run_ablation_27b.sh`)

```bash
MODEL=/path/Qwen3.6-27B DRAFT_DOMINO=/path/qwen3.6-27B-domino \
OUT=./ablation_out bash run_ablation_27b.sh
```

Reproduces `nano5_collect.sh` stages 1/1b and `nano5_followup.sh` parts A/B on H200,
TP=1, GRU table on, memory fraction 0.80, admission cap 2, graph max batch 2:

1. Fresh AR, Domino chain, and explicitly labeled `dominotree_flat` control, followed
   by budgets **15/32/64** at node_topk=8 and corr_topm=64.
2. At the winning budget, **node_topk {4,8,16} × corr_topm {32,64,128,0}**;
   **topM=0 means full vocabulary**. The (8,64) cell reuses its ladder observation.
3. Test budgets **8/12** at the chosen k/M and compare with the selected baseline.
4. Collect phase timings at budgets **8/15/32/64**, with
   `DOMINOTREE_STEP_TIMING=100` and two prompts per dataset. Timing adds
   synchronization; these runs are excluded from throughput and configuration selection.

Each selection uses the highest aggregate completion tokens / wall seconds
(rounded to one decimal, as in the originals) among **P9_GATE=PASS** configurations
only; ties keep the earlier observation. No passing initial budget means no selection
and a nonzero exit. Collection failures and comparator ERROR abort; FAIL/INCONCLUSIVE
configurations cannot win. All selection runs use the original 20-prompt, 1024-token
gate and fresh references/control from this run.

Each invocation creates a fresh child of `OUT` with the archived bundle's layout:
`ablations/budget_ladder/`, `ablations/node_topk_topm_grid/`, `ablations/timing/`,
and `losslessness_gate/gate_ref/`, plus server/client `logs/`. JSONL, verdicts,
summary files, and `timing_b*.steps.txt` retain the archived naming conventions.
`chosen_budget.txt`, `chosen_config.txt`, and `final_config.txt` record the successive
selections; the latter two contain `budget node_topk corr_topm`. Export those three
values as `DOMINOTREE_TREE_BUDGET`, `DOMINOTREE_NODE_TOPK`, and
`DOMINOTREE_CORR_TOPM` for subsequent benchmark runs. Outputs default to this script's
`ablation_out/`; the checked-in results directory is never an implicit destination.

## DominoTree's configuration on this device

Tree budget 15, `node_topk` 8, `corr_topm` 128, GRU input table on — this device's own
measured optimum (see `../../../results/serving/qwen36_27b_h200/ablations/`), different
from the Qwen3-4B/8B bundle's budget 16/32. `launch_server_27b.sh` defaults to exactly
this; override via `DOMINOTREE_TREE_BUDGET` / `DOMINOTREE_NODE_TOPK` /
`DOMINOTREE_CORR_TOPM` if you re-run the ablation on different hardware — **do not
assume this device's optimum transfers to yours.**

## Per-run ports + PID-scoped kills

Unlike [`../launch_server.sh`](../launch_server.sh) (which owns a fixed port per
invocation), `launch_server_27b.sh` defaults `PORT` to `20000 + PID % 20000` and
**refuses to launch if that port already answers** — the fix for an actual incident:
two collection jobs on the same host once shared a fixed port, and each job's
port-pattern `pkill` and health check silently accepted the *other* job's server for
about two hours (see `../../../results/serving/qwen36_27b_h200/PROVENANCE.txt`). The
`run_*_27b.sh` orchestrators check for an existing responder before spawning anything,
launch in a private session, and accept health only while their own child PID is alive.
Each launch gets a unique `--served-model-name`; `/v1/models` must report exactly that
identity, with PID liveness checked before and after the HTTP calls. Clients are
bracketed by the same checks and outputs are promoted only after successful collection
and verification. Health/identity curls have 2-second connect and 5-second total
timeouts. EXIT/INT/TERM traps terminate only the tracked client PID and server process
group, then reap them; no port-pattern `pkill` is used.

`DOMINOTREE_DIAG_FLAT_GDN` is unset for all ordinary methods and the positive gate
arm, even if inherited. Only the explicit diagnostic `METHOD=dominotree_flat` sets it
and gate/control outputs carry that flat-control label. Never use its throughput as a
benchmark result. HELMET writes its summary to `.part`, promotes the per-prompt
sidecar on success, and promotes the summary last, so partial output cannot trigger a
later method skip.

Servers, gate/ablation clients, concurrency, and HELMET run with
`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`; cache models and datasets beforehand.
Only bs=1 clients unset both variables, matching the originals' LiveCodeBench loader
requirements. The bs=1 wrapper sends no extra warmup request. Gate clients retain the
original 16-token warmup and concurrency retains its original configured warmup.

## Known limitation, out of scope for this benchmark

The GPU-native (frontier) tree builder's Python fallback path
(`_build_conditional_trees`) raises `IndexError` at tree budget > 16 when the frontier
builder is unavailable. Every script here launches with the frontier builder (the
plugin's default), so this is never hit in normal use — noted for anyone who disables
it and reruns at budget > 16. Not fixed on this branch; a known limitation, not a
regression these scripts introduce.

## Output

Same schemas as `../bs1/`, `../concurrency/`, `../helmet/` respectively (see their own
READMEs) — this target changes the launch flags and adds the gate, not the record
shapes. Regenerate every LaTeX table from your own run with
[`../../../results/serving/qwen36_27b_h200/gen_qwen36_27b_tables.py`](../../../results/serving/qwen36_27b_h200/gen_qwen36_27b_tables.py)
(point it at your own data directory, or drop your JSONL into that bundle's layout).
