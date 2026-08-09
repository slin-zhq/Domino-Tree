# Single-request (bs=1) serving benchmark

Measures **speculative-decoding acceptance length τ** (`meta_info.spec_accept_length`)
and single-request throughput inside SGLang, for the five methods listed in
[`../README.md`](../README.md), across the eight datasets and three temperatures used
in the paper.

This is the serving counterpart of the repo's HF-harness `benchmark.py`: it drives a
running SGLang server over HTTP but keeps that harness's methodology exactly —
the same dataset loading and `shuffle(seed=0).select(range(N))` subsetting, the same
chat templating, the same in-loop warmup prompt before timing, and the same
per-prompt seed. See the module docstring of [`bench_bs1.py`](bench_bs1.py) for the
point-by-point correspondence and for the few places where serving necessarily
differs (e.g. latency includes prefill and HTTP overhead).

## Run it

```bash
MODEL=Qwen/Qwen3-4B \
DRAFT_DOMINO=./Qwen3-4B-Domino-b16 \
DRAFT_DFLASH=./Qwen3-4B-DFlash-b16 \
DRAFT_EAGLE3=./Qwen3-4B_eagle3 \
  setsid bash run_bs1_all.sh > orch_bs1.log 2>&1 < /dev/null &
tail -F orch_bs1.log
```

For a larger target on 2 GPUs add `TP=2`. Useful knobs: `NSAMPLES` (prompts per
dataset, default 50), `MAXNEW` (default 2048), `DATASETS`, `TEMPS`, `MEMFRAC`,
`METHODS`, `OUT`.

**`DRAFT_DOMINO` / `DRAFT_DFLASH` / `DRAFT_EAGLE3`** must point at where you actually put
each checkpoint (see the table in [`../README.md`](../README.md)) — `./Qwen3-4B-...`
above is a placeholder. A path that does not exist fails **inside SGLang**, not in this
script, with a misleading `HFValidationError: Repo id must use alphanumeric chars...`
(it means "not found locally, so treated as a Hub repo id and rejected," not a launcher
bug). Use the real absolute path, e.g. `~/models/Qwen3-4B-Domino-b16`.
Also use `tail -F` (capital), not `-f`: the log file is created a moment *after*
`[1] <pid>` prints, so `-f` run immediately after can lose that race.

**Quick check before the full run** (~minutes, one dataset, one temperature):

```bash
MODEL=Qwen/Qwen3-4B DRAFT_DOMINO=./Qwen3-4B-Domino-b16 \
DATASETS=gsm8k TEMPS=0.0 NSAMPLES=5 MAXNEW=256 METHODS="dominotree ar" \
  bash run_bs1_all.sh
```

Confirm the log's `sanity spec_accept_length` is above 1.0 for every speculative
method — if it is 1.0 or `None`, speculation is not actually running and every
downstream number is meaningless.

The orchestrator skips any `(method, dataset, temperature)` whose output file already
exists, so an interrupted sweep resumes by re-running the same command.

## Output

One JSONL per `(method, dataset, temperature)` at
`out/<model>/<method>/<dataset>_T<temp>.jsonl`, one record per timed prompt (per turn
for MT-Bench):

```
{method, dataset, temperature, sample_idx, turn_index, num_output,
 decode_time, tps, mean_accept, ...}
```

`mean_accept` is τ. A method's throughput for a cell is the mean of `tps` over its
warm prompts; speedup is that divided by AR's on the same cell. Aggregate with your
preferred paired analysis over these records.

## Notes

- **bs=1 means bs=1.** Only one request is ever in flight, so the admission cap does
  not affect timing; the default (`MAXRUN=64`, `MEMFRAC=0.7`) simply matches the runs
  that produced the paper's numbers.
- **Temperature.** DominoTree drafts deterministically and samples the target,
  accepting a draft token iff it matches — the committed output is the target's own
  temperature-sampled sequence, so every method is lossless at every temperature.
- **The 4B and 8B grids are not cross-comparable** if you run them at different
  tensor-parallel degrees or on different GPUs. Read each grid internally.
