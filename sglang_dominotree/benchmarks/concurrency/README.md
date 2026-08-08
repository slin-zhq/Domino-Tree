# Concurrency (goodput) benchmark

Measures **aggregate output tokens/s** as the _offered_ concurrency rises, for the five
methods listed in [`../README.md`](../README.md), over the first-N prompts of real
datasets (GSM8K 128, MBPP 128, MT-Bench 80) — the same shape as Domino's own serving
tables, so the numbers are directly comparable.

Methodology per `(dataset, concurrency)` cell: flush the cache, run a warmup wave,
then issue the timed requests with `max_workers = concurrency`, reporting
`sum(completion_tokens) / wall_time` and the mean `spec_accept_length`. Prompts come
from the bs=1 driver's loader, so they are byte-identical to the bs=1 benchmark's.

## Read this before interpreting the numbers

`c` is the **offered** concurrency — how many requests the client keeps in flight.
`--max-running-requests` (the _cap_) bounds how many the server runs
**simultaneously**; the rest queue. Each method sustains a different cap on a given
GPU: a draft model costs weights plus its own KV, and tree verify additionally
materializes a `batch × draft × vocab` logits buffer, so on a memory-constrained card
DominoTree admits fewer concurrent requests than a chain.

Therefore:

- **At `c` ≤ every compared method's cap**, all methods genuinely run `c` requests and
  the comparison is like-for-like. This is the region that isolates the algorithm.
- **Past a method's cap**, its goodput plateaus because it is running its cap rather
  than the offered load. Differences there measure **admission capacity on your
  hardware**, not per-step cost. A telltale sign is a flat tail (e.g. `1142 → 1144 →
1151` across c = 8, 16, 32).
- **The cap is part of the result.** `run_conc_all.sh` records it in
  `status_<method>.done`; report it with the numbers. (That is the same filename
  `results/serving/verify_published_numbers.py` reads the published caps from, so your
  own run stays comparable to ours.)

A larger-memory GPU raises every cap and pushes each crossover to the right, so the
concurrency at which a chain overtakes a tree is a statement about the device, not
about the methods.

## Step 1 — find each method's cap

It boots at each candidate cap, bursts that many concurrent requests, and prints the
first cap that survives. Run it once **per method** — each takes a different draft
variable, and `ar` takes none:

```bash
CAPS="32 16 12 8 4"                      # candidates, tried high to low
COMMON="MODEL=./Qwen3-8B TP=2"           # target + TP; use TP=1 for the 4B target

env $COMMON METHOD=ar                                        bash find_caps.sh $CAPS
env $COMMON METHOD=dominotree   DRAFT_DOMINO=./Qwen3-8B-Domino-b16 bash find_caps.sh $CAPS
env $COMMON METHOD=domino_chain DRAFT_DOMINO=./Qwen3-8B-Domino-b16 bash find_caps.sh $CAPS
env $COMMON METHOD=dflash       DRAFT_DFLASH=./Qwen3-8B-DFlash-b16 bash find_caps.sh $CAPS
env $COMMON METHOD=eagle3       DRAFT_EAGLE3=./Qwen3-8B_eagle3     bash find_caps.sh $CAPS
```

Checkpoint identities for every method are in [`../README.md`](../README.md). Other
variables the script reads, with their defaults: `TP` (1), `PY` (`python`),
`PORT` (31690), `MEMFRAC` (0.85). `METHOD` and `MODEL` are required.

For reference, the paper's caps on **2×16 GB cards** (RTX 5080) were:

|                | AR  | EAGLE-3 | DFlash | Domino chain | DominoTree |
| -------------- | --- | ------- | ------ | ------------ | ---------- |
| Qwen3-4B, TP=1 | 32  | 32      | 32     | 16           | 12         |
| Qwen3-8B, TP=2 | 32  | 16      | 16     | 16           | 8          |

These are small-card numbers (16 GB). On a larger GPU expect all of them to rise --
possibly past 32, in which case no method's cap binds inside the default sweep and every
column is a like-for-like comparison. That is the cleaner experiment; we could not run it.

`run_conc_all.sh` also picks `MEMFRAC` by tensor-parallel degree (0.7 at TP=1, 0.85 at
TP>1), matching the validated runs. If a method OOMs mid-sweep, lower it for **all**
methods rather than just the one that failed.

## Step 2 — run the sweep

```bash
MODEL=Qwen/Qwen3-8B TP=2 \
CAP_AR=32 CAP_EAGLE3=16 CAP_DFLASH=16 CAP_DOMINO_CHAIN=16 CAP_DOMINOTREE=8 \
DRAFT_DOMINO=./Qwen3-8B-Domino-b16 \
DRAFT_DFLASH=./Qwen3-8B-DFlash-b16 \
DRAFT_EAGLE3=./Qwen3-8B_eagle3 \
  setsid bash run_conc_all.sh > orch_conc.log 2>&1 < /dev/null &
tail -f orch_conc.log
```

Knobs: `TASKS`, `CONC` (default `1,2,4,8,16,32`), `MAXNEW` (default 512), `MEMFRAC`,
`METHODS`, `OUT`. Confirm the log's `sanity spec_accept_length` exceeds 1.0 for every
speculative method before trusting a run.

`bench_concurrency.py --dry-run` exercises the whole driver without a server or a GPU,
which is a cheap way to check your task/concurrency arguments first.

## Output

One JSONL row per `(method, dataset, concurrency)` at `out/<model>/<method>/<method>.jsonl`:

```
{method, model, dataset, concurrency, tps, mean_accept, n_prompts,
 wall_s, completion_tokens, max_new_tokens, ...}
```

and a summary table as `<method>.md` alongside it. The paper's "Overall" is the **unweighted**
mean across the three datasets at each concurrency, so datasets contribute equally
despite unequal prompt counts.
