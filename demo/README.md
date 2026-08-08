# Live side-by-side demo

Watch a baseline and DominoTree decode the **same prompt** side by side in the
terminal, with live TPS, token/round counts, mean accepted length (τ), and
speedup badges — a quick visual sanity check that the tree is actually faster.

![record-then-replay: panes race on a shared clock](../results/README.md)
<!-- replace with an exported GIF once available -->

## How it works: record-then-replay

Each method is run **once** by its _own_ code, with the token stream and real
per-round timings written to a small JSON _cast_; then all casts are replayed on
one shared wall clock so the faster method visibly pulls ahead and finishes
first. This is the same design as the released DFlash/Domino demo, and it is
deliberate:

- **Clean timing.** Recording does zero terminal I/O inside the timed decode
  loop, so the TPS you see is not polluted by rendering.
- **Own code, unmodified.** `ar`/`domino` are driven by the released Domino
  drafter's `spec_generate`; DominoTree by this repo's tree. Domino's released
  code exposes no token-streaming hook, so replay is the only way to animate its
  pane without editing Domino's source.
- **Panes are not limited by GPU count.** Recording is sequential per GPU
  (parallel across GPUs); replay uses no GPU. Three panes on two (or one) GPUs
  is fine.

At `temperature 0`, every method converges to the _same_ text (DominoTree is
lossless w.r.t. the target's greedy decode), so the only visible difference is
speed. At `temperature > 0` each pane samples its own target and the texts
diverge; read TPS/τ, not text equality.

## Prerequisites

Same models as the benchmark (see the top-level README):

```bash
export DOMINO_CODE=/path/to/Domino/code        # released Domino repo's code/ dir
export MODEL_PATH=/path/to/Qwen3-4B            # target
export DRAFT_PATH=/path/to/Qwen3-4B-Domino-b16 # Domino drafter
pip install rich                               # only extra dependency (plus the benchmark's)
```

## Run

```bash
# typed prompt, three panes, GPUs auto-detected and load-balanced
python demo/compare.py --methods ar,domino,dominotree \
    --prompt "A cat eats nine sausages in 30 minutes. Compute the average time."

# or sample from a dataset
python demo/compare.py --methods ar,domino,dominotree --dataset gsm8k --sample-index 3
```

Knobs (all optional): `--temperature`, `--max-new-tokens`, `--budget` (tree node
budget, for `marg`/`dominotree`), `--corr-topm` (DominoTree correction width;
integer or `full_vocab` for the full-vocab correction), `--node-topk`.
Placement: `--gpus auto|cpu|0,1`. Replay: `--speed` (1.0 = real time),
`--baseline ar` (which pane the speedup× is measured against).

## Pieces

| File         | Role                                                                                           |
| ------------ | ---------------------------------------------------------------------------------------------- |
| `record.py`  | Run one method's own code once → write a cast JSON (`--dry-run` for a synthetic cast, no GPU). |
| `play.py`    | Replay ≥1 casts side by side (`rich` only, no torch): `python demo/play.py a.json b.json`      |
| `compare.py` | Orchestrate: record the chosen methods, then replay.                                           |

### UI check without a GPU

```bash
python demo/compare.py --dry-run --methods ar,domino,dominotree --prompt "hello"
```

## Notes

- The demo's `domino` pane runs eager (no CUDA graph) to match DominoTree's
  default Python builder — a like-for-like sanity check. It is a visual aid, not
  a substitute for the paper's benchmark protocol (`benchmark.py`), which is the
  source of record for reported numbers.
- Methods available: `ar`, `domino`, `marg`, `dominotree`.
