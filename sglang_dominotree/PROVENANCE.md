# Copied Domino head — provenance and copy proof

This plugin copies Domino's GRU-correction head rather than reimplementing it, so
`--speculative-algorithm DOMINO` runs the official Domino drafter code and DominoTree adds
only its own conditional tree builder on top. This document records what was copied, from
where, and every edit made to it, and ships a script that proves the copy mechanically.

## What was copied

| File in this package                      | Copied from (official Domino SGLang fork)         |
| ----------------------------------------- | ------------------------------------------------- |
| `src/dominotree_sglang/domino_helper.py`  | `python/sglang/srt/speculative/domino_helper.py`  |
| `src/dominotree_sglang/domino_kernels.py` | `python/sglang/srt/speculative/domino_kernels.py` |
| `src/dominotree_sglang/domino_rollout.py` | `python/sglang/srt/speculative/domino_rollout.py` |

- **Source repo:** <https://github.com/jianuo-huang/Domino>
- **Branch @ commit:** `sglang-feat/dflash-domino` @ `e0d78707a089780ae3b0a23967a1de450818c42b`

The head files exist only on that fork branch — Domino's `main` does not carry them — where
the GRU head is patched into `DFlashDraftModel`.

## The complete set of edits

Two kinds, both enumerated: import-path shims, which change no behaviour, and one
functional patch to `domino_rollout.py` required for tensor parallelism.

### (a) Import-path shims

The copied files import sibling modules under `sglang.srt.speculative.*` that upstream
SGLang does not have, so those imports are re-pointed into this package. Exhaustively:

| Our copy                                         | Official source                                                              |
| ------------------------------------------------ | ---------------------------------------------------------------------------- |
| `from .config import is_dflash_domino_projector` | `from sglang.srt.speculative.dflash_utils import is_dflash_domino_projector` |
| `from .domino_helper import DFlashDominoHelper`  | `from sglang.srt.speculative.domino_helper import DFlashDominoHelper`        |
| `from .domino_kernels import (`                  | `from sglang.srt.speculative.domino_kernels import (`                        |

Plus `# PORT SHIM:` comments marking those lines. No kernel or algorithm logic is changed
by any shim. Line accounting, verified 2026-08-04:

| File                | Official lines | Ours | Code lines changed                                                         |
| ------------------- | -------------- | ---- | -------------------------------------------------------------------------- |
| `domino_kernels.py` | 609            | 631  | **0** (imports unchanged; +22 comment lines)                               |
| `domino_helper.py`  | 111            | 135  | **0** (1 import re-pointed; +24 comment lines)                             |
| `domino_rollout.py` | 1111           | 1157 | **1 functional patch** (2 imports re-pointed; +46 comment lines) — see (b) |

### (b) The one functional patch: TP>1 collective safety

In `domino_rollout.py`, upstream decides whether to use the replicated scorer from _each
rank's own_ free memory (`torch.cuda.mem_get_info`):

```python
if int(free_bytes) < required_bytes + cushion * 1024 * 1024:
```

Two ranks can reach different verdicts. The losing ranks then skip a collective the others
enter, and the server deadlocks. The decision is all-reduced so the whole TP group takes
the same branch:

```python
local_enough = int(free_bytes) >= required_bytes + cushion * 1024 * 1024
flag = torch.tensor([1 if local_enough else 0], dtype=torch.int32,
                    device=local_lm_head_weight.device)
flag = tp_group.all_reduce(flag)
all_enough = int(flag.item()) == tp_size
if not all_enough:
```

The other two hunks are the log message and its extra argument.

This is a no-op at `--tp-size 1`, where a one-rank all-reduce cannot change the verdict, so
every Qwen3-4B result is unaffected. At TP>1 it does not change which scorer is preferred,
only that all ranks agree on it. Without it, Qwen3-8B at `--tp-size 2` hangs.

## Reproduce the proof

```bash
# 1. clone the official Domino repo and fetch the fork branch
git clone https://github.com/jianuo-huang/Domino
git -C Domino fetch origin sglang-feat/dflash-domino

# 2. run the verifier (exit 0 = verbatim, exit 1 = logic drifted)
python verify_vendored_head.py --domino /path/to/Domino
```

`verify_vendored_head.py` extracts each file from the official repo at the pinned commit,
re-points the three documented import shims, reverts the declared TP patch, and requires an
exact sha256 match on what remains — so an undeclared change still fails the check. It also
asserts that our only other additions are blank and `# PORT SHIM` comment lines, and prints
the declared patch and its rationale on every run. Non-zero exit makes it usable in CI as a
regression guard against accidental edits to a copied file.

## Attribution and license

The copied files are **Apache-2.0** (verified 2026-07-29). Domino's main repository ships
no LICENSE file, but these three files come from its SGLang fork branch
`sglang-feat/dflash-domino`, whose root carries SGLang's `LICENSE` — Apache License 2.0,
Copyright 2023-2024 SGLang Team. They are therefore Apache-2.0, not the MIT license that
covers the rest of this repository. The license ships as `LICENSE.apache-2.0`, each copied
file carries the Apache notice, and the section above is our statement of changes under
Apache-2.0 section 4(b).

The correction head is the intellectual and code contribution of the Domino authors; cite
Domino for the drafter. DominoTree claims only the conditional tree builder (`tree/`), the
GPU-native node expander (`tree/gpu_expander.py`), and the SGLang integration wiring, which
[`README.md`](README.md) describes.
