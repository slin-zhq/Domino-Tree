# Vendored Domino head — provenance & verbatim-copy proof

This plugin **vendors** (copies, does not reimplement) Domino's GRU-correction
head. `--speculative-algorithm DOMINO` therefore runs the *official* Domino
drafter code, and DominoTree adds only its own conditional tree builder on top.
This document records exactly what was copied, from where, and the complete set
of edits — and ships a script that mechanically proves the copy is verbatim.

## What is vendored

| File in this package | Copied from (official Domino SGLang fork) |
|---|---|
| `src/dominotree_sglang/domino_helper.py`  | `python/sglang/srt/speculative/domino_helper.py` |
| `src/dominotree_sglang/domino_kernels.py` | `python/sglang/srt/speculative/domino_kernels.py` |
| `src/dominotree_sglang/domino_rollout.py` | `python/sglang/srt/speculative/domino_rollout.py` |

- **Source repo:** <https://github.com/jianuo-huang/Domino>
- **Branch @ commit:** `sglang-feat/dflash-domino` @ `e0d78707a089780ae3b0a23967a1de450818c42b`

The head files live on that fork branch only (Domino's `main` does not carry
them); the fork patches the GRU head into `DFlashDraftModel`.

## The complete set of edits (nothing else changed)

Two kinds of edit, both enumerated: **import-path shims** (no behaviour change) and
**one functional patch** to `domino_rollout.py` required for tensor parallelism.

### (a) Import-path shims — no behaviour change

The copied files referenced sibling modules under `sglang.srt.speculative.*`
that do not exist in upstream SGLang, so those imports are re-pointed into this
package. Every such edit, exhaustively:

| Our copy | Official source |
|---|---|
| `from .config import is_dflash_domino_projector` | `from sglang.srt.speculative.dflash_utils import is_dflash_domino_projector` |
| `from .domino_helper import DFlashDominoHelper` | `from sglang.srt.speculative.domino_helper import DFlashDominoHelper` |
| `from .domino_kernels import (` | `from sglang.srt.speculative.domino_kernels import (` |

Plus the `# PORT SHIM:` comments that flag those lines. **No kernel or algorithm logic
was changed by any shim.** Last verified 2026-08-04:

| File | Official lines | Ours | Code lines changed |
|---|---|---|---|
| `domino_kernels.py` | 609 | 631 | **0** (imports unchanged; +22 comment lines) |
| `domino_helper.py` | 111 | 135 | **0** (1 import re-pointed; +24 comment lines) |
| `domino_rollout.py` | 1111 | 1157 | **1 functional patch** (2 imports re-pointed; +46 comment lines) — see (b) |

### (b) The one functional patch: TP>1 collective safety (`domino_rollout.py`)

> **Correction (2026-08-04).** An earlier version of this document claimed *"no line of
> kernel or algorithm logic was changed"* and listed `domino_rollout.py` as **0** code
> changes. That was accurate when written and became inaccurate when we added Qwen3-8B
> tensor-parallel support; it was not updated at the time. Corrected here, and
> `verify_vendored_head.py` now **declares** the patch rather than failing on it. The
> stale claim was caught by running our own verification runbook end to end.

**What changed.** Upstream decides whether to use the replicated scorer from *each rank's
own* free memory (`torch.cuda.mem_get_info`):

```python
if int(free_bytes) < required_bytes + cushion * 1024 * 1024:
```

Two ranks can reach different verdicts. The losing ranks then skip a collective the
others enter, and the server deadlocks. We all-reduce the decision so the whole TP group
takes the same branch:

```python
local_enough = int(free_bytes) >= required_bytes + cushion * 1024 * 1024
flag = torch.tensor([1 if local_enough else 0], dtype=torch.int32,
                    device=local_lm_head_weight.device)
flag = tp_group.all_reduce(flag)
all_enough = int(flag.item()) == tp_size
if not all_enough:
```

The other two hunks are the log message and its extra argument.

**Scope.** It is a **no-op at `--tp-size 1`** — a one-rank all-reduce cannot change the
verdict — so every Qwen3-4B result is unaffected. At TP>1 it makes the already-intended
decision consistent across ranks; it does not change which scorer is *preferred*, only
that all ranks agree on it. Without the patch, Qwen3-8B at `--tp-size 2` hangs, so no
8B measurement would be possible at all.

**How the proof handles it.** `verify_vendored_head.py` declares this patch in
`PATCH_MAP`, reverts it to the official text, and *then* requires an exact sha256 match.
The patch is therefore accounted for while any **undeclared** change still fails the
check — the guarantee is not weakened, only made accurate. The verifier prints the patch
and its rationale on every run.

## Reproduce the proof yourself

```bash
# 1. clone the official Domino repo and fetch the fork branch
git clone https://github.com/jianuo-huang/Domino
git -C Domino fetch origin sglang-feat/dflash-domino

# 2. run the verifier (exit 0 == verbatim, exit 1 == logic drifted)
python verify_vendored_head.py --domino /path/to/Domino
```

`verify_vendored_head.py` extracts each file from the official repo at the pinned
commit, re-points the three documented import shims, and asserts that every
remaining line of code is byte-identical (sha256) once the one declared functional
patch in (b) is reverted, and that the only other additions on our side are
blank/`# PORT SHIM` comment lines. It exits non-zero if any logic
line has drifted — so it doubles as a **regression guard** (run it in CI to catch
any future accidental edit to a vendored file).

## Attribution / license

**License of the vendored files: Apache-2.0** (verified 2026-07-29). The Domino
project's *main* repository ships no LICENSE file, but the three files above come
from its SGLang **fork branch** `sglang-feat/dflash-domino`, whose root carries
SGLang's `LICENSE` --- Apache License 2.0, Copyright 2023-2024 SGLang Team. Those
files are therefore Apache-2.0, **not** the MIT license that covers the rest of
this repository. A copy of the license ships as `LICENSE.apache-2.0`, each vendored
file carries the Apache notice, and the "complete set of edits" section above is our
statement of changes under Apache-2.0 section 4(b).

*(An earlier version of this paragraph said the files were redistributed "under
Domino's license". That was imprecise: no such license exists in Domino's main repo.
Corrected once the fork's LICENSE was checked.)*

The vendored head is the intellectual and code contribution of the Domino
authors. Cite Domino for the drafter;
DominoTree claims only the conditional tree builder (`tree/`), the GPU-native
node expander (`tree/gpu_expander.py`), and the SGLang integration wiring.

See `PORT_NOTES.md` for the full integration rationale (which upstream
methods are subclassed/wrapped vs copied, and why).
