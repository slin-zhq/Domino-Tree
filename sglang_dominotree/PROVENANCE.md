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

The **only** differences from the official source are import-path shims: the
copied files referenced two sibling modules under `sglang.srt.speculative.*`
that do not exist in upstream SGLang, so those imports are re-pointed into this
package. Every edit, exhaustively:

| Our copy | Official source |
|---|---|
| `from .config import is_dflash_domino_projector` | `from sglang.srt.speculative.dflash_utils import is_dflash_domino_projector` |
| `from .domino_helper import DFlashDominoHelper` | `from sglang.srt.speculative.domino_helper import DFlashDominoHelper` |
| `from .domino_kernels import (` | `from sglang.srt.speculative.domino_kernels import (` |

Plus the `# PORT SHIM:` comments that flag those three lines. **No line of
kernel or algorithm logic was changed.** Last verified:

| File | Official lines | Ours | Code lines changed |
|---|---|---|---|
| `domino_kernels.py` | 609 | 609 | **0** (byte-identical) |
| `domino_helper.py` | 111 | 113 | **0** (1 import re-pointed, +2 comment lines) |
| `domino_rollout.py` | 1111 | 1115 | **0** (2 imports re-pointed, + comments) |

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
remaining line of code is byte-identical (sha256), and that the only additions on
our side are blank/`# PORT SHIM` comment lines. It exits non-zero if any logic
line has drifted — so it doubles as a **regression guard** (run it in CI to catch
any future accidental edit to a vendored file).

## Attribution / license

The vendored head is the intellectual and code contribution of the Domino
authors, redistributed here under Domino's license. Cite Domino for the drafter;
DominoTree claims only the conditional tree builder (`tree/`), the GPU-native
node expander (`tree/gpu_expander.py`), and the SGLang integration wiring.

See `PORT_NOTES.md` for the full integration rationale (which upstream
methods are subclassed/wrapped vs copied, and why).
