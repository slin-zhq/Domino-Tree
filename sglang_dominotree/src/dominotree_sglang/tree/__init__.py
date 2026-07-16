"""DominoTree tree-construction package.

Phase 2 ships a fixed toy tree (``toy_tree``) to prove the EAGLE tree-verify
seam. The conditional best-first builder is Phase 3.
"""

from .toy_tree import (
    ToyTreeTopology,
    build_draft_tokens,
    build_full_attention_mask,
    build_intra_tree_mask,
    build_topology,
)

__all__ = [
    "ToyTreeTopology",
    "build_topology",
    "build_draft_tokens",
    "build_intra_tree_mask",
    "build_full_attention_mask",
]
