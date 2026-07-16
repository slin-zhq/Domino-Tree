"""DominoTree tree-construction package.

Phase 2 ships a fixed toy tree (``toy_tree``) to prove the EAGLE tree-verify
seam. The conditional best-first builder is Phase 3.
"""

from .best_first import TreeNode, build_best_first_tree
from .conditional_children import log_prob_topk, make_conditional_children_fn
from .gpu_expander import GraphNodeExpander
from .toy_tree import (
    ToyTreeTopology,
    build_draft_tokens,
    build_full_attention_mask,
    build_intra_tree_mask,
    build_intra_tree_mask_from_parents,
    build_topology,
)

__all__ = [
    "ToyTreeTopology",
    "build_topology",
    "build_draft_tokens",
    "build_intra_tree_mask",
    "build_intra_tree_mask_from_parents",
    "build_full_attention_mask",
    "TreeNode",
    "build_best_first_tree",
    "make_conditional_children_fn",
    "log_prob_topk",
    "GraphNodeExpander",
]
