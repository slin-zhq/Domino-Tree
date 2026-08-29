"""Frontier (batched depth-synchronous) tree builder for the research harness.

Why this file exists
--------------------
The research harness builds its draft tree with the host-side best-first heap
(``dominotree.build_best_first_tree``): one ``heapq`` pop per node, each pop
running the per-node scorer.  Its cost is therefore **linear in the node
budget**, which biases every throughput number in a budget sweep against large
budgets -- the parameter the sweep is measuring.  Measured on an RTX 5080 at
realistic Qwen3-4B-Domino shapes (``scripts/domino_tree/bench_builder.py``)::

    budget     16      32      64     128
    heap     2.465   3.990   7.358  13.724 ms   (linear)
    frontier 1.619   1.711   1.855   2.187 ms   (flat)

The SGLang plugin already ships a builder without that cost model: the batched
depth-synchronous frontier builder in
``sglang_dominotree/src/dominotree_sglang/tree/frontier.py``, which advances all
lanes one depth at a time on-device and selects the global top-B at the end.  It
is the plugin's default at every batch size.

This module loads **that exact file, by path** -- not a copy -- and adapts its
batched tensor interface to the harness's ``list[TreeNode]``.  Sharing one source
is deliberate: it makes "the frontier builder produces the same tree" a property
of one implementation rather than a claim about two.

Equivalence
-----------
``frontier.py`` reproduces ``build_best_first_tree`` exactly up to real-valued
score ties (the heap breaks ties by insertion order, the frontier by
depth-then-lane order); both are valid best-first trees and tau is statistically
identical.  The upstream gate is ``gpu_expander._frontier_equivalence_suite``.
``test_frontier_equiv.py`` re-runs that check through *this* adapter, on the real
drafter, and additionally compares end-to-end output signatures.
"""

from __future__ import annotations

import importlib.util
import math
import os

from dominotree import TreeNode

def _candidate_paths():
    """Where to look for ``frontier.py``, in priority order.

    Deployments differ: the full repo carries the plugin tree, but older harness
    snapshots on the GPU boxes do not, so allow the file to sit beside this
    module (or anywhere ``DOMINOTREE_FRONTIER_SRC`` points).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.get("DOMINOTREE_FRONTIER_SRC")
    return [p for p in (
        env,
        os.path.join(here, "sglang_dominotree", "src", "dominotree_sglang",
                     "tree", "frontier.py"),
        os.path.join(here, "frontier.py"),
    ) if p]


_module = None


def _load_frontier_module():
    """Import ``tree/frontier.py`` by path.

    Loading the file directly rather than the ``dominotree_sglang`` package keeps
    the harness free of the plugin's SGLang import chain; ``frontier.py`` itself
    depends only on ``torch``.
    """
    global _module
    if _module is None:
        cands = _candidate_paths()
        src = next((p for p in cands if os.path.exists(p)), None)
        if src is None:
            raise FileNotFoundError(
                "frontier builder source not found; looked at:\n  "
                + "\n  ".join(cands)
                + "\nSet DOMINOTREE_FRONTIER_SRC to the plugin's tree/frontier.py."
            )
        print(f"[frontier-build] loading builder from {src}")
        spec = importlib.util.spec_from_file_location(
            "dominotree_frontier_src", src
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _module = mod
    return _module


class FrontierBuilder:
    """Harness-facing wrapper: build once per round, return ``list[TreeNode]``.

    Construct once per (budget, node_topk, corr_topm) configuration -- the
    underlying builder allocates static buffers and captures a CUDA graph on
    first use, so re-constructing it per round would erase the benefit.
    """

    def __init__(
        self,
        *,
        draft,
        embed_tokens,
        k_draft: int,
        prefix_len: int,
        node_topk: int,
        corr_topm: int,
        budget: int,
        mask_token_id: int,
        device,
    ) -> None:
        mod = _load_frontier_module()
        self.budget = int(budget)
        self.k_draft = int(k_draft)
        self._builder = mod.FrontierTreeBuilder(
            draft=draft,
            embed_tokens=embed_tokens,
            k_draft=int(k_draft),
            prefix_len=int(prefix_len),
            node_topk=int(node_topk),
            corr_topm=int(corr_topm),
            budget=int(budget),
            # The harness bounds depth by k_draft (dominotree.build_best_first_tree
            # is called with max_depth=k_draft), so match that, not the plugin's
            # block_size.
            max_depth=int(k_draft),
            mask_token_id=int(mask_token_id),
            device=device,
        )

    def build(self, ph, base_logits, root_state, verified_token) -> list[TreeNode]:
        """One round.

        ``ph`` ``[k_draft, hidden]``, ``base_logits`` ``[k_draft, vocab]`` and
        ``root_state`` ``(1, 1, gru_dim)`` are the harness's per-round tensors --
        the same three the conditional ``children_fn`` closes over.
        ``verified_token`` is the committed root token id.
        """
        import torch

        dev = ph.device
        verified = torch.as_tensor([int(verified_token)], dtype=torch.long, device=dev)
        _tokens, _mask, aux = self._builder.build(
            ph[: self.k_draft].unsqueeze(0),
            base_logits[: self.k_draft].unsqueeze(0),
            root_state,
            verified,
            return_aux=True,
        )
        tok = _tokens[0].tolist()
        par = aux["parents"][0].tolist()
        dep = aux["depths"][0].tolist()
        sco = aux["cum_logprobs"][0].tolist()

        # Flat index 0 is the committed root; dead-leaf pad slots carry -inf
        # scores and sort last.  Remap flat indices onto the harness's node-list
        # indices (root -> -1) rather than assuming contiguity.
        remap = {0: -1}
        nodes: list[TreeNode] = []
        for flat in range(1, len(tok)):
            if not math.isfinite(sco[flat]):
                continue  # dead-leaf padding: not a real candidate
            remap[flat] = len(nodes)
            nodes.append(
                TreeNode(
                    token=int(tok[flat]),
                    # frontier depths count the root as 0; the harness gives the
                    # root's children depth 0.
                    depth=int(dep[flat]) - 1,
                    parent=remap[int(par[flat])],
                    cum_logprob=float(sco[flat]),
                )
            )
        return nodes
