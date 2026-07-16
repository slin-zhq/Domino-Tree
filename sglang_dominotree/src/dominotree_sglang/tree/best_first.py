"""Best-first draft-tree builder — ported VERBATIM from DominoTree's
``dominotree.py`` (``TreeNode`` + ``build_best_first_tree``).

This is the paper's actual tree-construction core: a global best-first heap over
cumulative drafter log-prob. The scorer (``children_fn``) is supplied by the
caller — for DominoTree it is the path-dependent Domino GRU-correction callback
in ``conditional_children.py``.

Invariant relied on downstream: ``node.parent < i`` for every ``nodes[i]`` — a
parent always appears earlier in the flattened list, so the list is a valid
topological (parents-before-children) order (checked by the source self-test,
dominotree.py:179,221).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable


@dataclass
class TreeNode:
    """One flattened draft-tree node.

    ``parent`` indexes into the returned node list; ``-1`` means the node is a
    direct child of the committed root/bonus token.
    """

    token: int
    depth: int
    parent: int
    cum_logprob: float


# children_fn(state, child_depth) -> (tokens, logprobs, child_states)
# ``state`` is opaque to this module. For the conditional Domino scorer it is
# the GRU state for the node's root-to-node path.
ChildrenFn = Callable[[object, int], "tuple[list[int], list[float], list[object]]"]


def build_best_first_tree(
    children_fn: ChildrenFn,
    root_state: object,
    budget: int,
    max_depth: int,
) -> list[TreeNode]:
    """Build up to ``budget`` nodes by global best-first expansion.

    The heap semantics match the DDTree/CaDDTree best-first order: pop the
    highest cumulative-log-prob prefix, append it, then push that node's
    children. The scorer is entirely supplied by ``children_fn``.
    """

    if budget <= 0 or max_depth <= 0:
        return []

    nodes: list[TreeNode] = []
    heap: list[tuple[float, int, int, int, int, object]] = []
    tie = 0

    tokens, logprobs, states = children_fn(root_state, 0)
    for token, logprob, state in zip(tokens, logprobs, states):
        heapq.heappush(heap, (-float(logprob), tie, int(token), 0, -1, state))
        tie += 1

    while heap and len(nodes) < budget:
        neg_cum, _, token, depth, parent_idx, state = heapq.heappop(heap)
        idx = len(nodes)
        nodes.append(
            TreeNode(token=token, depth=depth, parent=parent_idx, cum_logprob=-neg_cum)
        )

        if depth + 1 < max_depth:
            child_tokens, child_logprobs, child_states = children_fn(state, depth + 1)
            for child_token, child_logprob, child_state in zip(
                child_tokens, child_logprobs, child_states
            ):
                heapq.heappush(
                    heap,
                    (
                        neg_cum - float(child_logprob),
                        tie,
                        int(child_token),
                        depth + 1,
                        idx,
                        child_state,
                    ),
                )
                tie += 1

    return nodes
