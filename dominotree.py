"""Pure-Python DominoTree tree construction and verification bookkeeping.

The key invariant is that the same best-first heap builder is used for both:

* marg: a DDTree analogue whose children distribution depends only on depth.
* dominotree: the conditional scorer (this paper's method) whose children
  distribution is supplied by a path-dependent Domino GRU-correction callback.

This file intentionally excludes the experimental/deferred builders from the
research branch (beam, wave/condwave, hybrid, condadaptive, and CUDA/star
variants). Run the CPU self-test with:

    python dominotree.py
"""

from __future__ import annotations

import heapq
import math
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
        nodes.append(TreeNode(token=token, depth=depth, parent=parent_idx, cum_logprob=-neg_cum))

        if depth + 1 < max_depth:
            child_tokens, child_logprobs, child_states = children_fn(state, depth + 1)
            for child_token, child_logprob, child_state in zip(child_tokens, child_logprobs, child_states):
                heapq.heappush(
                    heap,
                    (neg_cum - float(child_logprob), tie, int(child_token), depth + 1, idx, child_state),
                )
                tie += 1

    return nodes


def build_attention_rows(nodes: list[TreeNode]) -> list[list[int]]:
    """Return tree-attention visibility rows for flattened positions.

    Position 0 is the committed root. Position ``1+i`` is ``nodes[i]``. Each
    node attends to the root, itself, and its ancestors. The caller adds the
    full prefix/context visibility.
    """

    rows: list[list[int]] = [[] for _ in range(1 + len(nodes))]
    rows[0] = [0]
    for i, node in enumerate(nodes):
        pos = 1 + i
        allowed = [0, pos]
        parent = node.parent
        while parent >= 0:
            allowed.append(1 + parent)
            parent = nodes[parent].parent
        rows[pos] = sorted(set(allowed))
    return rows


def position_ids(nodes: list[TreeNode], root_position: int) -> list[int]:
    """Absolute RoPE position ids for ``[root, *nodes]``."""

    return [root_position] + [root_position + 1 + node.depth for node in nodes]


def longest_accepted_path(
    nodes: list[TreeNode],
    root_posterior: int,
    node_posteriors: list[int],
) -> tuple[int, list[int]]:
    """Greedy tree verification under target posterior samples/argmaxes.

    ``root_posterior`` predicts the depth-0 token. ``node_posteriors[i]``
    predicts the child token after node ``i``. The returned path is the longest
    accepted root-to-node path.
    """

    children: list[list[int]] = [[] for _ in range(len(nodes))]
    roots: list[int] = []
    for i, node in enumerate(nodes):
        if node.parent == -1:
            roots.append(i)
        else:
            children[node.parent].append(i)

    best_len = 0
    best_path: list[int] = []

    def dfs(idx: int, path: list[int]) -> None:
        nonlocal best_len, best_path
        if len(path) > best_len:
            best_len = len(path)
            best_path = path.copy()
        pred = node_posteriors[idx]
        for child in children[idx]:
            if nodes[child].token == pred:
                dfs(child, path + [child])

    for root in roots:
        if nodes[root].token == root_posterior:
            dfs(root, [root])

    return best_len, best_path


def make_marginal_children_fn(per_depth_topk_tokens, per_depth_topk_logprobs):
    """Create a path-independent DDTree-style children function."""

    def children_fn(_state, child_depth):
        if child_depth >= len(per_depth_topk_tokens):
            return [], [], []
        tokens = per_depth_topk_tokens[child_depth]
        logprobs = per_depth_topk_logprobs[child_depth]
        return list(tokens), list(logprobs), [None] * len(tokens)

    return children_fn


def _self_test() -> None:
    toks = [[10, 11], [20, 21], [30, 31]]
    lps = [
        [math.log(0.7), math.log(0.3)],
        [math.log(0.6), math.log(0.4)],
        [math.log(0.8), math.log(0.2)],
    ]
    nodes = build_best_first_tree(make_marginal_children_fn(toks, lps), None, budget=4, max_depth=3)
    assert len(nodes) == 4
    assert nodes[0].token == 10 and nodes[0].depth == 0 and nodes[0].parent == -1
    cums = [node.cum_logprob for node in nodes]
    assert all(cums[i] >= cums[i + 1] - 1e-9 for i in range(len(cums) - 1)), cums
    for i, node in enumerate(nodes):
        assert node.parent < i

    rows = build_attention_rows(nodes)
    assert rows[0] == [0]
    for i, node in enumerate(nodes):
        pos = 1 + i
        assert 0 in rows[pos] and pos in rows[pos]
        assert max(rows[pos]) <= pos
        assert len(rows[pos]) - 2 == node.depth

    pids = position_ids(nodes, root_position=100)
    assert pids[0] == 100
    for i, node in enumerate(nodes):
        assert pids[1 + i] == 101 + node.depth

    chain = build_best_first_tree(
        make_marginal_children_fn([[10], [20], [30]], [[0.0], [0.0], [0.0]]),
        None,
        budget=3,
        max_depth=3,
    )
    acc_len, path = longest_accepted_path(chain, root_posterior=10, node_posteriors=[20, 30, 999])
    assert acc_len == 3 and [chain[i].token for i in path] == [10, 20, 30]
    acc_len, path = longest_accepted_path(chain, root_posterior=10, node_posteriors=[777, 30, 999])
    assert acc_len == 1 and [chain[i].token for i in path] == [10]
    acc_len, _ = longest_accepted_path(chain, root_posterior=999, node_posteriors=[20, 30, 999])
    assert acc_len == 0

    def cond_children_fn(state, child_depth):
        if child_depth >= 3:
            return [], [], []
        state_int = 0 if state is None else int(state)
        a, b = 40 + child_depth, 50 + child_depth
        if state_int % 2 == 0:
            tokens, logprobs = [a, b], [math.log(0.8), math.log(0.2)]
        else:
            tokens, logprobs = [b, a], [math.log(0.8), math.log(0.2)]
        return tokens, logprobs, [state_int + 1, state_int + 1]

    cond_nodes = build_best_first_tree(cond_children_fn, root_state=0, budget=5, max_depth=3)
    assert len(cond_nodes) == 5
    for i, node in enumerate(cond_nodes):
        assert node.parent < i
    for pos, row in enumerate(build_attention_rows(cond_nodes)):
        assert max(row) <= pos

    # A children_fn may be a stateful bound method (as the GPU-native expander
    # uses); the builder must not care. Same toy scorer as above, via a class.
    class _StatefulScorer:
        def children_fn(self, state, child_depth):
            return cond_children_fn(state, child_depth)

    method_nodes = build_best_first_tree(_StatefulScorer().children_fn, root_state=0, budget=5, max_depth=3)
    assert [(n.token, n.depth, n.parent) for n in method_nodes] == [
        (n.token, n.depth, n.parent) for n in cond_nodes
    ]

    # GPU-native builder equivalence suite (torch-dependent; skipped if absent).
    try:
        import dominotree_gpu
    except ImportError:
        print("dominotree self-test: torch not installed; skipping dominotree_gpu equivalence suite")
    else:
        dominotree_gpu._self_test()

    print("dominotree self-test: ALL PASSED")


if __name__ == "__main__":
    _self_test()
