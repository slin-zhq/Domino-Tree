"""Phase-2 toy tree topology for the Domino drafter.

P2 is a *plumbing proof*: prove that a branching draft tree from the Domino
drafter verifies end-to-end through SGLang's EAGLE tree-verify machinery,
losslessly at T=0. It is NOT the real conditional best-first builder (that is P3).

Shape (fixed "caterpillar"): with a verify node budget ``N`` (== the DFLASH
``block_size`` so all DFLASH KV/buffer sizing is reused) and ``num_branch``
extra branches::

    node 0            = root  (previous verified/bonus token)     depth 0
    node 1..S         = spine (Domino chain c_1..c_S)             depth 1..S
    node S+1..S+B     = branch siblings (2nd candidate)           depth 1..B

with ``S = N - 1 - num_branch`` and ``B = num_branch``. Branch ``k`` (depth
``k``) is a sibling of spine node ``k`` and a child of spine node ``k-1``.

The spine is the Domino chain, so the tree contains the chain's leading
``S``-token path; since ``S`` (e.g. 13) is far above the measured chain
acceptance (~2.7), the tree's accepted length is >= the chain's in practice, and
branches can only add acceptance. At T=0 greedy the tree verifier only emits
tokens the target would greedily produce, so output is lossless by construction.

This module builds only the *topology* (parent array, per-node tokens, the
intra-tree ancestor mask). The worker turns that into the six EAGLE topology
tensors: it feeds the intra mask to the ``reconstruct_indices_from_tree_mask``
sgl-kernel op (which derives ``positions`` + ``retrieve_index`` /
``retrieve_next_token`` / ``retrieve_next_sibling``), exactly as NGRAM does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass(frozen=True)
class ToyTreeTopology:
    """Fixed per-request tree topology (identical across the batch)."""

    num_nodes: int  # N
    spine_len: int  # S
    num_branch: int  # B
    parent: List[int]  # length N; parent[0] == -1 (root)
    depth: List[int]  # length N; depth[0] == 0
    # Node i's token source: ("root",), ("spine", j) -> spine_tokens[:, j],
    # or ("branch", k) -> branch_tokens[:, k].
    token_src: List[tuple]

    @property
    def max_depth(self) -> int:
        return max(self.depth)


def build_topology(num_nodes: int, num_branch: int) -> ToyTreeTopology:
    """Build the fixed caterpillar topology for ``N = num_nodes`` nodes.

    Requires ``num_branch >= 0`` and ``num_branch <= N - 2`` (need at least a
    root + one spine node). Each branch at depth ``k`` (1-indexed) hangs off
    spine node ``k-1``, so branches only exist for depths ``1..min(B, S)``.
    """
    n = int(num_nodes)
    b = int(num_branch)
    if n < 2:
        raise ValueError(f"toy tree needs at least 2 nodes, got N={n}.")
    if b < 0:
        raise ValueError(f"num_branch must be >= 0, got {b}.")
    s = n - 1 - b  # spine length (draft nodes on the main path)
    if s < 1:
        raise ValueError(
            f"num_branch={b} leaves spine_len={s} < 1 for N={n}; reduce num_branch."
        )
    if b > s:
        # A branch at depth k needs spine node k-1 as parent (k <= s), so at
        # most s branches can attach. Keep the toy shape well-formed.
        raise ValueError(
            f"num_branch={b} exceeds spine_len={s}; a branch at depth k needs "
            f"spine node k-1. Use num_branch <= {s}."
        )

    parent = [-1] * n
    depth = [0] * n
    token_src: List[tuple] = [("root",)]

    # Spine: nodes 1..s, node j has parent j-1, depth j.
    for j in range(1, s + 1):
        parent[j] = j - 1
        depth[j] = j
        token_src.append(("spine", j - 1))  # spine_tokens column j-1 == c_j

    # Branches: node s+k (k=1..b) at depth k, sibling of spine node k, child of
    # spine node k-1.
    for k in range(1, b + 1):
        node = s + k
        parent[node] = k - 1
        depth[node] = k
        token_src.append(("branch", k - 1))  # branch_tokens column k-1

    return ToyTreeTopology(
        num_nodes=n,
        spine_len=s,
        num_branch=b,
        parent=parent,
        depth=depth,
        token_src=token_src,
    )


def build_draft_tokens(
    topo: ToyTreeTopology,
    *,
    verified_id: torch.Tensor,  # [bs] int
    spine_tokens: torch.Tensor,  # [bs, >= S] int (Domino chain)
    branch_tokens: torch.Tensor,  # [bs, >= B] int (2nd candidates)
) -> torch.Tensor:
    """Assemble the flat verify-tree tokens ``[bs, N]`` (node order).

    Node 0 is the root (previous verified token); the remaining nodes follow the
    topology's ``token_src`` map.
    """
    bs = int(verified_id.shape[0])
    device = verified_id.device
    n = topo.num_nodes
    draft_tokens = torch.empty((bs, n), dtype=torch.int64, device=device)
    for node, src in enumerate(topo.token_src):
        if src[0] == "root":
            draft_tokens[:, node] = verified_id.to(torch.int64)
        elif src[0] == "spine":
            draft_tokens[:, node] = spine_tokens[:, src[1]].to(torch.int64)
        elif src[0] == "branch":
            draft_tokens[:, node] = branch_tokens[:, src[1]].to(torch.int64)
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown token source {src!r}")
    return draft_tokens


def build_intra_tree_mask(
    topo: ToyTreeTopology, *, bs: int, device: torch.device
) -> torch.Tensor:
    """Build the intra-tree ancestor mask ``[bs, N, N]`` (bool).

    ``mask[b, i, j] == True`` iff node ``j`` is an ancestor of node ``i`` or
    ``i == j`` (query ``i`` attends its own root-path). This is the exact input
    ``reconstruct_indices_from_tree_mask`` consumes to recover parent/child/
    sibling links and absolute positions. Identical across the batch for the
    fixed toy tree.
    """
    n = topo.num_nodes
    single = torch.zeros((n, n), dtype=torch.bool)
    for i in range(n):
        j = i
        # Walk up the parent chain from i to the root, marking ancestors + self.
        while j != -1:
            single[i, j] = True
            j = topo.parent[j]
    return single.to(device=device).unsqueeze(0).expand(bs, n, n).contiguous()


def build_full_attention_mask(
    intra_mask: torch.Tensor,  # [bs, N, N] bool
    *,
    seq_lens_cpu: torch.Tensor,  # [bs] int (prefix lengths L per req)
    device: torch.device,
) -> torch.Tensor:
    """Build the flattened FULL allow-mask for the target attention backend.

    Per request the verify attention mask is ``[N, L + N]`` = ``[ones(N, L) |
    intra_mask[N, N]]`` (each tree node attends the whole committed prefix plus
    its own tree ancestors), flattened and concatenated over the batch. Mirrors
    NGRAM's ``USE_FULL_MASK`` construction (ngram_worker.py:301-320). True means
    the (query, key) pair is allowed.
    """
    bs, n, _ = intra_mask.shape
    parts = []
    seq_lens_list = seq_lens_cpu.tolist()
    for i in range(bs):
        seq_len = int(seq_lens_list[i])
        prefix = torch.ones((n, seq_len), dtype=torch.bool, device=device)
        req_mask = torch.cat((prefix, intra_mask[i].to(device)), dim=1)
        parts.append(req_mask.reshape(-1))
    return torch.cat(parts, dim=0)
