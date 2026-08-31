"""Attention-mask utilities shared by DominoTree builders and verification."""

from __future__ import annotations

import torch


def build_intra_tree_mask_from_parents(
    parents: "list[list[int]]", *, n: int, device: torch.device
) -> torch.Tensor:
    """Build an intra-tree ancestor mask from per-request parent arrays.

    ``parents[b]`` is a length-``n`` flat parent array where
    ``parents[b][0] == -1`` (root) and ``parents[b][i] < i`` (topological
    order). ``mask[b, i, j]`` is true exactly when ``j`` is an ancestor of
    ``i`` or ``i == j``, matching the convention consumed by
    ``reconstruct_indices_from_tree_mask``.
    """
    bs = len(parents)
    mask = torch.zeros((bs, n, n), dtype=torch.bool)
    for b in range(bs):
        par = parents[b]
        for i in range(n):
            j = i
            while j != -1:
                mask[b, i, j] = True
                j = par[j]
    return mask.to(device=device).contiguous()


def build_full_attention_mask(
    intra_mask: torch.Tensor,
    *,
    seq_lens_cpu: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build the flattened full allow-mask for target tree verification.

    Per request the mask is ``[N, L + N]`` =
    ``[ones(N, L) | intra_mask(N, N)]``: every tree node attends the committed
    prefix and its own tree ancestors. The request masks are flattened and
    concatenated over the batch. True means the query-key pair is allowed.
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
