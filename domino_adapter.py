"""Thin Domino adapter for the public DominoTree benchmark.

This file vendors no Domino code. ``benchmark.py --domino-code`` loads the
user's cloned Domino repository, and this adapter only calls the passed-in
``target`` and ``draft`` objects after ``add_path`` exposes that clone.

It implements DominoTree's Domino-facing pieces:

* the path-dependent per-node GRU correction used by DominoTree,
* the marginal children function for the DDTree-analogue control,
* one draft-block forward,
* the corrected chain verifier,
* KV-cache compaction after tree verification.
"""

from __future__ import annotations

import sys


def add_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def log_prob_topk(logits_row, k: int):
    import torch

    logp = torch.log_softmax(logits_row.float(), dim=-1)
    vals, idx = torch.topk(logp, k=k, dim=-1)
    return idx.tolist(), vals.tolist()


def cache_gather(cache, keep_indices: list[int], device) -> None:
    """Keep only ``keep_indices`` in every DynamicCache layer, in order."""

    import torch

    idx = torch.tensor(keep_indices, device=device, dtype=torch.long)
    for layer in cache.layers:
        if getattr(layer, "keys", None) is not None:
            layer.keys = layer.keys.index_select(2, idx).contiguous()
            layer.values = layer.values.index_select(2, idx).contiguous()


def draft_block(
    *,
    target,
    draft,
    target_hidden,
    output_ids,
    position_ids,
    start: int,
    past_kv_draft,
    block_size: int,
    shift_label: bool,
):
    """Run one Domino draft block and return ``(ph, base_logits, root_state)``."""

    block_output_ids = output_ids[:, start : start + block_size].clone()
    noise = target.model.embed_tokens(block_output_ids)
    ph_full = draft(
        target_hidden=target_hidden,
        noise_embedding=noise,
        position_ids=position_ids[:, past_kv_draft.get_seq_length() : start + block_size],
        past_key_values=past_kv_draft,
        use_cache=True,
        is_causal=False,
    )
    if not shift_label:
        ph_full = ph_full[:, -block_size + 1 :, :]
    past_kv_draft.crop(start)
    base_logits = target.lm_head(ph_full)[0]
    ph = ph_full[0]
    _, root_state = draft.prefix_gru(target.model.embed_tokens(output_ids[:, start : start + 1]))
    return ph, base_logits, root_state


def make_marginal_children_fn(base_logits, k_draft: int, node_topk: int):
    import dominotree

    per = [log_prob_topk(base_logits[d], node_topk) for d in range(k_draft)]
    return dominotree.make_marginal_children_fn([p[0] for p in per], [p[1] for p in per])


def make_conditional_children_fn(
    *,
    target,
    draft,
    ph,
    base_logits,
    k_draft: int,
    prefix_len: int,
    node_topk: int,
    corr_topm: int,
    device,
):
    """Return the DominoTree (conditional) children function.

    For depths before Domino's pure-draft prefix, this uses backbone/base logits.
    After that, it applies Domino's GRU correction for each node state. When
    ``corr_topm > 0``, the correction is restricted to marginal top-M candidates
    per depth and re-ranked, matching the reported public configuration.
    """

    import torch

    cand = w2c = basec = None
    embed_proj = draft.embed_proj
    if corr_topm > 0:
        cand = [torch.topk(base_logits[d], corr_topm).indices for d in range(k_draft)]
        w2c = [embed_proj[2].weight[cand[d]].float() for d in range(k_draft)]
        basec = [base_logits[d][cand[d]].float() for d in range(k_draft)]

    def children_fn(state, depth):
        if depth >= k_draft:
            return [], [], []
        if depth < prefix_len:
            toks, lps = log_prob_topk(base_logits[depth], node_topk)
        elif cand is not None:
            s_feat = state.transpose(0, 1).squeeze(0)
            h = embed_proj[1](embed_proj[0](torch.cat([ph[depth].unsqueeze(0), s_feat], dim=-1))).float()
            corrected_logits = basec[depth] + (h @ w2c[depth].t())[0]
            vals, local_idx = torch.topk(torch.log_softmax(corrected_logits, dim=-1), k=node_topk)
            toks, lps = cand[depth][local_idx].tolist(), vals.tolist()
        else:
            s_feat = state.transpose(0, 1).squeeze(0)
            corr = draft.embed_proj(torch.cat([ph[depth].unsqueeze(0), s_feat], dim=-1))
            logp = torch.log_softmax((base_logits[depth].unsqueeze(0) + corr).float(), dim=-1)
            vals, idx = torch.topk(logp, k=node_topk, dim=-1)
            toks, lps = idx[0].tolist(), vals[0].tolist()

        emb = target.model.embed_tokens(torch.tensor(toks, device=device)).unsqueeze(1)
        h0 = state.expand(1, len(toks), state.shape[-1]).contiguous()
        _, hn = draft.prefix_gru(emb, h0)
        states = [hn[:, k : k + 1, :].contiguous() for k in range(len(toks))]
        return toks, lps, states

    return children_fn


def verify_domino_chain(
    *,
    target,
    draft,
    sample,
    extract_context_feature,
    output_ids,
    position_ids,
    start: int,
    k_draft: int,
    prefix_len: int,
    mask_token_id: int,
    base_logits,
    ph,
    past_kv,
    layer_ids,
    temperature: float,
    device,
    cuda_t,
):
    """Run Domino's published corrected chain verifier for one round.

    Returns ``(acc, target_hidden, stage_ms)``. ``stage_ms`` splits the round
    into the same ``build``/``verify``/``commit`` stages the tree path reports,
    with identical boundaries, so every method is timed the same way:
    ``build`` is the sequential GRU-correction that constructs the chain,
    ``verify`` is the single target forward, and ``commit`` is the acceptance
    check plus KV/output write. ``cuda_t`` is the caller's timer (it must
    synchronize the device before reading the clock).
    """

    import torch

    t0 = cuda_t()
    root_token_id = output_ids[0, start].item()
    verify_ids = torch.full((1, k_draft + 1), mask_token_id, dtype=torch.long, device=device)
    verify_ids[0, 0] = root_token_id
    for i in range(prefix_len):
        verify_ids[0, i + 1] = base_logits[i].argmax(dim=-1)
    _, state = draft.prefix_gru(target.model.embed_tokens(verify_ids[:, : 1 + prefix_len]))
    for i in range(prefix_len, k_draft):
        s_feat = state.transpose(0, 1).squeeze(0)
        corr = draft.embed_proj(torch.cat([ph[i].unsqueeze(0), s_feat], dim=-1))
        token_i = (base_logits[i].unsqueeze(0) + corr).argmax(dim=-1)
        verify_ids[0, i + 1] = token_i
        if i + 1 < k_draft:
            _, state = draft.prefix_gru(target.model.embed_tokens(token_i.view(1, 1)), state)
    build_ms = cuda_t() - t0

    t0 = cuda_t()
    vout = target(
        verify_ids,
        position_ids=position_ids[:, start : start + k_draft + 1],
        past_key_values=past_kv,
        use_cache=True,
        output_hidden_states=True,
    )
    verify_ms = cuda_t() - t0

    t0 = cuda_t()
    post = sample(vout.logits, temperature)
    acc = (verify_ids[:, 1:] == post[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
    output_ids[:, start : start + acc + 1] = verify_ids[:, : acc + 1]
    output_ids[:, start + acc + 1] = post[:, acc]
    past_kv.crop(start + acc + 1)
    target_hidden = extract_context_feature(vout.hidden_states, layer_ids)[:, : acc + 1, :]
    commit_ms = cuda_t() - t0
    return acc, target_hidden, {"build": build_ms, "verify": verify_ms, "commit": commit_ms}
