"""DominoTree conditional (Domino GRU-correction) children function.

Ported op-for-op from ``domino_adapter.py:120-194`` (the ``make_conditional_children_fn``
reference), adapted to the SGLang worker's model handles and restricted to the
GREEDY / T=0 path (no ``sample_draft`` branch — DominoTree is lossless by
construction under greedy tree verification, so the drafted candidate set is
deterministic top-k). Contract: ``docs/domino_tree_sglang_integration/dominotree_builder_contract.md``
Sections 1-3.

``children_fn(state, depth) -> (tokens: list[int], logprobs: list[float],
child_states: list[Tensor(1,1,gru_dim)])`` — length ``node_topk`` each, empty
past ``depth >= k_draft``. Three cases (Section 3):

* ``depth < prefix_len``            : uncorrected base-logit top-k (path-independent).
* ``depth >= prefix_len, corr_topm>0``: GRU correction restricted to the top-``corr_topm``
                                        marginal candidates per depth, re-ranked.
* ``depth >= prefix_len, corr_topm==0``: full-vocab GRU correction.
"""

from __future__ import annotations


def log_prob_topk(logits_row, k: int):
    """Deterministic top-k over ``log_softmax(logits_row)`` (domino_adapter.py:26-31)."""
    import torch

    logp = torch.log_softmax(logits_row.float(), dim=-1)
    vals, idx = torch.topk(logp, k=k, dim=-1)
    return idx.tolist(), vals.tolist()


def make_conditional_children_fn(
    *,
    ph,  # [k_draft, hidden] per-position draft hidden (already shift_label-sliced)
    base_logits,  # [k_draft, vocab] uncorrected marginals = target.lm_head(ph)
    draft_model,  # owns prefix_gru + embed_proj
    embed_tokens,  # target embedding module (Domino uses the TARGET table)
    node_topk: int,
    corr_topm: int,
    prefix_len: int,
    device,
):
    """Return the DominoTree conditional ``children_fn`` (greedy).

    Op-for-op transcription of ``domino_adapter.make_conditional_children_fn``
    with ``sample_draft=False``. ``k_draft = ph.shape[0]`` bounds the depth.
    """
    import torch

    k_draft = int(ph.shape[0])
    prefix_gru = draft_model.prefix_gru
    embed_proj = draft_model.embed_proj

    # Restricted-correction precompute (once per round), corr_topm > 0 only.
    cand = w2c = basec = None
    if corr_topm > 0:
        # node_topk <= corr_topm is required (dominotree_gpu.py:79-80): the
        # per-node topk selects from the corr_topm candidate rows.
        node_topk = min(int(node_topk), int(corr_topm))
        cand = [torch.topk(base_logits[d], corr_topm).indices for d in range(k_draft)]
        w2c = [embed_proj[2].weight[cand[d]].float() for d in range(k_draft)]
        basec = [base_logits[d][cand[d]].float() for d in range(k_draft)]

    def children_fn(state, depth):
        if depth >= k_draft:
            return [], [], []

        if depth < prefix_len:
            # Case 1: pure-draft prefix, uncorrected. Path-independent.
            toks, lps = log_prob_topk(base_logits[depth], node_topk)
        elif cand is not None:
            # Case 2: restricted correction (published default corr_topm=64).
            s_feat = state.transpose(0, 1).squeeze(0)  # (1, gru_dim)
            h = embed_proj[1](
                embed_proj[0](
                    torch.cat([ph[depth].unsqueeze(0), s_feat], dim=-1)
                )
            ).float()  # Linear(hidden+gru -> mlp) + SiLU
            corrected_logits = basec[depth] + (h @ w2c[depth].t())[0]
            vals, local_idx = torch.topk(
                torch.log_softmax(corrected_logits, dim=-1), k=node_topk
            )
            toks, lps = cand[depth][local_idx].tolist(), vals.tolist()
        else:
            # Case 3: full-vocab correction (corr_topm == 0).
            s_feat = state.transpose(0, 1).squeeze(0)
            corr = embed_proj(torch.cat([ph[depth].unsqueeze(0), s_feat], dim=-1))
            full_logits = (base_logits[depth].unsqueeze(0) + corr).float()
            vals, idx = torch.topk(
                torch.log_softmax(full_logits, dim=-1), k=node_topk, dim=-1
            )
            toks, lps = idx[0].tolist(), vals[0].tolist()

        # Both cases: advance the GRU one step per child to produce child states.
        emb = embed_tokens(torch.tensor(toks, device=device)).unsqueeze(1)
        h0 = state.expand(1, len(toks), state.shape[-1]).contiguous()
        _, hn = prefix_gru(emb, h0)
        states = [hn[:, k : k + 1, :].contiguous() for k in range(len(toks))]
        return toks, lps, states

    return children_fn
