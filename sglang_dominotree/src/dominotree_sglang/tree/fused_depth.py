"""Triton fusion of the frontier builder's per-depth body.

WHY THIS EXISTS
---------------
``frontier.py`` is GPU-**dispatch**-bound, and the cost model measured on the
RTX 5080 at Qwen3-4B shapes is exact: *build latency = kernels x 2.5 us*
(STATUS.md D64).  GPU time accounts for 100% of the captured graph's wall
clock, and the largest GEMM in the depth loop -- about 8 MFLOP -- takes 11.6 us,
roughly 250x its own arithmetic.  Nothing here is compute-bound, bandwidth-bound
or host-bound, so the ONLY thing that makes this builder faster is issuing fewer
kernels.

Round 1 (``DOMINOTREE_BUILDER_FUSION``) removed 155 launches by algebra alone
and bought -0.268 ms.  What is left is a per-depth body of ~34 tiny kernels x 16
depths.  This module collapses the two dispatch-heaviest stretches of that body
into one Triton kernel each:

``score_depth``   scores all W x k children of the frontier.  Replaces the
                  broadcast add, SiLU, two dtype casts, the ``w2[cand]``
                  index_select, the bmm, log_softmax, topk, the candidate
                  gather, the cumulative-score add and three ledger writes --
                  about 14 launches -- with 1.

``select_depth``  keeps the best W children and prepares the GRU inputs.
                  Replaces topk, the token gather, the parent-lane divide and
                  the lane-state gather -- about 6 launches -- with 1.

The remaining per-depth launches are the lane-half GEMM, the token embedding
and cuDNN's GRU cell, all of which stay in torch: they are real GEMMs, and
replacing cuDNN's GRU would perturb numerics for no dispatch win worth the risk.

EXACTNESS
---------
These kernels are a faithful re-expression of the torch ops they replace, in the
same precisions and the same order of operations -- the model-dtype add and
SiLU, the fp32 bmm (``input_precision="ieee"``, never TF32), the fp32
log_softmax, descending top-k with lowest-index tie-breaking.  They are NOT
bit-identical: floating-point reductions inside a Triton block accumulate in a
different order than cuBLAS/ATen, so results differ by ~1 ulp, exactly like the
round-1 weight split.  The gate is therefore the equivalence suite on TIE-FREE
inputs (``gpu_expander._self_test``), which asserts identical flat
``(token, parent, depth)`` lists against the per-request reference heap, plus an
end-to-end tau A/B -- not a bitwise comparison.

The kernels are CUDA-graph safe: static shapes, no host syncs, and Triton JIT
compilation happens during the builder's warmup iterations, before capture.

Gated and ON by default (``DOMINOTREE_BUILDER_FUSED_DEPTH=0`` restores the
torch depth loop in the same build).  The gate it passed: the CPU and CUDA
equivalence suites with the full fused stack, kernel engagement verified by
counting launches, and an end-to-end same-session A/B at 4B/TP=1 showing
+4.62% TPS [+4.43, +4.80] with tau unchanged to +0.01% (STATUS.md D67).

VERSION 2 (D74, the default; ``DOMINOTREE_BUILDER_FUSED_DEPTH=1`` keeps v1)
---------------------------------------------------------------------------
An external review (D73) showed the v1 inventory was double-counted and that
"the floor" had not been reached. v2 removes what was actually left:

* one GEMM per depth instead of two: the scorer's lane half and the GRU's
  hidden gates read the same lane states, so their weights are stacked;
* the selector ranks all W*k children in parallel instead of W serial
  argmaxes, and copies no state -- the GRU kernel gathers parent rows by
  index (a row-wise linear map commutes with a row gather);
* the scorer's per-lane top-k is a rank too, not K serial argmaxes;
* finalization (the stable sort, parent remap, padding, ancestor mask --
  ~40 launches) is two kernels, and ranks only the D*W KEPT lanes, which is
  exact when W >= B (see ``_final_rank_kernel``);
* the prefix rows reuse the candidate top-k and a split log-sum-exp instead
  of a one-SM softmax over the vocabulary.

Opt-in (``DOMINOTREE_BUILDER_GRU_TABLE=1``): the GRU input gates depend on the
token alone, so they are precomputed for every token (~0.93 GB at Qwen3
shapes) and the embedding lookup plus the largest GEMM leave the loop. It is
the biggest single lever (roughly halves the build) and the only one that
costs memory the server could otherwise give to requests, which is why it is
never on by default.

RTX 5080, Qwen3-4B shapes, bs=1, graph GPU time: budget 16 0.790 -> 0.515 ms
(v2) -> 0.257 ms (v2 + table); budget 32 0.805 -> 0.526 -> 0.293 ms. Faster at
every batch size measured (bs 1-32, budgets 16-64).

One trap, recorded because it cost a failed suite: any kernel that COMPARES a
computed vector against itself (rank = how many entries beat me) must read that
vector back from memory first. Otherwise the compiler may recompute it per
register layout, in a different summation order; the two copies then disagree
in the last bit, ranks collide, and a ledger slot is never written. It passed
at 1 warp and failed at 2+.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

try:  # Triton ships with torch, but never let its absence break the builder.
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover - exercised only on Triton-less installs
    HAVE_TRITON = False


def _pow2_at_least(n: int, floor: int = 1) -> int:
    """Block size for ``n``: a power of two, at least ``floor``.

    The equivalence suite deliberately runs tiny shapes (W as small as 2,
    E = 24, M = 8), so every block dimension is masked on load and on store
    rather than assumed to divide evenly.
    """
    b = 1
    while b < n:
        b *= 2
    return max(b, floor)


if HAVE_TRITON:

    @triton.jit
    def _score_depth_kernel(
        HPRE,  # [bs, W, E]   lane half of embed_proj[0], model dtype
        PHHALF,  # [bs, E]      ph half at this depth, model dtype
        W2,  # [V, E]       embed_proj[2].weight, model dtype
        CAND,  # [bs, M]      candidate token ids, int64
        BASEC,  # [bs, M]      base logits at those candidates, fp32
        LSCORE,  # [bs, W]      cumulative score of each frontier lane, fp32
        LNODE,  # [bs, W]      ledger index of the node in each lane, int64
        LED_S,  # [bs, W*k]    OUT cumulative child scores
        LED_T,  # [bs, W*k]    OUT child token ids
        LED_P,  # [bs, W]      OUT parent ledger index per lane
        SCR,  # [bs, W, BM]  scratch for the rank top-k (see below)
        s_hpre_b,
        s_hpre_w,
        s_ph_b,
        s_w2_v,
        s_cand_b,
        s_basec_b,
        s_lscore_b,
        s_lnode_b,
        s_leds_b,
        s_ledt_b,
        s_ledp_b,
        W: tl.constexpr,
        E: tl.constexpr,
        M: tl.constexpr,
        K: tl.constexpr,
        BM: tl.constexpr,
        BE: tl.constexpr,
        RANK_TOPK: tl.constexpr = False,
    ):
        """One program per (request, LANE).

        The first version ran one program per request and held the whole
        [M, E] candidate-weight tile in registers. Both were mistakes, and the
        profile said so: 38.6 us per launch against a 2.5 us dispatch floor.
        At bs=1 a grid of (bs,) occupies ONE of the 5080's 84 SMs, and a
        [64, 256] fp32 tile is 16k values -- 128 registers per thread at 4
        warps -- so it spilled to local memory. Splitting by lane gives W
        independent blocks (the top-k is per lane, so lanes never interact),
        and accumulating over E in BE-sized chunks caps the live tile at
        [BM, BE]. The candidate rows are re-read once per lane, which is
        redundant traffic but lands in L2 and is far cheaper than the spill.
        """
        b = tl.program_id(0)
        w = tl.program_id(1)
        om = tl.arange(0, BM)
        mm = om < M
        NEG = float("-inf")
        dt = HPRE.dtype.element_ty

        cand = tl.load(CAND + b * s_cand_b + om, mask=mm, other=0)

        # --- corrected = basec + silu(ph_half + lane_half) @ w2[cand].T ------
        # The add and the SiLU run in the MODEL dtype: torch's kernels widen to
        # fp32 internally and round back on store, so staying in fp32 here
        # would drift far more than an ulp at bf16.
        acc = tl.zeros([BM], dtype=tl.float32)
        for e0 in tl.range(0, E, BE):
            oe = e0 + tl.arange(0, BE)
            me = oe < E
            hpre = tl.load(
                HPRE + b * s_hpre_b + w * s_hpre_w + oe, mask=me, other=0.0
            )
            ph = tl.load(PHHALF + b * s_ph_b + oe, mask=me, other=0.0)
            pre = (hpre.to(tl.float32) + ph.to(tl.float32)).to(dt)
            p32 = pre.to(tl.float32)
            h = (p32 / (1.0 + tl.exp(-p32))).to(dt).to(tl.float32)  # [BE]
            w2 = tl.load(
                W2 + cand[:, None] * s_w2_v + oe[None, :],
                mask=mm[:, None] & me[None, :],
                other=0.0,
            )  # [BM, BE]
            acc += tl.sum(h[None, :] * w2.to(tl.float32), axis=1)

        basec = tl.load(BASEC + b * s_basec_b + om, mask=mm, other=0.0)
        corr = tl.where(mm, acc + basec, NEG)  # [BM]

        # --- log_softmax over the M candidates, fp32 --------------------------
        mx = tl.max(corr, 0)
        z = tl.where(mm, tl.exp(corr - mx), 0.0)
        logp = corr - mx - tl.log(tl.sum(z, 0))

        # --- top-k for this lane, cumulative score, ledger write --------------
        # Iterated argmax: K is a small constexpr (node_topk, 3-8), so this
        # unrolls into K masked reductions over BM -- cheaper than a sort, and
        # it reproduces torch.topk's descending order with lowest-index ties.
        lsc = tl.load(LSCORE + b * s_lscore_b + w)
        if RANK_TOPK:
            # Same order in one step: each candidate's rank is the number of
            # candidates that beat it (ties -> lower index), and the K best
            # scatter to their slots. Masked lanes hold -inf and a higher
            # index than every real candidate, so they never outrank one.
            #
            # The scores go through memory first, and that is load-bearing.
            # The comparison needs each score in two register layouts (as row
            # and as column), and without the round trip the compiler is free
            # to RECOMPUTE the reduction chain behind ``logp`` for the second
            # layout -- in a different summation order, so the two copies can
            # disagree in the last bit, the comparisons stop being a total
            # order, two candidates share a rank and a ledger slot is never
            # written. That is not hypothetical: it failed the equivalence
            # suite at 2+ warps and passed at 1. Loaded values are moved
            # between layouts, never recomputed, so both roles see one copy.
            scr = SCR + (b * W + w) * BM + om
            tl.store(scr, tl.where(mm, logp, NEG))
            tl.debug_barrier()
            lj = tl.load(scr)
            beats = (lj[None, :] > lj[:, None]) | (
                (lj[None, :] == lj[:, None]) & (om[None, :] < om[:, None])
            )
            rank = tl.sum(beats.to(tl.int32), axis=1)
            keep = mm & (rank < K)
            tl.store(LED_S + b * s_leds_b + w * K + rank, lsc + lj, mask=keep)
            tl.store(LED_T + b * s_ledt_b + w * K + rank, cand, mask=keep)
        else:
            cur = logp
            for i in tl.static_range(K):
                v = tl.max(cur, 0)
                j = tl.argmax(cur, 0)
                tok = tl.load(CAND + b * s_cand_b + j)
                tl.store(LED_S + b * s_leds_b + w * K + i, lsc + v)
                tl.store(LED_T + b * s_ledt_b + w * K + i, tok)
                cur = tl.where(om == j, NEG, cur)

        # Parent pointers are a property of the LANE, so one entry per lane.
        tl.store(LED_P + b * s_ledp_b + w, tl.load(LNODE + b * s_lnode_b + w))

    @triton.jit
    def _select_depth_kernel(
        LED_S,  # [bs, W*k]  cumulative child scores at this depth, fp32
        LED_T,  # [bs, W*k]  child token ids, int64
        LSTATE,  # [bs, W, G] current lane GRU states, model dtype
        H0,  # [bs, W, G] OUT parent state of each kept lane
        KTOK,  # [bs, W]    OUT kept token ids, int64
        LSCORE_OUT,  # [bs, W]    OUT new lane scores, fp32
        LNODE_OUT,  # [bs, W]    OUT new lane ledger indices, int64
        d_off,  # int        d * W * k, the global ledger offset
        s_leds_b,
        s_ledt_b,
        s_lstate_b,
        s_lstate_w,
        s_h0_b,
        s_h0_w,
        s_ktok_b,
        s_lso_b,
        s_lno_b,
        W: tl.constexpr,
        K: tl.constexpr,
        G: tl.constexpr,
        WK: tl.constexpr,
        BWK: tl.constexpr,
        BG: tl.constexpr,
    ):
        b = tl.program_id(0)
        owk = tl.arange(0, BWK)
        og = tl.arange(0, BG)
        mwk = owk < WK
        mg = og < G
        NEG = float("-inf")

        cur = tl.load(LED_S + b * s_leds_b + owk, mask=mwk, other=NEG)
        # W is a constexpr (the frontier width), so this unrolls into W masked
        # reductions over the flat candidate ledger -- the dispatch-cheap
        # equivalent of torch.topk(flat_cum, W) followed by three gathers.
        for i in tl.static_range(W):
            v = tl.max(cur, 0)
            j = tl.argmax(cur, 0)
            tok = tl.load(LED_T + b * s_ledt_b + j)
            tl.store(KTOK + b * s_ktok_b + i, tok)
            tl.store(LSCORE_OUT + b * s_lso_b + i, v)
            tl.store(LNODE_OUT + b * s_lno_b + i, j.to(tl.int64) + d_off)
            # keep_idx // k is the lane that produced this child.
            src = j // K
            row = tl.load(
                LSTATE + b * s_lstate_b + src * s_lstate_w + og, mask=mg, other=0.0
            )
            tl.store(H0 + b * s_h0_b + i * s_h0_w + og, row, mask=mg)
            cur = tl.where(owk == j, NEG, cur)


    @triton.jit
    def _gru_cell_kernel(
        GI,  # [R, 3G] input-side gates, x @ W_ih.T (+ b_ih)
        GH,  # [R, 3G] hidden-side gates, h @ W_hh.T (+ b_hh)
        H0,  # [R, G]  previous hidden state
        OUT,  # [R, G]  OUT next hidden state, model dtype
        s_gi_r,
        s_gh_r,
        s_h0_r,
        s_out_r,
        G: tl.constexpr,
        BG: tl.constexpr,
    ):
        """One program per row of the GRU's reset/update/new gate arithmetic.

        This is torch's own GRU cell, written out: r and z are sigmoids of the
        summed gates, n is a tanh whose hidden half is scaled by r, and the new
        state interpolates. Computed in fp32 and rounded once on store, which
        is what the fused RNN kernels do internally for bf16 -- so the only
        difference from cuDNN is reduction order, the same ~1 ulp the rest of
        this module already documents.
        """
        row = tl.program_id(0)
        og = tl.arange(0, BG)
        mg = og < G

        ir = tl.load(GI + row * s_gi_r + og, mask=mg, other=0.0).to(tl.float32)
        iz = tl.load(GI + row * s_gi_r + G + og, mask=mg, other=0.0).to(tl.float32)
        inn = tl.load(GI + row * s_gi_r + 2 * G + og, mask=mg, other=0.0).to(tl.float32)
        hr = tl.load(GH + row * s_gh_r + og, mask=mg, other=0.0).to(tl.float32)
        hz = tl.load(GH + row * s_gh_r + G + og, mask=mg, other=0.0).to(tl.float32)
        hn = tl.load(GH + row * s_gh_r + 2 * G + og, mask=mg, other=0.0).to(tl.float32)
        h = tl.load(H0 + row * s_h0_r + og, mask=mg, other=0.0).to(tl.float32)

        r = 1.0 / (1.0 + tl.exp(-(ir + hr)))
        z = 1.0 / (1.0 + tl.exp(-(iz + hz)))
        n = (2.0 / (1.0 + tl.exp(-2.0 * (inn + r * hn)))) - 1.0  # tanh
        out = (1.0 - z) * n + z * h
        tl.store(OUT + row * s_out_r + og, out.to(OUT.dtype.element_ty), mask=mg)


    @triton.jit
    def _select_rank_kernel(
        LED_S,  # [bs, W*k]  cumulative child scores at this depth, fp32
        LED_T,  # [bs, W*k]  child token ids, int64
        KTOK,  # [bs, W]    OUT kept token ids, int64
        LSCORE_OUT,  # [bs, W]    OUT new lane scores, fp32
        LNODE_OUT,  # [bs, W]    OUT new lane ledger indices, int64
        PLANE,  # [bs, W]    OUT lane that produced each kept child, int64
        d_off,  # int        d * W * k, the global ledger offset
        s_leds_b,
        s_ledt_b,
        s_ktok_b,
        s_lso_b,
        s_lno_b,
        s_plane_b,
        W: tl.constexpr,
        K: tl.constexpr,
        WK: tl.constexpr,
        BI: tl.constexpr,
        BJ: tl.constexpr,
    ):
        """Top-W by RANK instead of by W serial argmaxes.

        ``_select_depth_kernel`` finds the best W children one argmax at a
        time: W dependent reductions, each ending in a barrier, which is why
        it cost 2.2x the dispatch floor at budget 16 and grows with the
        budget. Here every candidate computes its own rank -- the number of
        candidates that beat it, ties broken toward the LOWER index, which is
        exactly the order the iterated argmax (and torch.topk) produces -- and
        the W winners scatter themselves into their slots. No serial chain,
        and the grid spreads over SMs for large ledgers.

        The kept lane's GRU state is NOT copied here any more: only the
        producing lane's index is written, and the GRU kernel gathers the
        rows it needs directly (see ``_gru_gather_kernel``).
        """
        b = tl.program_id(0)
        oi = tl.program_id(1) * BI + tl.arange(0, BI)
        mi = oi < WK
        NEG = float("-inf")
        si = tl.load(LED_S + b * s_leds_b + oi, mask=mi, other=NEG)
        rank = tl.zeros([BI], dtype=tl.int32)
        for j0 in tl.range(0, WK, BJ):
            oj = j0 + tl.arange(0, BJ)
            mj = oj < WK
            sj = tl.load(LED_S + b * s_leds_b + oj, mask=mj, other=NEG)
            beats = (sj[None, :] > si[:, None]) | (
                (sj[None, :] == si[:, None]) & (oj[None, :] < oi[:, None])
            )
            rank += tl.sum((beats & mj[None, :]).to(tl.int32), axis=1)
        keep = mi & (rank < W)
        tok = tl.load(LED_T + b * s_ledt_b + oi, mask=keep, other=0)
        tl.store(KTOK + b * s_ktok_b + rank, tok, mask=keep)
        tl.store(LSCORE_OUT + b * s_lso_b + rank, si, mask=keep)
        tl.store(LNODE_OUT + b * s_lno_b + rank, oi.to(tl.int64) + d_off, mask=keep)
        tl.store(PLANE + b * s_plane_b + rank, (oi // K).to(tl.int64), mask=keep)

    @triton.jit
    def _gru_gather_kernel(
        GI,  # [R, 3G] input-side gates (x @ W_ih.T + b_ih), or the [V, 3G] table
        X,  # [bs, W, E + 3G] per-lane GEMM; columns E: are h @ W_hh.T + b_hh
        LSTATE,  # [bs, W, G] current lane states (the parents)
        KTOK,  # [bs, W]    kept token ids (row of the table)
        PLANE,  # [bs, W]    parent lane of each kept child
        OUT,  # [bs, W, G] OUT next lane states, model dtype
        s_gi_r,
        s_x_b,
        s_x_w,
        s_ls_b,
        s_ls_w,
        s_ktok_b,
        s_plane_b,
        s_out_b,
        s_out_w,
        W: tl.constexpr,
        E: tl.constexpr,
        G: tl.constexpr,
        BG: tl.constexpr,
        USE_TABLE: tl.constexpr,
    ):
        """GRU cell for one kept lane, reading its inputs by INDEX.

        The hidden-side gates are computed for the CURRENT lanes, before
        selection, and gathered here by parent lane: a row-wise linear map
        commutes with a row gather, so ``gather(h) @ W_hh.T`` and
        ``gather(h @ W_hh.T)`` are the same numbers. That lets the hidden GEMM
        share one launch with the scorer's lane GEMM, and removes the state
        copy the selector used to do. With ``USE_TABLE`` the input-side gates
        are one row of the precomputed ``[V, 3G]`` table -- the input
        projection depends on the token alone -- so the embedding lookup and
        the input GEMM disappear too.
        """
        r = tl.program_id(0)
        b = r // W
        i = r % W
        og = tl.arange(0, BG)
        mg = og < G
        p = tl.load(PLANE + b * s_plane_b + i)
        if USE_TABLE:
            gi_row = GI + tl.load(KTOK + b * s_ktok_b + i) * s_gi_r
        else:
            gi_row = GI + r * s_gi_r
        gh_row = X + b * s_x_b + p * s_x_w + E

        ir = tl.load(gi_row + og, mask=mg, other=0.0).to(tl.float32)
        iz = tl.load(gi_row + G + og, mask=mg, other=0.0).to(tl.float32)
        inn = tl.load(gi_row + 2 * G + og, mask=mg, other=0.0).to(tl.float32)
        hr = tl.load(gh_row + og, mask=mg, other=0.0).to(tl.float32)
        hz = tl.load(gh_row + G + og, mask=mg, other=0.0).to(tl.float32)
        hn = tl.load(gh_row + 2 * G + og, mask=mg, other=0.0).to(tl.float32)
        h = tl.load(
            LSTATE + b * s_ls_b + p * s_ls_w + og, mask=mg, other=0.0
        ).to(tl.float32)

        rg = 1.0 / (1.0 + tl.exp(-(ir + hr)))
        z = 1.0 / (1.0 + tl.exp(-(iz + hz)))
        n = (2.0 / (1.0 + tl.exp(-2.0 * (inn + rg * hn)))) - 1.0  # tanh
        out = (1.0 - z) * n + z * h
        tl.store(
            OUT + b * s_out_b + i * s_out_w + og,
            out.to(OUT.dtype.element_ty),
            mask=mg,
        )


    @triton.jit
    def _final_rank_kernel(
        LED_S,  # [bs, L]      ledger scores, fp32
        LED_T,  # [bs, L]      ledger tokens, int64
        LED_P,  # [bs, D*W]    parent ledger index per LANE (-1 = root)
        KEPT,  # [bs, D*W]     ledger index of every KEPT lane (INDIRECT only)
        OUT_TOK,  # [bs, N]    OUT tokens (slots 1..B)
        OUT_PAR,  # [bs, N]    OUT flat parent positions
        OUT_DEP,  # [bs, N]    OUT flat depths
        OUT_SC,  # [bs, N]     OUT cumulative scores
        mask_tok,
        s_leds_b,
        s_ledt_b,
        s_ledp_b,
        s_kept_b,
        s_ot_b,
        s_op_b,
        s_od_b,
        s_os_b,
        L: tl.constexpr,  # entries ranked: D*W when INDIRECT, else the ledger
        B: tl.constexpr,
        K: tl.constexpr,
        WK: tl.constexpr,
        BI: tl.constexpr,
        BJ: tl.constexpr,
        INDIRECT: tl.constexpr,
    ):
        """Global top-B over the ledger, ordered, parent-remapped, padded.

        Replaces the stable descending sort and the ~20 gathers, scatters and
        ``where``s that followed it. A stable descending sort puts entry i at
        position rank(i) = #{j : s_j > s_i or (s_j == s_i and j < i)} -- so
        computing that count per entry IS the sort, and the top-B are the
        entries with rank < B. The parent's output position is its own rank,
        computed the same way, so the remap needs no second pass. Exact
        integer counting: bit-identical to the sort, ties included.

        INDIRECT ranks only the lanes the frontier KEPT (D*W entries, not
        D*W*k). That is exact whenever W >= B: an entry in the global top-B
        is beaten by fewer than B <= W entries overall, hence by fewer than W
        at its own depth, so it was kept at that depth; and everything that
        beats it is itself in the top-B, hence kept. So its rank among the
        kept set equals its rank in the whole ledger, and every kept entry
        outside the top-B still ranks >= B. The only entries this can reorder
        are -inf padding slots, whose outputs are identical by construction
        (mask token, root parent, depth 1). k-fold fewer comparisons.
        """
        b = tl.program_id(0)
        oi = tl.program_id(1) * BI + tl.arange(0, BI)
        mi = oi < L
        NEG = float("-inf")
        if INDIRECT:
            gi = tl.load(KEPT + b * s_kept_b + oi, mask=mi, other=0)
        else:
            gi = oi.to(tl.int64)
        si = tl.load(LED_S + b * s_leds_b + gi, mask=mi, other=NEG)
        p_led = tl.load(LED_P + b * s_ledp_b + gi // K, mask=mi, other=-1)
        has_p = p_led >= 0
        sp = tl.load(LED_S + b * s_leds_b + p_led, mask=mi & has_p, other=NEG)
        rank = tl.zeros([BI], dtype=tl.int32)
        prank = tl.zeros([BI], dtype=tl.int32)
        for j0 in tl.range(0, L, BJ):
            oj = j0 + tl.arange(0, BJ)
            mj = oj < L
            if INDIRECT:
                gj = tl.load(KEPT + b * s_kept_b + oj, mask=mj, other=0)
            else:
                gj = oj.to(tl.int64)
            sj = tl.load(LED_S + b * s_leds_b + gj, mask=mj, other=NEG)
            bi = (sj[None, :] > si[:, None]) | (
                (sj[None, :] == si[:, None]) & (gj[None, :] < gi[:, None])
            )
            bp = (sj[None, :] > sp[:, None]) | (
                (sj[None, :] == sp[:, None]) & (gj[None, :] < p_led[:, None])
            )
            rank += tl.sum((bi & mj[None, :]).to(tl.int32), axis=1)
            prank += tl.sum((bp & mj[None, :]).to(tl.int32), axis=1)
        keep = mi & (rank < B)
        # -inf = a slot the tree could not fill: dead leaf, a mask_token child
        # of the root at depth 1 (the per-request path's exact padding).
        valid = si > NEG
        tok = tl.load(LED_T + b * s_ledt_b + gi, mask=keep, other=0)
        tok = tl.where(valid, tok, mask_tok)
        # An unselected parent is impossible by the lemma; if it ever happened
        # the old gather produced -1 + 1 = 0, the root, and so does this.
        par = tl.where(has_p & (prank < B), prank + 1, 0)
        par = tl.where(valid, par, 0)
        dep = tl.where(valid, gi // WK + 1, 1)
        slot = rank + 1
        tl.store(OUT_TOK + b * s_ot_b + slot, tok, mask=keep)
        tl.store(OUT_PAR + b * s_op_b + slot, par.to(tl.int64), mask=keep)
        tl.store(OUT_DEP + b * s_od_b + slot, dep.to(tl.int64), mask=keep)
        tl.store(OUT_SC + b * s_os_b + slot, si, mask=keep)

    @triton.jit
    def _final_mask_kernel(
        VER,  # [bs]        verified root token
        OUT_TOK,  # [bs, N]
        OUT_PAR,  # [bs, N]
        OUT_DEP,  # [bs, N]
        OUT_SC,  # [bs, N]
        OUT_MASK,  # [bs, N, N] uint8 view of the bool mask
        s_ot_b,
        s_op_b,
        s_od_b,
        s_os_b,
        s_om_b,
        s_om_i,
        N: tl.constexpr,
        D: tl.constexpr,
        BN: tl.constexpr,
    ):
        """Root slot + intra-tree ancestor mask, one program per request.

        mask[i, j] = 1 iff j is i itself or one of its ancestors. Every row
        walks up its own parent chain in lockstep -- at most D hops, since no
        node is deeper than D -- reading the parents ``_final_rank_kernel``
        just wrote. Pure integer bookkeeping, so exact.
        """
        b = tl.program_id(0)
        on = tl.arange(0, BN)
        mn = on < N
        tl.store(OUT_TOK + b * s_ot_b, tl.load(VER + b))
        tl.store(OUT_PAR + b * s_op_b, -1)
        tl.store(OUT_DEP + b * s_od_b, 0)
        tl.store(OUT_SC + b * s_os_b, 0.0)
        # Slot 0 (the root) is not read back from memory -- it was stored by
        # this same program just above -- it is simply the end of every chain.
        cur = tl.load(OUT_PAR + b * s_op_b + on, mask=mn & (on > 0), other=-1)
        m = (on[None, :] == on[:, None]) & mn[:, None]
        for _ in tl.static_range(D):
            m = m | ((on[None, :] == cur[:, None]) & (cur[:, None] >= 0))
            nxt = tl.load(OUT_PAR + b * s_op_b + cur, mask=cur > 0, other=-1)
            cur = tl.where(cur > 0, nxt, -1)
        tl.store(
            OUT_MASK + b * s_om_b + on[:, None] * s_om_i + on[None, :],
            m.to(tl.uint8),
            mask=mn[:, None] & mn[None, :],
        )

    @triton.jit
    def _lse_partial_kernel(
        X,  # [bs, pd, V] base logits (model dtype)
        PART,  # [bs*pd, NC, 2] OUT (chunk max, chunk sum of exp(x - max))
        s_x_b,
        s_x_d,
        PD: tl.constexpr,
        V: tl.constexpr,
        NC: tl.constexpr,
        BV: tl.constexpr,
    ):
        """Split log-sum-exp over the vocabulary: one program per chunk.

        ``log_softmax`` over one 152k-wide row runs as ONE thread block --
        one SM of 84 -- and cost 25 us for a single row. Splitting the row
        across NC programs and combining (max, sum) pairs afterwards is the
        standard online-softmax merge.
        """
        r = tl.program_id(0)
        c = tl.program_id(1)
        o = c * BV + tl.arange(0, BV)
        m = o < V
        x = tl.load(
            X + (r // PD) * s_x_b + (r % PD) * s_x_d + o, mask=m, other=float("-inf")
        ).to(tl.float32)
        mx = tl.max(x, 0)
        sm = tl.sum(tl.where(m, tl.exp(x - mx), 0.0), 0)
        tl.store(PART + (r * NC + c) * 2, mx)
        tl.store(PART + (r * NC + c) * 2 + 1, sm)

    @triton.jit
    def _lse_final_kernel(
        PART,  # [bs*pd, NC, 2]
        BASEC,  # [bs, K, M] fp32 top-M logits (descending)
        CAND,  # [bs, K, M] int64 their token ids
        OUT_LP,  # [bs, pd, k] OUT prefix log-probs
        OUT_TOK,  # [bs, pd, k] OUT prefix tokens
        s_bc_b,
        s_bc_d,
        s_cd_b,
        s_cd_d,
        s_lp_b,
        s_lp_d,
        s_tk_b,
        s_tk_d,
        PD: tl.constexpr,
        NC: tl.constexpr,
        BNC: tl.constexpr,
        KT: tl.constexpr,
        BK: tl.constexpr,
    ):
        """Merge the chunk partials and emit ``x - max - log(sum)`` for the
        top-k, in the same operation order as torch's log_softmax."""
        r = tl.program_id(0)
        b = r // PD
        d = r % PD
        oc = tl.arange(0, BNC)
        mc = oc < NC
        cm = tl.load(PART + (r * NC + oc) * 2, mask=mc, other=float("-inf"))
        cs = tl.load(PART + (r * NC + oc) * 2 + 1, mask=mc, other=0.0)
        mx = tl.max(cm, 0)
        tot = tl.sum(tl.where(mc, cs * tl.exp(cm - mx), 0.0), 0)
        ok = tl.arange(0, BK)
        mk = ok < KT
        x = tl.load(BASEC + b * s_bc_b + d * s_bc_d + ok, mask=mk, other=0.0)
        t = tl.load(CAND + b * s_cd_b + d * s_cd_d + ok, mask=mk, other=0)
        tl.store(OUT_LP + b * s_lp_b + d * s_lp_d + ok, (x - mx) - tl.log(tot), mask=mk)
        tl.store(OUT_TOK + b * s_tk_b + d * s_tk_d + ok, t, mask=mk)


class FusedDepth:
    """Launcher for the two fused depth kernels, specialized to one builder.

    Holds nothing but shapes; the builder owns every buffer.  ``available``
    is False when Triton is missing or the shapes are outside what the kernels
    support, and the caller must then fall back to the torch path -- these
    kernels are an optimization, never a semantic requirement.
    """

    def __init__(self, *, W: int, E: int, M: int, K: int, G: int) -> None:
        self.W, self.E, self.M, self.K, self.G = W, E, M, K, G
        self.gru_weights: tuple | None = None
        # Depth-body version. 1 = the D67 body (scorer + serial-argmax selector
        # that copies parent states + cuDNN-free GRU gates). 2 = the D74 body:
        # the scorer's lane GEMM and the GRU's hidden GEMM share ONE launch,
        # the selector ranks in parallel and copies nothing, and the GRU
        # kernel gathers its rows by parent index. ``DOMINOTREE_BUILDER_FUSED_
        # DEPTH=1`` keeps the D67 body in the same build for the A/B.
        self.version = 2 if os.environ.get("DOMINOTREE_BUILDER_FUSED_DEPTH", "2") == "2" else 1
        self.w_lane: torch.Tensor | None = None  # [E + 3G, G] (v2)
        self.b_lane: torch.Tensor | None = None  # [E + 3G] or None
        self.gi_table: torch.Tensor | None = None  # [V, 3G] (opt-in)
        # Warp counts are tunable because these blocks are TINY (M = 64
        # candidates, W*k = 128 ledger entries): more warps buys no parallelism
        # but costs a barrier per reduction step. SWEPT on the 5080 at
        # Qwen3-4B shapes, budget 16 (graph wall, ms):
        #     select=4  select=1
        #   score=4   0.830    0.788   <- default
        #   score=2   0.850    0.808
        #   score=1   0.842    0.801
        # The selection kernel is a chain of W serial argmax reductions, so
        # extra warps are pure barrier cost; the scoring kernel has a real
        # [BM, BE] tile to spread and prefers 4.
        # v2 (D74) re-swept with the rank top-k: BE=128 / 8 warps ~= BE=256 /
        # 4 warps (0.353 vs 0.352 ms); 128 keeps the live tile smaller for a
        # wider drafter. The iterated-argmax v1 scorer keeps its 4 warps.
        self.warps_score = int(
            os.environ.get(
                "DOMINOTREE_WARPS_SCORE",
                "8" if os.environ.get("DOMINOTREE_BUILDER_FUSED_DEPTH", "2") == "2" else "4",
            )
        )
        self.warps_select = int(os.environ.get("DOMINOTREE_WARPS_SELECT", "1"))
        self.rank_topk = (
            os.environ.get("DOMINOTREE_SCORE_RANK_TOPK", "1" if self.version >= 2 else "0")
            == "1"
        )
        self.warps_rank = int(os.environ.get("DOMINOTREE_WARPS_RANK", "4"))
        self.rank_bi = int(os.environ.get("DOMINOTREE_RANK_BI", "16"))
        self.rank_bj = int(os.environ.get("DOMINOTREE_RANK_BJ", "64"))
        self.WK = W * K
        self.BM = _pow2_at_least(M, floor=1)
        # E is accumulated in chunks so the live [BM, BE] tile stays in
        # registers; 32 keeps it at 2k values (16 regs/thread at 4 warps).
        self.BE_CHUNK = min(
            _pow2_at_least(E, floor=1),
            int(os.environ.get("DOMINOTREE_SCORE_BE", "128" if self.version >= 2 else "32")),
        )
        self.BWK = _pow2_at_least(self.WK, floor=1)
        self.BG = _pow2_at_least(G, floor=1)
        self.available = HAVE_TRITON and M > 0

    def bind_gru(self, gru) -> bool:
        """Cache transposed GRU weights so the cell can be run as 2 GEMMs + 1.

        Returns False for anything this cell does not implement (multi-layer,
        bidirectional, projections); the caller must then keep torch's GRU.
        Refusing loudly beats silently drafting with the wrong recurrence.
        """
        if not self.available:
            return False
        if gru.num_layers != 1 or gru.bidirectional or getattr(gru, "proj_size", 0):
            return False
        # Keep the weights in torch's own [3G, in] layout and go through
        # F.linear (an A x B.T GEMM). MEASURED 2026-09-04: pre-transposing to
        # [in, 3G] and using matmul makes cuBLAS pick
        # cutlass_80_tensorop_bf16_s16816gemm, which costs 31 us per depth
        # against 11 us for the A x B.T form on the same bytes -- a skinny
        # M=16 GEMM is layout-sensitive and the "obvious" contiguous transpose
        # is the slow one. Do not re-transpose these.
        w_ih = gru.weight_ih_l0  # [3G, H]
        w_hh = gru.weight_hh_l0  # [3G, G]
        b_ih = getattr(gru, "bias_ih_l0", None)
        b_hh = getattr(gru, "bias_hh_l0", None)
        self.gru_weights = (w_ih, w_hh, b_ih, b_hh)
        return True

    def gru_cell(self, emb: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """One GRU step for every lane: ``[bs, W, H] x [bs, W, G] -> [bs, W, G]``.

        Replaces torch's fused RNN call, which on this shape costs two device
        memcpys plus three kernels per depth. The two GEMMs stay in cuBLAS --
        they read the [H, 3G] and [G, 3G] weight matrices, so they are
        bandwidth-bound and already near their floor; only the gate arithmetic
        is fused.
        """
        w_ih, w_hh, b_ih, b_hh = self.gru_weights
        bs, W, _ = emb.shape
        R, G = bs * W, self.G
        h0r = h0.reshape(R, G)
        gi = torch.nn.functional.linear(emb.reshape(R, -1), w_ih, b_ih)
        gh = torch.nn.functional.linear(h0r, w_hh, b_hh)
        out = torch.empty((R, G), dtype=emb.dtype, device=emb.device)
        _gru_cell_kernel[(R,)](
            gi, gh, h0r, out,
            gi.stride(0), gh.stride(0), h0r.stride(0), out.stride(0),
            G=G, BG=_pow2_at_least(G),
            num_warps=4,
        )
        return out.reshape(bs, W, G)

    def score(
        self,
        *,
        hpre: torch.Tensor,
        ph_half_d: torch.Tensor,
        w2: torch.Tensor,
        cand_d: torch.Tensor,
        basec_d: torch.Tensor,
        lane_scores: torch.Tensor,
        lane_node: torch.Tensor,
        led_s_d: torch.Tensor,
        led_t_d: torch.Tensor,
        led_p_d: torch.Tensor,
    ) -> None:
        bs = hpre.shape[0]
        _score_depth_kernel[(bs, self.W)](
            hpre,
            ph_half_d,
            w2,
            cand_d,
            basec_d,
            lane_scores,
            lane_node,
            led_s_d,
            led_t_d,
            led_p_d,
            torch.empty(
                (bs, self.W, self.BM) if self.rank_topk else (1,),
                dtype=torch.float32,
                device=hpre.device,
            ),
            hpre.stride(0),
            hpre.stride(1),
            ph_half_d.stride(0),
            w2.stride(0),
            cand_d.stride(0),
            basec_d.stride(0),
            lane_scores.stride(0),
            lane_node.stride(0),
            led_s_d.stride(0),
            led_t_d.stride(0),
            led_p_d.stride(0),
            W=self.W,
            E=self.E,
            M=self.M,
            K=self.K,
            BM=self.BM,
            BE=self.BE_CHUNK,
            RANK_TOPK=self.rank_topk,
            num_warps=self.warps_score,
        )

    def select(
        self,
        *,
        led_s_d: torch.Tensor,
        led_t_d: torch.Tensor,
        lane_states: torch.Tensor,
        h0: torch.Tensor,
        kept_tok: torch.Tensor,
        lane_scores_out: torch.Tensor,
        lane_node_out: torch.Tensor,
        d_off: int,
    ) -> None:
        bs = led_s_d.shape[0]
        _select_depth_kernel[(bs,)](
            led_s_d,
            led_t_d,
            lane_states,
            h0,
            kept_tok,
            lane_scores_out,
            lane_node_out,
            d_off,
            led_s_d.stride(0),
            led_t_d.stride(0),
            lane_states.stride(0),
            lane_states.stride(1),
            h0.stride(0),
            h0.stride(1),
            kept_tok.stride(0),
            lane_scores_out.stride(0),
            lane_node_out.stride(0),
            W=self.W,
            K=self.K,
            G=self.G,
            WK=self.WK,
            BWK=self.BWK,
            BG=self.BG,
            num_warps=self.warps_select,
        )

    # ------------------------------------------------------------------
    # Version-2 depth body (D74).
    # ------------------------------------------------------------------

    def bind_lane(self, w0_lane: torch.Tensor) -> bool:
        """Stack the scorer's lane weight on top of the GRU hidden weight.

        ``w0_lane`` is the lane half of ``embed_proj[0].weight`` in torch's
        native ``[E, G]`` layout. Both GEMMs read the SAME current lane states,
        so one ``F.linear`` over ``[E + 3G, G]`` replaces two launches. The
        scorer half gets a zero bias, which is exact (x + 0.0 == x).
        """
        if self.gru_weights is None:
            return False
        _, w_hh, _, b_hh = self.gru_weights
        self.w_lane = torch.cat([w0_lane, w_hh], dim=0).contiguous()
        if b_hh is not None:
            self.b_lane = torch.cat([b_hh.new_zeros(self.E), b_hh]).contiguous()
        return True

    @torch.no_grad()
    def bind_table(self, embed_tokens, rows: int, chunk: int = 8192) -> int:
        """Precompute the GRU input gates for EVERY token: ``[V, 3G]``.

        ``x @ W_ih.T + b_ih`` with ``x = embed(token)`` depends on the token
        alone, so it can be computed once and gathered per depth -- removing
        the embedding lookup and the largest GEMM of the depth loop. It is the
        same op on the same rows; only the GEMM's tiling differs, so values can
        move by ~1 ulp, like every other fusion here. Returns bytes used.

        The table lives in VRAM for the builder's lifetime (~0.93 GB at
        Qwen3 shapes), which is memory the server cannot give to requests --
        hence opt-in, and measured under concurrency before any default.
        """
        w_ih, _, b_ih, _ = self.gru_weights
        table = torch.empty(rows, 3 * self.G, dtype=w_ih.dtype, device=w_ih.device)
        for s in range(0, rows, chunk):
            ids = torch.arange(s, min(s + chunk, rows), device=w_ih.device)
            table[s : s + ids.numel()] = torch.nn.functional.linear(
                embed_tokens(ids), w_ih, b_ih
            )
        self.gi_table = table
        return table.numel() * table.element_size()

    def lane_gemm(self, lane_states: torch.Tensor, with_gru: bool) -> torch.Tensor:
        """``[bs, W, G] -> [bs, W, E (+ 3G)]``: scorer lane half + GRU hidden gates."""
        if with_gru:
            return torch.nn.functional.linear(lane_states, self.w_lane, self.b_lane)
        return torch.nn.functional.linear(lane_states, self.w_lane[: self.E])

    def select_rank(
        self,
        *,
        led_s_d: torch.Tensor,
        led_t_d: torch.Tensor,
        kept_tok: torch.Tensor,
        lane_scores_out: torch.Tensor,
        lane_node_out: torch.Tensor,
        parent_lane: torch.Tensor,
        d_off: int,
    ) -> None:
        bs = led_s_d.shape[0]
        BI = min(self.BWK, self.rank_bi)
        _select_rank_kernel[(bs, triton.cdiv(self.WK, BI))](
            led_s_d,
            led_t_d,
            kept_tok,
            lane_scores_out,
            lane_node_out,
            parent_lane,
            d_off,
            led_s_d.stride(0),
            led_t_d.stride(0),
            kept_tok.stride(0),
            lane_scores_out.stride(0),
            lane_node_out.stride(0),
            parent_lane.stride(0),
            W=self.W,
            K=self.K,
            WK=self.WK,
            BI=BI,
            BJ=self.rank_bj,
            num_warps=self.warps_rank,
        )

    def gru_gather(
        self,
        *,
        kept_tok: torch.Tensor,
        parent_lane: torch.Tensor,
        x: torch.Tensor,
        lane_states: torch.Tensor,
        embed_tokens,
    ) -> torch.Tensor:
        """Next lane states from the kept children, gathering parents by index."""
        bs, W, G = lane_states.shape
        R = bs * W
        if self.gi_table is not None:
            gi, use_table = self.gi_table, True
        else:
            w_ih, _, b_ih, _ = self.gru_weights
            gi = torch.nn.functional.linear(
                embed_tokens(kept_tok).reshape(R, -1), w_ih, b_ih
            )
            use_table = False
        out = torch.empty_like(lane_states)
        _gru_gather_kernel[(R,)](
            gi,
            x,
            lane_states,
            kept_tok,
            parent_lane,
            out,
            gi.stride(0),
            x.stride(0),
            x.stride(1),
            lane_states.stride(0),
            lane_states.stride(1),
            kept_tok.stride(0),
            parent_lane.stride(0),
            out.stride(0),
            out.stride(1),
            W=W,
            E=self.E,
            G=G,
            BG=self.BG,
            USE_TABLE=use_table,
            num_warps=4,
        )
        return out

    # ------------------------------------------------------------------
    # Finalization and prefix rows (D74).
    # ------------------------------------------------------------------

    def finalize(
        self, *, st, led_scores, led_tokens, led_parent, led_kept, D, B, mask_token_id
    ):
        """Ledger -> ordered flat tree + ancestor mask, in two launches.

        ``led_kept`` ([bs, D, W] ledger indices of the kept lanes, or None)
        selects the k-fold cheaper INDIRECT ranking; it is exact only when
        W >= B, so the caller passes None otherwise.
        """
        bs = led_scores.shape[0]
        Lfull = led_scores.shape[1] * led_scores.shape[2]
        ls = led_scores.reshape(bs, Lfull)
        lt = led_tokens.reshape(bs, Lfull)
        lp = led_parent.reshape(bs, -1)
        indirect = led_kept is not None
        kept = led_kept.reshape(bs, -1) if indirect else lp
        L = kept.shape[1] if indirect else Lfull
        # Small blocks: the work is L x L comparisons, so spread it over SMs.
        BI = min(_pow2_at_least(L), 32)
        _final_rank_kernel[(bs, triton.cdiv(L, BI))](
            ls, lt, lp, kept,
            st.S_out_tokens, st.S_out_parents, st.S_out_depths, st.S_out_scores,
            mask_token_id,
            ls.stride(0), lt.stride(0), lp.stride(0), kept.stride(0),
            st.S_out_tokens.stride(0), st.S_out_parents.stride(0),
            st.S_out_depths.stride(0), st.S_out_scores.stride(0),
            L=L, B=B, K=self.K, WK=self.WK, BI=BI, BJ=64, INDIRECT=indirect,
            num_warps=4,
        )
        N = st.S_out_tokens.shape[1]
        om = st.S_out_mask.view(torch.uint8)
        _final_mask_kernel[(bs,)](
            st.S_verified,
            st.S_out_tokens, st.S_out_parents, st.S_out_depths, st.S_out_scores,
            om,
            st.S_out_tokens.stride(0), st.S_out_parents.stride(0),
            st.S_out_depths.stride(0), st.S_out_scores.stride(0),
            om.stride(0), om.stride(1),
            N=N, D=D, BN=_pow2_at_least(N),
            num_warps=4,
        )

    def prefix_rows(self, *, base_logits, pd, st):
        """Top-k log-probs of the first ``pd`` rows, reusing the top-M
        candidates (``S_basec``/``S_cand``) and a split log-sum-exp."""
        bs, _, V = base_logits.shape
        R = bs * pd
        BV = 4096
        NC = triton.cdiv(V, BV)
        part = torch.empty((R, NC, 2), dtype=torch.float32, device=base_logits.device)
        _lse_partial_kernel[(R, NC)](
            base_logits, part,
            base_logits.stride(0), base_logits.stride(1),
            PD=pd, V=V, NC=NC, BV=BV,
            num_warps=8,
        )
        k = st.S_prefix_toks.shape[-1]
        _lse_final_kernel[(R,)](
            part, st.S_basec, st.S_cand, st.S_prefix_lps, st.S_prefix_toks,
            st.S_basec.stride(0), st.S_basec.stride(1),
            st.S_cand.stride(0), st.S_cand.stride(1),
            st.S_prefix_lps.stride(0), st.S_prefix_lps.stride(1),
            st.S_prefix_toks.stride(0), st.S_prefix_toks.stride(1),
            PD=pd, NC=NC, BNC=_pow2_at_least(NC), KT=k, BK=_pow2_at_least(k),
            num_warps=1,
        )
