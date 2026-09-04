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
        self.warps_score = int(os.environ.get("DOMINOTREE_WARPS_SCORE", "4"))
        self.warps_select = int(os.environ.get("DOMINOTREE_WARPS_SELECT", "1"))
        self.WK = W * K
        self.BM = _pow2_at_least(M, floor=1)
        # E is accumulated in chunks so the live [BM, BE] tile stays in
        # registers; 32 keeps it at 2k values (16 regs/thread at 4 warps).
        self.BE_CHUNK = min(_pow2_at_least(E, floor=1), 32)
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
