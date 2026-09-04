"""Depth-synchronous batched GPU frontier builder (Option B).

Implements ``batch_builder_design.md`` §2 Option B: replace the per-request
best-first heap (``best_first.build_best_first_tree`` driven by the
``conditional_children`` scorer / ``gpu_expander``) with a batched depth-loop
that builds ALL ``bs`` trees on-device with **zero host syncs** — no
``.item()`` / ``.tolist()`` anywhere between the batched inputs and the emitted
``draft_tokens_2d [bs, N]`` + ``intra_mask [bs, N, N]`` device tensors.

Algorithm (per decode step, one call to :meth:`FrontierTreeBuilder.build`):

1. Maintain a frontier of ``W = budget`` lanes per request. Lane 0 starts as the
   root (score 0); the other lanes start at ``-inf`` (static shapes: invalid
   lanes flow through every op, their ``-inf`` cumulative scores make them and
   all their descendants unselectable).
2. At each depth ``d`` (``D = min(max_depth, k_draft)`` depths) score all
   ``W * k`` children of the frontier in ONE batched op — the same three
   correction cases as ``conditional_children.make_conditional_children_fn``,
   transcribed with a leading ``[bs, W]`` batch — and log every scored
   candidate ``(cum_logprob, token, parent ledger index)`` into a ledger
   ``[bs, D, W*k]``. Keep the top-``W`` per request on GPU and GRU-advance only
   the kept lanes.
3. After ``D`` depths, select the final tree as the **global top-B by
   cum_logprob over the ledger** (one stable descending sort — see the
   tie-break note below), remap parent pointers to the ordered node list,
   dead-leaf-pad short trees, and emit the flat tokens + intra-tree ancestor
   mask as device tensors.

When this builder is and is not the right choice
------------------------------------------------
This is a **batching** optimisation, not a universally faster builder. It deliberately
trades arithmetic for synchronisation: it scores ``D * W`` lane-corrections per request
per step (240 at our defaults) where the best-first heap scores only the ``B`` nodes it
actually expands (15) -- roughly 16x the correction work -- in exchange for removing the
heap's per-pop device-to-host sync.

That trade pays under continuous batching, where those syncs serialise across every
in-flight request and idle the GPU between pops. It does **not** obviously pay at batch
size 1, where there is no batch to serialise and a per-node CUDA graph already makes the
syncs cheap; there the extra arithmetic is close to pure cost. This is why the paper's
single-stream (bs=1) results use the heap builder with the GPU-native per-node expander
(``dominotree_gpu.GraphNodeExpander``) and the served results use this one: each regime
runs the builder suited to it, and neither choice is an oversight.

Correctness — the monotone-score lemma (``batch_builder_design.md`` §2 intro):
a child's cumulative log-prob is <= its parent's, so ``build_best_first_tree``
pops exactly the global top-B nodes of the candidate tree, and that set is
ancestor-closed. With per-depth width ``W = B``, the kept frontier at each
depth contains every top-B node of that depth (fewer than B <= W candidates can
outscore it), so every top-B node's children get scored, so the global top-B
over the ledger reproduces the best-first tree **exactly — up to real-valued
score ties** (the heap breaks ties by insertion order; this builder by
depth-then-lane order). On a tie the two trees may differ; both are valid
best-first trees, tau is statistically identical, and losslessness is untouched
(the verify path is not modified). The equivalence suite
(``gpu_expander._frontier_equivalence_suite``) therefore asserts exact equality
on tie-free random inputs. ``W < budget`` (via the ``width`` argument) is a
genuine approximation and is NOT the default.

Topological order: the flat node list must satisfy parents-before-children
(``best_first.py:9-12``; relied on by ``build_intra_tree_mask_from_parents``
and ``reconstruct_indices_from_tree_mask``). Selection + ordering happen in one
stable descending sort of the ledger scores: the ledger flat index is
depth-major (``d * W*k + lane*k + child``), so the stable sort's tie-break is
depth-ascending — a parent (score >= child, depth < child) always sorts before
its child even on an exact score tie, and on tie-free inputs the order equals
the heap's pop order (pop scores are non-increasing). Ancestor closure of the
selected set also survives ties for the same reason, so every selected node's
parent is itself selected (or the root).

Dead-leaf padding: trees with fewer than ``budget`` finite-scored nodes are
padded with the EXACT semantics of the per-request path
(``worker._build_conditional_tree_for_req``): ``mask_token_id`` children of the
root, leaf-only, appended AFTER all real nodes (``-inf`` sorts last), so they
can never match the target argmax and never alter the accepted path.

Memory / graph-capture discipline (``batch_builder_design.md`` §2A memory,
§3.5): the ``[bs, K, V]`` logits are consumed transiently by
:meth:`_load_inputs` (reduced to ``cand``/``basec`` ``[bs, K, M]`` + prefix-row
statics) and the ``[bs, K, M, E]`` candidate weights are never materialized —
``w2[cand[:, d]]`` is gathered inside the depth body. All shapes are static in
``(bs, K, W, k, M)``; per-``bs`` state is cached. The split is
capture-shaped: ``_load_inputs`` writes static input buffers (outside a future
``torch.cuda.graph`` capture), ``_body`` reads only statics + model weights and
writes only static outputs (the capture target — a per-bs-bucket graph pool is
the drop-in follow-up, pattern ``domino_rollout._domino_loop_graph_pool``).
CAVEAT: the ``corr_topm == 0`` full-vocab path reads the transient
``[bs, K, V]`` logits inside the depth loop and is therefore eager-only as
structured (the production default is ``corr_topm = 64``).

When ``DOMINOTREE_FRONTIER_GRAPH`` is not ``"0"``, the static-shape body is
captured once per batch-size bucket and replayed on subsequent calls.  Capture
is best-effort: any error permanently selects the eager path for this builder.

Dispatch-cost fusions (``DOMINOTREE_BUILDER_FUSION``, default on)
-----------------------------------------------------------------
Profiling the captured graph (``scripts/domino_tree/profile_frontier.py``,
RTX 5080, Qwen3-4B shapes) showed the build is **GPU-dispatch-bound**: ~747
dependent kernels of ~2.3 us each, with cost linear in DEPTH and inert in
width.  Latency is therefore bought by removing KERNELS, not FLOPs.  Three
rewrites do that without changing the tree:

1. ``embed_proj[0]`` is split at the ``cat`` boundary.  It is a bias-free
   ``Linear(H + G -> E)`` over ``cat([ph_d, lane_states])`` and the ``ph``
   half does not depend on the lane, so that half is lifted out of the depth
   loop into ONE ``[bs, D, H] x [H, E]`` GEMM.  Removes D concatenations
   (2 ``memcpy128`` each) and shrinks each per-depth GEMM's contraction to G.
2. The parent ledger holds one entry per LANE, not per candidate: the k-fold
   broadcast folds into the final gather (``sel_idx // k``), removing D
   ``repeat_interleave`` allocations and shrinking D ledger writes k-fold.
3. The intra-tree ancestor mask is a boolean transitive closure by repeated
   SQUARING (``ceil(log2 D)`` rounds) instead of D sequential parent-pointer
   hops -- the largest kernel-count item outside the depth loop.
4. ``prefix_gru.flatten_parameters()`` is called once at construction, because
   cuDNN otherwise re-compacts the GRU weights on EVERY call -- once per depth,
   inside the hot loop.

Measured on the RTX 5080 at Qwen3-4B shapes, budget 16: 747 -> 592 kernels and
1.725 -> 1.457 ms of graph wall (-15.5%).  Note what the inventory says about
what is left: with ~2.5 us per kernel and 592 of them accounting for the full
1.457 ms, this builder's latency is simply ``kernels x 2.5 us``.  The D-sweep
agrees (0.084 ms per depth level, ~34 kernels/depth, plus a ~48-kernel tail).
So the only remaining lever of any size is launching FEWER kernels per depth --
a fused depth body -- and no amount of arithmetic reduction will help.

(2) and (3) are exact reindexing / exact 0-1 integer arithmetic in fp32, so
they are bit-identical to the originals.  (1) is exact in real arithmetic but
splits one floating-point reduction into two, so it differs by ~1 ulp of the
fp32 accumulator -- orders of magnitude below the bf16 drift floor documented
above, but a real change, which is why it is gated and why the gate is an
end-to-end tau A/B and not an assertion.  Setting
``DOMINOTREE_BUILDER_FUSION=0`` restores all three legacy paths in the SAME
build, so the A/B is an env flip rather than a cross-build comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


@dataclass
class _State:
    """Per-batch-size static buffers (inputs written by ``_load_inputs``,
    outputs written by ``_body``). Everything else is an intermediate."""

    # ---- static inputs ----
    S_ph: torch.Tensor  # [bs, K, H] model dtype
    S_root: torch.Tensor  # [bs, G] model dtype
    S_verified: torch.Tensor  # [bs] long
    S_cand: torch.Tensor | None  # [bs, K, M] long (corr_topm > 0)
    S_basec: torch.Tensor | None  # [bs, K, M] float32 (corr_topm > 0)
    S_prefix_toks: torch.Tensor | None  # [bs, PD, k] long (PD = min(prefix_len, D))
    S_prefix_lps: torch.Tensor | None  # [bs, PD, k] float32
    # ---- static outputs ----
    S_out_tokens: torch.Tensor  # [bs, N] long
    S_out_mask: torch.Tensor  # [bs, N, N] bool
    S_out_parents: torch.Tensor  # [bs, N] long (root = -1; flat parent indices)
    S_out_depths: torch.Tensor  # [bs, N] long (root = 0)
    S_out_scores: torch.Tensor  # [bs, N] float32 (root = 0.0, dead leaves = -inf)


class FrontierTreeBuilder:
    """Batched drop-in replacement for the per-request best-first tree build.

    Construct once per worker; call :meth:`build` once per decode step with the
    batched per-step tensors. ``draft`` is the ``DominoDraftModel`` (owns
    ``prefix_gru`` + ``embed_proj``); ``embed_tokens`` is the target embedding
    module (``get_input_embeddings()``) — the same handles the reference scorer
    uses, so the per-lane math transcribes
    ``conditional_children.make_conditional_children_fn`` line-by-line with a
    leading ``[bs, W]`` batch.
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
        max_depth: int,
        mask_token_id: int,
        device,
        width: int | None = None,
    ) -> None:
        device = torch.device(device)
        if k_draft <= 0 or node_topk <= 0:
            raise ValueError("k_draft and node_topk must be positive")
        if budget <= 0 or max_depth <= 0:
            raise ValueError("budget and max_depth must be positive")
        if corr_topm > 0:
            # Same clamp as make_conditional_children_fn (the per-node topk
            # selects from the corr_topm candidate rows).
            node_topk = min(int(node_topk), int(corr_topm))

        self.draft = draft
        self.embed_tokens = embed_tokens
        self.k_draft = int(k_draft)
        self.prefix_len = int(prefix_len)
        self.node_topk = int(node_topk)
        self.corr_topm = int(corr_topm)
        self.budget = int(budget)
        self.max_depth = int(max_depth)
        self.mask_token_id = int(mask_token_id)
        self.device = device

        # D depth levels: children_fn(depth) is empty at depth >= k_draft, and
        # best_first never scores children at depth >= max_depth.
        self.D = min(self.max_depth, self.k_draft)
        # W = budget is the EXACT setting (the lemma); smaller W is an
        # approximation and must be measured, not assumed.
        self.W = self.budget if width is None else int(width)
        if self.W <= 0:
            raise ValueError("width must be positive")
        self.N = self.budget + 1  # flat tree size incl. root
        self.L = self.D * self.W * self.node_topk  # ledger entries per request
        if self.L < self.budget:
            raise ValueError(
                f"ledger too small for the node budget: D*W*k = {self.L} < "
                f"budget = {self.budget} (raise width/node_topk)"
            )

        self.dtype = next(draft.parameters()).dtype
        self._gru_dim = int(draft.prefix_gru.hidden_size)
        self._hidden = int(draft.embed_proj[0].in_features) - self._gru_dim
        self._vocab = int(draft.embed_proj[2].out_features)
        if self.corr_topm > self._vocab:
            raise ValueError("corr_topm must be <= vocab")

        # ``embed_proj[0]`` is a bias-free Linear(H + G -> E) applied to
        # ``cat([ph_d, lane_states], -1)`` once per depth. The ph half is
        # LANE-INDEPENDENT, so the concatenation and D-1 of the D GEMMs are
        # redundant: split the weight once here, lift the ph half out of the
        # depth loop as a single [bs, D, H] x [H, E] GEMM, and per depth run
        # only the [bs, W, G] x [G, E] lane half plus a broadcast add. This
        # removes the per-depth ``torch.cat`` (2 memcpy128 each) and shrinks
        # the per-depth GEMM's contraction from H + G to G.
        #
        # Algebraically exact; in floating point the reduction is split in two,
        # so results differ by ~1 ulp of the fp32 accumulator — far below the
        # bf16 drift floor the builder already documents, but it is why this is
        # gated by DOMINOTREE_BUILDER_FUSION and validated by an end-to-end tau
        # A/B rather than assumed. The halves are cached because the drafter is
        # frozen at inference; they must be rebuilt if its weights ever change.
        self._fusion = os.environ.get("DOMINOTREE_BUILDER_FUSION", "1") != "0"
        if self._fusion:
            w0 = draft.embed_proj[0].weight  # [E, H + G]
            self._w0_ph = w0[:, : self._hidden].t().contiguous()  # [H, E]
            self._w0_lane = w0[:, self._hidden :].t().contiguous()  # [G, E]
            # cuDNN compacts a GRU's weights on every call when they are not
            # already one contiguous block, and torch's "RNN module weights are
            # not part of single contiguous chunk of memory" warning is present
            # in this builder's own profile. MEASURED 2026-09-03: this call
            # changes NOTHING at bf16 on torch 2.11/cu130 -- kernel count and
            # graph wall are byte-identical with and without it, and the warning
            # still fires, because flatten_parameters() early-returns when cuDNN
            # will not accept the dtype. Kept because it is free and would apply
            # to an fp16/fp32 drafter; do NOT re-open it as a bf16 lever, and do
            # not attribute any of the 30 memcpy128 per build to the GRU.
            try:
                draft.prefix_gru.flatten_parameters()
            except Exception:  # pragma: no cover - never block the build on it
                logger.warning("[DominoTree] prefix_gru.flatten_parameters() failed")
        else:
            self._w0_ph = self._w0_lane = None

        # Triton fusion of the per-depth body (see fused_depth.py). Requires the
        # round-1 fusion because it writes the LANE-keyed parent ledger, and the
        # restricted-correction path because the full-vocab scorer has no fused
        # form. Off by default until the end-to-end tau A/B has gated it.
        self._fused_depth = None
        if (
            os.environ.get("DOMINOTREE_BUILDER_FUSED_DEPTH", "0") != "0"
            and self._fusion
            and self.corr_topm > 0
            and self.device.type == "cuda"
        ):
            from .fused_depth import FusedDepth

            fd = FusedDepth(
                W=self.W,
                E=int(draft.embed_proj[0].out_features),
                M=self.corr_topm,
                K=self.node_topk,
                G=self._gru_dim,
            )
            if fd.available and not fd.bind_gru(draft.prefix_gru):
                logger.warning(
                    "[DominoTree] fused depth body: unsupported GRU "
                    "(multi-layer/bidirectional/projected); keeping torch's"
                )
                fd.gru_weights = None
            self._fused_depth = fd if fd.available else None
            if self._fused_depth is None:
                logger.warning(
                    "[DominoTree] fused depth body requested but unavailable "
                    "(Triton missing?); falling back to the torch depth loop"
                )

        self._states: dict[int, _State] = {}
        # Graphs own the captured allocations; retain their associated static
        # state in the per-bs pool for the lifetime of this builder.
        self._graph_pool: dict[int, dict] = {}
        self._graph_enabled = (
            os.environ.get("DOMINOTREE_FRONTIER_GRAPH", "1") != "0"
            and self.device.type == "cuda"
            and self.corr_topm > 0
        )
        self._graph_failed = False
        # Stamp the arm into the log: a bs=1 verdict can turn on ~0.5 ms/step,
        # so "which code produced this number" must never be a guess.
        logger.info(
            "[DominoTree] frontier builder: fusion=%s fused_depth=%s graph=%s "
            "budget=%d W=%d D=%d",
            self._fusion,
            self._fused_depth is not None,
            self._graph_enabled,
            self.budget,
            self.W,
            self.D,
        )

    # ------------------------------------------------------------------
    # Per-bs static state.
    # ------------------------------------------------------------------

    def _get_state(self, bs: int) -> _State:
        st = self._states.get(bs)
        if st is not None:
            return st
        dev, k = self.device, self.node_topk
        pd = min(self.prefix_len, self.D)
        st = _State(
            S_ph=torch.zeros(bs, self.k_draft, self._hidden, dtype=self.dtype, device=dev),
            S_root=torch.zeros(bs, self._gru_dim, dtype=self.dtype, device=dev),
            S_verified=torch.zeros(bs, dtype=torch.long, device=dev),
            S_cand=(
                torch.zeros(bs, self.k_draft, self.corr_topm, dtype=torch.long, device=dev)
                if self.corr_topm > 0
                else None
            ),
            S_basec=(
                torch.zeros(
                    bs, self.k_draft, self.corr_topm, dtype=torch.float32, device=dev
                )
                if self.corr_topm > 0
                else None
            ),
            S_prefix_toks=(
                torch.zeros(bs, pd, k, dtype=torch.long, device=dev) if pd > 0 else None
            ),
            S_prefix_lps=(
                torch.zeros(bs, pd, k, dtype=torch.float32, device=dev) if pd > 0 else None
            ),
            S_out_tokens=torch.zeros(bs, self.N, dtype=torch.long, device=dev),
            S_out_mask=torch.zeros(bs, self.N, self.N, dtype=torch.bool, device=dev),
            S_out_parents=torch.zeros(bs, self.N, dtype=torch.long, device=dev),
            S_out_depths=torch.zeros(bs, self.N, dtype=torch.long, device=dev),
            S_out_scores=torch.zeros(bs, self.N, dtype=torch.float32, device=dev),
        )
        self._states[bs] = st
        return st

    def _get_or_capture_graph(self, bs: int, st: _State) -> dict:
        """Return the CUDA graph for ``bs``, capturing it on first use.

        ``_load_inputs`` has already copied this step's inputs into ``st``.
        The warmup and capture deliberately operate only on those static
        buffers, so replay can use the same copy-in -> replay -> read-out
        sequence as the Domino rollout graph pool.
        """
        entry = self._graph_pool.get(bs)
        if entry is not None:
            return entry

        # cuDNN GRU and allocator work must finish before capture.  Use a side
        # stream so this follows the rollout graph-pool warmup discipline
        # without perturbing the caller's current stream ordering.
        current_stream = torch.cuda.current_stream(device=self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(3):
                self._body(st, None)
        current_stream.wait_stream(warmup_stream)
        torch.cuda.synchronize(device=self.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._body(st, None)
        entry = {"graph": graph, "state": st}
        self._graph_pool[bs] = entry
        logger.info("[DominoTree] captured CUDA graph for frontier builder bs=%d", bs)
        return entry

    # ------------------------------------------------------------------
    # Input load: transient [bs, K, V] logits -> reduced statics. Runs OUTSIDE
    # graph capture (analog of GraphNodeExpander.begin_round + _setup_impl,
    # batched).
    # ------------------------------------------------------------------

    def _load_inputs(
        self,
        st: _State,
        ph: torch.Tensor,
        base_logits: torch.Tensor,
        root_states: torch.Tensor,
        verified: torch.Tensor,
    ) -> None:
        bs = st.S_ph.shape[0]
        st.S_ph.copy_(ph[:, : self.k_draft])
        st.S_root.copy_(root_states.reshape(bs, self._gru_dim))
        st.S_verified.copy_(verified.view(-1))
        if self.corr_topm > 0:
            # == per-depth torch.topk(base_logits[d], corr_topm) of the
            # reference (same model-dtype rows -> same indices), batched.
            cand = torch.topk(base_logits, self.corr_topm, dim=-1).indices
            st.S_cand.copy_(cand)
            st.S_basec.copy_(torch.gather(base_logits, -1, cand).float())
        pd = min(self.prefix_len, self.D)
        if pd > 0:
            # == log_prob_topk(base_logits[d], node_topk): path-independent, so
            # once per step is value-identical to the reference's per-node call.
            logp = torch.log_softmax(base_logits[:, :pd].float(), dim=-1)
            vals, idx = torch.topk(logp, k=self.node_topk, dim=-1)
            st.S_prefix_toks.copy_(idx)
            st.S_prefix_lps.copy_(vals)

    # ------------------------------------------------------------------
    # The depth-synchronous body. Reads only static inputs + model weights,
    # writes only static outputs; every intermediate has a static shape.  It is
    # CUDA-graph captured for corr_topm > 0 only; the full-vocab path also
    # reads transient logits and therefore remains eager-only.
    # ------------------------------------------------------------------

    def _body(self, st: _State, base_logits_full: torch.Tensor | None) -> None:
        bs = st.S_ph.shape[0]
        W, k, D, N = self.W, self.node_topk, self.D, self.N
        G, H = self._gru_dim, self._hidden
        wk = W * k
        dev = self.device
        neg_inf = float("-inf")

        # Frontier lanes: lane 0 = root (score 0), the rest invalid (-inf).
        lane_scores = torch.full((bs, W), neg_inf, dtype=torch.float32, device=dev)
        lane_scores[:, 0] = 0.0
        lane_states = st.S_root.unsqueeze(1).expand(bs, W, G).contiguous()
        # Global ledger index of the node occupying each lane (-1 = root).
        lane_node = torch.full((bs, W), -1, dtype=torch.long, device=dev)

        led_scores = torch.full((bs, D, wk), neg_inf, dtype=torch.float32, device=dev)
        led_tokens = torch.zeros((bs, D, wk), dtype=torch.long, device=dev)
        # Parent pointers are a property of the LANE, not of each of its k
        # children, so the ledger stores one entry per lane and the k-fold
        # broadcast is folded into the final gather (``sel_idx // k`` is exactly
        # ``d * W + lane`` because the flat ledger index is
        # ``d*W*k + lane*k + child`` with ``child < k``). Storing [bs, D, W]
        # instead of [bs, D, W*k] removes one ``repeat_interleave`` alloc+copy
        # AND shrinks the per-depth ledger write k-fold. Pure reindexing — the
        # selected parents are bit-identical to the wide-ledger version.
        led_parent = torch.full(
            (bs, D, W if self._fusion else wk), -1, dtype=torch.long, device=dev
        )

        ep = self.draft.embed_proj
        # Lane-independent half of ep[0] for ALL depths in one GEMM (see the
        # weight-split note in __init__).
        ph_half = (
            torch.matmul(st.S_ph[:, :D], self._w0_ph) if self._fusion else None
        )  # [bs, D, E]

        fd = self._fused_depth
        for d in range(D):
            # --- score all W*k children of the frontier (3 correction cases,
            # conditional_children.py:64-92 with a leading [bs, W] batch) ---
            # The fused kernel covers case 2 only; the prefix rows keep the
            # torch path (there is nothing to fuse -- they are one broadcast).
            fused_here = fd is not None and d >= self.prefix_len
            if fused_here:
                # Case 2, fused: the whole scorer -- broadcast add, SiLU, casts,
                # w2[cand] gather, bmm, log_softmax, top-k, candidate gather,
                # cumulative add and all three ledger writes -- in ONE kernel.
                # Only the lane-half GEMM stays outside, because it is a real
                # GEMM and cuBLAS should keep it.
                fd.score(
                    hpre=torch.matmul(lane_states, self._w0_lane),
                    ph_half_d=ph_half[:, d],
                    w2=ep[2].weight,
                    cand_d=st.S_cand[:, d],
                    basec_d=st.S_basec[:, d],
                    lane_scores=lane_scores,
                    lane_node=lane_node,
                    led_s_d=led_scores[:, d],
                    led_t_d=led_tokens[:, d],
                    led_p_d=led_parent[:, d],
                )
            elif d < self.prefix_len:
                # Case 1: uncorrected prefix rows — path-independent, identical
                # for every lane.
                vals = st.S_prefix_lps[:, d].unsqueeze(1).expand(bs, W, k)
                toks = st.S_prefix_toks[:, d].unsqueeze(1).expand(bs, W, k)
            elif self.corr_topm > 0:
                # Case 2: restricted correction (published default corr_topm=64).
                if self._fusion:
                    pre = ph_half[:, d].unsqueeze(1) + torch.matmul(
                        lane_states, self._w0_lane
                    )  # [bs, W, E]
                else:
                    ph_d = st.S_ph[:, d].unsqueeze(1).expand(bs, W, H)
                    pre = ep[0](torch.cat([ph_d, lane_states], dim=-1))
                h = ep[1](pre).float()
                # Gather w2[cand] INSIDE the depth body — never materialize the
                # [bs, K, M, E] candidate-weight tensor (§2A memory).
                w2c_d = ep[2].weight[st.S_cand[:, d]].float()  # [bs, M, E]
                corrected = st.S_basec[:, d].unsqueeze(1) + torch.bmm(
                    h, w2c_d.transpose(1, 2)
                )  # [bs, W, M]
                vals, local_idx = torch.topk(
                    torch.log_softmax(corrected, dim=-1), k=k, dim=-1
                )
                # MEASURED 2026-09-03: gathering from the [bs, M] row instead
                # (``gather(S_cand[:, d], 1, local_idx.reshape(bs, W*k))``) is a
                # NO-OP — identical kernel count and wall time — because gather
                # reads the stride-0 expand without materializing it. Left as
                # is; do not "optimize" this expand away.
                toks = torch.gather(
                    st.S_cand[:, d].unsqueeze(1).expand(bs, W, self.corr_topm),
                    2,
                    local_idx,
                )
            else:
                # Case 3: full-vocab correction (corr_topm == 0; eager-only, see
                # module docstring).
                ph_d = st.S_ph[:, d].unsqueeze(1).expand(bs, W, H)
                corr = ep(torch.cat([ph_d, lane_states], dim=-1))  # [bs, W, V]
                full = (base_logits_full[:, d].unsqueeze(1) + corr).float()
                vals, toks = torch.topk(
                    torch.log_softmax(full, dim=-1), k=k, dim=-1
                )

            if not fused_here:
                # Cumulative path scores; -inf lanes poison all their
                # descendants.
                cum = lane_scores.unsqueeze(-1) + vals  # [bs, W, k] float32
                flat_cum = cum.reshape(bs, wk)
                flat_tok = toks.reshape(bs, wk)

                # --- ledger: log EVERY scored candidate ---
                led_scores[:, d] = flat_cum
                led_tokens[:, d] = flat_tok
                led_parent[:, d] = (
                    lane_node if self._fusion else lane_node.repeat_interleave(k, dim=1)
                )

            # --- keep top-W per request, GRU-advance the kept lanes ---
            if d + 1 < D:
                if fd is not None:
                    # topk + token gather + parent-lane divide + lane-state
                    # gather, fused. Fresh buffers per depth (as before): shapes
                    # are static, so graph capture owns the allocations.
                    kept_tok = torch.empty((bs, W), dtype=torch.long, device=dev)
                    h0 = torch.empty(
                        (bs, W, G), dtype=lane_states.dtype, device=dev
                    )
                    keep_vals = torch.empty(
                        (bs, W), dtype=torch.float32, device=dev
                    )
                    next_node = torch.empty((bs, W), dtype=torch.long, device=dev)
                    fd.select(
                        led_s_d=led_scores[:, d],
                        led_t_d=led_tokens[:, d],
                        lane_states=lane_states,
                        h0=h0,
                        kept_tok=kept_tok,
                        lane_scores_out=keep_vals,
                        lane_node_out=next_node,
                        d_off=d * wk,
                    )
                else:
                    keep_vals, keep_idx = torch.topk(flat_cum, W, dim=1)  # [bs, W]
                    kept_tok = torch.gather(flat_tok, 1, keep_idx)
                    parent_lane = keep_idx // k
                    h0 = torch.gather(
                        lane_states, 1, parent_lane.unsqueeze(-1).expand(bs, W, G)
                    )
                    next_node = d * wk + keep_idx  # global ledger index
                # One GRU step per kept lane, batched over bs*W (the reference
                # runs the same module with batch = node_topk per pop).
                emb = self.embed_tokens(kept_tok)  # [bs, W, E]
                if fd is not None and fd.gru_weights is not None:
                    lane_states = fd.gru_cell(emb, h0)
                else:
                    _, hn = self.draft.prefix_gru(
                        emb.reshape(bs * W, 1, -1), h0.reshape(1, bs * W, G)
                    )
                    lane_states = hn.reshape(bs, W, G)
                lane_scores = keep_vals
                lane_node = next_node

        # --- global top-B over the ledger: selection + topological order in
        # ONE stable descending sort. Flat ledger index is depth-major, so the
        # stable tie-break is depth-ascending: parents (score >= child, depth <
        # child) always precede children, even on exact score ties; on tie-free
        # inputs the order equals the heap's pop order. ---
        L, B = self.L, self.budget
        scores_flat = led_scores.reshape(bs, L)
        sorted_scores, order = torch.sort(
            scores_flat, dim=1, descending=True, stable=True
        )
        sel_idx = order[:, :B]  # [bs, B] ledger indices, final node order
        sel_scores = sorted_scores[:, :B]
        valid = torch.isfinite(sel_scores)  # -inf = dead-leaf slot (sorts last)

        sel_tokens = torch.gather(led_tokens.reshape(bs, L), 1, sel_idx)
        if self._fusion:
            # sel_idx // k == d * W + lane (see the led_parent comment above).
            sel_parent_led = torch.gather(
                led_parent.reshape(bs, D * W),
                1,
                torch.div(sel_idx, k, rounding_mode="floor"),
            )
        else:
            sel_parent_led = torch.gather(led_parent.reshape(bs, L), 1, sel_idx)
        sel_depth = sel_idx // wk + 1  # flat tree depth (root = 0)

        # Remap ledger parent indices -> positions in the ordered node list.
        # Ancestor closure guarantees every valid node's parent is selected
        # (root = -1); if it ever were not (impossible by the lemma), pos
        # gathers -1 and the +1 lands on the root — still a well-formed tree,
        # so the verify stays safe by construction.
        pos = torch.full((bs, L), -1, dtype=torch.long, device=dev)
        pos.scatter_(
            1, sel_idx, torch.arange(B, dtype=torch.long, device=dev).repeat(bs, 1)
        )
        parent_pos = torch.gather(pos, 1, sel_parent_led.clamp(min=0))
        flat_parent_sel = torch.where(
            sel_parent_led < 0, torch.zeros_like(parent_pos), parent_pos + 1
        )

        # Dead-leaf padding: EXACT per-request semantics (worker.py flat pad):
        # mask_token_id children of the root at flat depth 1, leaf-only,
        # appended after all real nodes.
        sel_tokens = torch.where(
            valid, sel_tokens, torch.full_like(sel_tokens, self.mask_token_id)
        )
        flat_parent_sel = torch.where(
            valid, flat_parent_sel, torch.zeros_like(flat_parent_sel)
        )
        sel_depth = torch.where(valid, sel_depth, torch.ones_like(sel_depth))

        # --- flat outputs (root at index 0) ---
        st.S_out_tokens[:, 0] = st.S_verified
        st.S_out_tokens[:, 1:] = sel_tokens
        st.S_out_parents[:, 0] = -1
        st.S_out_parents[:, 1:] = flat_parent_sel
        st.S_out_depths[:, 0] = 0
        st.S_out_depths[:, 1:] = sel_depth
        st.S_out_scores[:, 0] = 0.0
        st.S_out_scores[:, 1:] = sel_scores

        # --- intra-tree ancestor mask [bs, N, N] (same convention as
        # build_intra_tree_mask_from_parents: mask[b, i, j] == True iff j is an
        # ancestor of i or i == j), by BOOLEAN TRANSITIVE CLOSURE via repeated
        # squaring instead of D sequential parent-pointer hops.
        #
        # Why: the hop loop costs D iterations x ~7 kernels; on a GPU-dispatch-
        # bound builder (747 tiny kernels, 2.3 us each) that is the single
        # largest kernel-count item in the depth-independent tail. Squaring
        # reaches hop distance 2^t, so ceil(log2(D)) rounds cover every flat
        # depth <= D: 4 rounds x 2 kernels at D=16.
        #
        # Exactness: A holds 0/1 in float32 and every partial sum is bounded by
        # N (<= 2^24), so the products and the clamp are exact integer
        # arithmetic — the result is bit-identical to the boolean hop loop, not
        # an approximation.
        parent_full = st.S_out_parents
        if self._fusion:
            # A[b, i, j] = 1 iff parent(i) == j (the root's -1 contributes
            # nothing), plus the identity on the diagonal so each squaring also
            # carries the SHORTER paths forward.
            adj = F.one_hot(parent_full.clamp(min=0), N).to(torch.float32)
            adj = adj * (parent_full >= 0).unsqueeze(-1).to(torch.float32)
            # Diagonal via a strided VIEW, never mask[:, arange, arange] —
            # advanced indexing with a device tensor triggers a CUDA sync
            # (caught by the zero-sync gate).
            adj.diagonal(dim1=1, dim2=2).fill_(1.0)
            for _ in range(max(D - 1, 0).bit_length()):
                adj = torch.clamp(torch.bmm(adj, adj), max=1.0)
            st.S_out_mask.copy_(adj > 0.5)
        else:
            mask = torch.zeros((bs, N, N), dtype=torch.bool, device=dev)
            mask.diagonal(dim1=1, dim2=2).fill_(True)
            cur = parent_full.clone()
            for _ in range(D):  # max flat depth <= D -> <= D hops to the root
                alive = cur >= 0
                hop = F.one_hot(cur.clamp(min=0), N).bool() & alive.unsqueeze(-1)
                mask |= hop
                cur = torch.where(
                    alive, torch.gather(parent_full, 1, cur.clamp(min=0)), cur
                )
            st.S_out_mask.copy_(mask)

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def build(
        self,
        ph: torch.Tensor,  # [bs, K, H] shift_label-sliced draft hidden
        base_logits: torch.Tensor,  # [bs, K, V] = lm_head(ph); TRANSIENT
        root_states: torch.Tensor,  # (1, bs, G) GRU hidden after the verified token
        verified: torch.Tensor,  # [bs] committed root token ids (DEVICE tensor)
        *,
        return_aux: bool = False,
    ):
        """Build all ``bs`` trees; return ``(draft_tokens_2d [bs, N] long,
        intra_mask [bs, N, N] bool)`` device tensors (plus an aux dict of
        parents/depths/cum_logprobs when ``return_aux``).  Eager and steady-
        state graph replay have zero host syncs; first graph capture warms up
        and synchronizes once.
        """
        if ph.dtype != self.dtype or base_logits.dtype != self.dtype:
            raise TypeError(
                f"dtype mismatch: ph={ph.dtype}, base_logits={base_logits.dtype}, "
                f"expected {self.dtype} (a silent cast would change numerics)"
            )
        if ph.dim() != 3 or ph.shape[1] < self.k_draft or ph.shape[2] != self._hidden:
            raise ValueError(f"ph shape {tuple(ph.shape)} incompatible with builder")
        bs = int(ph.shape[0])
        st = self._get_state(bs)
        self._load_inputs(st, ph, base_logits, root_states, verified)
        if self._graph_enabled and not self._graph_failed:
            try:
                self._get_or_capture_graph(bs, st)["graph"].replay()
            except Exception:
                # Capture/replay must be an optimization only.  A failed CUDA
                # graph can leave the just-captured outputs unreliable, so
                # re-run this step eagerly and do not retry capture later.
                self._graph_failed = True
                self._graph_pool.clear()
                logger.exception(
                    "[DominoTree] frontier CUDA graph failed; permanently "
                    "falling back to eager execution"
                )
                self._body(st, None)
        else:
            self._body(st, base_logits if self.corr_topm == 0 else None)
        tokens = st.S_out_tokens.clone()
        mask = st.S_out_mask.clone()
        if not return_aux:
            return tokens, mask
        aux = {
            "parents": st.S_out_parents.clone(),
            "depths": st.S_out_depths.clone(),
            "cum_logprobs": st.S_out_scores.clone(),
        }
        return tokens, mask, aux
