"""Domino spec-decode worker: upstream ``DFlashWorkerV2`` + the Domino rollout.

Phase 1 (CHAIN only). The only Domino-specific change vs plain DFLASH is the
per-step draft-token selection: upstream runs a single greedy argmax over the
target LM head; Domino replaces it with the sequential GRU-corrected rollout
(``DFlashDominoRollout.rollout_draft_block``). The block construction, the
target verify (linear/chain), and the KV path are all inherited unchanged.

Override strategy
-----------------
Upstream's draft+verify live in one ~490-line method,
``DFlashWorkerV2.forward_batch_generation`` (dflash_worker_v2.py:1200-1692). The
greedy-sample call we must replace is a single statement buried in the middle
(dflash_worker_v2.py:1493-1496). Copying the whole method to patch 4 lines is
brittle, so instead we wrap ``forward_batch_generation`` and, for the duration
of the call, temporarily:

  1. capture the FULL draft-block hidden states by wrapping
     ``draft_model_runner.forward`` (the greedy seam only receives
     ``draft_hidden[:, 1:, :]``, dropping slot 0, which the rollout needs when
     ``shift_label=True``); and
  2. rebind ``_greedy_sample_from_vocab_parallel_head`` to a closure that runs
     the Domino rollout on the captured hidden + verified token.

Both are restored in ``finally``. In the DFLASH decode path,
``draft_model_runner.forward`` and ``_greedy_sample_from_vocab_parallel_head``
are each called exactly once, so the capture is unambiguous. See PORT_NOTES.md.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

from .config import is_dflash_domino_projector
from .domino_helper import DFlashDominoHelper
from .domino_rollout import DFlashDominoRollout

logger = logging.getLogger(__name__)


# Every upstream attribute this plugin reads through a defensive `getattr(..., default)`.
# A rename upstream makes such a read return the DEFAULT instead of raising, which is how
# this plugin has already been bitten three times in one upgrade:
#   * spec_info.verified_id      -> bonus_tokens      (correction silently skipped, tau 4.0 -> 1.0)
#   * model_runner.hybrid_gdn_config -> mambaish_config()  (hybrid guard silently stopped firing)
#   * DFlashDraftInputV2.verify_done  removed          (constructor kwarg silently wrong)
# The pattern is always the same: a *safety* or *dispatch* read degrades to a plausible
# default and nothing crashes. This check turns that class of failure into a loud startup
# error instead of a silent behaviour change.
#
# Format: (getter, attribute, why it matters). `getter` takes the worker and returns the
# object to inspect, or None if that object is not available at check time (then skipped).
_UPSTREAM_CONTRACT = [
    ("server_args", "speculative_use_rejection_sampling",
     "losslessness guard: rejection sampling is lossy for a greedy draft"),
    ("server_args", "enable_custom_logit_processor",
     "losslessness guard: custom logit processors bypass the verifier"),
    ("server_args", "speculative_draft_window_size",
     "compact draft-cache guard"),
]

# LATE-BOUND attributes: assigned after the draft worker is constructed (e.g.
# decode_cuda_graph_runner is set during target CUDA-graph capture,
# model_runner.py:929). hasattr() at init would be False even though nothing was
# renamed -- checking them here would fail-closed on a healthy build, which is
# exactly what happened the first time this check ran. They are verified at their
# point of use instead; see _require_late_bound below.


def _assert_upstream_contract(worker) -> None:
    """Fail loudly if an upstream attribute we read defensively has disappeared."""
    resolve = {
        "server_args": lambda w: getattr(
            getattr(getattr(w, "target_worker", None), "model_runner", None),
            "server_args", None),
        "target_model_runner": lambda w: getattr(
            getattr(w, "target_worker", None), "model_runner", None),
    }
    missing = []
    for holder, attr, why in _UPSTREAM_CONTRACT:
        obj = resolve[holder](worker)
        if obj is None:
            continue  # not constructed yet on this build; not evidence of a rename
        if not hasattr(obj, attr):
            missing.append(f"  {holder}.{attr}  --  {why}")
    if missing:
        raise RuntimeError(
            "DOMINOTREE: the upstream SGLang contract changed. These attributes are "
            "read defensively by this plugin and no longer exist, so the reads would "
            "silently return their defaults and the associated guard/dispatch would "
            "stop working:\n" + "\n".join(missing) +
            "\nRefusing to start rather than run with a silently disabled guard."
        )



def _require_late_bound(obj, attr, why):
    """Read a late-bound upstream attribute, distinguishing ABSENT from None.

    `getattr(obj, attr, None)` collapses two very different cases: the attribute
    exists and is legitimately None (e.g. CUDA graphs disabled), or the attribute
    is GONE because upstream renamed it -- in which case the guard that depends on
    it silently stops working. hasattr() separates them.
    """
    if not hasattr(obj, attr):
        logger.error(
            "DOMINOTREE: %s.%s no longer exists on this SGLang build. %s. The "
            "associated guard is now INACTIVE -- treat any results from this run "
            "as untrusted.",
            type(obj).__name__, attr, why,
        )
        return None
    return getattr(obj, attr)


def _detect_mamba_target(target_mr):
    """Is the TARGET a Mamba / hybrid-linear-attention model?

    Returns ``(is_hybrid, detector_name)``. ``detector_name is None`` means NO
    detector could be evaluated -- the caller must then refuse to launch rather
    than assume "not hybrid".

    Why this is more than a getattr. Our original check read
    ``model_runner.hybrid_gdn_config`` / ``.mamba2_config``. Current upstream
    removed both from ModelRunner and exposes the same information as a function,
    ``sglang.srt.configs.hybrid_arch.mambaish_config(model_config)``. A plain
    ``getattr(..., None)`` therefore returns None on new SGLang and the guard
    silently STOPS FIRING -- the exact failure mode it exists to prevent, and the
    third instance in this plugin of "defensive getattr silently absorbs an
    upstream rename". Detectors are tried newest-first and the one that answered
    is reported so the log says how we decided.
    """
    # 1) current upstream: a function over the model config
    try:
        from sglang.srt.configs.hybrid_arch import mambaish_config

        mc = getattr(target_mr, "model_config", None)
        if mc is not None:
            return mambaish_config(mc) is not None, "hybrid_arch.mambaish_config"
    except Exception:
        pass
    # 2) some builds cache it on the KV-cache configurator / runner
    for attr in ("mambaish_config", "hybrid_gdn_config", "mamba2_config"):
        try:
            if hasattr(target_mr, attr):
                return getattr(target_mr, attr) is not None, f"model_runner.{attr}"
        except Exception:
            pass
    # 3) last resort: the attention backend grew the mamba commit hook. Only
    #    meaningful once the backend exists (it does not at draft-worker init on
    #    every build), so absence here is NOT evidence of "not hybrid".
    try:
        backend = getattr(target_mr, "attn_backend", None)
        if backend is not None:
            return (
                hasattr(backend, "update_mamba_state_after_mtp_verify"),
                "attn_backend.update_mamba_state_after_mtp_verify",
            )
    except Exception:
        pass
    return False, None


def _draft_bonus_tokens(draft_input):
    """The previous round's accepted/bonus token(s), across SGLang versions.

    Upstream renamed DFlashDraftInputV2.verified_id -> bonus_tokens after
    1adb53f14. Both name the same thing: the last accepted token per request,
    which seeds the next draft.
    """
    v = getattr(draft_input, "bonus_tokens", None)
    if v is None:
        v = getattr(draft_input, "verified_id", None)
    if v is None:
        raise AttributeError(
            f"{type(draft_input).__name__} exposes neither bonus_tokens nor "
            "verified_id -- the upstream draft-input contract changed again."
        )
    return v


def assert_domino_server_args_supported(server_args, algo_name: str) -> None:
    """Fail fast at LAUNCH on server configs the Domino plugin does not support.

    Called from the spec class's ``handle_server_args`` (main process, before any
    model load) and re-checked in the worker ``__init__`` (scheduler subprocess),
    so programmatic launches that bypass the speculative arg hook still fail
    fast instead of silently producing wrong output (P5 correctness gate).
    """
    # TP>1 is supported: the Domino chain's global-argmax reduction callbacks are
    # implemented (DominoWorkerV2._global_argmax_from_local_{logits,max}) and the
    # DOMINOTREE tree build all-gathers a full-vocab, rank-replicated base-logits
    # tensor per step (DominoTreeWorkerV2._tp_full_base_logits). The
    # DOMINOTREE-specific guards below apply at every tensor-parallel size.
    if algo_name == "DOMINOTREE":
        page_size = int(getattr(server_args, "page_size", 1) or 1)
        if page_size != 1:
            raise NotImplementedError(
                f"DOMINOTREE does not support --page-size {page_size}: the tree "
                "verify KV path (reserved-slot dual-use in the draft block, "
                "accepted-path compaction, front-slot draft-KV commit) assumes "
                "page_size == 1. Run with --page-size 1."
            )
        if getattr(server_args, "speculative_draft_window_size", None) is not None:
            raise NotImplementedError(
                "DOMINOTREE does not support --speculative-draft-window-size "
                "(compact draft cache): the tree draft block reads the "
                "scheduler-reserved verify slots directly and assumes the "
                "non-windowed DFLASH draft-KV layout."
            )
        # P6: T>0 uses tree_speculative_sampling with THRESHOLD acceptance
        # (draft_probs=zeros). Rejection sampling would instead require a
        # target-vocab draft proposal distribution, which the greedy Domino draft
        # does not produce (eagle_utils.py:498-505 raises on missing draft_probs).
        if getattr(server_args, "speculative_use_rejection_sampling", False):
            raise NotImplementedError(
                "DOMINOTREE does not support --speculative-use-rejection-sampling: "
                "rejection sampling needs a target-vocab draft proposal "
                "distribution (draft_probs), but the Domino draft is greedy/argmax "
                "and produces none. Use the default threshold acceptance "
                "(speculative_accept_threshold_*), which matches the official "
                "Domino SGLang baseline."
            )
        # P6 losslessness: the T>0 tree verify (tree_speculative_sampling_target_only)
        # is exact ONLY at threshold 1.0/1.0. Lower thresholds enable SGLang's
        # "typical acceptance", which over-accepts and makes the committed T>0
        # distribution deviate from the target's -> silently NOT lossless.
        thr_single = getattr(server_args, "speculative_accept_threshold_single", 1.0)
        thr_acc = getattr(server_args, "speculative_accept_threshold_acc", 1.0)
        if thr_single != 1.0 or thr_acc != 1.0:
            raise NotImplementedError(
                "DOMINOTREE requires speculative_accept_threshold_single == 1.0 and "
                f"speculative_accept_threshold_acc == 1.0 (exact spec sampling); got "
                f"single={thr_single}, acc={thr_acc}. Non-default thresholds enable "
                "SGLang typical-acceptance, which breaks the T>0 lossless guarantee. "
                "Run with the defaults (1.0)."
            )
        # P6 losslessness: eagle_sample does NOT apply custom logit processors
        # (unlike the DFLASH chain, dflash_utils.py:170-174), so at T>0 they would
        # be silently ignored -> wrong distribution.
        if getattr(server_args, "enable_custom_logit_processor", False):
            raise NotImplementedError(
                "DOMINOTREE does not support --enable-custom-logit-processor: the "
                "tree verify (eagle_sample) does not apply custom logit processors, "
                "so they would be silently ignored at T>0. Use "
                "--speculative-algorithm DOMINO (chain), which applies them."
            )


class DominoWorkerV2(DFlashWorkerV2):
    """DFLASH v2 worker with the Domino GRU-corrected chain rollout."""

    _algo_name = "DOMINO"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Re-assert launch invariants in the scheduler subprocess (P5): the
        # handle_server_args guard runs in the main process only; programmatic
        # launches may skip it. TP>1 is now supported (see the TP helpers on
        # DominoWorkerV2 / DominoTreeWorkerV2); only the non-TP invariants remain.
        assert_domino_server_args_supported(self.server_args, self._algo_name)

        self.domino_helper: Optional[DFlashDominoHelper] = None
        self.domino_rollout: Optional[DFlashDominoRollout] = None

        # Target ORG vocab size (full, unsharded). Under TP>1 the tree build
        # slices the all-gathered full-vocab base logits to exactly this width.
        self._target_vocab_size = int(
            getattr(self.target_worker.model_runner.model_config, "vocab_size", 0) or 0
        )

        projector_type = getattr(self.draft_model, "projector_type", None)
        if not is_dflash_domino_projector(projector_type):
            logger.warning(
                "DominoWorkerV2 loaded a draft model without a Domino projector "
                "(projector_type=%r); falling back to plain DFLASH chain draft.",
                projector_type,
            )
            return

        self.domino_helper = DFlashDominoHelper(self.draft_model)

        self.domino_rollout = DFlashDominoRollout(
            domino_helper=self.domino_helper,
            block_size=int(self.block_size),
            target_vocab_size=self._target_vocab_size,
            # TP>1 reduction callbacks are only invoked by the rollout's
            # vocab-parallel fallback (when DFLASH_DOMINO_TP_REPLICATE_SCORER is
            # disabled or auto-disabled by a memory check); TP=1 never calls them.
            global_argmax_from_local_logits=self._global_argmax_from_local_logits,
            global_argmax_from_local_max=self._global_argmax_from_local_max,
        )
        logger.info(
            "DominoWorkerV2 initialized Domino chain rollout "
            "(projector_type=%r, block_size=%d, shift_label=%s).",
            projector_type,
            int(self.block_size),
            bool(getattr(self.draft_model, "shift_label", False)),
        )

    def forward_batch_generation(self, model_worker_batch, on_publish=None):
        if self.domino_rollout is None:
            return super().forward_batch_generation(model_worker_batch, on_publish)

        spec_info = getattr(model_worker_batch, "spec_info", None)
        # Version-tolerant: upstream renamed verified_id -> bonus_tokens. Reading
        # only the old name here silently disabled the Domino correction on newer
        # SGLang (see the loud warning in the fallback below).
        verified_id = None
        if spec_info is not None:
            verified_id = getattr(spec_info, "bonus_tokens", None)
            if verified_id is None:
                verified_id = getattr(spec_info, "verified_id", None)

        captured: dict = {}
        orig_draft_forward = self.draft_model_runner.forward
        orig_greedy = self._greedy_sample_from_vocab_parallel_head

        def _capture_draft_forward(forward_batch, *a, **k):
            out = orig_draft_forward(forward_batch, *a, **k)
            logits_output = getattr(out, "logits_output", None)
            hidden_states = (
                getattr(logits_output, "hidden_states", None)
                if logits_output is not None
                else None
            )
            if hidden_states is not None:
                captured["draft_hidden"] = hidden_states
            return out

        def _domino_sample_draft_block(*, hidden_states, lm_head, chunk_size: int = 256):
            if not getattr(self, "_domino_hook_confirmed", False):
                self._domino_hook_confirmed = True
                # Permanent positive signal. Its ABSENCE is the alarm: upstream's
                # folded draft sampler once bypassed this hook entirely and tau
                # silently fell 4.0 -> 1.0 with no error. If this line is missing
                # from a server log, the GRU correction is not running.
                logger.info(
                    "DOMINO: GRU correction active (draft sampling hook installed)."
                )
            draft_hidden = captured.get("draft_hidden")
            if draft_hidden is None or verified_id is None:
                # Fall back to plain greedy drafting. This is a CORRECTNESS-SAFE but
                # QUALITY-DESTROYING path: without the GRU correction the DFlash head
                # runs on Domino weights and acceptance collapses to ~1.0 -- i.e. the
                # speculation stops paying for itself while still costing a draft pass.
                #
                # It must never be silent again. It was: upstream renamed
                # verified_id -> bonus_tokens, this lookup started returning None, and
                # DOMINO quietly dropped from tau 4.0 to 1.008 with no error anywhere.
                if not getattr(self, "_warned_greedy_fallback", False):
                    self._warned_greedy_fallback = True
                    logger.error(
                        "DOMINO: falling back to PLAIN GREEDY drafting -- the Domino "
                        "GRU correction is NOT being applied (draft_hidden=%s, "
                        "verified_id/bonus_tokens=%s). Acceptance length will collapse "
                        "to ~1.0. This usually means the upstream draft-input contract "
                        "changed again; check DFlashDraftInputV2's field names.",
                        "missing" if draft_hidden is None else "ok",
                        "missing" if verified_id is None else "ok",
                    )
                return orig_greedy(
                    hidden_states=hidden_states, lm_head=lm_head, chunk_size=chunk_size
                )
            verified = verified_id.view(-1)
            bs = int(verified.shape[0])
            draft_hidden = draft_hidden.reshape(bs, int(self.block_size), -1)
            target_model = self.target_worker.model_runner.model
            # rollout_draft_block returns [bs, block_size - 1]; the caller
            # reshapes to [bs, block_size - 1], so flatten to match the greedy
            # contract ([bs * (block_size - 1)] flat).
            draft_next = self.domino_rollout.rollout_draft_block(
                draft_hidden=draft_hidden,
                verified_id=verified,
                target_model=target_model,
                lm_head=lm_head,
            )
            return draft_next.reshape(-1)

        self.draft_model_runner.forward = _capture_draft_forward
        self._greedy_sample_from_vocab_parallel_head = _domino_sample_draft_block
        try:
            return super().forward_batch_generation(model_worker_batch, on_publish)
        finally:
            self.draft_model_runner.forward = orig_draft_forward
            self._greedy_sample_from_vocab_parallel_head = orig_greedy

    # -- TP>1 global-argmax reductions (chain vocab-parallel fallback) --------
    #
    # These implement the two callbacks the Domino chain rollout invokes ONLY on
    # its vocab-parallel path (when the replicated scorer is off): each rank has
    # scored its own vocab shard, so the globally-winning token must be selected
    # across ranks before the next GRU step. Mirrors the upstream reference
    # ``DFlashWorkerV2._greedy_sample_from_vocab_parallel_head``
    # (dflash_worker_v2.py ~765-819): all-gather per-rank max VALUES + the
    # per-rank winning GLOBAL token ids, argmax over the rank axis, then gather
    # the winning ids. The all-gather + argmax are deterministic and identical on
    # every rank, so all ranks pick the same token and the Domino GRU state stays
    # in lockstep.

    def _tp_global_argmax_reduce(
        self, local_max: torch.Tensor, global_ids: torch.Tensor
    ) -> torch.Tensor:
        """Given each rank's per-row local max value and the corresponding GLOBAL
        token id, return the per-row globally-winning token id ([bs] int64),
        identical on every rank."""
        global_ids = global_ids.to(torch.int64)
        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        if tp_size == 1:
            return global_ids

        local_max = local_max.contiguous()
        global_ids = global_ids.contiguous()
        n = int(local_max.shape[0])
        # 1-D gather buffers, rank-major, then view(tp_size, n) -- exactly the
        # reference idiom (dflash_worker_v2.py:804-819).
        gathered_max = torch.empty(
            (tp_size * n,), dtype=local_max.dtype, device=local_max.device
        )
        gathered_ids = torch.empty(
            (tp_size * n,), dtype=global_ids.dtype, device=global_ids.device
        )
        tp_group.all_gather_into_tensor(gathered_max, local_max)
        tp_group.all_gather_into_tensor(gathered_ids, global_ids)
        gathered_max = gathered_max.view(tp_size, n)
        gathered_ids = gathered_ids.view(tp_size, n)

        best_rank = torch.argmax(gathered_max, dim=0)  # [n]
        selected = torch.gather(gathered_ids, 0, best_rank.unsqueeze(0))  # [1, n]
        return selected.view(-1).to(torch.int64)

    def _global_argmax_from_local_logits(
        self, *, local_logits, local_vocab_start, local_token_ids=None
    ) -> torch.Tensor:
        """TP global argmax from each rank's LOCAL logits.

        ``local_logits`` is ``[bs, local_width]``. Without ``local_token_ids`` the
        columns are contiguous local shard positions, so a local argmax + the
        shard's ``local_vocab_start`` offset yields the global id. With
        ``local_token_ids`` (``[bs, local_width]`` of GLOBAL ids, the candidate
        path), the winning global id is gathered directly from that table. The
        per-rank (value, global_id) pair is then reduced across the TP group.
        """
        local_max, local_arg = torch.max(local_logits, dim=-1)  # [bs], [bs]
        if local_token_ids is None:
            global_ids = local_arg.to(torch.int64) + int(local_vocab_start)
        else:
            global_ids = torch.gather(
                local_token_ids, 1, local_arg.to(torch.int64).unsqueeze(1)
            ).view(-1)
        return self._tp_global_argmax_reduce(local_max, global_ids)

    def _global_argmax_from_local_max(
        self, *, local_max, global_ids
    ) -> torch.Tensor:
        """TP global argmax from precomputed per-rank max VALUES + GLOBAL ids.

        Used on the fused-kernel path, where the local scorer already produced the
        winning value and (via ``+ org_vocab_start`` at the call site) its global
        token id; here we only reduce across the TP group.
        """
        return self._tp_global_argmax_reduce(local_max, global_ids)


# ---------------------------------------------------------------------------
# Phase 2: tree-verify (toy fixed tree)
# ---------------------------------------------------------------------------


def _compact_accept_to_front(
    x: torch.Tensor, accept_index: torch.Tensor, bs: int, nd: int
) -> torch.Tensor:
    """Gather the accepted tree path to the front of each per-req block.

    Reimplements ``EAGLEWorkerV2._compact_accept_to_front``
    (eagle_worker_v2.py:1595-1611): ``x`` is node-indexed over the whole tree
    (``[bs * nd, ...]``); ``accept_index`` is ``[bs, s1]`` global node indices
    (-1 padded, clamped to 0 — padded entries land past accept_lens and are
    never read). Returns a copy with the accepted path at the front.
    """
    s1 = accept_index.shape[1]
    safe = accept_index.to(torch.int64).clamp(min=0).reshape(-1)
    gathered = x[safe]
    out = x.clone()
    out.view(bs, nd, *x.shape[1:])[:, :s1] = gathered.view(bs, s1, *x.shape[1:])
    return out


class DominoTreeWorkerV2(DominoWorkerV2):
    """DFLASH-drafter worker whose *verify* is an EAGLE tree instead of a chain.

    The Domino block-parallel DRAFT is unchanged; only the decode-time verify is
    swapped from DFLASH's linear chain to a tree run through EAGLE's tree verifier
    (``eagle_prepare_for_verify`` -> target verify -> ``eagle_sample``), with the
    accepted path compacted to a prefix (EAGLE's
    ``move_accept_tokens_to_target_kvcache`` + ``_compact_accept_to_front``) so
    DFLASH's prefix-only draft-KV writer is reused verbatim.

    Three tree builders share that verify seam:

    * **conditional (P3, DEFAULT)** — the paper's method: a per-request adaptive,
      variable-width **best-first** tree built by the Domino GRU-correction scorer
      (``tree/best_first.py`` + ``tree/conditional_children.py``). Env
      ``DOMINOTREE_NODE_TOPK`` (8), ``DOMINOTREE_CORR_TOPM`` (64; 0=full-vocab),
      ``prefix_len = draft_model.pure_draft_prefix_len``, budget = block_size-1.
    * **toy (P2, A/B via ``DOMINOTREE_BUILDER=toy``)** — a fixed caterpillar tree.
    * **frontier (Option B, ``DOMINOTREE_BUILDER=frontier``)** — the batched
      depth-synchronous frontier builder (``tree/frontier.py``,
      batch_builder_design.md §2B): same conditional scorer math, all bs trees
      built on-device with zero host syncs, equal to the best-first tree up to
      real-valued score ties. Opt-in until the §3 validation gates pass on GPU;
      the default stays the per-request best-first builder.

    Losslessness is BY CONSTRUCTION at every temperature: at T=0 the greedy tree
    verify only accepts argmax-matching tokens; at T>0 (P6) eagle_sample's
    tree_speculative_sampling_target_only is the distribution-preserving threshold
    sampler (draft stays greedy; only acceptance is temperature-aware). Constraints:
    page_size==1, non-mamba target, no compact-draft-cache window, no rejection
    sampling (greedy draft produces no draft_probs). TP>1 IS supported: the tree
    build all-gathers a full-vocab, rank-replicated base-logits tensor per step
    (``_tp_full_base_logits``), so every rank builds the same tree and the
    per-request host syncs need no cross-rank coordination. Anything else falls
    back to the lossless Domino chain.

    P4: the tree verify now runs under the decode CUDA graph (the DOMINOTREE
    spec_class enables the custom-mask DFLASH verify graph in handle_server_args),
    so ``--disable-cuda-graph`` is OPTIONAL — kept as an eager fallback. See
    PORT_NOTES.md.

    P5 (correctness gate): unsupported configs FAIL FAST instead of running
    silently (page_size!=1, compact draft cache, Mamba/hybrid target at
    server level; return_logprob / grammar per request), and any chain-verify
    FALLBACK on a cuda-graph server is forced EAGER (``_chain_fallback``): the
    P4b hook captures the target-verify graph WITH a ``custom_mask_buf``, but
    upstream's chain verify passes ``custom_mask=None`` (dflash_worker_v2.py:1504),
    so a graph replay would reuse a stale mask -> wrong commits.
    """

    _algo_name = "DOMINOTREE"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # P5 server-level guard: Mamba/hybrid targets. Upstream's chain verify
        # commits Mamba states after verify (dflash_worker_v2.py:1518-1521), but
        # our tree verify path does not port commit_mamba_states_after_verify
        # (spec_utils.py:577), so accepted-path Mamba states would silently never
        # be committed. Detect from the target model CONFIG (hybrid_gdn_config /
        # mamba2_config), which is available at draft-worker init — the target
        # attn_backend is constructed LATER, so reading `.attn_backend` here
        # AttributeErrors on the pinned build. Best-effort + fully defensive: a
        # detection failure must never crash init for the common non-Mamba case.
        target_mr = self.target_worker.model_runner
        is_mamba_target, detect_via = _detect_mamba_target(target_mr)
        if detect_via is None:
            # We could not evaluate ANY known detector. Refusing to launch beats
            # running a hybrid target unguarded: this guard exists precisely
            # because the failure it prevents is silent state corruption, not a
            # crash. (Historically this block used
            # `except Exception: is_mamba_target = False`, i.e. it failed OPEN.)
            raise RuntimeError(
                "DOMINOTREE cannot determine whether the target is a Mamba/hybrid "
                "model: none of the known detectors are available on this SGLang "
                "build (configs.hybrid_arch.mambaish_config, model_runner."
                "hybrid_gdn_config / .mamba2_config, or the attention backend's "
                "update_mamba_state_after_mtp_verify hook). Refusing to launch "
                "rather than risk running a hybrid target unguarded -- accepted-path "
                "recurrent states would silently never be committed. Please file "
                "this with the SGLang version you are running."
            )
        logger.info(
            "DOMINOTREE hybrid-target detection: %s (via %s)",
            "HYBRID" if is_mamba_target else "not hybrid",
            detect_via,
        )
        _assert_upstream_contract(self)
        if is_mamba_target:
            raise NotImplementedError(
                "DOMINOTREE does not support Mamba/hybrid target models: the "
                "tree verify path does not port commit_mamba_states_after_verify, "
                "so accepted-path Mamba states would never be committed. Use a "
                "non-Mamba target, or --speculative-algorithm DOMINO / DFLASH "
                "(chain verify, which handles Mamba upstream)."
            )

        # Verify node budget N == DFLASH block_size, so ALL DFLASH KV/buffer
        # sizing (reserved per-decode tokens, cuda-graph widths) is reused with
        # no server-arg override.
        self.tree_num_nodes = int(self.block_size)

        def _env_int(name, default):
            try:
                return int(os.environ.get(name, str(default)))
            except ValueError:
                return default

        # DEFAULT = the Option B batched frontier builder (tree/frontier.py): GPU-
        # validated (commits 92a11a4 + e6fe5f0), zero host syncs, and with
        # DOMINOTREE_FRONTIER_GRAPH on by default (frontier.py) it is the fastest,
        # deployment-and-paper-headline config — so a bare `--speculative-algorithm
        # DOMINOTREE` runs the best builder out of the box. DOMINOTREE_BUILDER
        # overrides are kept for science/reproducibility + robustness, NOT for
        # routine use: `conditional` = the P3 per-request best-first heap (the
        # paper's Python-vs-GPU-native builder ablation, with DOMINOTREE_GPU_BUILDER
        # toggling its CUDA-graph node expander); `toy` = the P2 fixed caterpillar.
        # If the frontier builder fails to construct on a given GPU it falls back to
        # the conditional builder automatically (_build_frontier_trees).
        self.tree_builder = os.environ.get("DOMINOTREE_BUILDER", "frontier").strip()
        self.tree_node_topk = _env_int("DOMINOTREE_NODE_TOPK", 8)
        self.tree_corr_topm = _env_int("DOMINOTREE_CORR_TOPM", 64)
        self.tree_prefix_len = int(getattr(self.draft_model, "pure_draft_prefix_len", 1))
        self.tree_num_branch = _env_int("DOMINOTREE_NUM_BRANCH", 2)

        # P4: GPU-native CUDA-graph node expander (default ON; 0 = P3 pure-Python).
        # Constructed + captured LAZILY on first decode (needs the loaded model +
        # a live CUDA context, mid-run, without colliding with SGLang init).
        self.tree_gpu_builder = _env_int("DOMINOTREE_GPU_BUILDER", 1) != 0
        self._gpu_expander = None
        self._gpu_expander_failed = False

        # Option B batched frontier builder (DOMINOTREE_BUILDER=frontier).
        # Constructed lazily on first decode, like the GPU expander.
        self._frontier_builder = None
        self._frontier_failed = False

        self.tree_topology = None
        # `domino_rollout is not None` <=> the draft model has the Domino
        # projector (prefix_gru + embed_proj), which BOTH builders require.
        if self.domino_rollout is not None:
            from .tree.toy_tree import build_topology

            # Clamp branches so the toy spine keeps >= ~half the block.
            max_branch = max(0, (self.tree_num_nodes - 1) // 2)
            if self.tree_num_branch > max_branch:
                logger.warning(
                    "DOMINOTREE num_branch=%d too large for N=%d; clamping to %d.",
                    self.tree_num_branch,
                    self.tree_num_nodes,
                    max_branch,
                )
                self.tree_num_branch = max_branch
            self.tree_topology = build_topology(
                num_nodes=self.tree_num_nodes, num_branch=self.tree_num_branch
            )
            logger.info(
                "DominoTreeWorkerV2 ready: builder=%s, N=%d, node_topk=%d, "
                "corr_topm=%d, prefix_len=%d, shift_label=%s (toy: spine_len=%d, "
                "num_branch=%d).",
                self.tree_builder,
                self.tree_num_nodes,
                self.tree_node_topk,
                self.tree_corr_topm,
                self.tree_prefix_len,
                bool(getattr(self.draft_model, "shift_label", False)),
                self.tree_topology.spine_len,
                self.tree_topology.num_branch,
            )

    def forward_batch_generation(self, model_worker_batch, on_publish=None):
        # P5 per-request guards — BEFORE any routing (tree OR chain), so an
        # unsupported request fails with a clear error instead of silently
        # producing wrong output. First line of defense is upstream's
        # validate_dflash_request (scheduler.py:2120, fires because we report
        # is_dflash()=True), which gracefully aborts these requests before
        # scheduling; these raises are the worker-level backstop.
        self._assert_request_supported(model_worker_batch)

        # Only the DECODE-verify stage differs from DFLASH/Domino. Route prefill/
        # extend/idle and any non-tree case back to the Domino chain worker
        # (which is itself lossless), then handle the greedy decode-verify here.
        # Any fallback that can reach the chain DECODE verify must go through
        # _chain_fallback (stale custom-mask graph replay; see class docstring).
        if self.tree_topology is None:
            return self._chain_fallback(model_worker_batch, on_publish)

        mode = model_worker_batch.forward_mode
        if (
            mode.is_extend()
            or getattr(model_worker_batch, "is_extend_in_batch", False)
            or mode.is_idle()
            or model_worker_batch.spec_info is None
        ):
            return self._chain_fallback(model_worker_batch, on_publish)

        # P6: route BOTH greedy (T=0) and sampled (T>0) decode-verify through the
        # TREE. The draft stays greedy at every temperature (our convention,
        # matched to official Domino-on-SGLang which also drafts greedily); only
        # the ACCEPTANCE differs, and eagle_sample dispatches internally on
        # sampling_info.is_all_greedy: verify_tree_greedy (T=0) vs
        # tree_speculative_sampling_target_only (T>0), the distribution-preserving
        # threshold sampler EAGLE uses. sampling_info is None only for
        # non-generation batches -> chain fallback.
        sampling_info = getattr(model_worker_batch, "sampling_info", None)
        if sampling_info is None:
            return self._chain_fallback(model_worker_batch, on_publish)

        return self._tree_decode_forward(model_worker_batch, on_publish)

    # -- P5 correctness gate: guards + safe chain fallback -------------------

    def _assert_request_supported(self, model_worker_batch) -> None:
        """Reject per-request features the DOMINOTREE paths cannot honor."""
        if getattr(model_worker_batch, "return_logprob", False):
            raise ValueError(
                "DOMINOTREE does not support return_logprob: the tree verify "
                "path does not port compute_spec_v2_logprobs. Re-send the "
                "request without logprobs."
            )
        if getattr(model_worker_batch, "return_hidden_states", False):
            raise ValueError(
                "DOMINOTREE does not support return_hidden_states: the tree and "
                "chain-fallback verify paths consume and clear the target hidden "
                "states, so the client would silently receive empty hidden "
                "states. (Upstream's validate_dflash_request rejects this only "
                "when overlap is ENABLED; this plugin forces overlap off, so that "
                "check never fires — hence this worker-level guard.) Re-send "
                "without return_hidden_states."
            )
        if getattr(model_worker_batch, "has_grammar", False):
            raise ValueError(
                "DOMINOTREE does not support grammar-constrained decoding "
                "(json_schema/regex/ebnf/structural_tag): the tree verify calls "
                "eagle_sample without a grammar vocab mask, so constraints "
                "would be silently ignored. Re-send without grammar constraints."
            )
        # P6 losslessness: eagle_sample applies only temperature/top_k/top_p — NOT
        # min_p — so a min_p request would sample the tree from the wrong T>0
        # distribution silently.
        sampling_info = getattr(model_worker_batch, "sampling_info", None)
        if sampling_info is not None and getattr(
            sampling_info, "need_min_p_sampling", False
        ):
            raise ValueError(
                "DOMINOTREE does not support min_p sampling: the tree verify "
                "(eagle_sample) applies only temperature/top_k/top_p, so min_p "
                "would be silently ignored (wrong distribution at T>0). Re-send "
                "without min_p, or use --speculative-algorithm DOMINO (chain)."
            )

    def _chain_fallback(self, model_worker_batch, on_publish):
        """Run the inherited Domino chain path, forcing its target verify EAGER
        when the decode CUDA graph is enabled.

        Why: the P4b hook makes the decode-graph capture allocate a flashinfer
        ``custom_mask_buf`` for the DOMINOTREE target-verify graph (needed by
        the tree's custom mask). Upstream's chain verify, however, builds
        ``DFlashVerifyInput`` with ``custom_mask=None``
        (dflash_worker_v2.py:1502-1513) and never refreshes that buffer, while
        ``can_run_graph`` (decode_cuda_graph_runner.py:400-469) has no mask
        gate — so a chain-verify graph REPLAY would rerun the captured
        MaskMode.CUSTOM kernel against a stale/garbage mask buffer and commit
        wrong tokens. Nulling ``decode_cuda_graph_runner`` for the duration
        makes BOTH decision points (DFlashVerifyInput.prepare_for_verify:74-87
        and ModelRunner._forward_raw:2978-2982) take the eager branch, which is
        exactly the GPU-validated ``--disable-cuda-graph`` chain flow
        (prepare_for_verify plans eager attn metadata; the verify forward runs
        with metadata marked ready). The draft runner's own graphs and the
        Domino rollout graph live on the DRAFT model runner and are untouched.

        Post-P6 (T>0 routed to a sampling TREE verify) this helper remains only
        for genuinely-unsupported fallbacks (e.g. non-Domino draft model) —
        still correct, marginally slower, rarely hit.

        NOTE: safe only because the worker is synchronous (handle_server_args
        forces --disable-overlap-schedule). Revisit if overlap is ever enabled.
        """
        mode = model_worker_batch.forward_mode
        reaches_chain_verify = not (
            mode.is_extend()
            or getattr(model_worker_batch, "is_extend_in_batch", False)
            or mode.is_idle()
        )
        target_model_runner = self.target_worker.model_runner
        graph_runner = _require_late_bound(
            target_model_runner,
            "decode_cuda_graph_runner",
            "the chain fallback must force EAGER, or a graph replay reuses a stale "
            "custom mask and commits the wrong tokens",
        )
        if not reaches_chain_verify or graph_runner is None:
            return super().forward_batch_generation(model_worker_batch, on_publish)

        target_model_runner.decode_cuda_graph_runner = None
        try:
            return super().forward_batch_generation(model_worker_batch, on_publish)
        finally:
            target_model_runner.decode_cuda_graph_runner = graph_runner

    # -- draft -------------------------------------------------------------

    def _domino_draft_block(self, model_worker_batch):
        """Run the Domino block draft; return (spine_tokens[bs,N-1], draft_hidden[bs,N,H]).

        Mirrors DFLASH's draft build (dflash_worker_v2.py:1326-1500). The draft
        block dual-uses the SAME scheduler-reserved verify slots that the tree
        verify will use (DFLASH-v2 pattern): ``assign_extend_cache_locs_func``
        *reads* the reserved slots from ``req_to_token[req, L:L+N]`` (it does not
        allocate). We must NOT allocate scratch and overwrite ``req_to_token``:
        ``eagle_prepare_for_verify`` later re-reads ``req_to_token[req, L:L+N]``
        for the verify out_cache_loc, so any overwrite (+ later free) would make
        the verify — and every subsequent step's prefix — read freed/wrong KV
        slots (silent losslessness break). Assumes page_size==1.
        """
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )
        # UPSTREAM MOVE: sglang.srt.speculative.eagle_info_v2 was DELETED upstream and
        # this helper now lives in the kernels package. Try the new home first, fall
        # back to the old one so the plugin still imports on SGLang <= 1adb53f14 (the
        # commit every published v2 number was measured on).
        try:
            from sglang.kernels.ops.speculative.cache_locs import (
                assign_extend_cache_locs_func,
            )
        except ImportError:  # pragma: no cover - legacy SGLang
            from sglang.srt.speculative.eagle_info_v2 import (
                assign_extend_cache_locs_func,
            )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        device = self.device
        bs = len(model_worker_batch.seq_lens)
        n = int(self.block_size)
        draft_input = model_worker_batch.spec_info
        prefix_lens = model_worker_batch.seq_lens
        verified = _draft_bonus_tokens(draft_input).view(-1)

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()

        block_ids = torch.full(
            (bs, n), int(self._mask_token_id), dtype=torch.int64, device=device
        )
        block_ids[:, 0].copy_(verified)
        positions_2d = prefix_lens.view(bs, 1) + torch.arange(
            n, device=device, dtype=prefix_lens.dtype
        ).view(1, n)
        input_embeds = embed_module(block_ids).view(-1, embed_module.weight.shape[-1])

        # Read (do NOT allocate) the reserved verify slots and dual-use them for
        # the draft-KV block; req_to_token stays intact for eagle_prepare.
        block_cache_loc = assign_extend_cache_locs_func(
            model_worker_batch.req_pool_indices,
            self.model_runner.req_to_token_pool.req_to_token,
            prefix_lens,
            prefix_lens + n,
            bs,
            n,
            device,
        )
        seq_lens_cpu = prefix_lens.to(device="cpu", dtype=torch.int32)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=model_worker_batch.req_pool_indices,
            seq_lens=prefix_lens,
            out_cache_loc=block_cache_loc,
            seq_lens_sum=int(prefix_lens.sum().item()),
            seq_lens_cpu=seq_lens_cpu,
            positions=positions_2d.reshape(-1),
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(forward_batch).logits_output

        draft_hidden = draft_out.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DOMINOTREE draft model returned no hidden states.")
        return draft_hidden.reshape(bs, n, -1)  # [bs, N, H]

    # -- toy caterpillar tree (P2, A/B via DOMINOTREE_BUILDER=toy) ----------

    def _toy_tree_tokens(self, draft_hidden, verified, bs, device):
        """Assemble the P2 fixed caterpillar tree tokens + intra mask."""
        from .tree.toy_tree import build_draft_tokens, build_intra_tree_mask

        target_model = self.target_worker.model_runner.model
        lm_head = getattr(target_model, "lm_head", None)
        spine_tokens = self.domino_rollout.rollout_draft_block(
            draft_hidden=draft_hidden,
            verified_id=verified,
            target_model=target_model,
            lm_head=lm_head,
        )  # [bs, N-1]
        branch_tokens = self._branch_candidates(draft_hidden, spine_tokens)
        if branch_tokens is None:
            branch_tokens = torch.empty((bs, 0), dtype=torch.int64, device=device)
        draft_tokens_2d = build_draft_tokens(
            self.tree_topology,
            verified_id=verified,
            spine_tokens=spine_tokens,
            branch_tokens=branch_tokens,
        )  # [bs, N]
        intra_mask = build_intra_tree_mask(self.tree_topology, bs=bs, device=device)
        return draft_tokens_2d, intra_mask

    def _branch_candidates(self, draft_hidden, spine_tokens):
        """2nd-candidate tokens for the branch depths (TP=1 dense LM head)."""
        b = self.tree_num_branch
        if b <= 0:
            return None
        bs = draft_hidden.shape[0]
        target_model = self.target_worker.model_runner.model
        lm_head = target_model.lm_head
        weight = lm_head.weight  # [V, H] (full at TP=1)
        # draft_hidden[:, d, :] predicts the depth-d token (shift_label=False);
        # branch depth k (1..b) reuses draft_hidden[:, k, :].
        z = draft_hidden[:, 1 : b + 1, :].reshape(bs * b, -1)
        z = z.to(weight.dtype) if z.dtype != weight.dtype else z
        logits = torch.matmul(z, weight.T)  # [bs*b, V]
        top2 = torch.topk(logits, k=2, dim=-1).indices.view(bs, b, 2).to(torch.int64)
        c_head = spine_tokens[:, :b].to(torch.int64)  # spine tokens at depths 1..b
        # Take the top token unless it equals the spine token, then the 2nd.
        branch = torch.where(top2[:, :, 0] != c_head, top2[:, :, 0], top2[:, :, 1])
        return branch  # [bs, b]

    # -- conditional best-first tree (P3 pure-Python / P4 GPU expander) ----

    def _effective_node_topk(self):
        """node_topk clamped to <= corr_topm (matches the pure-Python scorer +
        the GPU expander's constructor precondition)."""
        node_topk = int(self.tree_node_topk)
        if self.tree_corr_topm > 0:
            node_topk = min(node_topk, int(self.tree_corr_topm))
        return node_topk

    def _get_gpu_expander(self):
        """Lazily build + capture the CUDA-graph node expander (P4) on first
        decode. Returns None (once) if unavailable so we fall back to the
        pure-Python conditional builder.

        Constructed mid-run: graph capture needs the loaded model + a live CUDA
        context and must not collide with SGLang init. It captures 3 graphs on a
        side stream in inference_mode; safe under --disable-cuda-graph (the model
        is not graphed) and independent of the Domino rollout's own graph.
        """
        if self._gpu_expander is not None or self._gpu_expander_failed:
            return self._gpu_expander
        try:
            from .tree.gpu_expander import GraphNodeExpander

            n = int(self.block_size)
            shift_label = bool(getattr(self.draft_model, "shift_label", False))
            k_draft = n if shift_label else n - 1
            embed_tokens = self.target_worker.model_runner.model.get_input_embeddings()
            self._gpu_expander = GraphNodeExpander(
                draft=self.draft_model,
                embed_tokens=embed_tokens,
                k_draft=k_draft,
                prefix_len=self.tree_prefix_len,
                node_topk=self._effective_node_topk(),
                corr_topm=self.tree_corr_topm,
                device=self.device,
            )
            logger.info(
                "DOMINOTREE GPU node expander ready (use_graphs=%s, k_draft=%d, "
                "node_topk=%d, corr_topm=%d, prefix_len=%d).",
                self._gpu_expander.use_graphs,
                k_draft,
                self._effective_node_topk(),
                self.tree_corr_topm,
                self.tree_prefix_len,
            )
        except Exception as e:
            self._gpu_expander_failed = True
            self._gpu_expander = None
            logger.warning(
                "DOMINOTREE GPU node expander unavailable; using pure-Python "
                "conditional builder. Reason: %s",
                e,
            )
        return self._gpu_expander

    def _build_conditional_tree_for_req(self, ph, base_logits, root_state, verified_scalar):
        """Build one request's conditional best-first tree (batch=1).

        Returns ``(tokens[N], parent[N], depth[N])`` PADDED to exactly N nodes,
        where index 0 is the root (= verified token). Short trees are padded with
        DEAD leaf nodes (children of root carrying ``mask_token_id``) that can
        never match the target's greedy argmax and are leaves, so they can never
        be accepted and never alter the accepted path.

        Phase 0 (batch_builder_design.md §3): the per-request SETUP — shift_label
        slicing, the LM-head matmul, and the root-state GRU — is computed batched
        in ``_build_conditional_trees`` and passed in per-request:

        * ``ph``: ``[k_draft, H]`` this request's shift_label-sliced draft hidden
          (domino_adapter.py:92-96);
        * ``base_logits``: ``[k_draft, V]`` = target.lm_head(ph) (TP=1 dense head,
          domino_adapter.py:95);
        * ``root_state``: ``(1, 1, gru_dim)`` = prefix_gru(embed_tokens(verified))
          hidden (domino_adapter.py:97);
        * ``verified_scalar``: the committed root token id (host int).
        """
        from .tree.best_first import build_best_first_tree
        from .tree.conditional_children import make_conditional_children_fn

        n = int(self.block_size)
        device = self.device
        target_model = self.target_worker.model_runner.model
        embed_tokens = target_model.get_input_embeddings()
        draft_model = self.draft_model

        # children_fn: P4 GPU CUDA-graph expander (default) or P3 pure-Python.
        # The GPU expander is bit-equivalent to make_conditional_children_fn (its
        # own equivalence suite proves it), so acceptance is unchanged.
        expander = self._get_gpu_expander() if self.tree_gpu_builder else None
        if expander is not None and (
            ph.dtype != expander.S_ph_all.dtype
            or base_logits.dtype != expander.S_base_all.dtype
        ):
            # Draft-hidden and lm_head are normally the same server dtype (bf16),
            # so this should not trigger; if it does (mixed dtypes), fall back to
            # pure-Python rather than let the expander's no-silent-cast guard
            # raise mid-decode.
            logger.warning(
                "DOMINOTREE GPU expander dtype mismatch (ph=%s base_logits=%s vs "
                "expander=%s); using pure-Python conditional builder.",
                ph.dtype,
                base_logits.dtype,
                expander.S_ph_all.dtype,
            )
            self._gpu_expander_failed = True
            self._gpu_expander = None
            expander = None
        if expander is not None:
            expander.begin_round(ph, base_logits)
            children_fn = expander.children_fn
        else:
            children_fn = make_conditional_children_fn(
                ph=ph,
                base_logits=base_logits,
                draft_model=draft_model,
                embed_tokens=embed_tokens,
                node_topk=self._effective_node_topk(),
                corr_topm=self.tree_corr_topm,
                prefix_len=self.tree_prefix_len,
                device=device,
            )

        # budget = N-1 so root + (N-1) nodes = N (the DFLASH-reserved width).
        nodes = build_best_first_tree(
            children_fn, root_state, budget=n - 1, max_depth=n
        )

        tokens = [int(verified_scalar)]
        parent = [-1]
        depth = [0]
        for nd in nodes:
            tokens.append(int(nd.token))
            # flat parent: root (flat 0) if node.parent == -1, else 1 + node.parent.
            parent.append(0 if nd.parent == -1 else 1 + int(nd.parent))
            depth.append(1 + int(nd.depth))  # flat tree-depth = node.depth + 1
        # Pad short trees with dead leaves = children of root carrying mask_token_id.
        while len(tokens) < n:
            tokens.append(int(self._mask_token_id))
            parent.append(0)
            depth.append(1)
        return tokens[:n], parent[:n], depth[:n]

    def _tp_full_base_logits(self, ph_cast, lm_head, valid_vocab_size):
        """FULL-VOCAB target base logits, replicated identically on every TP rank.

        The tree builders consume ``base_logits`` as ``[..., vocab]`` marginals and
        emit token ids by indexing that vocab axis. Under TP>1 ``lm_head.weight``
        is only this rank's org-vocab SHARD, so a naive ``ph @ weight.T`` would
        cover a partial vocab and produce wrong token ids. We instead all-gather
        the per-rank shard LOGITS (one collective per tree build) into the full
        org vocab, laid out as a direct padded-org concat, and slice to the target
        org vocab. The result is identical on all ranks, so the entire downstream
        per-request builder (and its ``.tolist()`` host syncs) runs rank-replicated
        with no further cross-rank coordination.

        This mirrors the shard-layout assumptions of
        ``DFlashDominoRollout._get_domino_tp_full_lm_head_weight`` (all-gather the
        first ``num_org_padded`` rows per rank, concatenate in rank order, slice to
        the org vocab) but gathers the LOGITS rather than the WEIGHT, which is
        cheaper and avoids the memory-fragile replicated-scorer weight path.
        """
        weight = lm_head.weight
        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        # TP=1 (or an unsharded head): dense weight -> behave EXACTLY as before.
        if tp_size == 1 or not hasattr(lm_head, "shard_indices"):
            return torch.matmul(ph_cast, weight.T)

        shard = lm_head.shard_indices
        num_added = int(shard.num_added_elements)
        if num_added != 0:
            raise NotImplementedError(
                "DOMINOTREE tree build does not support added-vocab lm_head shards "
                f"under TP>1 (num_added_elements={num_added})."
            )
        num_org_padded = int(shard.num_org_elements_padded)
        org_vocab_start = int(shard.org_vocab_start_index)
        valid = int(valid_vocab_size)
        if num_org_padded <= 0:
            raise RuntimeError(
                "DOMINOTREE tree build: lm_head shard has empty padded org vocab "
                f"(num_org_elements_padded={num_org_padded})."
            )
        # Layout assumption (identical to _get_domino_tp_full_lm_head_weight): the
        # target vocab is a DIRECT padded-org concat, i.e. rank r owns global org
        # token ids [r*num_org_padded, ...). Gathering shards in rank order is only
        # valid under this layout; otherwise token ids would be mislabeled.
        expected_org_start = int(tp_group.rank_in_group) * num_org_padded
        if org_vocab_start != expected_org_start:
            raise RuntimeError(
                "DOMINOTREE tree build requires a direct padded-org concat vocab "
                f"shard layout: org_vocab_start={org_vocab_start}, "
                f"expected={expected_org_start} (rank_in_group*num_org_padded)."
            )
        full_padded = tp_size * num_org_padded
        if valid <= 0 or valid > full_padded:
            raise RuntimeError(
                "DOMINOTREE tree build: target vocab out of range vs padded org "
                f"vocab: valid_vocab_size={valid}, full_padded={full_padded}."
            )
        if int(weight.shape[0]) < num_org_padded:
            raise RuntimeError(
                "DOMINOTREE tree build: lm_head shard smaller than padded org "
                f"shard: weight_rows={int(weight.shape[0])}, "
                f"num_org_padded={num_org_padded}."
            )

        # Local shard logits: [bs, k_draft, num_org_padded].
        local_logits = torch.matmul(
            ph_cast, weight[:num_org_padded].T
        ).contiguous()
        bs, k_draft, _ = local_logits.shape
        # All-gather over the TP group (rank-major): [tp_size, bs, k_draft, npad].
        gathered = torch.empty(
            (tp_size, bs, k_draft, num_org_padded),
            dtype=local_logits.dtype,
            device=local_logits.device,
        )
        tp_group.all_gather_into_tensor(gathered, local_logits)
        # Merge the rank axis with the local-vocab axis into the global vocab axis:
        #   gathered[r, b, k, j] -> full[b, k, r*num_org_padded + j]
        full = (
            gathered.permute(1, 2, 0, 3)
            .reshape(bs, k_draft, full_padded)
        )
        return full[..., :valid]

    def _build_conditional_trees(self, draft_hidden, verified, bs, n, device):
        """Build a per-request conditional tree; return (draft_tokens[bs,N], intra_mask[bs,N,N]).

        Phase 0 (batch_builder_design.md §3): the tree-build SETUP is hoisted out
        of the serial per-request loop and computed batched — one host sync + two
        batched kernels instead of one ``.item()`` sync plus two tiny kernels per
        request. The values fed to each request's builder are the same math as
        the old per-request setup (same dtype handling, same matmul operands,
        same GRU module); only WHERE the compute happens changes. The per-request
        expander + best-first heap (the Phase 1 target) are untouched.
        """
        from .tree.toy_tree import build_intra_tree_mask_from_parents

        target_model = self.target_worker.model_runner.model
        draft_model = self.draft_model
        shift_label = bool(getattr(draft_model, "shift_label", False))

        # (1) ONE GPU->CPU sync for ALL root token ids (was int(verified[b].item())
        # per request).
        verified_list = verified.tolist()

        # ph_all: per-position draft hidden, shift_label-sliced
        # (domino_adapter.py:92-96), whole batch at once.
        #   shift_label=False -> last (N-1) rows per request (k_draft = N-1)
        #   shift_label=True  -> all N rows                  (k_draft = N)
        if shift_label:
            ph_all = draft_hidden  # [bs, N, H]
        else:
            ph_all = draft_hidden[:, -(n - 1):, :]  # [bs, N-1, H]

        # (2) ONE batched LM-head matmul (was bs separate [k_draft,H]x[H,V]
        # GEMMs); torch.matmul folds [bs,k_draft,H]@[H,V] into a single
        # [bs*k_draft,H]@[H,V] mm. base_logits_all is a TRANSIENT (~139 MB bf16
        # at bs=32, V=151K) freed when this function returns — do NOT keep it as
        # a persistent buffer. Phase 1 (batch_builder_design.md §2A) reduces it
        # immediately to [bs,k_draft,corr_topm]-sized statics; Phase 0 only
        # batches the matmul.
        weight = target_model.lm_head.weight  # [V,H] full at TP=1; vocab shard at TP>1
        ph_cast = ph_all.to(weight.dtype) if ph_all.dtype != weight.dtype else ph_all
        # TP=1: dense head -> ph_cast @ weight.T, unchanged. TP>1: all-gather the
        # per-rank shard logits into a full-vocab, rank-replicated tensor so every
        # rank builds the same tree (the crux of the TP port).
        base_logits_all = self._tp_full_base_logits(
            ph_cast, target_model.lm_head, self._target_vocab_size
        )  # [bs, k_draft, V] (identical on every TP rank)

        # (3) ONE batched root-state GRU over the [bs] verified tokens (was bs
        # separate 1-token GRU calls). prefix_gru is batch_first, so
        # root_states[:, b:b+1, :] == the (1, 1, gru_dim) hidden the per-request
        # prefix_gru(embed_tokens(verified_b)) call produced.
        embed_tokens = target_model.get_input_embeddings()
        root_emb_all = embed_tokens(verified.view(bs, 1).to(torch.long))  # [bs,1,E]
        _, root_states = draft_model.prefix_gru(root_emb_all)  # (1, bs, gru_dim)

        all_tokens = []
        all_parents = []
        for b in range(bs):
            tokens, parent, _depth = self._build_conditional_tree_for_req(
                ph_all[b],
                base_logits_all[b],
                root_states[:, b : b + 1, :],
                verified_list[b],
            )
            all_tokens.append(tokens)
            all_parents.append(parent)
        draft_tokens_2d = torch.tensor(
            all_tokens, dtype=torch.int64, device=device
        )  # [bs, N]
        intra_mask = build_intra_tree_mask_from_parents(
            all_parents, n=n, device=device
        )  # [bs, N, N]
        return draft_tokens_2d, intra_mask

    # -- batched frontier tree (Option B, DOMINOTREE_BUILDER=frontier) ------

    def _get_frontier_builder(self):
        """Lazily build the Option B batched frontier builder
        (batch_builder_design.md §2B, ``tree/frontier.py``). Returns None
        (once) if unavailable so ``_build_frontier_trees`` falls back to the
        per-request best-first builder."""
        if self._frontier_builder is not None or self._frontier_failed:
            return self._frontier_builder
        try:
            from .tree.frontier import FrontierTreeBuilder

            n = int(self.block_size)
            shift_label = bool(getattr(self.draft_model, "shift_label", False))
            k_draft = n if shift_label else n - 1
            embed_tokens = self.target_worker.model_runner.model.get_input_embeddings()
            self._frontier_builder = FrontierTreeBuilder(
                draft=self.draft_model,
                embed_tokens=embed_tokens,
                k_draft=k_draft,
                prefix_len=self.tree_prefix_len,
                node_topk=self._effective_node_topk(),
                corr_topm=self.tree_corr_topm,
                budget=n - 1,
                max_depth=n,
                mask_token_id=int(self._mask_token_id),
                device=self.device,
            )
            logger.info(
                "DOMINOTREE frontier builder ready (k_draft=%d, node_topk=%d, "
                "corr_topm=%d, prefix_len=%d, width=%d, depths=%d).",
                k_draft,
                self._effective_node_topk(),
                self.tree_corr_topm,
                self.tree_prefix_len,
                self._frontier_builder.W,
                self._frontier_builder.D,
            )
        except Exception as e:
            self._frontier_failed = True
            self._frontier_builder = None
            logger.warning(
                "DOMINOTREE frontier builder unavailable; using the per-request "
                "best-first conditional builder. Reason: %s",
                e,
            )
        return self._frontier_builder

    def _build_frontier_trees(self, draft_hidden, verified, bs, n, device):
        """Option B (batch_builder_design.md §2B): depth-synchronous batched
        frontier build of ALL bs trees on-device with ZERO host syncs.

        Same batched setup as Phase 0's ``_build_conditional_trees`` (shift_label
        slicing, one LM-head matmul, one root-state GRU), but ``verified`` stays
        a DEVICE tensor (no ``.tolist()``), the per-request heap loop is replaced
        by the batched frontier depth-loop, and the flat tokens + intra mask are
        emitted as device tensors (no ``torch.tensor(all_tokens)`` H2D, no
        Python mask build). ``base_logits_all`` remains a TRANSIENT freed on
        return (the builder reduces it to [bs, k_draft, corr_topm] statics).

        Equivalence: the frontier build reproduces ``build_best_first_tree`` +
        the conditional scorer exactly up to real-valued score ties and batched-
        GEMM ULPs (frontier.py docstring; ``_frontier_equivalence_suite`` is the
        gate) — the verify path is untouched, so losslessness is unchanged by
        construction, and tau is preserved statistically rather than
        bit-identically.
        """
        builder = self._get_frontier_builder()
        if builder is None:
            return self._build_conditional_trees(draft_hidden, verified, bs, n, device)

        target_model = self.target_worker.model_runner.model
        draft_model = self.draft_model
        shift_label = bool(getattr(draft_model, "shift_label", False))

        if shift_label:
            ph_all = draft_hidden  # [bs, N, H]
        else:
            ph_all = draft_hidden[:, -(n - 1):, :]  # [bs, N-1, H]

        weight = target_model.lm_head.weight  # [V,H] full at TP=1; vocab shard at TP>1
        ph_cast = ph_all.to(weight.dtype) if ph_all.dtype != weight.dtype else ph_all
        # TP>1: all-gather the per-rank shard logits into a full-vocab,
        # rank-replicated tensor (identical on every rank). TP=1 is unchanged.
        base_logits_all = self._tp_full_base_logits(
            ph_cast, target_model.lm_head, self._target_vocab_size
        )  # transient [bs, k_draft, V]

        embed_tokens = target_model.get_input_embeddings()
        root_emb_all = embed_tokens(verified.view(bs, 1).to(torch.long))  # [bs,1,E]
        _, root_states = draft_model.prefix_gru(root_emb_all)  # (1, bs, gru_dim)

        try:
            return builder.build(ph_all, base_logits_all, root_states, verified)
        except TypeError as e:
            # Mixed server dtypes (see the GPU-expander guard above): fall back
            # to the per-request builder rather than crash mid-decode.
            logger.warning(
                "DOMINOTREE frontier builder dtype mismatch (%s); using the "
                "per-request best-first conditional builder.",
                e,
            )
            self._frontier_failed = True
            self._frontier_builder = None
            return self._build_conditional_trees(draft_hidden, verified, bs, n, device)

    # -- decode + tree verify ---------------------------------------------

    def _tree_decode_forward(self, model_worker_batch, on_publish):
        from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

        from sglang.srt.managers.scheduler import GenerationBatchResult
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
        from sglang.srt.speculative.eagle_info import EagleVerifyInput
        from sglang.srt.speculative.eagle_utils import (
            eagle_prepare_for_verify,
            eagle_sample,
        )
        from sglang.srt.speculative.spec_utils import (
            move_accept_tokens_to_target_kvcache,
        )

        from .tree.toy_tree import build_full_attention_mask

        device = self.device
        batch = model_worker_batch
        bs = len(batch.seq_lens)
        n = int(self.block_size)
        prefix_lens = batch.seq_lens
        draft_input = batch.spec_info
        verified = _draft_bonus_tokens(draft_input).view(-1)

        # 1) Domino block draft (raw per-position hidden).
        draft_hidden = self._domino_draft_block(batch)  # [bs, N, H]

        # 2) Assemble the tree: DEFAULT = per-request conditional best-first tree
        # (P3, the paper's method); DOMINOTREE_BUILDER=toy = P2 fixed caterpillar;
        # DOMINOTREE_BUILDER=frontier = Option B batched frontier (same
        # conditional scorer, zero host syncs; opt-in until GPU-validated).
        if self.tree_builder == "toy":
            draft_tokens_2d, intra_mask = self._toy_tree_tokens(
                draft_hidden, verified, bs, device
            )
        elif self.tree_builder == "frontier":
            draft_tokens_2d, intra_mask = self._build_frontier_trees(
                draft_hidden, verified, bs, n, device
            )
        else:
            draft_tokens_2d, intra_mask = self._build_conditional_trees(
                draft_hidden, verified, bs, n, device
            )

        # 3) Derive positions + retrieve links from the intra mask (NGRAM path).
        positions = torch.empty((bs * n,), dtype=torch.int64, device=device)
        retrieve_index = torch.empty((bs, n), dtype=torch.int64, device=device)
        retrieve_next_token = torch.empty((bs, n), dtype=torch.int64, device=device)
        retrieve_next_sibling = torch.empty((bs, n), dtype=torch.int64, device=device)
        reconstruct_indices_from_tree_mask(
            intra_mask.reshape(-1).contiguous(),
            prefix_lens,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            bs,
            n,
        )

        # 4) Full attention allow-mask for the target backend.
        seq_lens_cpu = (
            batch.seq_lens_cpu
            if batch.seq_lens_cpu is not None
            else prefix_lens.to(device="cpu", dtype=torch.int32)
        )
        custom_mask = build_full_attention_mask(
            intra_mask, seq_lens_cpu=seq_lens_cpu, device=device
        )

        # 5) EAGLE verify input (irregular tree -> tree_topk=-1). spec_steps=n-1
        # makes max_tree_depth=n so accept_index width == node budget.
        verify_input = EagleVerifyInput(
            draft_token=draft_tokens_2d.reshape(-1),
            custom_mask=custom_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=n - 1,
            topk=-1,
            draft_token_num=n,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=int(seq_lens_cpu.sum()),
            seq_lens_cpu=seq_lens_cpu,
        )
        batch.spec_info = verify_input

        # 6) Prepare + run target verify (EAGLE allocates the n verify slots).
        verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
            verify_input,
            self.model_runner.req_to_token_pool,
            batch,
            self.target_worker,
        )
        # NOTE: do NOT pass skip_attn_backend_init=True here. Unlike DFLASH's
        # DFlashVerifyInput.prepare_for_verify (which plans the target attention
        # backend itself), eagle_prepare_for_verify only plans it in the
        # cuda-graph path; with --disable-cuda-graph it *defers* planning to the
        # target forward_extend. Passing skip=True would mark the batch metadata
        # ready and skip that deferred planning, so the target verify would reuse
        # the previous forward's (prefill's) attention metadata -- exactly the
        # `q.shape[0] (N) != qo_indptr[-1]` crash. Omitting the flag (EAGLE's
        # behavior, eagle_worker_v2.py:1480-1484) lets forward_extend re-plan
        # the attention backend for the N tree nodes + custom tree mask.
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
        )
        logits_output = target_out.logits_output

        # 7) Tree acceptance. eagle_sample dispatches greedy (T=0) vs
        # threshold-sampled (T>0) internally on sampling_info; accept_lens
        # includes the bonus token.
        # NOTE: the 4th argument is passed POSITIONALLY on purpose. Upstream renamed
        # it (`vocab_mask: torch.Tensor` -> `grammar_mask: Optional[GrammarMask]`)
        # after 1adb53f14; both accept None, so a positional None is the one form
        # that binds on either version. Do NOT "clean this up" into a keyword -- that
        # would pin us to exactly one SGLang.
        predict, accept_lens, accept_index = eagle_sample(
            verify_input, batch, logits_output, None
        )
        new_seq_lens = prefix_lens + accept_lens
        if on_publish is not None:
            on_publish(new_seq_lens)

        # 8) Compact the accepted (possibly non-contiguous) tree path to the
        # front so target KV + hidden look like DFLASH's contiguous chain.
        allocator = self.target_worker.model_runner.token_to_kv_pool_allocator
        move_accept_tokens_to_target_kvcache(
            batch, accept_index, accept_lens - 1, allocator
        )
        predict = _compact_accept_to_front(predict, accept_index, bs, n)
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DOMINOTREE verify requires target hidden states.")
        hidden = _compact_accept_to_front(hidden, accept_index, bs, n)

        # 9) Commit the accepted prefix into the draft KV (DFLASH writer). Front
        # slots per req == req_to_token[req, L : L+n] after the KV move above.
        commit_lens = accept_lens.to(torch.int32)
        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        col_idx = prefix_lens.view(bs, 1) + torch.arange(
            n, device=device, dtype=torch.int64
        ).view(1, n)
        front_cache_loc_2d = req_to_token[
            batch.req_pool_indices.view(bs, 1).to(torch.int64), col_idx
        ]  # [bs, n]
        positions_commit_2d = prefix_lens.view(bs, 1).to(torch.int64) + torch.arange(
            n, device=device, dtype=torch.int64
        ).view(1, n)
        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=front_cache_loc_2d.reshape(-1),
            positions=positions_commit_2d.reshape(-1),
            cache_loc_2d=front_cache_loc_2d,
            commit_lens=commit_lens,
        )
        logits_output.hidden_states = None

        # 10) Bonus/next verified id = last accepted token per req.
        bonus = torch.gather(
            predict.view(bs, n), 1, (accept_lens - 1).view(bs, 1).to(torch.int64)
        ).view(-1)
        # UPSTREAM CONTRACT CHANGE (post-1adb53f14). DFlashDraftInputV2 was
        # restructured: `verified_id` -> `bonus_tokens`, and `verify_done` plus
        # `cur_allocated_seq_lens_cpu` were REMOVED outright (`verify_done` has zero
        # occurrences in current upstream -- the cross-iteration CUDA-event handshake
        # is gone). `_make_next_draft_input_decode` kept its name but lost two
        # keyword args, so call it with the new shape and fall back to the old one.
        try:
            next_draft_input = self._make_next_draft_input_decode(
                bonus_tokens=bonus, new_seq_lens=new_seq_lens
            )
        except TypeError:  # legacy SGLang <= 1adb53f14
            next_draft_input = self._make_next_draft_input_decode(
                verified_id=bonus,
                new_seq_lens=new_seq_lens,
                cur_allocated_seq_lens_cpu=draft_input.reserved_seq_lens_cpu,
            )
            verify_done = torch.get_device_module(device).Event()
            verify_done.record()
            next_draft_input.verify_done = verify_done

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            accept_lens=commit_lens,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=n,
            new_seq_lens=new_seq_lens,
        )
