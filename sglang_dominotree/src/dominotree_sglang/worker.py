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

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

from .config import is_dflash_domino_projector
from .domino_helper import DFlashDominoHelper
from .domino_rollout import DFlashDominoRollout

logger = logging.getLogger(__name__)


def _tp_argmax_not_implemented(*_args, **_kwargs):
    raise NotImplementedError(
        "Domino TP>1 rollout requires the global-argmax reductions "
        "(_global_argmax_from_local_logits / _global_argmax_from_local_max) that "
        "the fork's DFlashWorker exposed. They are unported for Phase 1, which "
        "targets single-GPU (TP=1) draft. Run with tensor-parallel size 1."
    )


class DominoWorkerV2(DFlashWorkerV2):
    """DFLASH v2 worker with the Domino GRU-corrected chain rollout."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.domino_helper: Optional[DFlashDominoHelper] = None
        self.domino_rollout: Optional[DFlashDominoRollout] = None

        projector_type = getattr(self.draft_model, "projector_type", None)
        if not is_dflash_domino_projector(projector_type):
            logger.warning(
                "DominoWorkerV2 loaded a draft model without a Domino projector "
                "(projector_type=%r); falling back to plain DFLASH chain draft.",
                projector_type,
            )
            return

        self.domino_helper = DFlashDominoHelper(self.draft_model)

        target_model_config = self.target_worker.model_runner.model_config
        target_vocab_size = int(getattr(target_model_config, "vocab_size", 0) or 0)

        self.domino_rollout = DFlashDominoRollout(
            domino_helper=self.domino_helper,
            block_size=int(self.block_size),
            target_vocab_size=target_vocab_size,
            # TP>1 reduction callbacks are only invoked by the rollout's
            # TP-eager path (tp_size != 1); TP=1 never calls them.
            global_argmax_from_local_logits=_tp_argmax_not_implemented,
            global_argmax_from_local_max=_tp_argmax_not_implemented,
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
            draft_hidden = captured.get("draft_hidden")
            if draft_hidden is None or verified_id is None:
                # Defensive: no captured hidden / verified id -> plain greedy.
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

    Phase 2 plumbing proof. The Domino block-parallel DRAFT is unchanged; only
    the decode-time verify is swapped from DFLASH's linear chain to a small fixed
    tree run through EAGLE's tree verifier (``eagle_prepare_for_verify`` ->
    target verify -> ``eagle_sample``), with the accepted path compacted to a
    prefix (EAGLE's ``move_accept_tokens_to_target_kvcache`` +
    ``_compact_accept_to_front``) so DFLASH's prefix-only draft-KV writer can be
    reused verbatim.

    Constraints (Phase 2): T=0 greedy only, TP=1 only, page_size==1, non-mamba
    target, no compact-draft-cache window. Anything else falls back to the
    lossless Domino chain (``DominoWorkerV2``). See PORT_NOTES.md.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Verify node budget N == DFLASH block_size, so ALL DFLASH KV/buffer
        # sizing (reserved per-decode tokens, cuda-graph widths) is reused with
        # no server-arg override. The tree is: root + (N-1-B) spine (Domino
        # chain) + B branch siblings at the shallowest depths.
        self._warned_tree_greedy_only = False
        self.tree_num_nodes = int(self.block_size)
        try:
            self.tree_num_branch = int(os.environ.get("DOMINOTREE_NUM_BRANCH", "2"))
        except ValueError:
            self.tree_num_branch = 2

        self.tree_topology = None
        if self.domino_rollout is not None:
            from .tree.toy_tree import build_topology

            # Clamp branches so the spine keeps at least ~half the block: the
            # spine must stay well above the measured chain acceptance for the
            # "tree accept >= chain accept" property to hold in practice.
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
                "DominoTreeWorkerV2 ready: N=%d nodes, spine_len=%d, num_branch=%d, "
                "max_depth=%d.",
                self.tree_topology.num_nodes,
                self.tree_topology.spine_len,
                self.tree_topology.num_branch,
                self.tree_topology.max_depth,
            )

    def forward_batch_generation(self, model_worker_batch, on_publish=None):
        # Only the DECODE-verify stage differs from DFLASH/Domino. Route prefill/
        # extend/idle and any non-tree case back to the Domino chain worker
        # (which is itself lossless), then handle the greedy decode-verify here.
        if self.tree_topology is None:
            return super().forward_batch_generation(model_worker_batch, on_publish)

        mode = model_worker_batch.forward_mode
        if (
            mode.is_extend()
            or getattr(model_worker_batch, "is_extend_in_batch", False)
            or mode.is_idle()
            or model_worker_batch.spec_info is None
        ):
            return super().forward_batch_generation(model_worker_batch, on_publish)

        sampling_info = getattr(model_worker_batch, "sampling_info", None)
        if sampling_info is None or not sampling_info.is_all_greedy:
            # Phase 2 tree verify is greedy-only; fall back to lossless Domino chain.
            if not self._warned_tree_greedy_only:
                logger.warning(
                    "DOMINOTREE tree verify supports T=0 greedy only; falling back "
                    "to the Domino chain for non-greedy sampling."
                )
                self._warned_tree_greedy_only = True
            return super().forward_batch_generation(model_worker_batch, on_publish)

        return self._tree_decode_forward(model_worker_batch, on_publish)

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
        from sglang.srt.speculative.eagle_info_v2 import assign_extend_cache_locs_func
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        device = self.device
        bs = len(model_worker_batch.seq_lens)
        n = int(self.block_size)
        draft_input = model_worker_batch.spec_info
        prefix_lens = model_worker_batch.seq_lens
        verified = draft_input.verified_id.view(-1)

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
        draft_hidden = draft_hidden.reshape(bs, n, -1)

        # Spine tokens: the Domino GRU-corrected chain (same as Phase 1).
        lm_head = getattr(target_model, "lm_head", None)
        spine_tokens = self.domino_rollout.rollout_draft_block(
            draft_hidden=draft_hidden,
            verified_id=verified,
            target_model=target_model,
            lm_head=lm_head,
        )  # [bs, n-1]
        return spine_tokens, draft_hidden

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

        from .tree.toy_tree import (
            build_draft_tokens,
            build_full_attention_mask,
            build_intra_tree_mask,
        )

        device = self.device
        batch = model_worker_batch
        bs = len(batch.seq_lens)
        n = int(self.block_size)
        prefix_lens = batch.seq_lens
        draft_input = batch.spec_info
        verified = draft_input.verified_id.view(-1)
        topo = self.tree_topology

        # 1) Domino block draft -> spine chain + branch candidates.
        spine_tokens, draft_hidden = self._domino_draft_block(batch)
        branch_tokens = self._branch_candidates(draft_hidden, spine_tokens)
        if branch_tokens is None:
            branch_tokens = torch.empty((bs, 0), dtype=torch.int64, device=device)

        # 2) Assemble tree tokens + intra-tree ancestor mask.
        draft_tokens_2d = build_draft_tokens(
            topo,
            verified_id=verified,
            spine_tokens=spine_tokens,
            branch_tokens=branch_tokens,
        )  # [bs, n]
        intra_mask = build_intra_tree_mask(topo, bs=bs, device=device)  # [bs,n,n]

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

        # 7) Tree acceptance (greedy). accept_lens includes the bonus token.
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
