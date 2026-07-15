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
from typing import Optional

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
