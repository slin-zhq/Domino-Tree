"""Domino draft model: upstream ``DFlashDraftModel`` + the GRU correction head.

Upstream SGLang's ``DFlashDraftModel`` is the plain block-parallel DFlash
drafter and has NO Domino head. This subclass adds, when the draft checkpoint's
``projector_type`` selects Domino:

    - ``prefix_gru``  : nn.GRU(hidden_size -> gru_hidden_dim), bias=False
    - ``embed_proj``  : Linear(hidden+gru -> emb_dim) -> SiLU -> Linear(emb_dim -> vocab)

The attribute names match the checkpoint parameter names exactly (per the fork's
``models/dflash.py``), so the standard SGLang weight loader picks the head
weights up with no custom mapping. The transformer forward is unchanged — the
head is never invoked inside ``forward()``; it only adds parameters that the
worker's Domino rollout consumes after the draft-block forward.

The SGLang model registry keys by class ``__name__`` via ``EntryClass``. The
public Domino checkpoint's architecture is ``"DFlashDraftModel"``, so this
plugin does NOT rely on ``EntryClass`` auto-discovery — ``register_plugin()``
rebinds ``ModelRegistry.models["DFlashDraftModel"]`` to this class so the loader
selects it for the Domino draft config.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.models.dflash import DFlashDraftModel

from .config import is_dflash_domino_projector, parse_domino_draft_config

logger = logging.getLogger(__name__)


class DominoDraftModel(DFlashDraftModel):
    """DFlash draft model with the optional Domino GRU correction head."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)

        draft_config = parse_domino_draft_config(draft_hf_config=config)

        # Expose the Domino config on the model so the worker can detect the
        # head and drive the rollout (mirrors fork models/dflash.py:299-303).
        self.projector_type: Optional[str] = draft_config.projector_type
        self.pure_draft_prefix_len: int = int(draft_config.pure_draft_prefix_len)
        self.shift_label: bool = bool(draft_config.shift_label)
        self.gru_hidden_dim: Optional[int] = draft_config.gru_hidden_dim
        self.emb_dim: Optional[int] = draft_config.emb_dim

        if not is_dflash_domino_projector(self.projector_type):
            # Not a Domino checkpoint: behave exactly like plain DFlash.
            return

        hidden_size = int(config.hidden_size)
        vocab_size = int(getattr(config, "vocab_size", 0))
        if vocab_size <= 0:
            raise ValueError(
                f"DFLASH Domino requires positive vocab_size, got {vocab_size}."
            )
        if self.gru_hidden_dim is None or self.emb_dim is None:
            raise ValueError(
                "DFLASH Domino requires gru_hidden_dim and emb_dim. "
                f"gru_hidden_dim={self.gru_hidden_dim}, emb_dim={self.emb_dim}."
            )

        self.prefix_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=int(self.gru_hidden_dim),
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size + int(self.gru_hidden_dim), int(self.emb_dim), bias=False),
            nn.SiLU(),
            nn.Linear(int(self.emb_dim), vocab_size, bias=False),
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # The base loader keys off ``self.named_parameters()``, which now
        # includes ``prefix_gru.*`` and ``embed_proj.*`` (their checkpoint names
        # match), so the head weights load through the generic path. We only add
        # the cuDNN contiguity fixup the GRU needs afterwards.
        super().load_weights(weights)
        if getattr(self, "prefix_gru", None) is not None:
            self.prefix_gru.flatten_parameters()


EntryClass = DominoDraftModel
