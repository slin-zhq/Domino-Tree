"""Domino draft-config parsing for the out-of-tree SGLang plugin.

Upstream SGLang's ``DFlashDraftConfig`` / ``parse_dflash_draft_config`` only
know the base DFLASH fields (layers, block_size, target_layer_ids, mask token).
The Domino correction head adds five more fields that live in the draft
checkpoint's ``dflash_config`` sub-dict (with ``emb_dim`` also accepted at the
HF-config top level, matching the public ``Huang2020/*-Domino-b16``
checkpoints):

    projector_type        "domino" (public) / "causal_v5" (legacy alias)
    gru_hidden_dim        GRU state width G
    emb_dim               embed_proj bottleneck width (fc1 out / fc2 in)
    pure_draft_prefix_len must be 1 (slot 0 of the draft block == verified token)
    shift_label           label-alignment flag consumed by the rollout

This module REUSES the upstream base parser for the shared fields and only
extends it for the Domino-specific ones, so it stays in lock-step with upstream
DFLASH config semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sglang.srt.speculative.dflash_utils import (
    DFlashDraftConfig,
    parse_dflash_draft_config,
)

# Projectors the Domino inference path understands. Public checkpoints use
# "domino"; "causal_v5" is a backward-compatible alias for older internal
# checkpoints (mirrors the fork's _DFLASH_DOMINO_PROJECTORS).
DOMINO_PROJECTORS = frozenset({"domino", "causal_v5"})


def is_dflash_domino_projector(projector_type: Optional[str]) -> bool:
    """True if ``projector_type`` selects the Domino GRU-correction head.

    Re-exported so the copied ``domino_helper`` / ``domino_rollout`` modules can
    import it from this package instead of the (fork-only) upstream symbol.
    """
    return projector_type in DOMINO_PROJECTORS


def _get_dflash_config_dict(config: Any) -> dict:
    """Return the ``dflash_config`` sub-dict from an HF config or plain dict.

    Mirrors upstream ``dflash_utils._get_dflash_config`` but kept local so the
    plugin does not depend on an underscore-private upstream helper.
    """
    if isinstance(config, dict):
        cfg = config.get("dflash_config", None)
    else:
        cfg = getattr(config, "dflash_config", None)
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    try:
        return dict(cfg)
    except Exception:
        return {}


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _parse_optional_int(
    value: Any, *, field_name: str, min_value: Optional[int] = None
) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception as e:
        raise ValueError(f"Invalid {field_name}={value!r}.") from e
    if min_value is not None and parsed < int(min_value):
        comparator = "positive" if int(min_value) == 1 else f">= {int(min_value)}"
        raise ValueError(f"{field_name} must be {comparator}, got {parsed}.")
    return parsed


@dataclass(frozen=True)
class DominoDraftConfig:
    """Base DFLASH config plus the Domino correction-head fields."""

    base: DFlashDraftConfig
    projector_type: Optional[str]
    gru_hidden_dim: Optional[int]
    emb_dim: Optional[int]
    pure_draft_prefix_len: int
    shift_label: bool

    @property
    def is_domino(self) -> bool:
        return is_dflash_domino_projector(self.projector_type)

    # Convenience pass-throughs to the base config used by the draft model.
    def resolve_block_size(self, *, default: Optional[int] = None) -> Optional[int]:
        return self.base.resolve_block_size(default=default)

    @property
    def block_size(self) -> Optional[int]:
        return self.base.block_size

    @property
    def mask_token(self) -> str:
        return self.base.mask_token

    @property
    def mask_token_id(self) -> Optional[int]:
        return self.base.mask_token_id


def parse_domino_draft_config(*, draft_hf_config: Any) -> DominoDraftConfig:
    """Parse base DFLASH fields (upstream) + Domino head fields (here)."""
    base = parse_dflash_draft_config(draft_hf_config=draft_hf_config)

    dflash_cfg = _get_dflash_config_dict(draft_hf_config)

    projector_type = dflash_cfg.get("projector_type", None)
    if projector_type is not None and (
        not isinstance(projector_type, str) or not projector_type
    ):
        raise ValueError(
            "DFLASH dflash_config.projector_type must be a non-empty string, "
            f"got {projector_type!r}."
        )

    gru_hidden_dim = _parse_optional_int(
        dflash_cfg.get("gru_hidden_dim", None),
        field_name="DFLASH dflash_config.gru_hidden_dim",
        min_value=1,
    )

    # emb_dim may be at the HF-config top level or inside dflash_config.
    emb_dim_raw = _cfg_get(draft_hf_config, "emb_dim", None)
    if emb_dim_raw is None:
        emb_dim_raw = dflash_cfg.get("emb_dim", None)
    emb_dim = _parse_optional_int(
        emb_dim_raw, field_name="DFLASH emb_dim", min_value=1
    )

    pure_draft_prefix_len_raw = dflash_cfg.get("pure_draft_prefix_len", 0)
    parsed_prefix_len = _parse_optional_int(
        pure_draft_prefix_len_raw,
        field_name="DFLASH dflash_config.pure_draft_prefix_len",
        min_value=0,
    )
    pure_draft_prefix_len = 0 if parsed_prefix_len is None else int(parsed_prefix_len)

    shift_label_raw = dflash_cfg.get("shift_label", False)
    if not isinstance(shift_label_raw, bool):
        raise ValueError(
            "DFLASH dflash_config.shift_label must be a bool, "
            f"got {shift_label_raw!r} (type={type(shift_label_raw).__name__})."
        )
    shift_label = bool(shift_label_raw)

    if is_dflash_domino_projector(projector_type):
        if gru_hidden_dim is None:
            raise ValueError(
                "DFLASH Domino requires dflash_config.gru_hidden_dim to be set."
            )
        if emb_dim is None:
            raise ValueError(
                "DFLASH Domino requires config.emb_dim or "
                "dflash_config.emb_dim to be set."
            )
        # The rollout implementation (domino_rollout.py) assumes exactly one
        # pure-draft prefix token: slot 0 of the draft block is the verified
        # token, slots 1..block_size-1 are drafted.
        if pure_draft_prefix_len != 1:
            raise NotImplementedError(
                "DFLASH Domino currently requires dflash_config.pure_draft_prefix_len=1, "
                f"got {pure_draft_prefix_len}."
            )

    return DominoDraftConfig(
        base=base,
        projector_type=projector_type,
        gru_hidden_dim=gru_hidden_dim,
        emb_dim=emb_dim,
        pure_draft_prefix_len=pure_draft_prefix_len,
        shift_label=shift_label,
    )
