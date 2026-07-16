"""DominoTree SGLang plugin.

Phase 1: register the Domino block-parallel drafter (GRU correction head, CHAIN
verify) as an out-of-tree speculative algorithm named ``DOMINO`` on latest
upstream SGLang, reusing all of upstream's DFLASH plumbing.

Entry point (declared in pyproject.toml)::

    [project.entry-points."sglang.srt.plugins"]
    dominotree = "dominotree_sglang:register_plugin"

Load it at launch with ``SGLANG_PLUGINS=dominotree`` and select it with
``--speculative-algorithm DOMINO``. See PORT_NOTES.md for the exact command.

``DOMINOTREE`` (the conditional draft tree) is Phase 3 and is intentionally NOT
registered here.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_registered = False


def _build_spec_class():
    """Build a ``CustomSpecAlgo`` subclass that makes ``DOMINO`` behave like the
    builtin ``DFLASH`` everywhere except worker creation.

    Upstream gates generic scheduler / model-runner / server-arg logic on
    ``spec_algorithm.is_dflash()`` and on the algo's ``handle_server_args`` /
    ``create_future_map`` / ``carries_draft_hidden_states`` / ``need_topk``
    methods. A bare ``CustomSpecAlgo`` returns the wrong answers (or lacks the
    method), so we mirror the ``SpeculativeAlgorithm.DFLASH`` enum member for
    each. Worker creation still routes through our factory
    (``CustomSpecAlgo.create_worker`` -> ``self.factory``), so the Domino worker
    is used while all DFLASH scaffolding fires unchanged. Built lazily so the
    base class import happens at plugin-load time, not module import.
    """
    from sglang.srt.speculative.spec_registry import CustomSpecAlgo

    class DominoSpecAlgo(CustomSpecAlgo):
        # --- Make DOMINO indistinguishable from DFLASH for builtin branches. ---
        def is_dflash(self) -> bool:
            return True

        def supports_target_verify_for_draft(self) -> bool:
            # DFLASH enum returns is_dflash() -> True.
            return True

        def carries_draft_hidden_states(self) -> bool:
            # DFLASH enum returns is_eagle() -> False. Present because the
            # scheduler calls it (scheduler.py:1121/1126/1174/1179) and the base
            # CustomSpecAlgo does not define it.
            return False

        def need_topk(self) -> bool:
            # DFLASH: is_eagle() or is_standalone() -> False. Called via
            # FutureMap (overlap_utils.py:152).
            return False

        def create_future_map(
            self, device, req_to_token_pool, needs_cpu_seq_lens: bool = True
        ):
            # Mirror the enum's implementation (scheduler.py:1236 calls this
            # unconditionally). Base CustomSpecAlgo does not define it.
            from sglang.srt.managers.overlap_utils import FutureMap

            return FutureMap(device, self, req_to_token_pool, needs_cpu_seq_lens)

        def handle_server_args(self, server_args) -> None:
            # Reuse ALL of DFLASH's server-arg normalization (num_steps=1,
            # topk=1, block_size inference from the draft config, mixed-chunk
            # disable, max_running_requests default, ...).
            from sglang.srt.arg_groups.speculative_hook import _handle_dflash

            _handle_dflash(server_args)

            # Phase 1 runs synchronously (supports_overlap=False). The Domino
            # rollout captures its own CUDA graph and we have not validated it
            # under overlap's multi-stream scheduling, so force overlap off here
            # (a supported DFLASH v2 mode) instead of erroring in create_worker.
            if not server_args.disable_overlap_schedule:
                logger.warning(
                    "DOMINO (Phase 1) forces --disable-overlap-schedule "
                    "(synchronous). Overlap scheduling is not validated for the "
                    "Domino rollout yet."
                )
                server_args.disable_overlap_schedule = True

    return DominoSpecAlgo


def register_plugin() -> None:
    """Plugin entry point (invoked for side effects by SGLang's plugin loader).

    1. Rebind the ``DFlashDraftModel`` architecture in the model registry to our
       ``DominoDraftModel`` (the Domino checkpoint's architecture string is
       ``"DFlashDraftModel"``, and upstream's class has no GRU head).
    2. Register the ``DOMINO`` (chain) and ``DOMINOTREE`` (toy tree verify)
       speculative algorithms.
    """
    global _registered
    if _registered:
        return

    from sglang.srt.models.registry import ModelRegistry
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    from .draft_model import DominoDraftModel
    from .worker import DominoTreeWorkerV2, DominoWorkerV2

    # (1) Substitute our subclass for the DFlash draft architecture so the model
    # loader instantiates the head-carrying model for Domino checkpoints.
    ModelRegistry.models["DFlashDraftModel"] = DominoDraftModel
    # Also expose under our own name in case a checkpoint declares it.
    ModelRegistry.models.setdefault("DominoDraftModel", DominoDraftModel)

    # (2) Register the DOMINO (chain) algorithm. supports_overlap=False for
    # Phase 1 (synchronous); handle_server_args forces overlap off so
    # create_worker does not reject the run.
    spec_class = _build_spec_class()

    @SpeculativeAlgorithm.register(
        "DOMINO", supports_overlap=False, spec_class=spec_class
    )
    def _domino_worker_factory(server_args):
        return DominoWorkerV2

    # (3) Register DOMINOTREE (Phase 2 toy tree verify). Same DFLASH-draft
    # gating as DOMINO (is_dflash()=True keeps the DFLASH draft plumbing); the
    # tree verify is driven entirely inside DominoTreeWorkerV2, not gated by the
    # scheduler. A distinct spec_class instance is required (register stores one
    # instance per name).
    tree_spec_class = _build_spec_class()

    @SpeculativeAlgorithm.register(
        "DOMINOTREE", supports_overlap=False, spec_class=tree_spec_class
    )
    def _dominotree_worker_factory(server_args):
        return DominoTreeWorkerV2

    _registered = True
    logger.info(
        "Registered DOMINO (chain) and DOMINOTREE (toy tree verify) "
        "speculative algorithms."
    )
