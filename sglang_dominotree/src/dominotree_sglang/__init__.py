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


def _install_dflash_custom_mask_graph_hook() -> None:
    """Let DOMINOTREE's tree verify replay the decode CUDA graph (drop the old
    ``--disable-cuda-graph`` requirement), subprocess-robust and DOMINOTREE-scoped.

    **Root cause of the old requirement.** The decode CUDA-graph runner captures
    the target-verify graph via ``get_spec_info``. For a DFLASH-gated algorithm it
    calls ``resolve_dflash_verify_mask_policy(attn_backend)``, which returns
    ``build_custom_mask=False`` for backends in
    ``_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS`` (flashinfer/fa3/triton/...) —
    correct for DFLASH's LINEAR verify (causal == no custom mask), so the captured
    wrapper has NO ``custom_mask_buf``. Our tree verify emits an ``EagleVerifyInput``
    WITH a custom tree mask, so at replay flashinfer errors
    ``custom_mask_buf must be initialized ... in cuda graph mode``. NGRAM (also an
    irregular tree) avoids this because its ``get_spec_info`` uses
    ``custom_mask=buffers.custom_mask``.

    **Why a wrapper here (not handle_server_args).** The graph capture runs in the
    scheduler SUBPROCESS. ``handle_server_args`` runs only in the MAIN process
    (``run_scheduler_process`` receives a pre-constructed ``server_args``;
    ``__post_init__`` / ``handle_speculative_decoding`` do not re-run), and on spawn
    the subprocess re-imports ``dflash_utils`` fresh, losing a main-process patch.
    ``register_plugin`` (this hook's caller) DOES run in the subprocess via
    ``load_plugins()`` (scheduler.py:4181), before ``Scheduler(...)`` / graph
    capture. So we install an algorithm-aware wrapper on
    ``resolve_dflash_verify_mask_policy`` here.

    **Mechanism.** On the first call while the server's ``speculative_algorithm``
    is ``DOMINOTREE`` (i.e. during graph capture, after the global server args are
    set), the wrapper empties ``_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS``. Because
    that frozenset is a module global read at CALL time by
    ``resolve_dflash_verify_mask_policy``, ALL callers become consistent — the
    graph capture (``get_spec_info``'s local import -> our wrapper) AND the
    chain-fallback runtime verify (which may hold the original function reference)
    both then see ``build_custom_mask=True``. So the graph is captured with a
    ``custom_mask_buf`` and the chain fallback builds a matching causal custom mask.

    **Scope / safety.** No effect on a DOMINO (chain) server — the wrapper checks
    the algorithm and never empties the set there. No-op under
    ``--disable-cuda-graph`` (no graph is captured). Idempotent (marks the wrapper;
    only empties once). The only side effect on a DOMINOTREE server is the T>0
    chain fallback building a causal custom mask instead of the built-in causal
    path — correct, marginally slower, rarely hit.
    """
    try:
        import sglang.srt.speculative.dflash_utils as _du
        from sglang.srt.server_args import get_global_server_args
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("DOMINOTREE: custom-mask graph hook not installed (%s).", e)
        return

    current = getattr(_du, "resolve_dflash_verify_mask_policy", None)
    if current is None or getattr(current, "_dominotree_wrapped", False):
        return
    _orig = current

    def _wrapper(attn_backend):
        try:
            algo = get_global_server_args().speculative_algorithm
        except Exception:
            algo = None
        if algo == "DOMINOTREE" and _du._DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS:
            logger.info(
                "DOMINOTREE: enabling custom-mask DFLASH verify CUDA graph "
                "(emptying _DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS=%s); "
                "--disable-cuda-graph no longer required.",
                sorted(_du._DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS),
            )
            _du._DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS = frozenset()
        return _orig(attn_backend)

    _wrapper._dominotree_wrapped = True
    _du.resolve_dflash_verify_mask_policy = _wrapper


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

    # (3) Register DOMINOTREE (adaptive/toy tree verify). Same DFLASH-draft
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

    # (4) Enable the custom-mask decode CUDA graph for DOMINOTREE (subprocess-
    # robust, algorithm-scoped) so its tree verify replays a graph instead of
    # requiring --disable-cuda-graph. No-op for a DOMINO server / under
    # --disable-cuda-graph. See the hook's docstring for the full rationale.
    _install_dflash_custom_mask_graph_hook()

    _registered = True
    logger.info(
        "Registered DOMINO (chain) and DOMINOTREE (tree verify) "
        "speculative algorithms."
    )
