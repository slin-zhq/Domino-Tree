"""DominoTree SGLang plugin.

Registers two out-of-tree speculative algorithms on upstream SGLang, reusing all
of upstream's DFLASH plumbing:

  ``DOMINO``      the Domino block-parallel drafter (GRU correction head) with
                  CHAIN verify;
  ``DOMINOTREE``  the conditional best-first draft tree, verified with SGLang's
                  EAGLE tree-verify.

Entry point (declared in pyproject.toml)::

    [project.entry-points."sglang.srt.plugins"]
    dominotree = "dominotree_sglang:register_plugin"

Load it at launch with ``SGLANG_PLUGINS=dominotree`` and select an algorithm with
``--speculative-algorithm DOMINOTREE`` (or ``DOMINO``). See README.md for the
full launch command.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_registered = False


def _build_spec_class(algo_name: str = "DOMINO"):
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

            from .worker import assert_domino_server_args_supported

            _handle_dflash(server_args)

            # P5 fail-fast guards (main process, before any model load):
            # DOMINOTREE-specific configs (page_size!=1, compact draft cache,
            # rejection sampling, non-default accept thresholds, custom logit
            # processor). Tensor parallelism is supported at any size. Re-checked
            # in the worker __init__ (scheduler subprocess). Mamba/hybrid targets
            # are guarded in the worker __init__ only (needs the live backend).
            assert_domino_server_args_supported(server_args, algo_name)

            # A tree budget counts drafted non-root nodes. The target verifies
            # the root plus that budget, while the draft block still bounds depth.
            if algo_name == "DOMINOTREE":
                raw_budget = os.environ.get("DOMINOTREE_TREE_BUDGET")
                if raw_budget is not None:
                    try:
                        budget = int(raw_budget)
                    except ValueError as exc:
                        raise ValueError(
                            "DOMINOTREE_TREE_BUDGET must be an integer, "
                            f"got {raw_budget!r}."
                        ) from exc
                    if budget < 1:
                        raise ValueError(
                            "DOMINOTREE_TREE_BUDGET must be >= 1, "
                            f"got {budget}."
                        )
                    draft_block_size = int(server_args.speculative_num_draft_tokens)
                    server_args.speculative_num_draft_tokens = budget + 1
                    logger.info(
                        "DOMINOTREE tree budget=%d drafted nodes; target verify "
                        "slots=%d (root included); draft block_size=%d (depth).",
                        budget,
                        budget + 1,
                        draft_block_size,
                    )

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
    set), the wrapper empties ``_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS``, so the
    target-verify graph is captured WITH ``custom_mask=buffers.custom_mask`` and
    the flashinfer wrapper gets a static ``custom_mask_buf``. At replay, the tree
    verify's ``EagleVerifyInput`` mask is copied into that buffer each step.

    **P5 correction (chain-fallback interaction).** The only runtime consumer of
    this policy is the graph CAPTURE (decode_cuda_graph_runner.py:1063-1073).
    Upstream's chain verify does NOT consult it — it hardcodes
    ``custom_mask=None`` (dflash_worker_v2.py:1504) — so a chain-verify graph
    replay on a DOMINOTREE server would rerun the captured custom-mask kernel
    against a STALE mask buffer (wrong commits). ``DominoTreeWorkerV2._chain_fallback``
    therefore forces every chain-fallback verify EAGER; this hook alone is NOT
    sufficient for correctness of the fallback.

    **Scope / safety.** No effect on a DOMINO (chain) server — the wrapper checks
    the algorithm and never empties the set there. No-op under
    ``--disable-cuda-graph`` (no graph is captured). Idempotent (marks the wrapper;
    only empties once).
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


# Set by _install_hybrid_tree_verify_hooks; read by the worker's hybrid guard.
_HYBRID_TREE_HOOKS_INSTALLED = False
# DIAGNOSTIC ONLY (never for published numbers): DOMINOTREE_DIAG_FLAT_GDN=1 skips
# the hooks so the GDN layers verify the tree as a flat chain, i.e. the P9 bug.
# It is the negative control that shows the losslessness gate can detect it.
_HYBRID_DIAG_FLAT_GDN = os.environ.get("DOMINOTREE_DIAG_FLAT_GDN", "0") == "1"


def _is_dominotree_server() -> bool:
    try:
        from sglang.srt.server_args import get_global_server_args

        return get_global_server_args().speculative_algorithm == "DOMINOTREE"
    except Exception:
        return False


def _install_hybrid_tree_verify_hooks() -> None:
    """Make hybrid Gated-DeltaNet targets verify our tree as a TREE (P9).

    **Why.** Three quarters of Qwen3.6-27B's layers are recurrent (GDN). A
    recurrent layer has no attention mask, so it only respects tree topology if
    the engine walks parent pointers. Upstream switches that on from
    ``speculative_eagle_topk > 1``, a startup-time fanout integer that DFLASH
    forces to 1 and that no single value of which describes our irregular tree.
    Left at 1, the GDN kernels process the n flattened nodes as one flat chain:
    each node absorbs the state of whatever precedes it, INCLUDING rejected
    siblings, so the verify logits and the committed state are both wrong and
    nothing fails (``p9_hybrid_gdn_design.md``, "decisive finding").

    **What (the two startup-time mechanisms that key on topk).**

    1. ``MambaAttnBackendBase.topk`` gates reading ``retrieve_next_token`` /
       ``retrieve_next_sibling`` into the forward metadata, eager and CUDA-graph
       (``hybrid_linear_attn_backend.py:125,423,489``). We raise it on the TARGET
       GDN backend only. It is read nowhere else in that class.
    2. ``conv_window_dedup_enabled`` picks the conv-intermediate layout when the
       memory pool is ALLOCATED. The deduplicated sliding-window layout is only
       correct for a linear chain (its own docstring); we force the dense layout.
       ``fused_conv_window_scatter_with_mask`` reads either layout through
       strides, so the post-verify commit needs no change.

    The third mechanism (verify-path metadata propagation, ``gdn_backend.py``)
    consumes the forward metadata from (1). The state commit itself is called
    from the worker (``commit_mamba_states_after_verify``).

    The GDN kernels walk nodes in flattened order and load each node's PARENT
    state, so a parent must precede its children. Upstream's own
    ``reconstruct_indices_from_tree_mask`` (which derives our retrieve links)
    already requires that order, and both of our builders emit it.

    **Why here, not handle_server_args, and why not raise topk globally.**
    ``speculative_eagle_topk`` is read in 36 files (CUDA-graph sizing, KV
    reservation, backend predicates) along the DFLASH-derived path our worker
    reuses. Patching the two consumers that matter keeps everything else on the
    exact configuration the published numbers were measured on. The pool and
    backend are built in the scheduler SUBPROCESS, where ``register_plugin``
    runs before model load (see ``_install_dflash_custom_mask_graph_hook``), and
    the global server args are set before the memory pool is allocated
    (``model_runner.py:525`` vs ``:812``).

    **Scope.** Active only while the server's algorithm is ``DOMINOTREE``, and it
    only reaches objects that exist for hybrid-GDN targets (``MambaPool``,
    ``GDNAttnBackend``). Conventional attention targets such as Qwen3-4B/8B never
    construct either, so they are untouched by construction; a DOMINO/DFLASH/
    EAGLE server is untouched by the algorithm check.
    """
    global _HYBRID_TREE_HOOKS_INSTALLED
    if _HYBRID_DIAG_FLAT_GDN:
        logger.warning(
            "DOMINOTREE_DIAG_FLAT_GDN=1: hybrid tree-verify hooks NOT installed. "
            "GDN layers will verify trees as a flat chain. DIAGNOSTIC ONLY -- "
            "output is NOT lossless; never publish numbers from this server."
        )
        return
    try:
        import sglang.srt.mem_cache.memory_pool as _mp
        from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    except Exception as e:  # pragma: no cover - defensive
        # The worker refuses hybrid targets while the flag stays False.
        logger.warning("DOMINOTREE: hybrid tree-verify hooks not installed (%s).", e)
        return

    orig_dedup = getattr(_mp, "conv_window_dedup_enabled", None)
    if orig_dedup is None:
        logger.warning(
            "DOMINOTREE: memory_pool.conv_window_dedup_enabled not found; "
            "hybrid tree-verify hooks not installed."
        )
        return
    if not getattr(orig_dedup, "_dominotree_wrapped", False):

        def _dedup(is_npu, is_cpu, speculative_eagle_topk):
            if _is_dominotree_server():
                return False  # dense per-node conv windows: tree-safe
            return orig_dedup(is_npu, is_cpu, speculative_eagle_topk)

        _dedup._dominotree_wrapped = True
        _mp.conv_window_dedup_enabled = _dedup

    orig_init = GDNAttnBackend.__init__
    if not getattr(orig_init, "_dominotree_wrapped", False):

        def _gdn_init(self, model_runner, *args, **kwargs):
            orig_init(self, model_runner, *args, **kwargs)
            if _is_dominotree_server() and not self.is_draft_worker:
                self.topk = max(int(self.topk or 0), 2)
                self._dominotree_tree_verify = True
                logger.info(
                    "DOMINOTREE: GDN target backend reads tree topology in verify "
                    "(topk gate raised to %d on this backend only).",
                    self.topk,
                )

        _gdn_init._dominotree_wrapped = True
        GDNAttnBackend.__init__ = _gdn_init

    _HYBRID_TREE_HOOKS_INSTALLED = True


def register_plugin() -> None:
    """Plugin entry point (invoked for side effects by SGLang's plugin loader).

    1. Rebind the ``DFlashDraftModel`` architecture in the model registry to our
       ``DominoDraftModel`` (the Domino checkpoint's architecture string is
       ``"DFlashDraftModel"``, and upstream's class has no GRU head).
    2. Register the ``DOMINO`` (chain) and ``DOMINOTREE`` (adaptive tree verify)
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
    spec_class = _build_spec_class("DOMINO")

    @SpeculativeAlgorithm.register(
        "DOMINO", supports_overlap=False, spec_class=spec_class
    )
    def _domino_worker_factory(server_args):
        return DominoWorkerV2

    # (3) Register DOMINOTREE (adaptive tree verify). Same DFLASH-draft
    # gating as DOMINO (is_dflash()=True keeps the DFLASH draft plumbing); the
    # tree verify is driven entirely inside DominoTreeWorkerV2, not gated by the
    # scheduler. A distinct spec_class instance is required (register stores one
    # instance per name).
    tree_spec_class = _build_spec_class("DOMINOTREE")

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

    # (5) Hybrid Gated-DeltaNet targets (Qwen3.5/3.6): make the GDN layers verify
    # the tree as a tree. DOMINOTREE-scoped; inert for attention-only targets.
    _install_hybrid_tree_verify_hooks()

    _registered = True
    logger.info(
        "Registered DOMINO (chain) and DOMINOTREE (tree verify) "
        "speculative algorithms."
    )
