# DominoTree SGLang plugin — Port Notes (Phase 1: chain draft)

This plugin makes the **Domino** block-parallel drafter (GRU correction head,
**linear/chain** verify) run as an out-of-tree speculative algorithm named
**`DOMINO`** on latest upstream SGLang, without patching the SGLang source tree.

- **Runtime target:** upstream SGLang at
  `SpecDec-Optimize/ref_repo/sglang` (imported at runtime; NOT vendored).
- **Provenance:** the correction-head wiring was reverse-engineered from the
  fork branch `sglang-feat/dflash-domino` (commit `e0d7870`, base ≈ upstream
  early-May 2026, pre-DFLASH-v2). See
  `docs/sglang_integration/domino_fork_head_wiring.md`.
- **Scope:** `DOMINO` only. `DOMINOTREE` (the conditional draft tree) is Phase 3
  and is deliberately not registered.

## Package contents

| File | Role |
|---|---|
| `pyproject.toml` | `dominotree-sglang` dist, src-layout, entry point `dominotree = "dominotree_sglang:register_plugin"` under group `sglang.srt.plugins`. |
| `src/dominotree_sglang/__init__.py` | `register_plugin()` — registry rebind + `SpeculativeAlgorithm.register("DOMINO", ...)`. |
| `src/dominotree_sglang/config.py` | `parse_domino_draft_config` (reuses upstream base parser) + `is_dflash_domino_projector`. |
| `src/dominotree_sglang/draft_model.py` | `DominoDraftModel(DFlashDraftModel)` — adds `prefix_gru` + `embed_proj`. |
| `src/dominotree_sglang/worker.py` | `DominoWorkerV2(DFlashWorkerV2)` — swaps greedy sample → Domino rollout. |
| `src/dominotree_sglang/domino_{helper,rollout,kernels}.py` | copied from the fork (see shim note below). |

## (a) Exact upstream methods/symbols overridden

All upstream file:line refs are in `ref_repo/sglang/python/sglang/srt/`.

1. **`DFlashWorkerV2.forward_batch_generation`** (`speculative/dflash_worker_v2.py:1200-1692`)
   — **wrapped, not copied.** `DominoWorkerV2.forward_batch_generation` installs
   two temporary rebinds around `super().forward_batch_generation(...)`:
   - captures the full draft-block hidden by wrapping
     `self.draft_model_runner.forward` (upstream draft-block forward call at
     `dflash_worker_v2.py:1484-1487`);
   - rebinds `self._greedy_sample_from_vocab_parallel_head`
     (`dflash_worker_v2.py:611`) — the exact statement replaced is the greedy
     draft sample at **`dflash_worker_v2.py:1493-1496`** — to a closure that runs
     `DFlashDominoRollout.rollout_draft_block(...)`. Restored in `finally`.
   Rationale: the greedy call is 4 lines buried in a ~490-line method; copying
   the whole method to patch its middle is brittle. In the DFLASH decode path
   both `draft_model_runner.forward` and `_greedy_sample_from_vocab_parallel_head`
   are called exactly once, so the capture/swap is unambiguous. (Prefill/extend
   path at `1211-1281` calls neither on the draft runner, so the wrap is inert
   there.)

2. **`DFlashWorkerV2.__init__`** (`dflash_worker_v2.py:71`) — extended: after
   `super().__init__(*args, **kwargs)`, build `DFlashDominoHelper` +
   `DFlashDominoRollout` when `self.draft_model.projector_type` is a Domino
   projector. Uses the already-initialized `self.draft_model` (`:165`),
   `self.block_size` (`:171/173`), `self.target_worker` (`:258`).

3. **`DFlashDraftModel`** (`models/dflash.py:308`, `EntryClass` at `:459`) —
   subclassed by `DominoDraftModel`. Adds `prefix_gru`/`embed_proj` (mirrors fork
   `models/dflash.py:296-331`); `load_weights` calls `super().load_weights` then
   `prefix_gru.flatten_parameters()` (the base loader already loads the head
   weights because they are now in `named_parameters()` with matching names).

4. **`ModelRegistry.models["DFlashDraftModel"]`** (`models/registry.py:130`,
   keyed by class `__name__`) — rebound to `DominoDraftModel` in
   `register_plugin()`. The public checkpoint's architecture string is
   `"DFlashDraftModel"`, and upstream's class lacks the head, so the arch entry
   must point at our subclass.

5. **`CustomSpecAlgo`** (`speculative/spec_registry.py:24`) — subclassed by
   `_DominoSpecAlgo` (built in `__init__._build_spec_class`) to mirror the
   `SpeculativeAlgorithm.DFLASH` enum member so all builtin `is_dflash()`-gated
   branches fire for `DOMINO`:
   - `is_dflash() -> True` (scheduler gates: `managers/scheduler.py:897,1484,2120,2690`; verify-input dispatch `speculative/spec_info.py:327`).
   - `supports_target_verify_for_draft() -> True` (enum `spec_info.py:118`).
   - `carries_draft_hidden_states() -> False` (called `scheduler.py:1121,1126,1174,1179`; absent on base `CustomSpecAlgo`).
   - `need_topk() -> False` (called `managers/overlap_utils.py:152`; absent on base).
   - `create_future_map(...)` mirrors enum `spec_info.py:132` (called `scheduler.py:1236`; absent on base).
   - `handle_server_args(server_args)` calls `arg_groups/speculative_hook._handle_dflash` (dispatched at `speculative_hook.py:128`), then forces `disable_overlap_schedule = True` (see (c)).

6. **Algorithm registration** — `SpeculativeAlgorithm.register("DOMINO", supports_overlap=False, spec_class=...)` (`spec_info.py:59` / `spec_registry.py:189`); factory returns `DominoWorkerV2`. Dispatch: `from_string` (`spec_info.py:43`), worker built at `scheduler.py:811-812` via `CustomSpecAlgo.create_worker -> self.factory` (`spec_registry.py:95-111`).

### Import shim in the copied files (documented, minimal)

The three copied fork modules referenced fork-only paths; only their import
lines were edited (no logic change):
- `domino_helper.py`: `from sglang.srt.speculative.dflash_utils import is_dflash_domino_projector` → `from .config import is_dflash_domino_projector` (upstream `dflash_utils` has no such symbol).
- `domino_rollout.py`: `from sglang.srt.speculative.domino_{helper,kernels}` → `from .domino_{helper,kernels}` (those modules do not exist upstream; they live in this package).

## (b) v1 → v2 translation decisions

- **Worker base:** fork patched the v1 `DFlashWorker` (`dflash_worker.py`); latest
  upstream dispatches `DFLASH` to `DFlashWorkerV2`. We subclass **v2**.
- **Swap site moved:** fork swapped at v1 `dflash_worker.py:705-724`; the v2
  analogue is the greedy call at `dflash_worker_v2.py:1493-1496`, inside the
  monolithic `forward_batch_generation`. We use the wrap-and-rebind seam instead
  of the fork's inline `if is_dflash_domino_projector(...)` branch.
- **TP-argmax callbacks:** the fork's `DFlashWorker` exposed
  `_global_argmax_from_local_logits` / `_global_argmax_from_local_max`
  (`dflash_worker.py:205-206`) that the rollout needs **only on the TP>1
  eager path** (`domino_rollout.py:1035-1045`). v2 has no such helpers, so for
  Phase 1 we pass callbacks that raise `NotImplementedError`. At **TP=1** they
  are never called and the CUDA-graph fused path runs. → **Phase 1 is TP=1 only.**
- **Config:** reuse upstream `parse_dflash_draft_config` for the 6 base fields;
  `parse_domino_draft_config` adds only the 5 Domino fields (fork
  `dflash_utils.py:404-483`). We do not depend on upstream underscore-private
  helpers — the tiny `dflash_config`/int parsers are re-implemented locally.
- **Server-arg setup:** upstream moved the DFLASH `== "DFLASH"` string-gated
  block out of `server_args.py` into `_handle_dflash` (`speculative_hook.py`).
  We route `DOMINO` through `_handle_dflash` via the spec-class
  `handle_server_args`, so num_steps/topk→1, block_size inference, and
  max_running_requests default all apply for free.
- **Naming:** followed the `speculative-naming` skill for new identifiers. The
  copied rollout keeps its existing param name `verified_id` (Rule 2 would
  prefer `bonus_token`); left unchanged to avoid diverging the copied API — flag
  for a later rename pass if upstreaming.

## (c) Overlap scheduler question → `supports_overlap = False`

**Decision: `supports_overlap=False` for Phase 1**, and `handle_server_args`
force-sets `server_args.disable_overlap_schedule = True`.

Reasoning:
- The Domino rollout captures **its own CUDA graph** for the per-step loop
  (`domino_rollout._get_or_capture_domino_loop_graph`, keyed on shapes) and
  replays it on the current stream. It also monkeypatches
  `draft_model_runner.forward` for the duration of each `forward_batch_generation`
  call.
- v2 **overlap** scheduling runs draft/verify across multiple CUDA streams and
  can overlap consecutive steps. That interacts unverified-ly with (i) the
  rollout's captured graph / stream assumptions and (ii) our per-call monkeypatch
  seam (a second concurrent call could observe the swapped `forward`).
- The fork forbade overlap outright for DFLASH-Domino (v1). Upstream v2 DFLASH
  *supports* both, and **non-overlap is a first-class, supported mode**
  ("scheduler runs it synchronously when overlap is disabled",
  `spec_info.py:201`). So forcing overlap off is safe and standard, not a hack.
- With `supports_overlap=False`, `CustomSpecAlgo.create_worker`
  (`spec_registry.py:96-111`) would *raise* unless overlap is already disabled;
  forcing `disable_overlap_schedule=True` in `handle_server_args` (which runs
  earlier, `speculative_hook.py:128`) makes the launch "just work" without the
  user passing `--disable-overlap-schedule`.

Phase 2+ can revisit `supports_overlap=True` after validating the rollout graph
under multi-stream scheduling on a GPU.

## (d) Not verifiable without a GPU (static-only environment)

`python -m py_compile` passes for all 7 modules. The following need a GPU +
real Domino checkpoint to confirm:
1. **Weight loading of the head** — that the checkpoint's `prefix_gru.*` and
   `embed_proj.{0,2}.*` names match `DominoDraftModel.named_parameters()` so the
   base loader binds them (no silent "ignore unexpected weights"). High risk.
2. **Registry-rebind timing** — that `register_plugin()` (via `load_plugins()`
   at `launch_server.py:65`, before `prepare_server_args`) actually wins over the
   import-time `ModelRegistry.register("sglang.srt.models")`. Verified by reading
   the launch order; not executed.
3. **The full-hidden capture** — that wrapping `draft_model_runner.forward`
   captures the pre-`view` `[bs*block_size, H]` tensor and `reshape(bs, block_size, -1)`
   reproduces the layout the rollout expects (esp. slot-0 for `shift_label=True`).
4. **`shift_label` of the public checkpoint** — official code defaults it to
   `False` (`ref_repo/Domino/code/dflash.py:347-348`); the exact value in
   `Huang2020/Qwen3-8B-Domino-b16/config.json` was not inspected (offline).
5. **DFLASH target aux-hidden capture** — DFLASH requires the target model to be
   configured to emit per-layer aux hidden states; inherited from upstream and
   unverified here.
6. **CUDA-graph capture** inside the rollout on a live stream, and the TP=1 fused
   Triton kernels.

## (e) Validation launch command

TP=1, single GPU (Phase 1 constraint). Block size 16 = the `-b16` checkpoint.

```bash
SGLANG_PLUGINS=dominotree \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --speculative-algorithm DOMINO \
  --speculative-draft-model-path Huang2020/Qwen3-8B-Domino-b16 \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 16 \
  --tp-size 1 \
  --trust-remote-code \
  --port 30000
```

Notes:
- Install first (editable) in the SGLang env: `pip install -e .` from this dir
  (registers the `sglang.srt.plugins` entry point).
- `--speculative-num-steps 1` / `--speculative-eagle-topk 1` are forced to 1 by
  `_handle_dflash` anyway; passing them avoids a warning. `--speculative-num-draft-tokens`
  is the DFLASH block size and must equal the checkpoint's `block_size` (16).
- Overlap is auto-disabled by the plugin (see (c)); no flag needed.
- Smoke check that the plugin loaded: the log should show
  "Registered DOMINO speculative algorithm" and
  "DominoWorkerV2 initialized Domino chain rollout".
```
