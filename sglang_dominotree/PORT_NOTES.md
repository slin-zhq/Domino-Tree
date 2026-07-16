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
- **Scope:** Phase 1 = `DOMINO` (chain). Phase 2 = `DOMINOTREE` (toy tree
  verify) — see the "Phase 2" section at the end. The real conditional
  best-first tree builder is Phase 3.

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
  "Registered DOMINO ... and DOMINOTREE ..." and
  "DominoWorkerV2 initialized Domino chain rollout".
```

---

# Phase 2 — `DOMINOTREE` (toy tree verify)

Goal: prove a **branching** draft tree from the Domino drafter verifies
end-to-end through SGLang's EAGLE tree-verify machinery, **losslessly at T=0**,
with `spec_accept_length >= the DOMINO chain's`. Plumbing proof only — the real
conditional best-first builder is Phase 3.

## New files / additions

| File | Role |
|---|---|
| `src/dominotree_sglang/tree/toy_tree.py` | Pure-torch fixed "caterpillar" tree: topology (`build_topology`), tokens (`build_draft_tokens`), intra-tree ancestor mask (`build_intra_tree_mask`), full attention allow-mask (`build_full_attention_mask`). |
| `src/dominotree_sglang/tree/__init__.py` | Re-exports. |
| `src/dominotree_sglang/worker.py` | Adds `DominoTreeWorkerV2(DominoWorkerV2)` + `_compact_accept_to_front` helper. |
| `src/dominotree_sglang/__init__.py` | Registers `DOMINOTREE` alongside `DOMINO`. |

## Tree-construction approach chosen (and why)

**Hand-emit the intra-tree ancestor mask, then let the `reconstruct_indices_from_tree_mask`
sgl-kernel op derive positions + the three retrieve tensors — the NGRAM pattern
(ngram_worker.py:288-344), NOT EAGLE's `build_tree_kernel_efficient`.** Rationale:
`build_tree_kernel_efficient` is EAGLE-rigid (fixed `topk`/`spec_steps` fanout,
level-wise scored inputs); the seam doc (§2, §13-14) flags it as the wrong tool
for an irregular tree. NGRAM proves the verifier accepts a hand-built irregular
tree via `reconstruct_indices_from_tree_mask` + `tree_topk = -1`. We reuse
`EagleVerifyInput` directly with `spec_steps = N-1` (so `max_tree_depth = N`) and
`topk = -1` — no verify-input subclass needed.

**Tree shape (fixed caterpillar), N = block_size:** node 0 = root (prev
verified/bonus token); nodes 1..S = spine = the Domino GRU-corrected chain
(`S = N-1-num_branch`); nodes S+1..S+B = branch siblings at the shallowest
depths (2nd LM-head candidate over the draft hidden). `num_branch` is env
`DOMINOTREE_NUM_BRANCH` (default 2, clamped to ≤ (N-1)//2). For N=16, B=2 →
spine of 13, branches at depths 1-2 (validated topology).

**Why N = block_size (not block_size + branches):** the verify node budget
equals the DFLASH `block_size`, so **all** DFLASH KV reservation, cuda-graph
widths, and buffer sizing are reused with zero server-arg changes. Cost: the
spine is `N-1-B` (=13) rather than the full `N-1` (=15) chain, so the strict
"tree contains the whole chain" guarantee weakens to "tree contains the chain's
first 13 tokens." Since 13 ≫ the measured chain acceptance (~2.7) and branches
only add acceptance, `accept_length(tree) >= accept_length(chain)` holds on
average (the P2 success metric) and per-step whenever chain acceptance ≤ 13.
Set `DOMINOTREE_NUM_BRANCH=0` for a guaranteed chain-as-tree (accept ==, still
exercises the full EAGLE tree-verify path).

## THE KEY DESIGN QUESTION — `is_dflash()` vs `is_eagle()` gating

**Decision: `DOMINOTREE` keeps `is_dflash() = True` (same spec_class as DOMINO);
it does NOT report `is_eagle()`.** Reasoning, from the actual gating call sites:

- The DOMINO drafter is a DFLASH block-parallel draft. All the DFLASH-gated
  scaffolding is draft/request/KV-sizing, not verify shape:
  scheduler `is_dflash()` gates at scheduler.py:897, 1484, 2120 (`validate_dflash_request`),
  2690; server-arg setup via `handle_server_args -> _handle_dflash`. DOMINOTREE
  needs every one of these (it keeps the DFLASH draft), so `is_dflash()=True`.
- **The verify shape is NOT gated by the algorithm anywhere in the
  scheduler/model-runner.** It is decided by the `SpecInput` the worker emits
  and the attention backend's dispatch on `is_verify_input()`. Our worker emits
  an `EagleVerifyInput` (type `EAGLE_VERIFY`) with a custom tree mask and calls
  `eagle_prepare_for_verify` / `eagle_sample` **directly inside
  `forward_batch_generation`**. So `is_dflash()=True` does **not** force a linear
  verify — there is no scheduler branch that would. Reporting `is_eagle()=True`
  instead would (wrongly) swap the draft worker/plumbing to EAGLE's and break the
  DFLASH draft. So the correct combination is **DFLASH draft gating + EAGLE
  verify invoked in-worker.**
- **One real interaction:** `create_dummy_verify_input` (spec_info.py:295-355)
  dispatches on `is_dflash()` and would build a *DFlash linear* dummy verify
  input for target-verify **CUDA-graph capture**, which mismatches our real
  `EagleVerifyInput` tree. Mitigation for P2: we run the target verify **eagerly**
  — `eagle_prepare_for_verify`'s `can_run_graph` returns False for the
  type/shape mismatch, so no graph is replayed. This is a GPU-validation risk
  (see risks) and a P2 correctness-over-speed choice.

## How the accepted path is compacted for KV commit

A branching tree accepts a **non-contiguous** set of nodes; DFLASH's
`_append_target_hidden_to_draft_kv_by_loc` commits a **dense prefix**
(dflash_worker_v2.py:823-953). Sequence:

1. `predict, accept_lens, accept_index = eagle_sample(...)` — `accept_lens`
   includes the bonus (eagle_utils.py:560-563).
2. `move_accept_tokens_to_target_kvcache(batch, accept_index, accept_lens-1, allocator)`
   (spec_utils.py:506-558) — moves the accepted nodes' **target** KV to the
   contiguous front slots `req_to_token[req, L : L+accept_len]`.
3. `_compact_accept_to_front(predict/hidden, accept_index, bs, N)` — gathers the
   accepted `predict` tokens and target hidden to the front of each per-req block
   (reimpl of eagle_worker_v2.py:1595-1611). After this the tree looks exactly
   like DFLASH's contiguous chain.
4. Reuse DFLASH's writer verbatim: `_append_target_hidden_to_draft_kv_by_loc(
   target_hidden=front_hidden, cache_loc_2d=front_slots, commit_lens=accept_lens, ...)`
   writes the accepted prefix's projected hidden into the **draft** KV. Front
   slots are gathered from `req_to_token[req, L:L+N]` (post-move).
5. `next_token_ids = compacted predict`; `accept_lens` (=commit_lens) carries the
   length; bonus / next `verified_id` = `predict[req, accept_len-1]`;
   `new_seq_lens = prefix_lens + accept_lens`.

## Upstream methods reused (file:line)

- `eagle_prepare_for_verify` (eagle_utils.py:281-354), `eagle_sample`
  (eagle_utils.py:357-563) — verify prep + greedy tree acceptance.
- `reconstruct_indices_from_tree_mask` (`sgl_kernel.speculative`; NGRAM usage
  ngram_worker.py:288-297) — positions + retrieve tensors from the intra mask.
- `move_accept_tokens_to_target_kvcache` (spec_utils.py:506-558),
  `assign_req_to_token_pool_func` (spec_utils) — KV compaction / draft-block map.
- `EagleVerifyInput` (eagle_info.py:30-96) with `topk=-1`, `spec_steps=N-1`.
- DFLASH draft build templated from dflash_worker_v2.py:1326-1500 (using freed
  backup/alloc/restore scratch, the fork-v1 pattern, since the tree verify
  allocates its own cache); `_append_target_hidden_to_draft_kv_by_loc`,
  `_make_next_draft_input_decode`, `_draft_block_spec_info` reused as-is.

## Phase-2 constraints

T=0 greedy only; TP=1 only (dense LM-head for branch candidates + Domino
rollout); page_size==1; non-mamba target; no compact-draft-cache window. Any
other case falls back to the lossless Domino chain (`DominoWorkerV2`).

## Not verifiable without a GPU (top risks to validate on MIRLab, 4B/TP=1)

1. **Intra-tree mask orientation / `reconstruct_indices_from_tree_mask`
   contract.** We build `mask[i,j]=True iff j is ancestor-or-self of i` (row =
   query node, col = key/ancestor) matching the NGRAM convention, but the exact
   row/col orientation and dtype the kernel expects are not verifiable offline.
   A wrong orientation yields wrong retrieve tensors → wrong acceptance. **First
   thing to check** (compare `retrieve_next_token`/`retrieve_index` against a
   hand-computed tree for bs=1).
2. **KV compaction + draft-KV commit correctness.** The
   `move_accept_tokens_to_target_kvcache` + front-slot gather +
   `_append_target_hidden_to_draft_kv_by_loc` chain must land the accepted
   path's target KV and projected draft KV at exactly `req_to_token[req, L:L+len]`.
   A mistake corrupts the cache → non-lossless output or drift. Validate
   byte-identity vs plain AR and vs DOMINO chain at T=0.
3. **Eager tree verify under `is_dflash()` gating.** That the target attention
   backend (flashinfer/fa3) correctly consumes our FULL custom tree mask for an
   `EAGLE_VERIFY` input built outside the EAGLE worker. **`--disable-cuda-graph`
   is REQUIRED for P2** (verify runs eager): `eagle_prepare_for_verify` routes a
   tree verify through the decode CUDA-graph runner otherwise, which errors
   `custom_mask_buf must be initialized ... in cuda graph mode`. Cuda-graph
   custom-mask support is a later phase.

## GPU bring-up fixes (validated iterations)

- **`custom_mask_buf must be initialized ... in cuda graph mode`** — the tree
  verify cannot run under the decode CUDA-graph runner in P2. Launch with
  `--disable-cuda-graph` (verify is eager). Documented as a hard P2 requirement.
- **`q.shape[0] (16) != qo_indptr[-1] (6)` at the target verify.** Root cause:
  the target verify forward was called with `skip_attn_backend_init=True`
  (copied from DFLASH). DFLASH's `DFlashVerifyInput.prepare_for_verify` plans the
  target attention backend itself, so DFLASH may skip. But
  `eagle_prepare_for_verify` only plans in the cuda-graph path; with
  `--disable-cuda-graph` it **defers** planning to the target `forward_extend`.
  `skip_attn_backend_init=True` maps to `ForwardBatch.mark_forward_metadata_ready()`
  (forward_batch_info.py:578-603), which makes `forward_extend` SKIP that
  deferred planning — so the verify reused the previous forward's (prefill's)
  attention metadata (`qo_indptr[-1]=6` = the ~6-token warmup/prefill), while our
  tree passes N=16 nodes. **Fix:** omit `skip_attn_backend_init` on the target
  verify call (EAGLE's behavior, eagle_worker_v2.py:1480-1484) so
  `forward_extend` re-plans for the 16 tree nodes + custom mask.
  **Warmup needed no special handling:** the target warmup uses the
  `is_dflash()` DFlash-linear dummy from `create_dummy_verify_input`
  (base_runner.py:603), which is self-consistent; only the real tree verify
  needed the re-plan.
- **NOT LOSSLESS (over-accept ~3.33 vs chain 2.7, output diverges) — even at
  NUM_BRANCH=0 (pure chain).** Root cause: the draft block corrupted the
  scheduler's reserved verify-slot mapping. `assign_extend_cache_locs_func`
  (cache_locs.py `assign_extend_cache_locs` kernel) *reads* the reserved slots
  from `req_to_token[req, L:L+N]` — the scheduler pre-reserves them for the
  decode; it does NOT allocate. My `_domino_draft_block` instead did
  `allocator.alloc(bs*N)` + `assign_req_to_token_pool_func(...)`, which
  **overwrote** `req_to_token[req, L:L+N]` with fresh scratch slots, then
  `allocator.restore_state()` freed the scratch but left `req_to_token` pointing
  at the freed ids. `eagle_prepare_for_verify` then re-reads
  `req_to_token[req, L:L+N]` for the verify `out_cache_loc`, so the target verify
  wrote committed KV into **freed** slots; once those slots were reused by a
  later step, the committed prefix KV was corrupted → the target verify at the
  next block saw wrong context → accepted non-greedy tokens (over-accept) and
  the output diverged. The KV stores are mask-independent, which is why the
  first block(s) looked fine and divergence appeared a few tokens in.
  **Fix:** the draft block now `assign_extend_cache_locs_func`-reads the reserved
  slots and dual-uses them for the draft-KV forward (draft pool) while the tree
  verify uses the same slots for target KV (target pool) — exactly DFLASH-v2's
  non-compact draft path (dflash_worker_v2.py:1393-1402). No alloc/restore, no
  `req_to_token` overwrite; `eagle_prepare_for_verify` re-reads the intact
  reserved slots. This makes NUM_BRANCH=0 a true chain-as-tree (accept == chain,
  byte-identical) and keeps the target KV correct for branching trees too.

## Validation launch command (DOMINOTREE)

```bash
SGLANG_PLUGINS=dominotree \
DOMINOTREE_NUM_BRANCH=2 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B \
  --speculative-algorithm DOMINOTREE \
  --speculative-draft-model-path Huang2020/Qwen3-4B-Domino-b16 \
  --speculative-num-draft-tokens 16 \
  --tp-size 1 \
  --trust-remote-code \
  --disable-cuda-graph \
  --port 30000
```

`--disable-cuda-graph` is **required** in P2 (eager tree verify). Compare at
temperature 0 against: (a) plain AR (no spec), (b)
`--speculative-algorithm DOMINO` — output must be byte-identical to both, and
`spec_accept_length` must be ≥ the DOMINO run's. Set `DOMINOTREE_NUM_BRANCH=0`
for the guaranteed chain-as-tree sanity baseline first.

---

# Phase 3 — conditional best-first tree (the paper's actual builder)

P3 replaces P2's fixed caterpillar with DominoTree's **per-request adaptive,
variable-width, best-first** tree, built by the Domino GRU-correction scorer,
and feeds it through the **exact same** P2 verify seam (reconstruct_indices →
`EagleVerifyInput` tree_topk=-1 → eagle_prepare_for_verify → target verify →
eagle_sample → move_accept + KV commit). Conditional is now the DEFAULT
DOMINOTREE path; `DOMINOTREE_BUILDER=toy` keeps the P2 caterpillar for A/B.

## New files / additions

| File | Role |
|---|---|
| `tree/best_first.py` | `TreeNode` + `build_best_first_tree` — ported VERBATIM from `dominotree.py` (heap over cumulative drafter log-prob; `parent < i` topological invariant). |
| `tree/conditional_children.py` | `make_conditional_children_fn` + `log_prob_topk` — ported op-for-op from `domino_adapter.py:120-194`, GREEDY/T=0 only, all 3 correction cases. |
| `tree/toy_tree.py` | added `build_intra_tree_mask_from_parents` (per-request parent-array → ancestor mask). |
| `worker.py` | `_domino_draft_block` now returns raw `draft_hidden` only; added `_build_conditional_tree_for_req`, `_build_conditional_trees`, `_toy_tree_tokens`; `_tree_decode_forward` dispatches on `tree_builder`. |

## How ph / base_logits / root_state are computed (with shift_label)

Per request (batch=1 slice `draft_hidden[b]` = `[N, H]`), mirroring
`domino_adapter.draft_block` (`:92-97`):

- **ph** (per-position draft hidden for the GRU correction): shift_label slice.
  `shift_label=False` → `ph = draft_hidden[b][-(N-1):]` (rows 1..N-1, `k_draft=N-1`);
  `shift_label=True` → `ph = draft_hidden[b]` (all N rows, `k_draft=N`). The public
  Qwen3-Domino checkpoints are shift_label=False, so k_draft = block_size-1.
- **base_logits** = `target.lm_head(ph)` → `[k_draft, V]`. Computed as
  `ph @ lm_head.weight.T` (TP=1 dense head; `matmul` in the head dtype, cast to
  float inside the scorer as the reference does).
- **root_state** = `draft_model.prefix_gru(target_embed_tokens(verified))[1]` →
  `(1,1,gru_dim)`, the GRU hidden after consuming ONLY the committed root token
  (recomputed fresh each round; no cross-round carry — contract §2.4).

The scorer's per-node GRU correction uses `draft_model.prefix_gru` +
`draft_model.embed_proj` + the **target** `embed_tokens` table exactly as the
reference; the 3 cases (depth<prefix_len uncorrected; depth≥prefix_len with
corr_topm>0 restricted; corr_topm==0 full-vocab) are transcribed line-for-line.
`node_topk` is clamped to `≤ corr_topm` when corr_topm>0 (dominotree_gpu.py:79-80).

## Tree budget / sizing

`build_best_first_tree(children_fn, root_state, budget=N-1, max_depth=N)` with
`N = block_size`. budget=N-1 draft nodes + the implicit root = N flat positions =
the DFLASH-reserved verify width, so **all P2 KV/buffer sizing is reused
unchanged**. Because a node at depth d needs d+1 nodes on its root path (all
counted against budget), the deepest reachable node is depth N-2 → flat tree-depth
N-1 → RoPE position L+(N-1), always within the N reserved slots. (max_depth=N is
the loop bound; children_fn's `depth ≥ k_draft` check is the real cap.)

## Padding scheme for short trees

The heap can yield `< N-1` nodes (narrow node_topk / shallow depth). The flat
tree is padded to exactly N nodes with **dead leaves**: `token = mask_token_id`,
`parent = 0` (child of root), `depth = 1`. These are safe by construction:
- they are leaves (no children), so acceptance can never extend through them;
- their token is `mask_token_id`, which the target's greedy argmax never emits
  mid-sequence, so they never match `target_predict[root]` and are never the
  accepted child;
- they only add siblings under the root, which `verify_tree_greedy` skips over
  when finding the argmax-matching child — they cannot alter the accepted path,
  positions, or retrieve links.
Validated offline: flat parent arrays stay topological (`parent[i] < i`),
root-reachable, cycle-free, with `depth[i] == ancestor-count(i)` including full
and empty trees.

## Batch handling

The best-first heap + per-pop host sync is inherently per-request (contract §5).
`_build_conditional_trees` loops `b in range(bs)`, builds a tree per request, and
assembles `draft_token[bs,N]` + per-request `intra_mask[bs,N,N]` (via
`build_intra_tree_mask_from_parents`). The shared verify tail (reconstruct →
eagle_sample → compaction) already handles the batch. Target regime is bs=1;
larger bs works but pays bs× the per-request Python heap + one GPU→CPU sync per
heap pop (expected for P3; the batched/GPU-native builder is P4).

## Flat-tree ↔ reconstruct convention

Node `nodes[i]` → flat index `1+i`; `flat_parent[1+i] = 0` if `nodes[i].parent==-1`
else `1+nodes[i].parent`; `flat_depth = nodes[i].depth + 1`. `reconstruct_indices_from_tree_mask`
derives positions (`L + flat_depth`) and retrieve links from the ancestor mask —
identical to P2, matching `dominotree.position_ids` (`[start]+[start+1+depth]`).

## Top-3 risks to validate on GPU (MIRLab, 4B/TP=1, --disable-cuda-graph)

1. **ph / base_logits / root_state numerical fidelity vs the reference.** The
   shift_label slice, the dense-head `matmul` vs HF `lm_head(...)`, and the GRU
   `embed_proj` transcription must match `domino_adapter.py` so the drafted
   candidate set (and thus acceptance) matches the paper. Sanity: coherent output
   + accept_length ≥ the toy tree's (an adaptive best-first tree should accept
   ≥ a fixed caterpillar). Losslessness itself is guaranteed by the verify.
2. **Dead-leaf padding under real vocab.** Confirm `mask_token_id` is never the
   target's greedy argmax (would let a dead node be "accepted"). It is a special
   token, so this should hold; watch for any prompt where the target legitimately
   emits it.
3. **Per-pop GPU→CPU sync throughput / correctness.** `children_fn` returns
   logprobs via `.tolist()` (host sync per pop). Verify it runs (eager,
   `--disable-cuda-graph`) and that the heap order matches the drafter log-prob
   priority; also confirm `node_topk ≤ corr_topm` clamping didn't silently shrink
   the tree below intent.

## Validation launch (DOMINOTREE, conditional builder = default)

```bash
SGLANG_PLUGINS=dominotree \
DOMINOTREE_NODE_TOPK=8 DOMINOTREE_CORR_TOPM=64 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B \
  --speculative-algorithm DOMINOTREE \
  --speculative-draft-model-path Huang2020/Qwen3-4B-Domino-b16 \
  --speculative-num-draft-tokens 16 \
  --tp-size 1 --trust-remote-code --disable-cuda-graph --port 30000
```

Validation (P3 is lossless by construction — do NOT byte-compare vs DOMINO):
coherent+correct output, and `spec_accept_length` ≥ the toy tree's
(`DOMINOTREE_BUILDER=toy DOMINOTREE_NUM_BRANCH=2`, which was ~3.4). `corr_topm=0`
exercises the full-vocab correction path.

---

# Phase 4 (part 1) — GPU-native CUDA-graph node expander

Replaces P3's per-node eager GRU-correction (many small kernel launches + a
GPU→CPU sync per heap pop) with `dominotree_gpu.py`'s CUDA-graph replay: the
systems contribution that turns the accept-length win into a per-step-latency
win. The best-first heap stays in Python (data-dependent); only the fixed-shape
per-node math inside `children_fn` is graphed.

## New file / additions

| File | Role |
|---|---|
| `tree/gpu_expander.py` | `GraphNodeExpander` ported near-verbatim from `dominotree_gpu.py` (3 captured graphs: setup / corr / base-expand; static-buffer copy-in → replay → read-out). Bundles the reference equivalence suite (adapted to this package's modules). |
| `worker.py` | lazy `_get_gpu_expander()` + `_effective_node_topk()`; `_build_conditional_tree_for_req` uses `expander.begin_round(ph, base_logits)` + `expander.children_fn` when available, else P3 pure-Python. |

## The `embed_tokens` adaptation (the ONE change vs the reference)

The reference `_expand_impl` hardcodes `self.target.model.embed_tokens(toks)`. The
SGLang target exposes its embedding via `get_input_embeddings()`, so the port
drops the `target` ctor arg, takes an `embed_tokens` module argument, and calls
`self.embed_tokens(toks)`. `draft` is our `DominoDraftModel` (already has
`prefix_gru` + `embed_proj`); `dtype/gru_dim/vocab/mlp_dim/hidden` are inferred
exactly as the reference. Nothing else changed — the math is byte-identical, so
the equivalence suite still proves it matches the pure-Python scorer.

## Where/when the expander is constructed + captured

**Lazily, on first decode**, in `_get_gpu_expander()` (cached on `self`,
constructed at most once; a construction/capture failure sets a flag and falls
back to pure-Python for the rest of the run). It is built inside
`_build_conditional_tree_for_req`, i.e. **after** the draft-block forward and
**before** the target verify. Capture is cleanly bracketed:
`_capture()` first `torch.cuda.synchronize()` (drains the draft forward), warms
up the exact graph bodies 3× on a **side stream** in `torch.inference_mode()`,
re-syncs, then records each graph with `torch.cuda.graph(...)` on the default
stream (no other work is submitted — the worker is single-threaded/synchronous
under `--disable-cuda-graph`), then syncs again. This mirrors how the Domino
rollout captures its own graph mid-run; the two never coexist here because the
conditional path never calls `rollout_draft_block` (that's the toy/chain path).

## GPU-vs-pure-Python gating

`DOMINOTREE_GPU_BUILDER` (default `1` = GPU expander; `0` = P3 pure-Python).
`DOMINOTREE_GPU_EAGER=1` runs the expander's graph bodies eagerly (no capture) —
a GPU debugging fallback that is still bit-equivalent. `node_topk` is clamped to
`≤ corr_topm` (`_effective_node_topk()`) for BOTH paths so they stay identical
and the expander's constructor precondition holds. `DOMINOTREE_BUILDER=toy` (P2)
and the Domino-chain fallback remain reachable. **Acceptance must be unchanged**
vs P3 (3.92) — the expander is bit-equivalent by construction; a different number
is a port bug.

## dtype reconciliation

The expander's static buffers use `dtype = next(draft.parameters()).dtype`, and
`begin_round` **raises** on a `ph`/`base_logits` dtype mismatch (a silent cast
would change numerics). In the standard config the draft model and the target
`lm_head` are the same server dtype (bf16), so `ph` (draft-hidden dtype) and
`base_logits = ph_cast @ lm_head.weight.T` (lm_head dtype) both equal the buffer
dtype — no mismatch. As belt-and-suspenders the worker checks the dtypes before
`begin_round` and falls back to pure-Python (with a warning) instead of letting
the guard raise mid-decode.

## Local self-test status

`python -m dominotree_sglang.tree.gpu_expander` runs the equivalence suite
(CPU eager-static vs pure-Python; +CUDA graph-replay vs pure-Python when a GPU is
present). **Not runnable on the dev Mac (no torch)** — run it on MIRLab. The port
was verified statically: each graph body (`_setup/_corr/_corr_full/_base_expand/_expand`)
matches both the reference GPU bodies and this package's `conditional_children.py`
op-for-op; the only edit is the `embed_tokens` handle.

## Top-3 GPU risks to validate (MIRLab, 4B/TP=1, --disable-cuda-graph)

1. **CUDA-graph capture coexistence.** The #1 risk: 3 graphs captured mid-run on
   a side stream. Confirm capture doesn't collide with anything (it's bracketed
   by syncs; SGLang model graphs are off; the rollout graph is never captured in
   the conditional path) and leaves the default stream clean. Watch for capture
   errors from the cuDNN GRU (the 3× warmup primes its workspace/algo — the same
   pattern the P1 rollout uses successfully) and from `VocabParallelEmbedding`
   (graph-safe at TP=1: direct lookup, no all-reduce).
2. **Equivalence.** `accept_length` from the GPU builder MUST equal the
   pure-Python 3.92 (a delta = a port/capture bug). First run
   `DOMINOTREE_GPU_EAGER=1` (eager-static, no capture) to isolate math from
   capture, then the captured path; also run `python -m dominotree_sglang.tree.gpu_expander`.
3. **Latency actually improves.** The point of P4 is fewer launches + the per-pop
   sync collapsing to one small D→H copy per corr pop. Confirm per-step latency /
   TPS drops vs P3 (the accept-length win becoming a real throughput win); the
   `corr_topm=0` full-vocab path is heavier — check both `corr_topm=64` and `0`.

---

# Phase 4 (part 2) — tree verify under the decode CUDA graph (drop `--disable-cuda-graph`)

DOMINOTREE previously **required** `--disable-cuda-graph` (found in P2): the decode
CUDA-graph runner captured the target-verify graph without a `custom_mask_buf`, so
replaying our tree verify crashed with
`custom_mask_buf must be initialized ... in cuda graph mode`. Running eager is a
real handicap vs the DOMINO chain (which uses the decode graph). P4b makes the
tree verify replay a graph like EAGLE/NGRAM.

## Root cause + how EAGLE/NGRAM provide `custom_mask_buf`

The decode graph runner captures the verify graph via `get_spec_info`
(decode_cuda_graph_runner.py:1025-1104). It dispatches on the algorithm:
- **EAGLE / NGRAM** build the dummy verify input with `custom_mask=self.buffers.custom_mask`
  (1044 / 1095) → flashinfer's `init_forward_metadata_capture_cuda_graph`
  (flashinfer_backend.py:803-808) sees `spec_info.custom_mask is not None` →
  `use_custom_mask=True` → the prefill wrapper is created with a static
  `custom_mask_buf` (= `self.cuda_graph_custom_mask`, flashinfer_backend.py:737,
  760-773). At replay, the real tree mask (from `generate_attn_arg_prefill`) is
  copied into that buffer by flashinfer's `begin_forward` and the graph replays.
- **DFLASH** (what DOMINOTREE reports via `is_dflash()=True`) instead calls
  `resolve_dflash_verify_mask_policy(attn_backend)` (1071); for backends in
  `_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS` (flashinfer/fa3/triton/trtllm) it
  returns `build_custom_mask=False` — correct for DFLASH's LINEAR verify
  (causal == no custom mask), so the capture uses `custom_mask=None` and the
  wrapper has **no** `custom_mask_buf`. Our tree verify emits an
  `EagleVerifyInput` WITH a tree mask → replay wants a `custom_mask_buf` that was
  never allocated → the P2 crash. `can_run_graph` itself has no mask-specific
  gate (decode_cuda_graph_runner.py:400-469), so nothing else blocks it.

## What changed (plugin-only)

`__init__.py`: `register_plugin()` calls `_install_dflash_custom_mask_graph_hook()`,
which installs an **algorithm-aware wrapper** on
`sglang.srt.speculative.dflash_utils.resolve_dflash_verify_mask_policy`. On the
first call while the server's `speculative_algorithm == "DOMINOTREE"` (i.e. during
target-verify graph capture, after the global server args are set at
`model_runner.py:525`), the wrapper empties the module global
`_DFLASH_VERIFY_SKIP_CUSTOM_MASK_BACKENDS`. Since that frozenset is read at CALL
time, `build_custom_mask` becomes True for all callers, so the DFLASH verify graph
is captured WITH `custom_mask=buffers.custom_mask` (the static buffer, sized
`(max_bs*seq_len_fill + max_num_token) * num_tokens_per_bs` in base_runner.py:93 —
large enough for our tree FULL_MASK). At replay our tree mask is copied into that
buffer (flashinfer), so the graph replays the correct tree attention.

**Why a wrapper installed in `register_plugin` (not `handle_server_args`):** the
graph capture runs in the scheduler **subprocess**, but `handle_server_args` runs
only in the **main process** — `run_scheduler_process` receives a pre-constructed
`server_args`, so `ServerArgs.__post_init__` / `handle_speculative_decoding` never
re-run in the subprocess, and on spawn the subprocess re-imports `dflash_utils`
fresh, losing a main-process patch. `register_plugin` DOES run in the subprocess
via `load_plugins()` (scheduler.py:4181), before `Scheduler(...)`/graph capture,
so the hook must live there. `get_spec_info` reads
`resolve_dflash_verify_mask_policy` via a **local import** each call, so it picks
up our wrapper; other callers (chain-fallback runtime) read the now-emptied
frozenset and stay consistent.

The worker's `_tree_decode_forward` was **already** graph-ready: it mirrors EAGLE
(`eagle_prepare_for_verify` returns `can_run_cuda_graph`; when True it
`load_batch`es + `mark_forward_metadata_ready()`, and the target
`forward_batch_generation(is_verify=True)` then replays the graph). No worker
change was needed beyond removing the `--disable-cuda-graph` requirement.

## Scoping + safety

- **Algorithm-scoped at call time** (the wrapper checks `get_global_server_args().speculative_algorithm`),
  so a DOMINO-chain server is unaffected — the wrapper never empties the set there.
- **Subprocess-robust**: installed in `register_plugin`, which `load_plugins()`
  runs in the scheduler subprocess before graph capture (works under spawn OR
  fork).
- Ordering: the TARGET model runner captures before the DRAFT; the global algo is
  `DOMINOTREE` at target capture, so the skip set is emptied exactly when the
  target-verify custom-mask graph is captured. Idempotent (wrapper is marked;
  only empties once).
- Contained to a DOMINOTREE server. Its only side effect is the Domino-chain
  fallback (T>0) building a causal custom mask instead of the built-in causal
  path — correct, marginally slower, rarely hit.
- **No-op with `--disable-cuda-graph`** (no decode graph is captured), so that
  remains a working eager fallback.
- The draft-block forward under graph is unchanged from DFLASH (draft worker
  `get_spec_info` keeps `custom_mask=None` because `is_draft_worker=True`,
  decode_cuda_graph_runner.py:1080 — so the draft graph stays a plain block
  forward, exactly as DFLASH runs it).

## Coexistence with the expander / rollout graphs

- **Model decode graphs** (draft + target) are captured at **init** (before
  serving). **The P4a GPU node-expander** captures its 3 graphs **lazily on the
  first decode**, on a side stream with sync bracketing, while we're in eager
  Python between the draft forward and the target verify — so it never captures
  concurrently with the model graphs. The Domino **rollout** graph is never
  captured in the conditional path (only the toy/chain path calls
  `rollout_draft_block`). At replay all coexist as separate `CUDAGraph` objects.
  This is the #1 thing to GPU-validate (see risks).

## If it can't be graphed — the minimal blocker (documented, not forced)

If empting the skip set does not yield a working graph on GPU (e.g. flashinfer
refuses the capture/replay for an externally-built `EagleVerifyInput`, or the
expander capture can't coexist with the model graph pool), the minimal blocker is
architectural: **`get_spec_info` hardcodes the verify spec-info + mask policy per
`is_*()` branch, and DOMINOTREE must stay `is_dflash()=True` for the draft
plumbing while needing NGRAM/EAGLE-style custom-mask capture.** The only in-plugin
lever is the skip-set global; a cleaner fix would be an upstream hook letting a
`CustomSpecAlgo` choose the capture mask policy (out of scope: no upstream edits).
In that case keep `--disable-cuda-graph` (still fully lossless, just eager).

## Top-3 GPU risks to validate

1. **Graph coexistence.** Model decode graph (target verify custom-mask) + the 3
   expander graphs (captured mid-first-decode on a side stream) + draft-block
   graph must all capture and replay without allocator/stream collisions. This is
   why `--disable-cuda-graph` was used so far — validate capture succeeds and no
   "operation not permitted during stream capture" / pool errors.
2. **Lossless + same accept.** Output must stay coherent/correct and
   `accept_length` must match the eager run (4.68) — the graph must apply the tree
   mask, not causal-attend the flattened tree (that was the P2 over-accept bug).
   Cross-check the graphed run's accept vs `--disable-cuda-graph`.
3. **TPS actually rises.** The point of dropping `--disable-cuda-graph` is a
   per-step latency win from the target-verify (and draft) graphs. Confirm TPS
   improves vs the eager P4a run; if the expander capture forces eager or coexist
   fails, fall back to `--disable-cuda-graph`.

## Validation launch (no --disable-cuda-graph)

```bash
SGLANG_PLUGINS=dominotree \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B \
  --speculative-algorithm DOMINOTREE \
  --speculative-draft-model-path Huang2020/Qwen3-4B-Domino-b16 \
  --speculative-num-draft-tokens 16 \
  --tp-size 1 --trust-remote-code --port 30000
```

Compare accept_length + TPS against the same command **with** `--disable-cuda-graph`
(eager P4a): accept must be identical (4.68), TPS should be higher without it.
