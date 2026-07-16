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
