# KERNEL_NOTES — GPU-native (CUDA-graph) DominoTree builder

Branch: `feat/gpu-native-builder`. Authored and CPU-tested **without GPU access**;
every performance statement below is a hypothesis **to be measured on GPU**.

## 1. What this is

The DominoTree conditional build (`dominotree@B`) is bottlenecked by per-node GPU
compute inside `children_fn` (measured profile on RTX 5080, dominotree@16,
corr-topm=64: build = 4.24 ms/round, 78.7% of it correction compute — per-node
`prefix_gru` + state slicing 1.44 ms, correction select 0.96 ms, per-round
corr_topm setup 0.84 ms). Each of the ~16 best-first pops per round launches
~15–20 tiny kernels from Python.

Best-first expansion is data-dependent (the heap decides the next pop), so the
whole build loop cannot be captured. Instead we adapt the proven pattern from
Domino's released `DraftCorrectionGraphRunner`
(`ref Domino code/kernel/domino.py`, driven in `dflash.py:spec_generate`):
capture the **fixed-shape single-node compute** once, then per pop
`copy_` inputs into static buffers → `graph.replay()` → read static outputs.
The Python heap (`dominotree.build_best_first_tree`) is untouched.

Implementation: `dominotree_gpu.GraphNodeExpander`, opt-in via
`benchmark.py --gpu-native-build`. The pure-Python `children_fn` remains the
default; nothing changes unless the flag is passed.

## 2. The three captured graphs

| Graph | Replayed | Body |
|---|---|---|
| `g_setup` | once per round (only if `corr_topm > 0`) | for each depth `d < k_draft`: `topk(base_logits[d], corr_topm)` candidate ids; gather `embed_proj[2].weight[cand]` and `base_logits[d][cand]` (both `.float()`) into `S_cand_all` / `S_w2c_all` / `S_basec_all` |
| `g_corr` | once per pop at `depth >= prefix_len` | fc1+SiLU on `cat([ph_d, s_feat])` → corrected logits (candidate-restricted `h @ w2c.t()` if `corr_topm > 0`, else full-vocab `embed_proj` add) → `log_softmax` → `topk(node_topk)` tokens/logprobs → embed tokens → batched 1-step `prefix_gru` → `node_topk` child states |
| `g_base` | once per pop at `depth < prefix_len` (only if `prefix_len > 0`) | embed the supplied per-depth base top-k tokens → batched 1-step `prefix_gru` → child states |

For `depth < prefix_len` the token/logprob selection is path-independent, so it
is computed **once per round** in `begin_round` (identical math to the per-node
`log_prob_topk` the Python path re-evaluates), and only the state expansion is
replayed per pop.

Per pop, host-side work is: `S_state.copy_(state)`, `S_depth.fill_(depth)`
(corr path) or `S_exp_tokens.copy_(idx_dev)` (prefix path), one `replay()`, one
device synchronize to read tokens/logprobs into pinned host buffers, and one
`S_out_states.clone()` snapshot (the next replay overwrites the static output;
heap entries hold views into the clone).

## 3. Equivalence to the Python path, by construction

The graph bodies (`_setup_impl`, `_corr_impl`, `_corr_full_impl`,
`_expand_impl`) are line-by-line transcriptions of
`domino_adapter.make_conditional_children_fn`, with only two mechanical changes:

1. Inputs are read from static buffers that are **bit-identical copies** of the
   round tensors (`S_ph_all ← ph`, `S_base_all ← base_logits`; `begin_round`
   raises on any dtype mismatch rather than silently casting).
2. Depth-dependent rows are selected with `index_select(0, S_depth)` on a
   static depth-index tensor instead of Python-side `[depth]` indexing.
   `index_select` materializes the same values with the same row-contiguous
   layout the Python views have.

All per-node op shapes, dtypes, op order, and reduction dims are unchanged
(`(1, hidden+gru)` cat → fc1 → SiLU → `.float()`; `(corr_topm,)` corrected
logits; `topk` on the same shapes with the same `k`; `(node_topk, 1, E)` GRU
input with `(1, node_topk, gru)` initial state). Same ops on same
shapes/dtypes dispatch the same kernels, so results are expected
**bit-identical**, not merely close — this is exactly what checklist item V2
verifies end-to-end. The one deliberate caution: `topk` tie-breaking among
equal bf16 logits is kernel-dependent, which is why every `topk` in the fast
path keeps the exact input shape of its Python counterpart (e.g. the setup
graph does per-row `topk` in a loop rather than one batched `topk`, so the
`(V,)`-shaped call matches `torch.topk(base_logits[d], corr_topm)` exactly).

The same body functions run either eagerly or under capture:
`use_graphs=False` (or env `DOMINOTREE_GPU_EAGER=1`) executes them directly
against the same static buffers. The CPU self-test therefore exercises the
*identical code* that gets captured on GPU, and on a CUDA machine the self-test
additionally captures/replays the real graphs on tiny models against the
pure-Python reference.

Known intentional no-ops (values unaffected):
- Child states are non-contiguous views into the per-pop clone instead of
  per-child `.contiguous()` copies; downstream use is `copy_`/`transpose`/
  `expand`, all layout-agnostic.
- Prefix-depth token/logprob lists computed once per round instead of per node
  (the function of `base_logits[d]` alone).

## 4. Static buffer layout

Shapes for `k_draft=K`, `node_topk=k`, `corr_topm=M`, GRU width `G`, draft
hidden `H`, `embed_proj` middle width `E`, vocab `V`. Dtype `dt` = draft dtype
(bf16 in the benchmark; the CPU test runs fp32).

| Buffer | Shape / dtype | Written | Read by |
|---|---|---|---|
| `S_ph_all` | `(K, H)` dt | per round (`begin_round`) | `g_corr` |
| `S_base_all` | `(K, V)` dt | per round | `g_setup` (topm>0), `g_corr` (topm==0), prefix rows |
| `S_cand_all` | `(K, M)` i64 | by `g_setup` | `g_corr` |
| `S_w2c_all` | `(K, M, E)` f32 | by `g_setup` | `g_corr` |
| `S_basec_all` | `(K, M)` f32 | by `g_setup` | `g_corr` |
| `S_state` | `(1, 1, G)` dt | per pop | `g_corr`, `g_base` |
| `S_depth` | `(1,)` i64 | per pop (corr) | `g_corr` (`index_select` selector) |
| `S_exp_tokens` | `(k,)` i64 | per pop (prefix) | `g_base` |
| `S_out_tokens` | `(k,)` i64 | by `g_corr` | host readback (pinned `H_tokens`) |
| `S_out_lps` | `(k,)` f32 | by `g_corr` | host readback (pinned `H_lps`) |
| `S_out_states` | `(1, k, G)` dt | by `g_corr` / `g_base` | per-pop `clone()` |

Approximate extra device memory at the paper config (K=16-ish, M=64, E=emb_dim,
V≈128K, bf16): `S_base_all` ≈ K·V·2 B ≈ 4 MB, `S_w2c_all` ≈ K·64·E·4 B
(≈ 17 MB at E=4096), everything else negligible, plus one private capture pool
per graph (three graphs; pools intentionally NOT shared because replay order is
data-dependent — see risks).

## 5. GPU VERIFICATION CHECKLIST (for the GPU-execution loop)

Environment: the benchmark venv on the GPU box (RTX 5080), repo on
`feat/gpu-native-builder`. Set these once:

```bash
export MODEL=/path/to/target-model DRAFT=/path/to/domino-draft DOMINO=/path/to/Domino/code
cd <Domino-Tree repo>
```

**V1 — unit: capture works and graphs match the pure-Python scorer.**
```bash
python dominotree.py
```
Must print BOTH `dominotree_gpu self-test (cpu, ...) ALL PASSED` and
`dominotree_gpu self-test (cuda, CUDA-graph replay vs pure-Python): ALL PASSED`.
This exercises capture + replay of all three graphs across 5
(corr_topm, prefix_len) configs × 2 rounds on tiny fp32 models. If capture
itself fails here (see risks R1), stop and report the traceback.

**V2 — losslessness/identity end-to-end at T=0 (the hard gate).**
Two identical runs, fast vs python, greedy:
```bash
python benchmark.py --model-name-or-path $MODEL --draft-name-or-path $DRAFT \
  --domino-code $DOMINO --dataset gsm8k --max-samples 20 --temperature 0.0 \
  --methods dominotree --budgets 16 --out results/gnb_python.jsonl
python benchmark.py --model-name-or-path $MODEL --draft-name-or-path $DRAFT \
  --domino-code $DOMINO --dataset gsm8k --max-samples 20 --temperature 0.0 \
  --methods dominotree --budgets 16 --gpu-native-build --out results/gnb_graph.jsonl
python - <<'EOF'
import json
a = [json.loads(l) for l in open("results/gnb_python.jsonl")]
b = [json.loads(l) for l in open("results/gnb_graph.jsonl")]
assert len(a) == len(b), (len(a), len(b))
bad = [(x["sample_idx"], x["turn_index"], x["out_sig"], y["out_sig"])
       for x, y in zip(a, b) if x["out_sig"] != y["out_sig"]]
acc = [(x["sample_idx"], x["mean_accept"], y["mean_accept"])
       for x, y in zip(a, b) if abs(x["mean_accept"] - y["mean_accept"]) > 1e-9]
print("rows:", len(a), "| out_sig mismatches:", bad or "NONE", "| accept mismatches:", acc or "NONE")
EOF
```
PASS = zero `out_sig` and zero `mean_accept` mismatches, row-for-row.
(A `--smoke` variant of both commands is a fast pre-flight; do it first.)
Repeat V2 on a second dataset (e.g. `--dataset humaneval`) before trusting it.

**V3 — build-time and TPS measurement (the point of the exercise).**
From the same two runs, compare the printed per-round stage table and/or:
```bash
python - <<'EOF'
import json, statistics
for name in ("results/gnb_python.jsonl", "results/gnb_graph.jsonl"):
    rows = [json.loads(l) for l in open(name)]
    for key in ("ms_build", "ms_draft", "ms_verify", "ms_commit", "tps"):
        print(name, key, round(statistics.fmean(r[key] for r in rows), 3))
EOF
```
Report mean `ms_build` python vs graph (baseline profile: 4.24 ms/round) and
end-to-end TPS. `ms_draft`/`ms_verify` must be statistically unchanged — the
flag must not touch other stages. No target number is claimed; measure it.

**V4 — isolate capture vs static-buffer refactor if V2 fails.**
```bash
DOMINOTREE_GPU_EAGER=1 python benchmark.py ... --gpu-native-build --out results/gnb_eager.jsonl
```
If eager-static matches python but graphed does not, the bug is in
capture/replay (stale buffer, uncaptured op); if eager-static already
mismatches, it is in the buffer plumbing. Report which.

**V5 — memory/stability spot-check.** During V2's graph run, note
`torch.cuda.max_memory_allocated()` delta vs the python run (expect tens of MB)
and that multi-prompt runs are stable (no capture-pool growth over rounds:
capture happens once at startup, `begin_round` only replays).

## 6. Risks / unverified (no GPU here — honest list)

- **R1: cuDNN GRU under capture.** `nn.GRU` is captured inside the graphs.
  Precedent: Domino's released runner captures `prefix_gru` the same way
  (`kernel/domino.py` line ~391), so this is proven on their stack, but not on
  this exact torch/cuDNN/sm_120 combination — V1 tests it directly. Escape
  hatch if capture errors: `DOMINOTREE_GPU_EAGER=1` (correctness-preserving,
  perf reverts to ~Python-path level).
- **R2: bf16 `nn.GRU` kernel path.** The unit test runs fp32; the benchmark
  runs bf16. Whatever bf16 GRU kernel the current Python path uses is what gets
  captured (same call), but bf16-under-capture specifically is only verified by
  V2.
- **R3: topk tie-breaking.** Bit-identity relies on same-shape same-kernel
  dispatch (Section 3). If a mismatch ever appears in V2, first check whether
  the diverging round involved tied bf16 logits at the candidate boundary
  (V4 will localize it).
- **R4: graph memory pools.** Three graphs with private pools; not shared
  because replay order is data-dependent (PyTorch only sanctions pool sharing
  for graphs replayed in capture order). Cost is a few private-pool MBs — V5.
- **R5: remaining sync floor.** One device synchronize per pop remains (the CPU
  heap needs token/logprob values). The measured 14.7% CPU-sync share is the
  floor this design cannot remove without changing the algorithm (forbidden).
  If post-V3 profiles show sync dominating, the next lever is a batched/wave
  builder — out of scope for this losslessness-preserving change.
- **R6: no measured speedup.** Expected direction: build well under 4.24 ms
  (launch-overhead removal + one setup replay instead of ~50 setup launches),
  but the correction ops are tiny GEMVs whose graphed cost on the 5080 is
  unknown. TO BE MEASURED (V3).
