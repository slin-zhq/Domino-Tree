# dominotree-sglang

DominoTree and the Domino chain as out-of-tree speculative-decoding algorithms for
[SGLang](https://github.com/sgl-project/sglang). The package registers itself through
SGLang's `sglang.srt.plugins` entry point, so SGLang's own source is unmodified: no fork,
no patches, no vendored engine.

Two algorithms are registered:

- `DOMINOTREE` — the conditional best-first draft tree, verified with SGLang's EAGLE
  tree-verify. This is the paper's method.
- `DOMINO` — the plain Domino chain draft, as the same-engine baseline.

Both are lossless by construction: the target verifier is untouched, so only tokens the
target would itself have emitted are committed.

## Install

```bash
pip install -e .
```

`sglang`, `torch`, and `triton` are deliberately not declared as dependencies — the plugin
is only ever imported into an environment that already has a matching SGLang and its
CUDA-matched wheels, and pinning them here risks clobbering that stack.

The plugin binds to SGLang internals (the DFLASH-v2 worker, the EAGLE verify helpers, the
model registry), so it tracks specific upstream versions rather than a range. It needs an
upstream SGLang with the `sglang.srt.plugins` entry point and DFLASH-v2, and is validated
against upstream commit `1adb53f14`. A startup contract check names any upstream attribute
it depends on that has moved, and refuses to start rather than silently degrading.

## Serve

```bash
SGLANG_PLUGINS=dominotree \
python -m sglang.launch_server \
  --model-path /path/to/Qwen3-4B \
  --speculative-algorithm DOMINOTREE \
  --speculative-draft-model-path /path/to/Qwen3-4B-Domino-b16 \
  --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 16 \
  --tp-size 1 --trust-remote-code --mem-fraction-static 0.7 --port 30000
```

`SGLANG_PLUGINS=dominotree` whitelists the plugin. Swap in
`--speculative-algorithm DOMINO` for the chain baseline. Tensor parallelism is supported
(`--tp-size 2` is what the Qwen3-8B results use). Serving is over SGLang's usual
OpenAI-compatible and `/generate` endpoints, at any temperature.

The defaults are the configuration the paper reports — the zero-sync frontier builder with
CUDA-graph capture, running under SGLang's decode CUDA graphs. No environment variables are
needed for it.

## Requirements the plugin enforces

Unsupported server configurations fail at startup, rather than running and returning
subtly wrong output.

- **`--page-size 1`** (SGLang's default). The tree verifier addresses the KV cache one
  token at a time. A page holding several tokens cannot represent a branching draft, where
  siblings diverge partway through what would be a single page.
- **A non-hybrid target model** — no Mamba / gated-DeltaNet layers. Rejecting a draft
  branch means rewinding the target's state to the fork point. Attention KV can be rewound
  slot by slot; a recurrent layer's state cannot, so a rejected branch would leave the
  model's state corrupted with nothing to detect it.

## Tuning knobs

Defaults are the reported configuration; these exist for ablation and reproducibility.

| Variable                    | Default    | Meaning                                                                                                                                               |
| --------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DOMINOTREE_BUILDER`        | `frontier` | `frontier` = the zero-sync batched builder; `conditional` = the per-request best-first heap used in the paper's Python-vs-GPU-native builder ablation |
| `DOMINOTREE_FRONTIER_GRAPH` | `1`        | Capture the frontier builder itself in a CUDA graph                                                                                                   |
| `DOMINOTREE_GPU_BUILDER`    | `1`        | Use the CUDA-graph node expander for the `conditional` builder                                                                                        |
| `DOMINOTREE_NODE_TOPK`      | `8`        | Branching cap: candidate children considered per expanded node (clamped to `<= DOMINOTREE_CORR_TOPM`)                                                 |
| `DOMINOTREE_CORR_TOPM`      | `64`       | Candidates the GRU correction is re-run over                                                                                                          |

`DOMINOTREE_CORR_TOPM` is the width knob the paper sweeps (the candidate-width saturation
table, M ∈ {16, 64, 128, 256, full}). `DOMINOTREE_NODE_TOPK` is held at 8 in every reported
run, offline and serving alike, and is not swept. At the reported 16-token budget it is a
slack constraint by construction: cumulative log-probability is non-increasing along a path
and siblings enter the heap ranked, so a node's 9th-ranked child can only be popped after
all eight better siblings have been — meaning a single parent would have to own nine of the
sixteen nodes before the cap could bind at all.

## How it attaches to SGLang

Useful if you are reading the source, or porting this to a newer SGLang.

- **Registration.** `register_plugin()` registers both algorithms and rebinds the model
  registry's `DFlashDraftModel` entry to a subclass that adds Domino's `prefix_gru` and
  `embed_proj`. The released checkpoint's architecture string is `DFlashDraftModel`, and
  upstream's class has no correction head, so the entry has to point at the subclass.
- **The draft swap.** `DominoWorkerV2` subclasses SGLang's `DFlashWorkerV2` and wraps
  `forward_batch_generation` rather than copying it: around the `super()` call it captures
  the draft-block hidden state and rebinds the greedy draft sample to Domino's rollout,
  restoring both in a `finally`. The greedy sample is four lines in the middle of a
  ~490-line method; copying the method to patch its middle would be brittle across upstream
  releases.
- **DFLASH gating, EAGLE verify.** The spec class reports `is_dflash() = True`, because the
  draft genuinely is a DFLASH block draft and all the DFLASH-gated scaffolding — scheduler
  gates, request validation, server-argument setup — is what it needs. Verify shape is not
  gated by the algorithm anywhere in the scheduler; it follows from the `SpecInput` the
  worker emits. The worker emits an `EagleVerifyInput` carrying its own tree mask, so the
  tree verify runs without the plugin claiming `is_eagle()`, which would swap the draft
  path over to EAGLE's and break the DFLASH draft.
- **Accepted-path commit.** A tree accepts a non-contiguous set of nodes, while DFLASH's
  KV writer commits a dense prefix, so accepted nodes are compacted to the front of each
  request's block before that writer is reused unchanged.

## Copied code

The Domino GRU-correction head is copied from Domino's official SGLang fork, not
reimplemented. [`PROVENANCE.md`](PROVENANCE.md) lists what was copied, records the complete
set of edits, and ships a one-command proof that the copy is byte-identical to the
official source at a pinned commit.

## Benchmarks

[`benchmarks/`](benchmarks/) holds the three serving benchmarks behind the paper's serving
tables — single request (`bs1/`), goodput under concurrency (`concurrency/`), and long
context on HELMET (`helmet/`) — each comparing DominoTree against AR, DFlash, EAGLE-3, and
the Domino chain under identical flags. Start at [`benchmarks/README.md`](benchmarks/README.md),
which also covers the per-method admission cap that governs the concurrency comparison.
