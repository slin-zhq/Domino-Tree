#!/usr/bin/env python3
"""Record one method's decode run into a replayable *cast* file.

Each method is driven by its OWN code, never a reimplementation:

  * ``ar`` / ``domino`` -> the released Domino drafter's
    ``dflash.DFlashDraftModel.spec_generate`` (``block_size=1`` for ``ar``,
    the drafter's block size for ``domino``), imported unmodified from the
    ``--domino-code`` clone.
  * ``dominotree``       -> this repository's ``dominotree`` + ``domino_adapter``
    best-first conditional tree (the same calls ``benchmark.py`` makes).

The cast is a small JSON: a list of per-round *reveal events* (time from the
start of decoding, number of tokens committed that round, and the decoded text
chunk) plus summary metrics (tps, rounds, mean accepted length, ttft). One or
more casts are replayed side by side on a shared clock by ``play.py``.

Why record-then-replay (and not live streaming): recording runs each method's
code with zero terminal I/O inside the timed decode loop, so the measured TPS is
clean. The animation is a separate step. Domino's released ``spec_generate`` also
exposes no token-streaming hook, so replay is the only way to show its pane
decoding without editing Domino's source.

Timing fidelity:
  * ``dominotree`` records *real* per-round wall-clock timestamps (we own the loop).
  * ``ar`` / ``domino`` come from ``spec_generate``, which returns aggregate decode
    time plus per-round ``acceptance_lengths`` but no per-round timestamps. We
    reconstruct events by spreading the measured decode time uniformly across
    rounds (the chain's per-round target forward is a fixed-size op, so per-round
    wall time is ~constant) and revealing that round's accepted tokens as a burst.
    Total decode time and total token count match the real measurement exactly;
    only the intra-run pacing is reconstructed.

``--dry-run`` writes a synthetic cast (no torch / no GPU) so the replay UI can be
exercised offline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

METHODS = ("ar", "domino", "marg", "dominotree")

DEFAULT_LABELS = {
    "ar": "Autoregressive",
    "domino": "Domino (chain)",
    "marg": "Marg tree",
    "dominotree": "DominoTree",
}
DEFAULT_COLORS = {
    "ar": "red",
    "domino": "green",
    "marg": "yellow",
    "dominotree": "cyan",
}


def parse_corr_topm(value: str) -> int:
    """``full_vocab`` (or ``full`` / ``0``) -> 0 (full-vocab correction)."""
    s = str(value).strip().lower()
    if s in {"full_vocab", "fullvocab", "full", "0"}:
        return 0
    n = int(s)
    if n < 0:
        raise argparse.ArgumentTypeError("--corr-topm must be >= 0 or 'full_vocab'")
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--out", required=True, help="Output cast JSON path.")
    p.add_argument("--label", default=None, help="Pane title (defaults per method).")
    p.add_argument("--color", default=None, help="Pane color (rich color name).")

    # Prompt source: either an explicit prompt or a dataset sample.
    src = p.add_mutually_exclusive_group()
    src.add_argument("--prompt", default=None, help="Raw user prompt text.")
    src.add_argument("--dataset", default=None, help="Dataset name (e.g. gsm8k, humaneval, mt_bench).")
    p.add_argument("--sample-index", type=int, default=0, help="Which dataset sample to use (after seed-0 shuffle).")

    # Decode knobs (exposed to the user per request).
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--budget", type=int, default=16, help="Tree node budget (marg/dominotree).")
    p.add_argument("--corr-topm", type=parse_corr_topm, default=64,
                   help="DominoTree correction candidate width; integer or 'full_vocab' (= full-vocab correction).")
    p.add_argument("--node-topk", type=int, default=8, help="Children per tree node (marg/dominotree).")
    p.add_argument("--gpu-native-build", action="store_true",
                   help="dominotree: build the tree with the CUDA-graph GraphNodeExpander (paper default) "
                        "instead of the pure-Python children_fn. Produces identical trees, lower build cost.")
    p.add_argument("--use-graph", action="store_true",
                   help="domino: run spec_generate with Domino's own DraftCorrectionGraphRunner "
                        "(the released --use-graph path). No effect on ar.")

    # Models / environment (all overridable; also read from env in run scripts).
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    p.add_argument("--draft-path", default=os.environ.get("DRAFT_PATH"))
    p.add_argument("--domino-code", default=os.environ.get("DOMINO_CODE"))
    p.add_argument("--device", default=None, help="e.g. cuda:0 (default: cuda if available else cpu).")

    p.add_argument("--no-warmup", action="store_true", help="Skip the untimed warmup generation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="Emit a synthetic cast with no torch / GPU.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Cast assembly
# --------------------------------------------------------------------------- #

def build_events(generated_ids, round_token_counts, round_times, tokenizer):
    """Turn (token ids, per-round counts, per-round cumulative times) into reveal events.

    ``round_times[i]`` is the wall-clock time (s, from the start of decoding) at
    which round ``i`` completes. Text is revealed by decoding cumulative token
    prefixes and diffing the strings, which keeps multi-token characters intact.
    """
    events = []
    cum = 0
    prev_text = ""
    n_total = len(generated_ids)
    for i, n in enumerate(round_token_counts):
        cum = min(cum + int(n), n_total)
        text = tokenizer.decode(generated_ids[:cum], skip_special_tokens=True)
        chunk = text[len(prev_text):]
        events.append({
            "t": round(float(round_times[i]), 6),
            "round": i + 1,
            "tokens": int(n),
            "cum_tokens": cum,
            "chunk": chunk,
        })
        prev_text = text
        if cum >= n_total:
            break
    return events


def make_cast(method, label, color, prompt_preview, config, num_input_tokens,
              ttft_s, decode_time_s, generated_ids, round_token_counts, round_times, tokenizer):
    events = build_events(generated_ids, round_token_counts, round_times, tokenizer)
    num_output = len(generated_ids)
    rounds = len(round_token_counts)
    tps = num_output / decode_time_s if decode_time_s > 0 else 0.0
    mean_accept = (num_output / rounds) if rounds else 0.0
    return {
        "schema": "dominotree-demo-cast/v1",
        "method": method,
        "label": label,
        "color": color,
        "prompt_preview": prompt_preview,
        "config": config,
        "num_input_tokens": int(num_input_tokens),
        "ttft_s": round(float(ttft_s), 6),
        "summary": {
            "num_output_tokens": int(num_output),
            "decode_time_s": round(float(decode_time_s), 6),
            "tps": round(float(tps), 4),
            "rounds": int(rounds),
            "mean_accept": round(float(mean_accept), 4),
        },
        "events": events,
    }


def write_cast(cast, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(cast, f, indent=2)
    s = cast["summary"]
    print(f"[record] {cast['method']:>10} -> {out}  "
          f"({s['num_output_tokens']} tok, {s['rounds']} rounds, "
          f"tps={s['tps']:.1f}, tau={s['mean_accept']:.2f})")


# --------------------------------------------------------------------------- #
# Dry-run synthetic cast (no torch)
# --------------------------------------------------------------------------- #

def dry_run_cast(args):
    """A plausible synthetic cast so play.py can be tested without a GPU.

    Relative speeds are illustrative only: ar < domino < marg < dominotree.
    """
    import random
    rng = random.Random(args.seed + hash(args.method) % 1000)
    prompt = args.prompt or f"[{args.dataset or 'gsm8k'} sample #{args.sample_index}]"
    words = ("We are given a problem that asks us to reason carefully step by step "
             "and then put the final answer in a box . Let us begin by restating "
             "what is known and what must be found , then proceed").split()
    # Per-method throughput profile (tokens/sec) and mean accepted length.
    profile = {
        "ar":         (95.0, 1.0),
        "domino":     (520.0, 7.2),
        "marg":       (540.0, 6.9),
        "dominotree": (585.0, 8.0),
    }[args.method]
    tps, tau = profile
    n_out = min(args.max_new_tokens, 180)
    decode_time = n_out / tps
    # Build rounds of ~tau tokens each.
    round_counts, round_times = [], []
    produced, t = 0, 0.0
    per_round_time = decode_time / max(1, (n_out / tau))
    gen_words = []
    while produced < n_out:
        k = max(1, int(round(rng.gauss(tau, tau * 0.2))))
        k = min(k, n_out - produced)
        for _ in range(k):
            gen_words.append(words[len(gen_words) % len(words)])
        produced += k
        t += per_round_time * (k / tau)
        round_counts.append(k)
        round_times.append(t)
    decode_time = round_times[-1]

    class _WordTok:
        def decode(self, ids, skip_special_tokens=True):
            return " ".join(gen_words[:len(ids)])
    generated_ids = list(range(n_out))
    cast = make_cast(
        method=args.method,
        label=args.label or DEFAULT_LABELS[args.method],
        color=args.color or DEFAULT_COLORS[args.method],
        prompt_preview=prompt[:200],
        config={"temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
                "budget": args.budget, "corr_topm": args.corr_topm, "node_topk": args.node_topk,
                "dry_run": True},
        num_input_tokens=len(prompt.split()),
        ttft_s=0.03,
        decode_time_s=decode_time,
        generated_ids=generated_ids,
        round_token_counts=round_counts,
        round_times=round_times,
        tokenizer=_WordTok(),
    )
    return cast


# --------------------------------------------------------------------------- #
# Real recording (torch)
# --------------------------------------------------------------------------- #

def _load_everything(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.domino_code:
        raise SystemExit("--domino-code (or $DOMINO_CODE) is required for a real run.")
    if not args.model_path or not args.draft_path:
        raise SystemExit("--model-path/--draft-path (or $MODEL_PATH/$DRAFT_PATH) are required.")

    sys.path.insert(0, args.domino_code)
    import importlib.util
    from dflash import is_domino_projector  # noqa: F401  (validated below)
    from model.utils import extract_context_feature, load_and_process_dataset, sample  # noqa: F401

    dbench_path = Path(args.domino_code) / "benchmark.py"
    spec = importlib.util.spec_from_file_location("domino_released_benchmark", dbench_path)
    dbench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dbench)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    target = AutoModelForCausalLM.from_pretrained(
        args.model_path, attn_implementation="sdpa", dtype=torch.bfloat16,
    ).to(device).eval()
    draft = dbench.load_draft_model_for_benchmark(args.draft_path, "sdpa").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    return torch, device, target, draft, tokenizer, extract_context_feature, sample, load_and_process_dataset


def _make_input_ids(args, tokenizer, load_and_process_dataset, device):
    if args.prompt is not None:
        user = args.prompt
    else:
        ds = load_and_process_dataset(args.dataset or "gsm8k")
        if len(ds) > 1:
            ds = ds.shuffle(seed=0)
        idx = max(0, min(args.sample_index, len(ds) - 1))
        user = ds[idx]["turns"][0]
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    return input_ids, user


def record_spec_generate(args, torch, device, target, draft, tokenizer, sample):
    """ar / domino via the released Domino ``spec_generate`` (unmodified).

    With ``--use-graph`` (domino only), Domino's own ``DraftCorrectionGraphRunner``
    (its released kernel) is built and passed in, matching Domino's
    ``benchmark.py --use-graph`` path. Nothing here is a reimplementation.
    """
    block = 1 if args.method == "ar" else int(draft.block_size)
    eos = tokenizer.eos_token_id

    graph_runner = None
    if args.use_graph and args.method == "domino":
        from kernel.domino import DraftCorrectionGraphRunner  # Domino's own kernel

        shift_label = bool(getattr(draft.config, "dflash_config", {}).get("shift_label", False))
        prefix_len = int(getattr(draft, "pure_draft_prefix_len", 0))
        k = block if shift_label else block - 1
        graph_runner = DraftCorrectionGraphRunner(
            draft_model=draft, target_model=target, batch_size=1, steps=k - prefix_len,
            hidden_dim=int(target.lm_head.weight.shape[1]),
            gru_hidden_dim=draft.prefix_gru.hidden_size,
            vocab_size=int(target.lm_head.weight.shape[0]),
            prefix_token_count=1 + prefix_len, device=device,
        )
        print(f"[record] domino: using Domino's DraftCorrectionGraphRunner (--use-graph), steps={k - prefix_len}")

    def run(ids, max_new):
        return draft.spec_generate(
            target=target, input_ids=ids, max_new_tokens=max_new,
            block_size=block, stop_token_ids=[eos] if eos is not None else None,
            temperature=args.temperature, graph_runner=graph_runner, use_bias=True, return_dict=True,
        )

    return block, run, eos


def record_dominotree(args, torch, device, target, draft, tokenizer, extract_context_feature, sample):
    """dominotree / marg via this repo's tree, capturing real per-round timings.

    Mirrors benchmark.py's tree path exactly (same domino_adapter / dominotree
    calls); the only addition is recording a wall-clock timestamp and the accepted
    token ids per round.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import dominotree
    import domino_adapter
    from transformers import DynamicCache

    prefix_len = int(getattr(draft, "pure_draft_prefix_len", 0))
    block_size = draft.block_size
    shift_label = bool(getattr(draft.config, "dflash_config", {}).get("shift_label", False))
    k_draft = block_size if shift_label else block_size - 1
    layer_ids = draft.target_layer_ids
    mask_token_id = draft.mask_token_id
    eos = tokenizer.eos_token_id
    budget = args.budget

    expander = None
    if args.method == "dominotree" and args.gpu_native_build:
        import dominotree_gpu
        expander = dominotree_gpu.GraphNodeExpander(
            target=target, draft=draft, k_draft=k_draft, prefix_len=prefix_len,
            node_topk=args.node_topk, corr_topm=args.corr_topm, device=device,
        )
        print(f"[record] dominotree: GPU-native builder "
              f"graphs={'on' if expander.use_graphs else 'OFF (eager-static)'}")

    def cuda_t():
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    @torch.inference_mode()
    def run(input_ids, max_new):
        num_input = input_ids.shape[1]
        max_length = num_input + max_new
        extra = block_size + 1 if shift_label else block_size
        output_ids = torch.full((1, max_length + extra), mask_token_id, dtype=torch.long, device=device)
        position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
        past_kv, past_kv_draft = DynamicCache(), DynamicCache()

        t_pref = cuda_t()
        out = target(input_ids, position_ids=position_ids[:, :num_input], past_key_values=past_kv,
                     use_cache=True, logits_to_keep=1, output_hidden_states=True)
        output_ids[:, :num_input] = input_ids
        output_ids[:, num_input:num_input + 1] = out.logits.argmax(dim=-1)
        target_hidden = extract_context_feature(out.hidden_states, layer_ids)
        ttft = cuda_t() - t_pref

        start = num_input
        round_counts, round_times = [], []
        t_decode = cuda_t()
        while start < max_length:
            prev = start
            ph, base_logits, root_state = domino_adapter.draft_block(
                target=target, draft=draft, target_hidden=target_hidden, output_ids=output_ids,
                position_ids=position_ids, start=start, past_kv_draft=past_kv_draft,
                block_size=block_size, shift_label=shift_label)
            if args.method == "marg":
                children_fn = domino_adapter.make_marginal_children_fn(
                    base_logits, k_draft, args.node_topk,
                    sample_draft=False, temperature=args.temperature)
                root_state_for_tree = None
            elif expander is not None:
                expander.begin_round(ph, base_logits)
                children_fn = expander.children_fn
                root_state_for_tree = root_state
            else:
                children_fn = domino_adapter.make_conditional_children_fn(
                    target=target, draft=draft, ph=ph, base_logits=base_logits, k_draft=k_draft,
                    prefix_len=prefix_len, node_topk=args.node_topk, corr_topm=args.corr_topm,
                    device=device, sample_draft=False, temperature=args.temperature)
                root_state_for_tree = root_state
            nodes = dominotree.build_best_first_tree(children_fn, root_state_for_tree, budget, k_draft)

            tree_len = 1 + len(nodes)
            ids = torch.empty((1, tree_len), dtype=torch.long, device=device)
            ids[0, 0] = output_ids[0, start].item()
            for i, node in enumerate(nodes):
                ids[0, 1 + i] = node.token
            pos = torch.tensor([dominotree.position_ids(nodes, start)], device=device)
            rows = dominotree.build_attention_rows(nodes)
            mask = torch.full((1, 1, tree_len, start + tree_len), float("-inf"), device=device, dtype=target.dtype)
            mask[..., :start] = 0.0
            row_idx = [r for r, cols in enumerate(rows) for _ in cols]
            col_idx = [start + c for cols in rows for c in cols]
            mask[0, 0, row_idx, col_idx] = 0.0

            tout = target(ids, position_ids=pos, attention_mask=mask, past_key_values=past_kv,
                          use_cache=True, output_hidden_states=True)
            from model.utils import sample as _sample  # noqa
            post = _sample(tout.logits, args.temperature)[0].tolist()
            acc_len, path = dominotree.longest_accepted_path(nodes, post[0], post[1:])
            accepted_tokens = [nodes[i].token for i in path]
            bonus = post[0] if not path else post[1 + path[-1]]
            for j, token in enumerate(accepted_tokens):
                output_ids[0, start + 1 + j] = token
            output_ids[0, start + 1 + acc_len] = bonus
            flat = [0] + [1 + i for i in path]
            keep = list(range(start)) + [start + f for f in flat]
            domino_adapter.cache_gather(past_kv, keep, device)
            th = extract_context_feature(tout.hidden_states, layer_ids)
            target_hidden = th[:, flat, :]
            start += acc_len + 1

            round_counts.append(start - prev)
            round_times.append(cuda_t() - t_decode)
            if eos is not None and eos in output_ids[0, num_input:start].tolist():
                break

        decode_time = round_times[-1] if round_times else (cuda_t() - t_decode)
        reported_end = min(start, max_length)
        generated = output_ids[0, num_input:reported_end].tolist()
        return generated, round_counts, round_times, ttft, decode_time, num_input

    return run


def main():
    args = parse_args()
    label = args.label or DEFAULT_LABELS[args.method]
    color = args.color or DEFAULT_COLORS[args.method]

    if args.dry_run:
        write_cast(dry_run_cast(args), args.out)
        return

    (torch, device, target, draft, tokenizer,
     extract_context_feature, sample, load_and_process_dataset) = _load_everything(args)
    input_ids, user = _make_input_ids(args, tokenizer, load_and_process_dataset, device)
    num_input = input_ids.shape[1]
    config = {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
              "budget": args.budget, "corr_topm": args.corr_topm, "node_topk": args.node_topk,
              "block_size": int(draft.block_size), "device": str(device), "method": args.method,
              "gpu_native_build": bool(args.gpu_native_build), "use_graph": bool(args.use_graph)}

    warmup_tokens = min(args.max_new_tokens, 16)
    warmup_ids = tokenizer.encode(
        tokenizer.apply_chat_template([{"role": "user", "content": "Warmup"}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False),
        return_tensors="pt").to(device)

    if args.method in ("ar", "domino"):
        block, run, eos = record_spec_generate(args, torch, device, target, draft, tokenizer, sample)
        if not args.no_warmup:
            run(warmup_ids, warmup_tokens)
        r = run(input_ids, args.max_new_tokens)
        generated = r.output_ids[0, r.num_input_tokens:].tolist()
        counts = [int(x) for x in r.acceptance_lengths] or [len(generated)]
        # Uniform per-round pacing reconstructed from measured decode time.
        decode_time = float(r.time_per_output_token) * int(r.num_output_tokens)
        n_rounds = len(counts)
        round_times = [(i + 1) / n_rounds * decode_time for i in range(n_rounds)]
        ttft = float(r.time_to_first_token)
        num_input = int(r.num_input_tokens)
    else:
        run = record_dominotree(args, torch, device, target, draft, tokenizer,
                                extract_context_feature, sample)
        if not args.no_warmup:
            run(warmup_ids, warmup_tokens)
        generated, counts, round_times, ttft, decode_time, num_input = run(input_ids, args.max_new_tokens)

    cast = make_cast(
        method=args.method, label=label, color=color, prompt_preview=user[:200], config=config,
        num_input_tokens=num_input, ttft_s=ttft, decode_time_s=decode_time,
        generated_ids=generated, round_token_counts=counts, round_times=round_times, tokenizer=tokenizer)
    write_cast(cast, args.out)


if __name__ == "__main__":
    main()
