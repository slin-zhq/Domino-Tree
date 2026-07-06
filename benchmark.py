#!/usr/bin/env python3
"""End-to-end DominoTree benchmark.

Methods:
  ar       : token-serial target autoregressive decoding (for speedup baselines)
  chain    : Domino's released corrected block chain
  marg@B   : marginal-tree DDTree analogue over Domino base logits
  cond@B   : DominoTree conditional best-first tree with per-node GRU correction

The public v1 intentionally ships only the pure-Python best-first tree backend.
CUDA/star, wave/condwave, beam, hybrid, and adaptive experimental paths from the
research branch are not exposed here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--draft-name-or-path", required=True)
    parser.add_argument("--domino-code", required=True, help="Path to the released Domino code directory.")
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--budgets", default="16", help="Comma-separated tree budgets for tree methods.")
    parser.add_argument("--node-topk", type=int, default=8)
    parser.add_argument("--corr-topm", type=int, default=64, help="DominoTree correction candidate set size; 0 = full vocab.")
    parser.add_argument(
        "--methods",
        default="chain,marg,dominotree",
        help="Subset of {ar,chain,marg,dominotree}; dominotree is the conditional draft tree (this paper's method), marg is the marginal-tree DDTree-analogue.",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--gpu-native-build",
        action="store_true",
        help=(
            "Opt-in: build the dominotree method's conditional children via the CUDA-graph "
            "node expander (dominotree_gpu.GraphNodeExpander) instead of the pure-Python "
            "children_fn. Must produce identical trees (same out_sig at T=0); the pure-Python "
            "path remains the default. DOMINOTREE_GPU_EAGER=1 disables graph capture (debug)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = {"ar", "chain", "marg", "dominotree"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}; allowed={sorted(allowed)}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    import dominotree
    import domino_adapter

    domino_adapter.add_path(args.domino_code)
    from dflash import is_domino_projector
    from model.utils import extract_context_feature, load_and_process_dataset, sample

    domino_benchmark_path = Path(args.domino_code) / "benchmark.py"
    spec = importlib.util.spec_from_file_location("domino_released_benchmark", domino_benchmark_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import Domino benchmark helper from {domino_benchmark_path}")
    domino_benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(domino_benchmark)
    load_draft_model_for_benchmark = domino_benchmark.load_draft_model_for_benchmark

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()
    draft = load_draft_model_for_benchmark(args.draft_name_or_path, "sdpa").to(device).eval()
    assert is_domino_projector(getattr(draft, "projector_type", None))

    prefix_len = int(getattr(draft, "pure_draft_prefix_len", 0))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    block_size = draft.block_size
    shift_label = bool(getattr(draft.config, "dflash_config", {}).get("shift_label", False))
    k_draft = block_size if shift_label else block_size - 1
    layer_ids = draft.target_layer_ids
    mask_token_id = draft.mask_token_id
    eos = tokenizer.eos_token_id

    graph_expander = None
    if args.gpu_native_build and "dominotree" in methods:
        import dominotree_gpu

        graph_expander = dominotree_gpu.GraphNodeExpander(
            target=target,
            draft=draft,
            k_draft=k_draft,
            prefix_len=prefix_len,
            node_topk=args.node_topk,
            corr_topm=args.corr_topm,
            device=device,
        )
        print(
            f"[gpu-native-build] node expander active: graphs={'on' if graph_expander.use_graphs else 'OFF (eager-static)'} "
            f"k_draft={k_draft} node_topk={args.node_topk} corr_topm={args.corr_topm} prefix_len={prefix_len}"
        )

    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    def cuda_t() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    @torch.inference_mode()
    def generate(input_ids, mode: str, budget: int, sample_idx: int, max_new_override=None):
        num_input = input_ids.shape[1]
        max_length = num_input + (max_new_override or args.max_new_tokens)
        extra = block_size + 1 if shift_label else block_size
        output_ids = torch.full((1, max_length + extra), mask_token_id, dtype=torch.long, device=device)
        position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
        past_kv, past_kv_draft = DynamicCache(), DynamicCache()

        out = target(
            input_ids,
            position_ids=position_ids[:, :num_input],
            past_key_values=past_kv,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        output_ids[:, :num_input] = input_ids
        output_ids[:, num_input : num_input + 1] = out.logits.argmax(dim=-1)
        target_hidden = extract_context_feature(out.hidden_states, layer_ids)

        start = num_input
        accepts: list[int] = []
        times = {"draft": 0.0, "build": 0.0, "verify": 0.0, "commit": 0.0}
        t_decode = cuda_t()

        while start < max_length:
            if mode == "ar":
                o = target(
                    output_ids[:, start : start + 1],
                    position_ids=position_ids[:, start : start + 1],
                    past_key_values=past_kv,
                    use_cache=True,
                )
                output_ids[0, start + 1] = sample(o.logits[:, -1:], args.temperature).view(-1)[0]
                start += 1
                accepts.append(1)
                if eos is not None and eos in output_ids[0, num_input:start].tolist():
                    break
                continue

            t0 = cuda_t()
            ph, base_logits, root_state = domino_adapter.draft_block(
                target=target,
                draft=draft,
                target_hidden=target_hidden,
                output_ids=output_ids,
                position_ids=position_ids,
                start=start,
                past_kv_draft=past_kv_draft,
                block_size=block_size,
                shift_label=shift_label,
            )
            times["draft"] += cuda_t() - t0

            if mode == "chain":
                acc_len, target_hidden, stage_ms = domino_adapter.verify_domino_chain(
                    target=target,
                    draft=draft,
                    sample=sample,
                    extract_context_feature=extract_context_feature,
                    output_ids=output_ids,
                    position_ids=position_ids,
                    start=start,
                    k_draft=k_draft,
                    prefix_len=prefix_len,
                    mask_token_id=mask_token_id,
                    base_logits=base_logits,
                    ph=ph,
                    past_kv=past_kv,
                    layer_ids=layer_ids,
                    temperature=args.temperature,
                    device=device,
                    cuda_t=cuda_t,
                )
                start += acc_len + 1
                for stage, value in stage_ms.items():
                    times[stage] += value
            else:
                t0 = cuda_t()
                if mode == "marg":
                    children_fn = domino_adapter.make_marginal_children_fn(base_logits, k_draft, args.node_topk)
                    root_state_for_tree = None
                elif graph_expander is not None:
                    graph_expander.begin_round(ph, base_logits)
                    children_fn = graph_expander.children_fn
                    root_state_for_tree = root_state
                else:
                    children_fn = domino_adapter.make_conditional_children_fn(
                        target=target,
                        draft=draft,
                        ph=ph,
                        base_logits=base_logits,
                        k_draft=k_draft,
                        prefix_len=prefix_len,
                        node_topk=args.node_topk,
                        corr_topm=args.corr_topm,
                        device=device,
                    )
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
                times["build"] += cuda_t() - t0

                t0 = cuda_t()
                tout = target(
                    ids,
                    position_ids=pos,
                    attention_mask=mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                )
                times["verify"] += cuda_t() - t0

                t0 = cuda_t()
                post = sample(tout.logits, args.temperature)[0].tolist()
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
                times["commit"] += cuda_t() - t0

            accepts.append(acc_len + 1)
            if eos is not None and eos in output_ids[0, num_input:start].tolist():
                break

        decode_time = cuda_t() - t_decode
        reported_end = min(start, max_length)
        num_output = int(reported_end - num_input)
        rounds = max(1, len(accepts))
        ms = {key: 1000.0 * value / rounds for key, value in times.items()}
        generated = output_ids[0, num_input:reported_end].tolist()
        out_sig = hashlib.md5(str(generated).encode()).hexdigest()[:12]
        return num_output, decode_time, accepts, ms, out_sig, generated[:128], generated

    runs = []
    for method in methods:
        if method in {"ar", "chain"}:
            runs.append((method, None))
        else:
            runs.extend((method, budget) for budget in budgets)

    # Warmup, matching the DDTree/CaDDTree/DFlash benchmark SOP: run each method
    # once on a short "Warmup" prompt so all CUDA kernels/caches are hot before any
    # timing. Every measured prompt is then warm from the start, so we take a plain
    # mean over all prompts with no warmup-row trimming (same as the references).
    warmup_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    warmup_ids = tokenizer.encode(warmup_text, return_tensors="pt").to(device)
    warmup_tokens = min(args.max_new_tokens, 16)
    for mode, budget in runs:
        generate(warmup_ids, mode, budget or 0, 0, max_new_override=warmup_tokens)

    records = []
    t_start = time.time()
    for mode, budget in runs:
        label = mode if budget is None else f"{mode}@{budget}"
        for sample_idx in range(len(dataset)):
            turns = dataset[sample_idx]["turns"]
            messages = []
            for turn_idx, user_content in enumerate(turns):
                messages.append({"role": "user", "content": user_content})
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
                torch.manual_seed(1000 + sample_idx * 100 + turn_idx)
                num_output, decode_time, accs, ms, out_sig, out_head, gen_full = generate(
                    input_ids,
                    mode,
                    budget or 0,
                    sample_idx,
                )
                rec = {
                    "method": label,
                    "mode": mode,
                    "budget": budget,
                    "sample_idx": sample_idx,
                    "turn_index": turn_idx,
                    "num_turns": len(turns),
                    "num_output": num_output,
                    "decode_time": decode_time,
                    "tps": num_output / decode_time if decode_time > 0 else 0.0,
                    "mean_accept": statistics.fmean(accs) if accs else 0.0,
                    "out_sig": out_sig,
                    "out_head": out_head,
                    "gpu_native_build": bool(args.gpu_native_build),
                }
                rec.update({f"ms_{key}": value for key, value in ms.items()})
                records.append(rec)
                messages.append({"role": "assistant", "content": tokenizer.decode(gen_full, skip_special_tokens=True)})
            if args.smoke and sample_idx >= 1:
                break

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n[benchmark] {len(records)} runs in {time.time() - t_start:.1f}s -> {args.out}")
    if args.smoke:
        print("[smoke] Same warmup + process as a full run, but only a couple of prompts, so TPS is")
        print("[smoke] noisy (small sample). Accept-length and out_sig are reliable; for stable,")
        print("[smoke] comparable throughput run the full protocol (no --smoke, more prompts).")
    print("\nPer-round stage times are in milliseconds (shown after the '|').")
    print(f"{'method':>13} {'TPS':>7} {'tok/rnd':>8} | {'draft':>6} {'build':>6} {'verify':>6} {'commit':>6} {'total':>6} {'n':>4}")
    print("-" * 72)
    for mode, budget in runs:
        label = mode if budget is None else f"{mode}@{budget}"
        rows = [rec for rec in records if rec["method"] == label]
        if not rows:
            continue
        avg = lambda key: statistics.fmean(rec.get(key, 0.0) for rec in rows)
        total = avg("ms_draft") + avg("ms_build") + avg("ms_verify") + avg("ms_commit")
        print(
            f"{label:>13} {avg('tps'):>7.1f} {avg('mean_accept'):>8.3f} | "
            f"{avg('ms_draft'):>6.2f} {avg('ms_build'):>6.2f} {avg('ms_verify'):>6.2f} "
            f"{avg('ms_commit'):>6.2f} {total:>6.2f} {len(rows):>4}"
        )


if __name__ == "__main__":
    main()
