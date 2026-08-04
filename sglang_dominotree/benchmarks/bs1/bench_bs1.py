#!/usr/bin/env python3
"""SGLang-serving bs=1 benchmark driver (paper Table 2 / Appendix E).

Reproduces this repo's HF-harness methodology (`benchmark.py` at the repo root,
referred to below as "ref_benchmark") EXACTLY, but drives a
running SGLang server over HTTP instead of running the model in-process:

  * dataset loading   : verbatim copy of the official Domino code's
                        model/utils.py::load_and_process_dataset, plus the
                        identical `shuffle(seed=0).select(range(max_samples))`.
  * prompt formatting : tokenizer.apply_chat_template(messages, tokenize=False,
                        add_generation_prompt=True, enable_thinking=False),
                        multi-turn exactly like ref_benchmark (append the
                        assistant response to `messages` after every turn).
  * warmup            : one "Warmup" chat prompt capped at 16 new tokens before
                        any timing, same as ref_benchmark's in-loop warmup.
  * per-prompt seed   : ref_benchmark calls torch.manual_seed(1000 +
                        sample_idx*100 + turn_idx) before each generate. The
                        serving analogue is sampling_params["sampling_seed"]
                        (this SGLang's per-request seed field; a bare "seed"
                        key would raise, because the server constructs
                        SamplingParams(**sampling_params)). If the server
                        rejects the field we fall back to unseeded sampling
                        with a loud warning (T=0 is greedy either way).
  * output schema     : same field names as ref_benchmark so the paper's
                        make_latex_table.py consumes the JSONL unchanged.

Differences forced by the serving API (uniform across methods, so
speedup-over-own-AR comparisons remain valid; flagged for the paper text):

  * decode_time is the wall-clock of the whole HTTP request, so unlike
    ref_benchmark it INCLUDES prefill and HTTP overhead (ref starts its timer
    after the initial prefill forward). meta_info.e2e_latency is recorded as
    an extra field for cross-checking.
  * Per-round stage times (ms_draft / ms_build / ms_verify / ms_commit) are
    NOT available from the serving API and are omitted. make_latex_table.py
    tolerates this: aggregate() reads them with .get(..., nan) and Table 1 /
    the pairwise tables never need them (only Table 2 does, which keeps using
    the paper's dedicated instrumented runs).
  * out_sig is the md5 of the returned TEXT (first 12 hex chars), not of the
    token-id list like ref_benchmark, because /generate returns detokenized
    text. Signatures are therefore comparable across SGLang runs but not
    against the HF-harness JSONLs. out_head is left [] for the same reason.
  * EOS handling: the server stops exactly at EOS; ref_benchmark stops at the
    end of the round in which EOS appears (may commit a few tokens past EOS).
    Sub-token-level difference, uniform across methods.

Usage (one method label per server; see RUN_SGLANG_BENCH.md):

  python bench_bs1.py --port 30000 --method dominotree@16 \\
      --dataset gsm8k --temperature 0.0 --model-path Qwen/Qwen3-4B \\
      --out raw_parts/dominotree@16/gsm8k_T0.0.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time

import requests


# --------------------------------------------------------------------------
# Dataset loading — verbatim copy of the official Domino code's
# model/utils.py::load_and_process_dataset (the exact function ref_benchmark
# imports), so prompt text is byte-identical to the paper harness.
# ASSUMPTION for the human to check on the GPU box: the `datasets` cache there
# must resolve the same HF datasets/revisions the paper runs used.
# --------------------------------------------------------------------------
def load_and_process_dataset(data_name: str):
    from datasets import Features, Sequence, Value, load_dataset

    # Math datasets
    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    # Chat datasets
    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(lambda x: {"formatted_input": (f"{x['instruction']}\n\nInput:\n{x['input']}" if x["input"] else x["instruction"])})
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})

    elif data_name == "mt-bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})

    # Coding datasets
    elif data_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})

    elif data_name == "livecodebench":
        base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
        allowed_files = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
        urls = [base + fn for fn in allowed_files]
        dataset = load_dataset("json", data_files={"test": urls})["test"]

        def format_lcb(doc):
            system_prompt = (
                "You are an expert Python programmer. You will be given a question (problem specification) "
                "and will generate a correct Python program that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            question_block = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                format_message = "### Format: Use the following code structure:"
                code_block = f"```python\n{doc['starter_code']}\n```"
            else:
                format_message = "### Format: Write your code in the following format:"
                code_block = "```python\n# YOUR CODE HERE\n```"
            answer_footer = "### Answer: (use the provided format with backticks)"
            return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"

        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]},
            remove_columns=dataset.column_names,
            features=target_features,
        )

    else:
        # Explicit error added here (the Domino original silently falls through
        # for unknown names); does not change behavior for any valid name.
        raise SystemExit(
            f"unknown dataset {data_name!r}; supported: gsm8k, math500, aime24, aime25, "
            "alpaca, mt-bench, humaneval, mbpp, livecodebench"
        )

    return dataset


# --------------------------------------------------------------------------
# Method label parsing: "dominotree@16" -> mode="dominotree", budget=16;
# "ar"/"chain" -> budget=None. Mirrors ref_benchmark's label = f"{mode}@{budget}".
# --------------------------------------------------------------------------
def split_method_label(label: str) -> tuple[str, int | None]:
    if "@" in label:
        mode, budget_str = label.rsplit("@", 1)
        try:
            return mode, int(budget_str)
        except ValueError:
            raise SystemExit(f"--method {label!r}: budget after '@' must be an integer")
    return label, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True, help="Port of the running SGLang server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host of the running SGLang server.")
    parser.add_argument(
        "--method",
        required=True,
        help="Label to stamp on every record, e.g. ar / chain / dominotree@16. "
        "Must describe the server that is actually running on --port; this driver "
        "cannot verify which speculative algorithm the server was launched with.",
    )
    parser.add_argument("--dataset", required=True, help="gsm8k|math500|aime25|humaneval|mbpp|livecodebench|mt-bench|alpaca")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--model-path", required=True, help="HF model path/name for the tokenizer + chat template (must match the served model).")
    parser.add_argument("--smoke", action="store_true", help="Only 2 prompts (same warmup + process as a full run).")
    parser.add_argument("--request-timeout", type=int, default=600, help="Per-request HTTP timeout in seconds.")
    return parser.parse_args()


def post_generate(base_url: str, text: str, sampling_params: dict, timeout_s: int) -> dict:
    try:
        resp = requests.post(
            base_url + "/generate",
            json={"text": text, "sampling_params": sampling_params},
            timeout=timeout_s,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"POST {base_url}/generate failed: {exc}") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"/generate returned HTTP {resp.status_code}: {resp.text[:500]}")
    out = resp.json()
    if not isinstance(out, dict):
        raise RuntimeError(f"expected a dict response for single-prompt /generate, got {type(out).__name__}")
    return out


def check_server(base_url: str, model_path: str, timeout_s: int) -> None:
    try:
        resp = requests.get(base_url + "/health", timeout=min(timeout_s, 30))
    except requests.RequestException as exc:
        raise SystemExit(
            f"cannot reach SGLang server at {base_url} ({exc}).\n"
            "Launch the server for this method first — see RUN_SGLANG_BENCH.md."
        )
    if resp.status_code != 200:
        raise SystemExit(f"{base_url}/health returned HTTP {resp.status_code}: {resp.text[:200]}")
    # Best-effort sanity check that the served model matches --model-path
    # (tokenizer/template mismatch would silently corrupt the methodology).
    try:
        info = requests.get(base_url + "/get_model_info", timeout=min(timeout_s, 30)).json()
        served = str(info.get("model_path", ""))
        if served and os.path.basename(served.rstrip("/")) != os.path.basename(str(model_path).rstrip("/")):
            print(f"[warn] served model {served!r} != --model-path {model_path!r}; chat template may not match the server.")
        else:
            print(f"[server] model: {served or '<unknown>'}")
    except Exception:
        print("[warn] could not query /get_model_info; skipping served-model sanity check.")


def flush_cache(base_url: str, timeout_s: int = 60) -> None:
    """GET /flush_cache so a measurement block starts with a cold prefix cache.

    Granularity matches the reference benchmarks -- per measurement CELL, not per
    prompt (verified 2026-07-29: DFlash `benchmark.py` flushes once per run before
    its warmup; Domino `benchmark_sglang.py` flushes once per concurrency cell).
    Flushing per prompt would be a STRICTER protocol than the baselines we are
    compared against and would understate our numbers relative to theirs.

    Warn, do not die: a failed flush degrades hygiene, it does not invalidate the run.
    """
    import urllib.request
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/flush_cache", timeout=timeout_s):
            pass
    except Exception as exc:  # pragma: no cover - network hiccup
        print(f"[warn] /flush_cache failed ({exc}); continuing (cache may be warm).")


def main() -> None:
    args = parse_args()
    # Convenience: accept the underscore spelling; canonical name (used by the
    # Domino loader and by make_latex_table.py's filenames) is "mt-bench".
    dataset_name = "mt-bench" if args.dataset == "mt_bench" else args.dataset
    mode, budget = split_method_label(args.method)
    base_url = f"http://{args.host}:{args.port}"

    check_server(base_url, args.model_path, args.request_timeout)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    dataset = load_and_process_dataset(dataset_name)
    # Identical subsetting to ref_benchmark: same seed, same order.
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    def base_sampling_params(max_new_tokens: int) -> dict:
        # top_p=1.0 / top_k=-1 pin the server to untruncated softmax sampling,
        # matching ref_benchmark's sample() (plain softmax multinomial at T>0,
        # argmax at T=0 — SGLang treats temperature 0.0 as greedy).
        return {
            "temperature": float(args.temperature),
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": int(max_new_tokens),
        }

    # This driver is invoked once per (dataset, temperature) cell, so a single flush
    # here IS the per-cell granularity the reference benchmarks use.
    flush_cache(base_url, args.request_timeout)

    # ---- Warmup, exactly like ref_benchmark: one "Warmup" chat prompt capped
    # at 16 new tokens, sent before any timing. Doubles as the probe for
    # per-request seed support ("sampling_seed" in this SGLang lineage).
    warmup_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    warmup_tokens = min(args.max_new_tokens, 16)
    seed_supported = True
    warmup_params = base_sampling_params(warmup_tokens)
    warmup_params["sampling_seed"] = 0
    try:
        post_generate(base_url, warmup_text, warmup_params, args.request_timeout)
    except RuntimeError as exc:
        msg = str(exc)
        if "sampling_seed" in msg or "unexpected keyword" in msg:
            seed_supported = False
            print("[warn] server rejected sampling_params['sampling_seed']; continuing WITHOUT per-prompt seeds.")
            print("[warn] T=0 runs are greedy and unaffected; T>0 runs will not be replayable bit-for-bit.")
            post_generate(base_url, warmup_text, base_sampling_params(warmup_tokens), args.request_timeout)
        else:
            raise SystemExit(f"warmup request failed: {msg}")
    print(f"[warmup] done (method={args.method}, seeds={'on' if seed_supported else 'OFF'})")

    # ---- Measured loop: mirrors ref_benchmark's per-prompt loop one-to-one.
    records = []
    t_start = time.time()
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
            sampling_params = base_sampling_params(args.max_new_tokens)
            # Serving analogue of ref_benchmark's
            # torch.manual_seed(1000 + sample_idx * 100 + turn_idx).
            seed = 1000 + sample_idx * 100 + turn_idx
            if seed_supported:
                sampling_params["sampling_seed"] = seed

            t0 = time.perf_counter()
            try:
                out = post_generate(base_url, text, sampling_params, args.request_timeout)
            except RuntimeError as exc:
                raise SystemExit(
                    f"generation failed at dataset={dataset_name} sample_idx={sample_idx} "
                    f"turn={turn_idx} method={args.method}: {exc}"
                )
            decode_time = time.perf_counter() - t0

            out_text = out.get("text")
            meta = out.get("meta_info") or {}
            if out_text is None or "completion_tokens" not in meta:
                raise SystemExit(
                    f"unexpected /generate response shape at sample_idx={sample_idx} turn={turn_idx}: "
                    f"keys={sorted(out.keys())}, meta keys={sorted(meta.keys())}"
                )
            num_output = int(meta["completion_tokens"])
            # spec_accept_length = completion_tokens / verify_ct, bonus token
            # included — the same tokens-per-round convention as ref_benchmark's
            # mean(accepts) where each round appends acc_len + 1. Absent on a
            # non-speculative (AR) server -> 1.0, matching ref's AR rounds.
            mean_accept = float(meta.get("spec_accept_length", 1.0) or 1.0)

            rec = {
                "method": args.method,
                "mode": mode,
                "budget": budget,
                "sample_idx": sample_idx,
                "turn_index": turn_idx,
                "num_turns": len(turns),
                "num_output": num_output,
                "decode_time": decode_time,
                "tps": num_output / decode_time if decode_time > 0 else 0.0,
                "mean_accept": mean_accept,
                # md5 of the returned text (serving harness convention; not
                # comparable to the HF harness' token-id out_sig).
                "out_sig": hashlib.md5(out_text.encode()).hexdigest()[:12],
                "out_head": [],
                # The serving path always builds trees with the GPU-native
                # builder (plugin default), hence stamped true.
                "gpu_native_build": True,
                # ms_draft/ms_build/ms_verify/ms_commit intentionally omitted:
                # not observable over the serving API. Table 1 / pairwise
                # aggregation never reads them; Table 2 keeps using the paper's
                # dedicated instrumented HF-harness runs.
                # ---- extra provenance fields (ignored by make_latex_table.py):
                "dataset": dataset_name,
                "temperature": float(args.temperature),
                "sampling_seed": seed if seed_supported else None,
                "e2e_latency": float(meta["e2e_latency"]) if "e2e_latency" in meta else None,
                "harness": "sglang-serving-bs1",
            }
            records.append(rec)
            # Multi-turn: feed the assistant response back, exactly like
            # ref_benchmark (which appends tokenizer.decode(gen_full,
            # skip_special_tokens=True); the server's `text` is detokenized
            # with skip_special_tokens=True by default).
            messages.append({"role": "assistant", "content": out_text})
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
    print(f"\n{'method':>15} {'TPS':>7} {'tok/rnd':>8} {'n':>4}")
    print("-" * 40)
    for label in dict.fromkeys(rec["method"] for rec in records):
        rows = [rec for rec in records if rec["method"] == label]
        avg = lambda key: statistics.fmean(rec[key] for rec in rows)
        print(f"{label:>15} {avg('tps'):>7.1f} {avg('mean_accept'):>8.3f} {len(rows):>4}")


if __name__ == "__main__":
    main()
