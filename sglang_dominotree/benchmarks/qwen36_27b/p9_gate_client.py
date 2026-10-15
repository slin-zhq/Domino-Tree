#!/usr/bin/env python3
"""P9 losslessness gate — client. Runs a fixed prompt set against ONE running server
at T=0 and saves every output's token ids, so outputs can be compared token by token
against plain autoregressive decoding (AR) by p9_gate_compare.py.

Why long generations: a verify that mixes a rejected sibling's recurrent state into
the accepted path does not crash. It shifts the target's own logits slightly, so the
first visible symptom is an early divergence from AR that compounds with length.
Short outputs would hide it.

Prompts: the first turn of the first N samples per dataset, loaded with the paper
harness' own loader (bench_bs1.load_and_process_dataset + shuffle(seed=0)), and
formatted with the served model's chat template, enable_thinking=False.

  python p9_gate_client.py --port 30000 --label ar --model-path <target> \
      --bench-dir <plugin>/benchmarks/bs1 --out gate/ar.jsonl
"""

import argparse
import json
import sys
import time

import requests

DATASETS = ("gsm8k", "humaneval", "mt-bench", "math500")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--bench-dir", required=True, help="dir containing bench_bs1.py")
    ap.add_argument("--per-dataset", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.bench_dir)
    from bench_bs1 import load_and_process_dataset  # paper harness loader
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    base = f"http://127.0.0.1:{args.port}"
    requests.post(base + "/flush_cache", timeout=60)

    prompts = []
    for ds in DATASETS:
        data = load_and_process_dataset(ds).shuffle(seed=0).select(range(args.per_dataset))
        for i, sample in enumerate(data):
            messages = [{"role": "user", "content": sample["turns"][0]}]
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            prompts.append((f"{ds}/{i}", text))

    # One untimed warmup so graph/JIT first-use is not attributed to a prompt.
    requests.post(base + "/generate", json={
        "text": "Warmup", "sampling_params": {"temperature": 0.0, "max_new_tokens": 16}},
        timeout=600).raise_for_status()

    with open(args.out, "w") as f:
        for pid, text in prompts:
            t0 = time.perf_counter()
            r = requests.post(base + "/generate", json={
                "text": text,
                "sampling_params": {"temperature": 0.0,
                                    "max_new_tokens": args.max_new_tokens},
            }, timeout=1800)
            r.raise_for_status()
            out = r.json()
            meta = out.get("meta_info") or {}
            ids = out.get("output_ids")
            if ids is None:  # older response shape: re-tokenize the text
                ids = tok.encode(out["text"], add_special_tokens=False)
            rec = {
                "label": args.label,
                "prompt_id": pid,
                "output_ids": ids,
                "text": out["text"],
                "completion_tokens": int(meta.get("completion_tokens", len(ids))),
                "tau": float(meta.get("spec_accept_length", 1.0) or 1.0),
                "wall_s": time.perf_counter() - t0,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{args.label} {pid}: {rec['completion_tokens']} tok, "
                  f"tau={rec['tau']:.2f}, {rec['wall_s']:.1f}s", flush=True)
    print(f"CLIENT_DONE {args.label} {len(prompts)} prompts")


if __name__ == "__main__":
    main()
