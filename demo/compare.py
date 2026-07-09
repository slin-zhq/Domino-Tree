#!/usr/bin/env python3
"""Record several methods on one prompt, then replay them side by side.

This is the single entry point for the live comparison demo. It records each
method (each driven by its own code; see ``record.py``), spreading the recordings
across whatever GPUs are visible, then hands the resulting casts to ``play.py``.

    # one typed prompt, three panes, auto GPU placement
    python compare.py --methods ar,domino,dominotree \
        --prompt "A cat eats nine sausages in 30 minutes. Compute the average time."

    # sample from a dataset instead
    python compare.py --methods ar,domino,dominotree --dataset gsm8k --sample-index 3

Panes are NOT limited by GPU count: recording is sequential per GPU (parallel
across GPUs), and replay uses no GPU at all. Env vars MODEL_PATH / DRAFT_PATH /
DOMINO_CODE are read as defaults (see README).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def detect_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        idx = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return idx
    except Exception:
        return []


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", default="ar,domino,dominotree",
                   help="Comma-separated subset of ar,domino,marg,dominotree (left-to-right pane order).")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--prompt", default=None)
    src.add_argument("--dataset", default=None)
    p.add_argument("--sample-index", type=int, default=0)

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--budget", type=int, default=16)
    p.add_argument("--corr-topm", default="64", help="Integer or 'full_vocab'.")
    p.add_argument("--node-topk", type=int, default=8)

    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    p.add_argument("--draft-path", default=os.environ.get("DRAFT_PATH"))
    p.add_argument("--domino-code", default=os.environ.get("DOMINO_CODE"))
    p.add_argument("--gpus", default="auto",
                   help="'auto' (detect), 'cpu', or comma list of GPU indices e.g. 0,1.")
    p.add_argument("--out-dir", default=str(HERE / "runs" / "last"))

    p.add_argument("--eager-domino", action="store_true",
                   help="Run the domino pane eager (default: Domino's own --use-graph CUDA graph runner).")
    p.add_argument("--python-builder", action="store_true",
                   help="Run dominotree with the pure-Python builder (default: GPU-native builder).")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--baseline", default=None)
    p.add_argument("--record-only", action="store_true", help="Record casts but do not replay.")
    p.add_argument("--play-only", action="store_true", help="Replay existing casts in --out-dir; skip recording.")
    p.add_argument("--dry-run", action="store_true", help="Synthetic casts (no torch/GPU); good for UI checks.")
    return p.parse_args()


def record_one(method, device_env, args, out_path):
    cmd = [sys.executable, str(HERE / "record.py"),
           "--method", method, "--out", out_path,
           "--temperature", str(args.temperature),
           "--max-new-tokens", str(args.max_new_tokens),
           "--budget", str(args.budget), "--corr-topm", str(args.corr_topm),
           "--node-topk", str(args.node_topk), "--device", "cuda:0"]
    if method == "domino" and not args.eager_domino:
        cmd.append("--use-graph")
    if method == "dominotree" and not args.python_builder:
        cmd.append("--gpu-native-build")
    if args.prompt is not None:
        cmd += ["--prompt", args.prompt]
    elif args.dataset is not None:
        cmd += ["--dataset", args.dataset, "--sample-index", str(args.sample_index)]
    for flag, val in (("--model-path", args.model_path), ("--draft-path", args.draft_path),
                      ("--domino-code", args.domino_code)):
        if val:
            cmd += [flag, val]
    if args.dry_run:
        cmd.append("--dry-run")
    env = dict(os.environ)
    if device_env is not None:
        env["CUDA_VISIBLE_DEVICES"] = device_env
    r = subprocess.run(cmd, env=env)
    return method, out_path, r.returncode


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cast_paths = [str(out_dir / f"{i:02d}_{m}.json") for i, m in enumerate(methods)]

    if not args.play_only:
        if args.dry_run or args.gpus == "cpu":
            gpus = [None] * len(methods)
        elif args.gpus == "auto":
            detected = detect_gpus()
            gpus = [detected[i % len(detected)] if detected else None for i in range(len(methods))]
        else:
            ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
            gpus = [ids[i % len(ids)] for i in range(len(methods))]

        # Bucket methods by GPU: parallel across GPUs, sequential within a GPU.
        buckets = {}
        for m, cast, g in zip(methods, cast_paths, gpus):
            buckets.setdefault(g, []).append((m, cast))

        print(f"[compare] recording {methods} on gpus={gpus} (dry_run={args.dry_run})")

        def run_bucket(gpu, items):
            for m, cast in items:
                method, path, rc = record_one(m, gpu, args, cast)
                if rc != 0:
                    print(f"[compare] ERROR recording {method} (exit {rc})", file=sys.stderr)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(buckets))) as ex:
            list(ex.map(lambda kv: run_bucket(kv[0], kv[1]), buckets.items()))

    missing = [p for p in cast_paths if not Path(p).exists()]
    if missing:
        print(f"[compare] missing casts, cannot replay: {missing}", file=sys.stderr)
        sys.exit(1)
    if args.record_only:
        print(f"[compare] casts written to {out_dir}")
        return

    play_cmd = [sys.executable, str(HERE / "play.py"), *cast_paths, "--speed", str(args.speed)]
    if args.baseline:
        play_cmd += ["--baseline", args.baseline]
    subprocess.run(play_cmd)


if __name__ == "__main__":
    main()
