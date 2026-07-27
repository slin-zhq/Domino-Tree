#!/usr/bin/env python3
"""Full-dataset concurrency (goodput) driver for the paper's serving
results (Table 3 / Appendix E).  Method-AGNOSTIC: it only hits a *running* SGLang server's `/generate`;
the caller labels which speculative method that server is running via `--method`.
This driver launches nothing and imports no torch/CUDA (safe to `py_compile` and
`--dry-run` on a laptop).

WHAT IT MEASURES
----------------
For each dataset it sweeps the *offered* concurrency (requests kept in flight)
over the FIRST-N prompts of real datasets (gsm8k:128, mbpp:128, mt-bench:80),
reporting aggregate output tok/s and mean accepted length per cell.

NOTE ON READING THE RESULTS: the server admits at most
`--max-running-requests` requests simultaneously; offered requests beyond that
queue.  A cross-method comparison is therefore only like-for-like at
concurrencies within EVERY compared method's cap -- past a method's cap its
goodput plateaus because it is running its cap, not the offered load.

METHODOLOGY (matches Domino's own serving driver,
`benchmark_sglang_tasks.py::_run_bench_requests` in the Domino release)
-----------------------------------------------------------------------
For every (dataset, concurrency) cell:
  1. `/flush_cache`  (GET, with retries) so radix/KV cache starts cold.
  2. WARMUP: issue `warmup_requests * concurrency` requests at
     `warmup_max_new_tokens`, `max_workers=concurrency`; output DISCARDED. This
     drives graphs/kernels to steady state before timing.
  3. SKIP-FIRST: issue `skip_first` more requests at `max_new_tokens`; still sent
     (so the server's request stream is identical) but DROPPED from the timed
     window and metrics.
  4. TIMED: submit ALL N timed prompts to a `ThreadPoolExecutor(max_workers=
     concurrency)` and time the whole batch with `time.perf_counter()`.  N is
     FIXED per dataset (e.g. 128) regardless of concurrency, so every level is
     measured on the SAME prompt set; concurrency only caps in-flight requests.
     Goodput metric:
         tps        = sum(meta_info.completion_tokens) / wall_seconds
         mean_accept = mean(meta_info.spec_accept_length)   (over timed reqs)
     These are Domino's `output_toks_per_s` and mean `spec_accept_length`.

PROMPT SELECTION (deliberate, documented deviation from Domino — see RUNBOOK)
----------------------------------------------------------------------------
Domino's ref concurrency driver takes the first-N prompts in NATURAL dataset
order.  We instead reuse OUR bs=1 driver's subsetting so the bs=1 and concurrency
tables share identical prompts:
      timed = load_and_process_dataset(name).shuffle(seed=0).select(range(N))
`load_and_process_dataset` is IMPORTED from the bs=1 driver `bench_bs1.py`
(the verbatim copy of Domino's `model/utils.py`), and each prompt is formatted
with the identical chat template call
      apply_chat_template([{"role":"user","content": turns[0]}],
                          tokenize=False, add_generation_prompt=True,
                          enable_thinking=False)
Because `shuffle(seed=0)` is deterministic, the bs=1 set (N=20) is exactly the
PREFIX of the concurrency set (N=128), so the two tables share prompts by
construction.  Warmup/skip prompts are drawn from the REMAINDER of the shuffled
dataset (indices [N:]), i.e. DISJOINT from the timed set (matching Domino, so the
radix cache is not pre-warmed with timed prompts); if the dataset is exhausted we
fall back to cycling the timed prompts (`warmup_source=timed_reuse`).

Concurrency requests are single-shot FIRST-TURN only (`turns[0]`), like Domino's
concurrency driver — multi-turn is meaningless for a pool of independent
requests.  For every single-turn dataset (gsm8k/mbpp/... all have one turn) this
is byte-identical to the bs=1 prompt; for mt-bench it is the bs=1 first turn.

Usage (server must already be running; this driver does NOT launch it):
  python conc_bench_datasets.py --host 127.0.0.1 --port 31650 \
      --method dflash --model-label qwen3-4b --model-path Qwen/Qwen3-4B \
      --tasks gsm8k:128,mbpp:128,mt-bench:80 --concurrencies 1,2,4,8,16,32 \
      --max-new-tokens 2048 --warmup-requests 3 --warmup-max-new-tokens 1024 \
      --skip-first 1 --out-jsonl out/conc_dflash_4b.jsonl --out-md out/conc_dflash_4b.md

Self-test (no server, no GPU, no datasets/transformers needed):
  python selftest_conc.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# `requests` is only needed for real (non-dry-run) runs; import lazily so the
# module still imports (and --dry-run / self-test still run) if it is absent.
try:
    import requests  # noqa: F401
    _HAVE_REQUESTS = True
except Exception:  # pragma: no cover
    _HAVE_REQUESTS = False


# --------------------------------------------------------------------------
# Reuse the bs=1 dataset loader verbatim (guarantees byte-identical prompts
# across the two benchmarks).  We look for bench_bs1.py next to this file and in
# ../bs1/.  The import only pulls in `requests`
# at module top (datasets/transformers are imported lazily *inside*
# load_and_process_dataset), so importing it here is cheap and laptop-safe.
# --------------------------------------------------------------------------
def _import_dataset_loader() -> Callable[[str], object]:
    here = Path(__file__).resolve().parent
    for cand in (here, here.parent / "bs1", here.parent):
        if (cand / "bench_bs1.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            break
    from bench_bs1 import load_and_process_dataset  # type: ignore
    return load_and_process_dataset


# --------------------------------------------------------------------------
# Argument / task parsing
# --------------------------------------------------------------------------
def parse_tasks(spec: str) -> list[tuple[str, int]]:
    """`"gsm8k:128,mbpp:128,mt-bench:80"` -> [("gsm8k",128), ...]."""
    tasks: list[tuple[str, int]] = []
    for raw in spec.replace("\n", ",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"task entry must be dataset:count, got {raw!r}")
        name, count_s = raw.split(":", 1)
        name = name.strip()
        # Accept the underscore spelling; canonical name is mt-bench (matches
        # the Domino loader and our bs=1 filenames).
        if name == "mt_bench":
            name = "mt-bench"
        count = int(count_s.strip())
        if not name or count <= 0:
            raise ValueError(f"invalid task entry: {raw!r}")
        tasks.append((name, count))
    if not tasks:
        raise ValueError("no tasks specified")
    return tasks


def parse_concurrencies(spec: str) -> list[int]:
    levels = [int(x) for x in spec.split(",") if x.strip()]
    levels = [c for c in levels if c >= 1]
    if not levels:
        raise ValueError("no concurrency levels specified")
    return levels


def cycle_take(pool: list[str], n: int) -> list[str]:
    """First `n` items of `pool`, cycling if `pool` is shorter than `n`."""
    if n <= 0:
        return []
    if not pool:
        raise RuntimeError("cannot take prompts from an empty pool")
    return [pool[i % len(pool)] for i in range(n)]


# --------------------------------------------------------------------------
# Goodput aggregation — PURE function, unit-tested offline by selftest_conc.py.
# --------------------------------------------------------------------------
def aggregate(metas: list[dict], wall_s: float) -> dict:
    """Domino's goodput math over a list of per-request `meta_info` dicts.

    tps          = sum(completion_tokens) / wall_s
    mean_accept  = mean(spec_accept_length) over requests that report it (else None)
    """
    total_tokens = sum(int(m.get("completion_tokens", 0) or 0) for m in metas)
    accepts: list[float] = []
    for m in metas:
        v = m.get("spec_accept_length")
        if v is None:
            continue
        try:
            accepts.append(float(v))
        except (TypeError, ValueError):
            pass
    verify_sum = sum(int(m.get("spec_verify_ct", 0) or 0) for m in metas)
    tps = (total_tokens / wall_s) if wall_s > 0 else 0.0
    return {
        "n": len(metas),
        "completion_tokens": int(total_tokens),
        "wall_s": float(wall_s),
        "tps": float(tps),
        "mean_accept": (float(statistics.mean(accepts)) if accepts else None),
        "spec_verify_ct_sum": int(verify_sum),
    }


# --------------------------------------------------------------------------
# Concurrency runner — PURE structure, unit-tested offline with a fake sender.
# Submits every prompt to a pool with max_workers=concurrency; returns per-request
# meta dicts in prompt order.  Used for warmup (discard), skip (discard) and timed.
# --------------------------------------------------------------------------
def run_concurrent(prompts: list[str], concurrency: int, send_fn: Callable[[str], dict]) -> list[dict]:
    metas: list[Optional[dict]] = [None] * len(prompts)
    if not prompts:
        return []
    with ThreadPoolExecutor(max_workers=max(int(concurrency), 1)) as pool:
        futs = {pool.submit(send_fn, p): i for i, p in enumerate(prompts)}
        for fut in as_completed(futs):
            # Request-level fault tolerance (Phase-C robustness): a single hung
            # or failed request (e.g. HTTP timeout) must NOT abort the whole
            # sweep. Record an empty meta (0 tokens, no accept) and continue —
            # "skip-and-continue". A healthy server never hits this; a dead one
            # is caught by the orchestrator's server-health circuit-breaker.
            try:
                metas[futs[fut]] = fut.result()
            except Exception as exc:  # noqa: BLE001 — deliberately broad
                metas[futs[fut]] = {}
                print(f"[warn] request idx={futs[fut]} failed, skipping: {exc}",
                      file=sys.stderr, flush=True)
    return [m if m is not None else {} for m in metas]


# --------------------------------------------------------------------------
# Senders
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DryCfg:
    tokens: int = 100      # canned completion_tokens per request
    accept: float = 3.0    # canned spec_accept_length per request
    latency: float = 0.01  # simulated per-request seconds (so parallelism shows)


def make_sender(*, dry_run: bool, base_url: str, sampling: dict, timeout_s: int, dry: DryCfg) -> Callable[[str], dict]:
    """Return a `prompt -> meta_info` callable for one phase.

    In --dry-run it returns a canned meta (no server, no network) after a tiny
    sleep, so the ThreadPool + goodput math run exactly as in a real sweep.
    """
    if dry_run:
        def _dry(_prompt: str) -> dict:
            if dry.latency > 0:
                time.sleep(dry.latency)
            return {
                "completion_tokens": dry.tokens,
                "spec_accept_length": dry.accept,
                # verify_ct implied by a "block" of ~accept tokens per round.
                "spec_verify_ct": max(1, round(dry.tokens / max(dry.accept, 1.0))),
            }
        return _dry

    if not _HAVE_REQUESTS:
        raise SystemExit("`requests` is required for a real run (only --dry-run works without it).")

    def _http(prompt: str) -> dict:
        try:
            resp = requests.post(
                base_url + "/generate",
                json={"text": prompt, "sampling_params": sampling},
                timeout=int(timeout_s),
            )
        except requests.RequestException as exc:  # type: ignore[name-defined]
            raise RuntimeError(f"POST {base_url}/generate failed: {exc}") from exc
        if resp.status_code != 200:
            raise RuntimeError(f"/generate returned HTTP {resp.status_code}: {resp.text[:300]}")
        out = resp.json()
        if not isinstance(out, dict):
            raise RuntimeError(f"expected a dict /generate response, got {type(out).__name__}")
        return out.get("meta_info", {}) or {}

    return _http


def flush_cache(base_url: str, timeout_s: int = 60) -> None:
    """GET /flush_cache with retries (Domino discipline). Warn, don't die."""
    if not _HAVE_REQUESTS:
        return
    last: Optional[Exception] = None
    for i in range(5):
        try:
            resp = requests.get(base_url + "/flush_cache", timeout=timeout_s)
            resp.raise_for_status()
            return
        except requests.HTTPError as exc:  # type: ignore[name-defined]
            last = exc
            if exc.response is None or exc.response.status_code != 400:
                raise
        except requests.RequestException as exc:  # type: ignore[name-defined]
            last = exc
        time.sleep(1.0 + i)
    print(f"[warn] /flush_cache failed after retries, continuing. err={last}")


def check_server(base_url: str, timeout_s: int = 30) -> None:
    if not _HAVE_REQUESTS:
        raise SystemExit("`requests` is required to reach a server.")
    try:
        resp = requests.get(base_url + "/health", timeout=timeout_s)
    except requests.RequestException as exc:  # type: ignore[name-defined]
        raise SystemExit(
            f"cannot reach SGLang server at {base_url} ({exc}). "
            "Launch the method's server first — see RUNBOOK_conc.md."
        )
    if resp.status_code != 200:
        raise SystemExit(f"{base_url}/health returned HTTP {resp.status_code}: {resp.text[:200]}")


# --------------------------------------------------------------------------
# Prompt building for one dataset (timed set + disjoint warmup/skip pool).
# --------------------------------------------------------------------------
def build_dataset_prompts(
    *,
    dataset_name: str,
    n_timed: int,
    dry_run: bool,
    tokenizer=None,
    loader: Optional[Callable[[str], object]] = None,
) -> tuple[list[str], list[str], str, int]:
    """Return (timed_prompts, pool_prompts, warmup_source, n_available).

    timed_prompts  : first `n_timed` of shuffle(seed=0) (== bs=1 prefix), cycled
                     if the dataset has fewer than n_timed rows.
    pool_prompts   : remaining shuffled prompts (indices [n_timed:]) for warmup/
                     skip, DISJOINT from timed; falls back to timed if exhausted.
    """
    if dry_run:
        timed = [f"[dry-run] {dataset_name} prompt {i}" for i in range(n_timed)]
        pool = [f"[dry-run] {dataset_name} pool {i}" for i in range(max(n_timed, 8))]
        return timed, pool, "dry_run_synthetic", n_timed

    assert loader is not None and tokenizer is not None
    dataset = loader(dataset_name)
    n_avail = len(dataset)
    if n_avail == 0:
        raise RuntimeError(f"dataset {dataset_name!r} is empty")
    shuffled = dataset.shuffle(seed=0)  # deterministic; bs=1 uses the same seed

    def fmt(idx: int) -> str:
        # Single-shot, first-turn only (like Domino's concurrency driver); the
        # chat-template call is identical to bs=1's per-turn formatting.
        user_content = shuffled[idx]["turns"][0]
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    timed = [fmt(i % n_avail) for i in range(n_timed)]
    if n_avail > n_timed:
        pool = [fmt(i) for i in range(n_timed, n_avail)]
        warmup_source = "post_timed"
    else:
        pool = list(timed)  # dataset exhausted by the timed set
        warmup_source = "timed_reuse"
    return timed, pool, warmup_source, n_avail


# --------------------------------------------------------------------------
# Markdown report (Domino-style per-dataset tables, for Table-2/3 comparability)
# --------------------------------------------------------------------------
def render_md(args, tasks, concurrencies, rows) -> str:
    by_key = {(r["dataset"], r["concurrency"]): r for r in rows}
    out: list[str] = []
    out.append(f"# Concurrency (goodput) sweep — `{args.method}` on `{args.model_label}`")
    out.append("")
    out.append("## Settings")
    out.append(f"- method (label): `{args.method}`")
    out.append(f"- model: `{args.model_label}`  (tokenizer/template: `{args.model_path}`)")
    out.append(f"- host:port: `{args.host}:{args.port}`")
    out.append(f"- max_new_tokens: `{args.max_new_tokens}`")
    out.append(f"- temperature/top_p/top_k: `{args.temperature}` / `{args.top_p}` / `{args.top_k}`")
    out.append(f"- warmup_requests (batches of size=conc): `{args.warmup_requests}`")
    out.append(f"- warmup_max_new_tokens: `{args.warmup_max_new_tokens}`")
    out.append(f"- skip_first: `{args.skip_first}`")
    out.append(f"- concurrencies: `{', '.join(str(c) for c in concurrencies)}`")
    out.append("- prompt_selection: `shuffle(seed=0).select(range(N))` (== bs=1 prefix)")
    out.append("- warmup_pool: `post_timed_shuffled (disjoint from timed) or timed_reuse`")
    out.append("- metric: `tps = sum(completion_tokens)/wall_s ; mean_accept = mean(spec_accept_length)`")
    out.append("")
    out.append("## Tasks")
    for name, count in tasks:
        out.append(f"- `{name}`: N=`{count}`")
    out.append("")

    def table(metric: str, fmt: str) -> None:
        header = ["dataset \\ conc"] + [str(c) for c in concurrencies]
        out.append("| " + " | ".join(header) + " |")
        out.append("| " + " | ".join(["---"] * len(header)) + " |")
        for name, _ in tasks:
            cells = [f"`{name}`"]
            for c in concurrencies:
                r = by_key.get((name, c))
                v = None if r is None else r.get(metric)
                cells.append("N/A" if v is None else format(v, fmt))
            out.append("| " + " | ".join(cells) + " |")
        out.append("")

    out.append("## Goodput (output tok/s)")
    table("tps", ",.2f")
    out.append("## Mean acceptance length (spec_accept_length)")
    table("mean_accept", ".3f")
    return "\n".join(out) + "\n"


def write_outputs(args, tasks, concurrencies, rows) -> None:
    jp = Path(args.out_jsonl)
    jp.parent.mkdir(parents=True, exist_ok=True)
    with jp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.out_md:
        mp = Path(args.out_md)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(render_md(args, tasks, concurrencies, rows), encoding="utf-8")


# --------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--method", required=True,
                   help="Label stamped on every row (ar/dflash/eagle3/domino-chain/dominotree@16). "
                        "Describes the server on --port; this driver cannot verify it.")
    p.add_argument("--model-label", default=None,
                   help="Short model label for the `model` row field (e.g. qwen3-4b). "
                        "Defaults to basename(--model-path).")
    p.add_argument("--model-path", default=None,
                   help="HF path/name for the tokenizer + chat template (must match the served "
                        "model). Required for a real run; not needed for --dry-run.")
    p.add_argument("--tasks", default="gsm8k:128,mbpp:128,mt-bench:80",
                   help="Comma list dataset:N, e.g. gsm8k:128,mbpp:128,mt-bench:80")
    p.add_argument("--concurrencies", default="1,2,4,8,16,32")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--warmup-requests", type=int, default=3,
                   help="Warmup BATCHES (each of size=concurrency) sent after flush; discarded.")
    p.add_argument("--warmup-max-new-tokens", type=int, default=1024)
    p.add_argument("--skip-first", type=int, default=1,
                   help="Timed requests issued-then-dropped after warmup.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1,
                   help="Untruncated softmax (-1) matches OUR bs=1 driver; Domino uses top_k=1. "
                        "At T=0 both are greedy (identical).")
    p.add_argument("--timeout-s", type=int, default=3600, help="Per-request HTTP timeout.")
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--out-md", default=None)
    # ---- offline self-test knobs -------------------------------------------
    p.add_argument("--dry-run", action="store_true",
                   help="No server/GPU/datasets: synthetic prompts + canned /generate responses. "
                        "Exercises the ThreadPool + goodput math offline.")
    p.add_argument("--dry-run-tokens", type=int, default=100)
    p.add_argument("--dry-run-accept", type=float, default=3.0)
    p.add_argument("--dry-run-latency", type=float, default=0.01)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tasks = parse_tasks(args.tasks)
    concurrencies = parse_concurrencies(args.concurrencies)
    base_url = f"http://{args.host}:{args.port}"

    if args.model_label is None:
        args.model_label = (os.path.basename(str(args.model_path).rstrip("/"))
                            if args.model_path else "unknown-model")

    dry = DryCfg(tokens=args.dry_run_tokens, accept=args.dry_run_accept, latency=args.dry_run_latency)

    tokenizer = None
    loader = None
    if not args.dry_run:
        if not args.model_path:
            raise SystemExit("--model-path is required for a real run (needed for the chat template).")
        check_server(base_url)
        loader = _import_dataset_loader()
        from transformers import AutoTokenizer  # type: ignore  (present on the GPU box)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        print(f"[server] {base_url} healthy; tokenizer={args.model_path}")
    else:
        print("[dry-run] no server/GPU/datasets; canned /generate "
              f"(tokens={dry.tokens}, accept={dry.accept}, latency={dry.latency}s)")

    def sampling_for(mnt: int) -> dict:
        return {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "max_new_tokens": int(mnt),
        }

    rows: list[dict] = []
    t_all = time.time()
    for dataset_name, n_timed in tasks:
        timed_prompts, pool_prompts, warmup_source, n_avail = build_dataset_prompts(
            dataset_name=dataset_name, n_timed=n_timed, dry_run=args.dry_run,
            tokenizer=tokenizer, loader=loader,
        )
        print(f"\n=== dataset={dataset_name} N={n_timed} (avail={n_avail}, warmup_source={warmup_source}) ===")
        for conc in concurrencies:
            if not args.dry_run:
                flush_cache(base_url)
            warmup_n = max(int(args.warmup_requests), 0) * int(conc)
            skip_n = max(int(args.skip_first), 0)
            warmup_prompts = cycle_take(pool_prompts, warmup_n)
            skip_prompts = cycle_take(pool_prompts, skip_n)

            timed_send = make_sender(dry_run=args.dry_run, base_url=base_url,
                                     sampling=sampling_for(args.max_new_tokens),
                                     timeout_s=args.timeout_s, dry=dry)
            warmup_send = make_sender(dry_run=args.dry_run, base_url=base_url,
                                      sampling=sampling_for(args.warmup_max_new_tokens),
                                      timeout_s=args.timeout_s, dry=dry)

            # WARMUP (discard) + SKIP-FIRST (issue, discard) — same concurrency.
            if warmup_prompts:
                run_concurrent(warmup_prompts, conc, warmup_send)
            if skip_prompts:
                run_concurrent(skip_prompts, conc, timed_send)

            # TIMED window.
            t0 = time.perf_counter()
            metas = run_concurrent(timed_prompts, conc, timed_send)
            wall_s = time.perf_counter() - t0

            agg = aggregate(metas, wall_s)
            row = {
                "method": args.method,
                "model": args.model_label,
                "dataset": dataset_name,
                "concurrency": int(conc),
                "tps": agg["tps"],
                "mean_accept": agg["mean_accept"],
                "n_prompts": agg["n"],
                "wall_s": agg["wall_s"],
                "completion_tokens": agg["completion_tokens"],
                "max_new_tokens": int(args.max_new_tokens),
                # ---- provenance (ignored by table builders) ----
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
                "warmup_requests": int(args.warmup_requests),
                "warmup_max_new_tokens": int(args.warmup_max_new_tokens),
                "skip_first": int(args.skip_first),
                "warmup_source": warmup_source,
                "spec_verify_ct_sum": agg["spec_verify_ct_sum"],
                "host": args.host,
                "port": int(args.port),
                "dry_run": bool(args.dry_run),
                "harness": "sglang-serving-conc-v2",
            }
            rows.append(row)
            ma = "None" if row["mean_accept"] is None else f"{row['mean_accept']:.3f}"
            print(f"[{args.method}] {dataset_name:>12} conc={conc:>3} N={row['n_prompts']:<4} "
                  f"tps={row['tps']:>9.2f} wall={wall_s:>6.2f}s tok={row['completion_tokens']:<7} "
                  f"accept={ma}")
            if (not args.dry_run and args.method != "ar"
                    and row["mean_accept"] is None and row["concurrency"] == concurrencies[0]):
                print(f"[warn] method={args.method!r} but no spec_accept_length seen — is speculation ON?")
            # Write incrementally so a mid-sweep drop keeps completed cells.
            write_outputs(args, tasks, concurrencies, rows)

    write_outputs(args, tasks, concurrencies, rows)
    print(f"\n[done] {len(rows)} cells in {time.time() - t_all:.1f}s -> {args.out_jsonl}"
          + (f" , {args.out_md}" if args.out_md else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
