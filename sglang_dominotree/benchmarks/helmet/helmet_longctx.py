#!/usr/bin/env python3
"""HELMET long-context bs=1 acceptance-length (tau) driver for the DominoTree
SGLang plugin — the v2 long-context section's PRIMARY instrument.

WHY HELMET (replaces the home-grown RULER-style NIAH probe)
-----------------------------------------------------------
`../longctx_probe/longctx_probe.py` built its own needle-in-a-haystack. HELMET
(Yen et al., ICLR 2025, arXiv:2410.02694) is the paper-grade replacement: it is a
rigorous, controllable-length long-context suite whose **Summarization** tasks
(inf-bench Sum + Multi-LexSum, generation cap **1,200** tokens) and **Cite** tasks
(ALCE ASQA + QAMPARI, cap **300**) give a genuinely LONG decode phase — which is
exactly what a decode-side metric like acceptance length tau needs. (Short-answer
long-context tasks decode ~10 tokens and cannot measure tau; see the RULER
faithfulness note.) HELMET also natively bins input length to
{8K,16K,32K,64K,128K} (Llama-2 tokens), so it subsumes our synthetic length sweep.

DIVISION OF LABOUR — no re-implementation of HELMET
---------------------------------------------------
Per the project hard rule ("NEVER re-implement official code"), THIS driver never
constructs a HELMET prompt. `helmet_prep.py` runs HELMET's OWN loader on the GPU box
to materialise the exact chat-templated prompt strings into a flat "cell" jsonl:

    prompts/<model>/<task>/<length_bin>.jsonl   # one prompt per line:
        {"task","length_bin","idx","text","gen_cap", ...}

`text` is the final string to POST verbatim as /generate `text` (HELMET's own
template already applied for the served model). This driver is deliberately dumb: it
replays those prompts against a RUNNING SGLang server, records tau + throughput, and
aggregates one row per (task, length_bin). It launches nothing, imports no
torch/CUDA, and (like the Phase-C/D drivers) is laptop-safe and HTTP-only.

METHOD-AGNOSTIC
---------------
`--method {ar,dflash,eagle3,domino_chain,dominotree}` is only a LABEL stamped on
each row; the driver cannot verify which speculative algorithm the server on
`--port` runs (a Step-2 sanity curl in the RUNBOOK does). Apply identical serving
flags across methods (the launcher does) so any tau/tps delta reflects the method.

SAMPLE-SIZE-AS-PREFIX (n=50 -> n=100 supplement)
------------------------------------------------
Cells are read in the STABLE on-disk order (helmet_prep.py does not shuffle), and
`--n-prompts N` takes the FIRST N lines. So an n=50 run is a strict prefix of an
n=100 run: re-running a marginal cell at n=100 only APPENDS prompts 51..100 and
tightens the CI; the first 50 are byte-identical. Never resamples.

OUTPUT — one aggregated JSONL row per (method, task, length_bin)
---------------------------------------------------------------
  {method, model, task, length_bin, tps, mean_accept, n_prompts,
   input_tokens, output_tokens, gen_tokens}                        (+ provenance)
`tps`         = sum(completion_tokens) / sum(decode_time)   (bs=1 goodput)
`mean_accept` = mean(spec_accept_length)                    (== tau; None->1.0 AR)
`input_tokens`/`output_tokens` = mean server-reported prompt/completion tokens.
`gen_tokens`  = the cell's HELMET gen_cap (per-task: 1200 summ / 300 cite), unless
                clamped with --gen-cap-clamp (a documented cost knob).

Usage (server already running for ONE method; see README.md):
  python helmet_longctx.py --host 127.0.0.1 --port 31600 \\
      --method dominotree --model-label qwen3-8b \\
      --prompts-dir prompts/qwen3-8b \\
      --tasks infbench_sum,multi_lexsum \\
      --length-bins 8192,16384,32768 --n-prompts 50 \\
      --out out/dominotree/helmet_8b.jsonl

Offline self-test (no server, no GPU, no network, no transformers):
  (dev self-test lives in the source repo)
  python helmet_longctx.py --dry-run --prompts-dir <fixture> \\
      --tasks demo --length-bins 1024 --n-prompts 3 --method dominotree --out /tmp/h.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Callable, Optional

try:
    import requests  # noqa: F401
    _HAVE_REQUESTS = True
except Exception:  # pragma: no cover
    _HAVE_REQUESTS = False

# HELMET per-task generation caps (repo configs/*.yaml @128K). Recorded here only
# as a fallback/sanity default; the authoritative value is the `gen_cap` field the
# prep step writes into each cell file.
HELMET_GEN_CAP = {
    "infbench_sum": 1200,
    "multi_lexsum": 1200,
    "alce_asqa": 300,
    "alce_qampari": 300,
    "narrativeqa": 100,   # borderline (short-ish); included for completeness
}
DEFAULT_LENGTH_BINS = [8192, 16384, 32768]


# --------------------------------------------------------------------------
# Cell loading — read helmet_prep.py's flat prompt files.  A "cell" is one
# (task, length_bin) file; each line is one ready-to-POST prompt.
# --------------------------------------------------------------------------
def cell_path(prompts_dir: str, task: str, length_bin: int) -> Path:
    return Path(prompts_dir) / task / f"{length_bin}.jsonl"


def load_cell(prompts_dir: str, task: str, length_bin: int, n_prompts: int) -> list[dict]:
    """Load up to the FIRST n_prompts prompts for one (task, length_bin) cell.

    Returns a list of {text, gen_cap, idx, ...} dicts in on-disk order (a strict
    prefix, so n=50 nests inside n=100). Raises SystemExit with an actionable
    message if the cell file is missing (prep step not run for this task/bin).
    """
    p = cell_path(prompts_dir, task, length_bin)
    if not p.exists():
        raise SystemExit(
            f"missing HELMET cell {p}\n"
            f"  run helmet_prep.py first to materialise task={task} bin={length_bin} "
            f"for this model (see README.md Step 2)."
        )
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "text" not in rec:
                raise SystemExit(f"{p}: prompt line missing required `text` field: {line[:120]}")
            rows.append(rec)
            if len(rows) >= n_prompts:
                break
    if not rows:
        raise SystemExit(f"{p}: no prompts found (empty cell file).")
    return rows


def cell_gen_cap(cell_rows: list[dict], task: str, clamp: Optional[int]) -> int:
    """Resolve the generation cap for a cell: the prep-written `gen_cap`
    (authoritative, HELMET's per-task value), falling back to HELMET_GEN_CAP, then
    optionally clamped by --gen-cap-clamp (a cost knob; logged as a deviation)."""
    cap = int(cell_rows[0].get("gen_cap", HELMET_GEN_CAP.get(task, 512)))
    if clamp is not None and clamp > 0:
        cap = min(cap, int(clamp))
    return cap


# --------------------------------------------------------------------------
# Senders.  Real: POST /generate, return meta_info.  Dry: canned meta (no
# server/network) so the bs=1 loop + aggregation run fully offline.
# (Mirrors ../longctx_probe/longctx_probe.py so behaviour is identical.)
# --------------------------------------------------------------------------
def make_sender(*, dry_run: bool, base_url: str, timeout_s: int,
                dry_accept: float, dry_latency: float) -> Callable[[str, dict], dict]:
    if dry_run:
        def _dry(text: str, sampling: dict) -> dict:
            if dry_latency > 0:
                time.sleep(dry_latency)
            want = int(sampling.get("max_new_tokens", 128))
            # completion == the cap (a full-length generation) for deterministic tests.
            return {
                "prompt_tokens": max(1, len(text.split())),
                "completion_tokens": want,
                "spec_accept_length": float(dry_accept),
                "spec_verify_ct": max(1, round(want / max(dry_accept, 1.0))),
            }
        return _dry

    if not _HAVE_REQUESTS:
        raise SystemExit("`requests` is required for a real run (only --dry-run works without it).")

    def _http(text: str, sampling: dict) -> dict:
        try:
            resp = requests.post(base_url + "/generate",
                                 json={"text": text, "sampling_params": sampling},
                                 timeout=int(timeout_s))
        except requests.RequestException as exc:  # type: ignore[name-defined]
            raise RuntimeError(f"POST {base_url}/generate failed: {exc}") from exc
        if resp.status_code != 200:
            raise RuntimeError(f"/generate returned HTTP {resp.status_code}: {resp.text[:300]}")
        out = resp.json()
        if not isinstance(out, dict):
            raise RuntimeError(f"expected a dict /generate response, got {type(out).__name__}")
        return out.get("meta_info", {}) or {}

    return _http


def check_server(base_url: str, model_label: str, timeout_s: int) -> None:
    if not _HAVE_REQUESTS:
        raise SystemExit("`requests` is required to reach a server.")
    try:
        resp = requests.get(base_url + "/health", timeout=min(timeout_s, 30))
    except requests.RequestException as exc:  # type: ignore[name-defined]
        raise SystemExit(
            f"cannot reach SGLang server at {base_url} ({exc}).\n"
            "Launch the method's server first — see README.md."
        )
    if resp.status_code != 200:
        raise SystemExit(f"{base_url}/health returned HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        info = requests.get(base_url + "/get_model_info", timeout=min(timeout_s, 30)).json()
        served = str(info.get("model_path", ""))
        print(f"[server] model: {served or '<unknown>'} (label={model_label})")
    except Exception:
        print("[warn] could not query /get_model_info; skipping served-model check.")


# --------------------------------------------------------------------------
# Aggregation — PURE function (unit-tested in the source repo).
# (Identical semantics to longctx_probe.aggregate.)
# --------------------------------------------------------------------------
def aggregate(per_prompt: list[dict], gen_tokens: int) -> dict:
    n = len(per_prompt)
    out_sum = sum(int(r["output_tokens"]) for r in per_prompt)
    time_sum = sum(float(r["decode_time"]) for r in per_prompt)
    accepts = [float(r["accept"]) for r in per_prompt]
    return {
        "n_prompts": n,
        "tps": (out_sum / time_sum) if time_sum > 0 else 0.0,
        "mean_accept": (statistics.fmean(accepts) if accepts else None),
        "input_tokens": (statistics.fmean(int(r["input_tokens"]) for r in per_prompt) if n else 0.0),
        "output_tokens": (statistics.fmean(int(r["output_tokens"]) for r in per_prompt) if n else 0.0),
        "gen_tokens": int(gen_tokens),
    }


def parse_int_list(spec: str, what: str) -> list[int]:
    vals = [int(x) for x in spec.split(",") if x.strip()]
    vals = [v for v in vals if v > 0]
    if not vals:
        raise ValueError(f"no {what} specified")
    return vals


def parse_str_list(spec: str, what: str) -> list[str]:
    vals = [x.strip() for x in spec.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"no {what} specified")
    return vals


# --------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--method", required=True,
                   help="Label stamped on every row (ar/dflash/eagle3/domino_chain/dominotree). "
                        "Describes the server on --port; this driver cannot verify it.")
    p.add_argument("--model-label", default="unknown-model",
                   help="Short model label for the `model` row field (e.g. qwen3-4b / qwen3-8b).")
    p.add_argument("--prompts-dir", required=True,
                   help="Root of prep output: <prompts-dir>/<task>/<length_bin>.jsonl")
    p.add_argument("--tasks", default="infbench_sum,multi_lexsum",
                   help="Comma list of HELMET tasks to sweep (must exist under --prompts-dir).")
    p.add_argument("--length-bins", default=",".join(str(x) for x in DEFAULT_LENGTH_BINS),
                   help="Comma list of input length bins, e.g. 8192,16384,32768.")
    p.add_argument("--n-prompts", type=int, default=50,
                   help="First-N prompts per cell (prefix; n=50 nests in n=100).")
    p.add_argument("--gen-cap-clamp", type=int, default=None,
                   help="COST KNOB: cap generation at this many tokens (min with HELMET's per-task "
                        "cap). Omit for the faithful full cap (1200 summ / 300 cite). Logged.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--request-timeout", type=int, default=1800, help="Per-request HTTP timeout (s).")
    p.add_argument("--out", required=True, help="Output JSONL path (one row per task x length_bin).")
    # ---- offline self-test knobs -------------------------------------------
    p.add_argument("--dry-run", action="store_true",
                   help="No server/GPU/network: canned /generate. Exercises cell-loading + "
                        "aggregation offline against a real prompt jsonl fixture.")
    p.add_argument("--dry-run-accept", type=float, default=3.0)
    p.add_argument("--dry-run-latency", type=float, default=0.005)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tasks = parse_str_list(args.tasks, "tasks")
    length_bins = parse_int_list(args.length_bins, "length bins")
    base_url = f"http://{args.host}:{args.port}"

    if not args.dry_run:
        check_server(base_url, args.model_label, args.request_timeout)
        print(f"[server] {base_url} healthy; label={args.model_label}")
    else:
        print(f"[dry-run] canned /generate (accept={args.dry_run_accept}, "
              f"latency={args.dry_run_latency}s) — no server/GPU/network.")

    if args.gen_cap_clamp:
        print(f"[warn] --gen-cap-clamp={args.gen_cap_clamp}: generation capped BELOW HELMET's "
              f"per-task cap (a cost deviation; note it in the paper).")

    send = make_sender(dry_run=args.dry_run, base_url=base_url, timeout_s=args.request_timeout,
                       dry_accept=args.dry_run_accept, dry_latency=args.dry_run_latency)

    # ---- warmup (real runs only): one short generation to reach steady state ----
    if not args.dry_run:
        try:
            probe_cell = load_cell(args.prompts_dir, tasks[0], length_bins[0], 1)
            send(probe_cell[0]["text"], {"temperature": float(args.temperature),
                                         "top_p": float(args.top_p), "top_k": int(args.top_k),
                                         "max_new_tokens": 16})
            print("[warmup] done")
        except (RuntimeError, SystemExit) as exc:
            print(f"[warn] warmup skipped/failed ({exc}); continuing to the measured loop.")

    rows: list[dict] = []
    # Per-PROMPT records (one per generation), written to a sidecar alongside --out.
    # Required for paired-bootstrap CIs and the n=50->100 supplement (append prompts
    # 51..100 and recompute a cell's CI). The per-cell `rows` above are only means.
    prompt_rows: list[dict] = []
    prompts_out = _prompts_path(args.out)
    t_all = time.time()
    for task in tasks:
        for L in length_bins:
            cell = load_cell(args.prompts_dir, task, L, int(args.n_prompts))
            gen_cap = cell_gen_cap(cell, task, args.gen_cap_clamp)
            sampling = {
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
                "max_new_tokens": int(gen_cap),
            }
            per_prompt: list[dict] = []
            for i, rec in enumerate(cell):
                t0 = time.perf_counter()
                try:
                    meta = send(rec["text"], sampling)
                except RuntimeError as exc:
                    raise SystemExit(f"generation failed at task={task} L={L} prompt={i} "
                                     f"method={args.method}: {exc}")
                decode_time = time.perf_counter() - t0
                in_tok = int(meta.get("prompt_tokens", meta.get("input_tokens", 0)) or 0)
                out_tok = int(meta.get("completion_tokens", 0) or 0)
                accept = float(meta.get("spec_accept_length", 1.0) or 1.0)
                per_prompt.append({
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "decode_time": decode_time,
                    "accept": accept,
                })
                # per-prompt record keyed for pairing across methods (same task,bin,idx)
                prompt_rows.append({
                    "method": args.method, "model": args.model_label,
                    "task": task, "length_bin": int(L),
                    "idx": int(rec.get("idx", i)),
                    "input_tokens": in_tok, "output_tokens": out_tok,
                    "decode_time": decode_time, "accept": accept,
                    "tps": (out_tok / decode_time) if decode_time > 0 else 0.0,
                    "gen_cap": int(gen_cap),
                })

            agg = aggregate(per_prompt, gen_cap)
            row = {
                # ---- required schema ----
                "method": args.method,
                "model": args.model_label,
                "task": task,
                "length_bin": int(L),
                "tps": agg["tps"],
                "mean_accept": agg["mean_accept"],
                "n_prompts": agg["n_prompts"],
                "input_tokens": agg["input_tokens"],
                "output_tokens": agg["output_tokens"],
                "gen_tokens": agg["gen_tokens"],
                # ---- provenance (ignored by any table builder) ----
                "benchmark": "helmet",
                "gen_cap_helmet": int(cell[0].get("gen_cap", HELMET_GEN_CAP.get(task, 512))),
                "gen_cap_clamped": bool(args.gen_cap_clamp),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
                "host": args.host,
                "port": int(args.port),
                "dry_run": bool(args.dry_run),
                "harness": "sglang-serving-bs1-helmet",
            }
            rows.append(row)
            ma = "None" if row["mean_accept"] is None else f"{row['mean_accept']:.3f}"
            print(f"[{args.method}] task={task:<14} bin={L:>6} "
                  f"in~{row['input_tokens']:.0f} out~{row['output_tokens']:.0f} "
                  f"gen_cap={gen_cap} tau={ma} tps={row['tps']:.2f} n={row['n_prompts']}")
            # incremental: a WiFi drop keeps completed cells (means + per-prompt).
            _write(args.out, rows)
            _write(prompts_out, prompt_rows)

    _write(args.out, rows)
    _write(prompts_out, prompt_rows)
    _print_summary(args.method, rows)
    print(f"\n[done] {len(rows)} cell rows ({len(prompt_rows)} per-prompt) in "
          f"{time.time() - t_all:.1f}s -> {args.out}\n           per-prompt -> {prompts_out}")
    return 0


def _write(out_path: str, rows: list[dict]) -> None:
    p = Path(out_path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _prompts_path(out_path: str) -> str:
    """Sidecar path for per-prompt records: `<stem>.prompts.jsonl` next to --out."""
    p = Path(out_path)
    return str(p.with_name(p.stem + ".prompts.jsonl"))


def _print_summary(method: str, rows: list[dict]) -> None:
    print(f"\n{'task':<16}{'bin':>8} {'in_tok':>8} {'tau':>7} {'tps':>9}   method={method}")
    print("-" * 60)
    for r in rows:
        ma = "None" if r["mean_accept"] is None else f"{r['mean_accept']:.3f}"
        print(f"{r['task']:<16}{r['length_bin']:>8} {r['input_tokens']:>8.0f} {ma:>7} {r['tps']:>9.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
