#!/usr/bin/env python3
"""Aggregate HELMET long-context per-prompt sidecars into the paper's long-context
table: per-cell tau + throughput per method, DominoTree speedup vs AR, and
paired-bootstrap 95% CIs on DominoTree-vs-each-baseline deltas.

INPUT — the per-prompt sidecars written by helmet_longctx.py:
    <root>/<method>/*.prompts.jsonl     # one record per prompt:
      {method, task, length_bin, idx, accept, tps, decode_time, input_tokens, ...}
Records are paired ACROSS methods by (task, length_bin, idx) — same HELMET example,
so the delta is a true paired comparison (prompt-specific effects cancel).

DELTA CONVENTION (matches the paper / partner report): ratio-of-per-cell-means, NOT
mean-of-per-prompt-ratios. Delta% = mean(metric[dominotree]) / mean(metric[baseline])
- 1, over the prompts common to both methods in that cell. CI is a paired bootstrap
(resample the shared prompt indices, recompute the ratio) — B=5000, seed fixed.

A cell whose Delta CI straddles 0 is flagged MARGINAL: a candidate for the n=50->100
supplement (re-run that cell at NPR=100; sidecars append prompts 51..100, then re-run
this script). tau vs AR is exact (AR tau==1).

Usage:
  python aggregate_helmet.py --root out/8b --model qwen3-8b --out-md table_helmet_8b.md
  python aggregate_helmet.py --selftest      # offline synthetic check, no data needed
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LEAD = "dominotree"
BASELINES = ["domino_chain", "eagle3", "dflash", "ar"]  # order for the delta table
METHOD_ORDER = ["dominotree", "eagle3", "domino_chain", "dflash", "ar"]


# --------------------------------------------------------------------------
def load_sidecars(root: str) -> dict:
    """Return records[method][(task,bin)][idx] = {'accept':..,'tps':..}."""
    recs: dict = defaultdict(lambda: defaultdict(dict))
    files = sorted(glob.glob(str(Path(root) / "**" / "*.prompts.jsonl"), recursive=True))
    if not files:
        raise SystemExit(f"no *.prompts.jsonl under {root} — run helmet_longctx.py first "
                         f"(its per-prompt sidecar), or point --root at the right dir.")
    for f in files:
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            m = r["method"]; cell = (r["task"], int(r["length_bin"])); idx = int(r["idx"])
            recs[m][cell][idx] = {"accept": float(r.get("accept", 1.0)),
                                  "tps": float(r.get("tps", 0.0))}
    return recs


def paired_bootstrap_ratio(av: list[float], bv: list[float], b: int, seed: int) -> tuple:
    """Point Delta% and 95% CI for mean(a)/mean(b)-1, paired resample (a,b aligned)."""
    n = len(av)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mb0 = statistics.fmean(bv)
    point = (statistics.fmean(av) / mb0 - 1.0) if mb0 else float("nan")
    rng = random.Random(seed)
    deltas = []
    for _ in range(b):
        picks = [rng.randrange(n) for _ in range(n)]
        ma = sum(av[j] for j in picks) / n
        mb = sum(bv[j] for j in picks) / n
        deltas.append((ma / mb - 1.0) if mb else float("nan"))
    deltas = [d for d in deltas if d == d]  # drop nan
    deltas.sort()
    if not deltas:
        return (point * 100, float("nan"), float("nan"))
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    return (point * 100, lo * 100, hi * 100)


def cell_delta(recs: dict, cell: tuple, metric: str, base: str, b: int, seed: int) -> dict:
    """Paired DominoTree-vs-base delta for one metric in one cell."""
    a = recs.get(LEAD, {}).get(cell, {})
    c = recs.get(base, {}).get(cell, {})
    common = sorted(set(a) & set(c))
    av = [a[i][metric] for i in common]
    cv = [c[i][metric] for i in common]
    pt, lo, hi = paired_bootstrap_ratio(av, cv, b, seed)
    marginal = (lo == lo and hi == hi and lo <= 0.0 <= hi)  # CI straddles 0
    return {"n": len(common), "delta_pct": pt, "lo": lo, "hi": hi, "marginal": marginal}


def cell_mean(recs: dict, cell: tuple, method: str, metric: str):
    d = recs.get(method, {}).get(cell, {})
    if not d:
        return None
    return statistics.fmean(v[metric] for v in d.values())


# --------------------------------------------------------------------------
def build_report(recs: dict, model: str, b: int, seed: int) -> str:
    cells = sorted({c for m in recs.values() for c in m}, key=lambda c: (c[0], c[1]))
    L: list[str] = []
    L.append(f"# HELMET long-context — {model}\n")
    present = [m for m in METHOD_ORDER if m in recs]
    L.append(f"Methods: {', '.join(present)} · paired bootstrap B={b}, seed={seed}\n")

    # --- per-cell tau / tps / speedup-vs-AR ---
    L.append("## Per-cell τ, throughput, and DominoTree speedup vs AR\n")
    L.append("| task | bin | metric | " + " | ".join(present) + " |")
    L.append("|" + "---|" * (3 + len(present)))
    for cell in cells:
        task, bn = cell
        taus = {m: cell_mean(recs, cell, m, "accept") for m in present}
        tpss = {m: cell_mean(recs, cell, m, "tps") for m in present}
        fmt = lambda x: "—" if x is None else f"{x:.2f}"
        L.append(f"| {task} | {bn} | τ | " + " | ".join(fmt(taus[m]) for m in present) + " |")
        L.append(f"| {task} | {bn} | tps | " + " | ".join(fmt(tpss[m]) for m in present) + " |")
        ar = tpss.get("ar")
        sp = lambda m: "—" if (ar in (None, 0) or tpss[m] is None) else f"{tpss[m]/ar:.2f}×"
        L.append(f"| {task} | {bn} | tps/AR | " + " | ".join(sp(m) for m in present) + " |")

    # --- DominoTree delta table (paired CI) ---
    for metric, label in (("accept", "Δτ%"), ("tps", "Δtps%")):
        L.append(f"\n## DominoTree {label} vs each baseline (paired 95% CI; * = MARGINAL, CI straddles 0)\n")
        bases = [x for x in BASELINES if x in recs and x != "ar" or (x == "ar" and metric == "tps")]
        # tau vs AR is trivially +inf-ish (AR tau=1); show tau deltas vs spec baselines only.
        L.append("| task | bin | " + " | ".join(bases) + " |")
        L.append("|" + "---|" * (2 + len(bases)))
        for cell in cells:
            task, bn = cell
            cellstrs = []
            for base in bases:
                d = cell_delta(recs, cell, metric, base, b, seed)
                if d["n"] == 0:
                    cellstrs.append("—")
                else:
                    star = "*" if d["marginal"] else ""
                    cellstrs.append(f"{d['delta_pct']:+.1f}% [{d['lo']:+.1f},{d['hi']:+.1f}]{star}")
            L.append(f"| {task} | {bn} | " + " | ".join(cellstrs) + " |")

    # --- macro rollup (mean over cells of per-cell means) ---
    L.append("\n## Macro-average over all cells\n")
    L.append("| metric | " + " | ".join(present) + " |")
    L.append("|" + "---|" * (1 + len(present)))
    for metric, lab in (("accept", "τ"), ("tps", "tps")):
        vals = {}
        for m in present:
            cms = [cell_mean(recs, c, m, metric) for c in cells]
            cms = [x for x in cms if x is not None]
            vals[m] = statistics.fmean(cms) if cms else None
        fmt = lambda x: "—" if x is None else f"{x:.2f}"
        L.append(f"| {lab} | " + " | ".join(fmt(vals[m]) for m in present) + " |")

    # --- marginal-cell callout for the n=50->100 supplement ---
    marg = []
    for metric in ("accept", "tps"):
        for cell in cells:
            for base in [x for x in BASELINES if x in recs]:
                if base == "ar" and metric == "accept":
                    continue
                d = cell_delta(recs, cell, metric, base, b, seed)
                if d["n"] and d["marginal"]:
                    marg.append(f"{cell[0]}@{cell[1]} {metric} vs {base} (Δ={d['delta_pct']:+.1f}%, "
                                f"CI[{d['lo']:+.1f},{d['hi']:+.1f}], n={d['n']})")
    L.append("\n## Marginal cells (CI straddles 0 → candidates for NPR=100 boost)\n")
    L.append("\n".join(f"- {x}" for x in marg) if marg else "- none — every delta CI is one-sided.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
def _selftest() -> int:
    """Offline synthetic check: dominotree strictly beats a baseline -> positive,
    tight, non-marginal delta; identical vectors -> ~0 and marginal."""
    import tempfile, os
    recs = defaultdict(lambda: defaultdict(dict))
    cell = ("infbench_sum", 8192)
    for i in range(50):
        recs["dominotree"][cell][i] = {"accept": 3.0 + (i % 3) * 0.01, "tps": 100.0}
        recs["dflash"][cell][i] = {"accept": 2.0 + (i % 3) * 0.01, "tps": 80.0}
        recs["ar"][cell][i] = {"accept": 1.0, "tps": 66.0}
    d = cell_delta(recs, cell, "accept", "dflash", 2000, 0)
    ok1 = 45 < d["delta_pct"] < 55 and not d["marginal"] and d["lo"] > 0
    # identical -> ~0 and marginal
    for i in range(50):
        recs["eagle3"][cell][i] = dict(recs["dominotree"][cell][i])
    d2 = cell_delta(recs, cell, "accept", "eagle3", 2000, 0)
    ok2 = abs(d2["delta_pct"]) < 1e-6 and d2["marginal"]
    rep = build_report(recs, "selftest", 500, 0)
    ok3 = "DominoTree Δτ%" in rep and "Macro-average" in rep
    print("  ok  clear win: +~50% tight CI, not marginal" if ok1 else "  FAIL clear-win case")
    print("  ok  identical: ~0% and MARGINAL" if ok2 else "  FAIL identical case")
    print("  ok  report builds with all sections" if ok3 else "  FAIL report build")
    print("ALL PASS" if (ok1 and ok2 and ok3) else "FAILURES")
    return 0 if (ok1 and ok2 and ok3) else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="Dir with <method>/*.prompts.jsonl sidecars (e.g. out/8b).")
    p.add_argument("--model", default="model", help="Label for the report header.")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-md", default=None, help="Write the markdown report here (also printed).")
    p.add_argument("--selftest", action="store_true", help="Offline synthetic check; no data needed.")
    args = p.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.root:
        raise SystemExit("--root is required (or use --selftest).")
    recs = load_sidecars(args.root)
    report = build_report(recs, args.model, args.bootstrap, args.seed)
    print(report)
    if args.out_md:
        Path(args.out_md).write_text(report, encoding="utf-8")
        print(f"[written] {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
