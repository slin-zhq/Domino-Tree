#!/usr/bin/env python3
"""Aggregate the v2 concurrency sweeps (4B/TP=1 and 8B/TP=2) into the paper's
Table S2 "Overall" rows: goodput (output tok/s) and acceptance length tau,
averaged over the three datasets (gsm8k, mbpp, mt-bench) at each concurrency.

Emits BOTH an unweighted dataset mean and a prompt-count-weighted mean so the
convention actually used in the paper can be verified against the published 4B row.

Usage:
  python agg_conc_overall.py <sweep_root> [--label 8B]
    <sweep_root> holds <method>/<method>.jsonl  (or <method>/*.jsonl)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

METHODS = ["ar", "eagle3", "dflash", "domino_chain", "dominotree"]
DATASETS = ["gsm8k", "mbpp", "mt-bench"]
CONCS = [1, 2, 4, 8, 16, 32]


def load(root: Path, method: str):
    d = root / method
    rows = []
    if not d.is_dir():
        return rows
    files = sorted(d.glob(f"{method}.jsonl")) or sorted(d.glob("*.jsonl"))
    for fp in files:
        for line in fp.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    label = "model"
    if "--label" in sys.argv:
        label = sys.argv[sys.argv.index("--label") + 1]
    root = Path(args[0]).resolve()

    # cell[(method, dataset, conc)] = row
    cell = {}
    npr = {}
    for m in METHODS:
        for r in load(root, m):
            cell[(m, r["dataset"], int(r["concurrency"]))] = r
            npr[r["dataset"]] = r.get("n_prompts")

    print(f"# Concurrency Overall aggregation — {label}  ({root})\n")
    print("## Coverage (cells present / %d expected)" % (len(DATASETS) * len(CONCS)))
    for m in METHODS:
        have = sum(1 for ds in DATASETS for c in CONCS if (m, ds, c) in cell)
        print(f"- {m:14s}: {have}/{len(DATASETS) * len(CONCS)}")
    print(f"\nn_prompts per dataset: {npr}\n")

    for metric, name, fmt in [("tps", "Goodput (output tok/s)", "{:.0f}"),
                              ("mean_accept", "Acceptance length tau", "{:.2f}")]:
        for weighted in (False, True):
            tag = "prompt-count-weighted" if weighted else "unweighted dataset mean"
            print(f"## {name} — Overall ({tag})")
            print("| method | " + " | ".join(f"c={c}" for c in CONCS) + " |")
            print("|" + "---|" * (1 + len(CONCS)))
            for m in METHODS:
                out = [m]
                for c in CONCS:
                    vals, wts = [], []
                    for ds in DATASETS:
                        r = cell.get((m, ds, c))
                        if r is None or r.get(metric) is None:
                            continue
                        vals.append(r[metric])
                        wts.append(r.get("n_prompts") or 1)
                    if not vals:
                        out.append("--")
                    elif weighted:
                        out.append(fmt.format(sum(v * w for v, w in zip(vals, wts)) / sum(wts)))
                    else:
                        out.append(fmt.format(sum(vals) / len(vals)))
                print("| " + " | ".join(out) + " |")
            print()

    # Speedup over AR (unweighted Overall), the paper's secondary read.
    print("## Speedup over served AR (unweighted Overall goodput ratio)")
    print("| method | " + " | ".join(f"c={c}" for c in CONCS) + " |")
    print("|" + "---|" * (1 + len(CONCS)))

    def overall(m, c, metric="tps"):
        vals = [cell[(m, ds, c)][metric] for ds in DATASETS
                if (m, ds, c) in cell and cell[(m, ds, c)].get(metric) is not None]
        return sum(vals) / len(vals) if vals else None

    for m in METHODS:
        out = [m]
        for c in CONCS:
            a, b = overall("ar", c), overall(m, c)
            out.append("--" if (a is None or b is None or a == 0) else f"{b / a:.2f}x")
        print("| " + " | ".join(out) + " |")
    print()

    # DominoTree vs Domino chain, the within-drafter spine.
    print("## DominoTree vs Domino chain (Overall)")
    print("| conc | tps DT | tps chain | delta% | tau DT | tau chain | delta% |")
    print("|---|---|---|---|---|---|---|")
    for c in CONCS:
        dt, ch = overall("dominotree", c), overall("domino_chain", c)
        dta, cha = overall("dominotree", c, "mean_accept"), overall("domino_chain", c, "mean_accept")
        if None in (dt, ch, dta, cha):
            continue
        print(f"| {c} | {dt:.0f} | {ch:.0f} | {100*(dt/ch-1):+.1f}% | "
              f"{dta:.2f} | {cha:.2f} | {100*(dta/cha-1):+.1f}% |")


if __name__ == "__main__":
    main()
