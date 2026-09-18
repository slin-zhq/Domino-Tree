#!/usr/bin/env python3
"""Aggregate the Qwen3-4B fused-frontier Table 1 recollection.

Usage: agg_tab1f_4b.py NEW_ROOT OLD_ROOT
Both roots may be a run directory containing ``out/`` or an output directory.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DATASETS = ("gsm8k", "math500", "aime25", "humaneval", "mbpp", "livecodebench", "mt-bench", "alpaca")
TEMPS = ("0.0", "0.5", "1.0")
BUDGETS = (16, 32, 64)


def out_dir(root: Path) -> Path:
    return root / "out" if (root / "out").is_dir() else root


def load(root: Path):
    rows = defaultdict(list)
    for path in sorted(out_dir(root).glob("*_T*.jsonl")):
        stem = path.name.removesuffix(".jsonl")
        dataset, temp = stem.rsplit("_T", 1)
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    rows[(dataset, temp, json.loads(line)["method"])].append(json.loads(line))
    return rows


def summary(rows):
    if not rows:
        return None
    return {
        "n": len(rows),
        "tau": statistics.fmean(row["mean_accept"] for row in rows),
        "tps": statistics.fmean(row["tps"] for row in rows),
        "build": statistics.fmean(row.get("ms_build", 0.0) for row in rows),
    }


def ar_tps(data, dataset):
    arm = summary(data.get((dataset, "0.0", "ar"), []))
    return arm["tps"] if arm else None


def with_speed(data, dataset, temp, budget):
    item = summary(data.get((dataset, temp, f"dominotree@{budget}"), []))
    base = ar_tps(data, dataset)
    if item:
        item["speedup"] = item["tps"] / base if base else None
    return item


def fmt(item):
    if item is None:
        return "-"
    speed = "-" if item["speedup"] is None else f"{item['speedup']:.3f}x"
    return f"n={item['n']} tau={item['tau']:.3f} tps={item['tps']:.1f} xAR={speed} build={item['build']:.3f}ms"


def overall(items):
    valid = [item for item in items if item]
    if not valid:
        return None
    return {
        "n": sum(item["n"] for item in valid),
        "tau": statistics.fmean(item["tau"] for item in valid),
        "tps": statistics.fmean(item["tps"] for item in valid),
        "build": statistics.fmean(item["build"] for item in valid),
        "speedup": statistics.fmean(item["speedup"] for item in valid if item["speedup"] is not None)
        if any(item["speedup"] is not None for item in valid) else None,
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} NEW_ROOT OLD_ROOT")
    new, old = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
    print("temperature dataset budget | NEW (rows, tau, tok/s, same-session AR speedup, ms_build) | OLD tab1_4b")
    for temp in TEMPS:
        for dataset in DATASETS:
            for budget in BUDGETS:
                nitem = with_speed(new, dataset, temp, budget)
                oitem = with_speed(old, dataset, temp, budget)
                print(f"{temp:>3} {dataset:>12} B={budget:<2} | {fmt(nitem):<63} | {fmt(oitem)}")
        for budget in BUDGETS:
            nitems = [with_speed(new, dataset, temp, budget) for dataset in DATASETS]
            oitems = [with_speed(old, dataset, temp, budget) for dataset in DATASETS]
            print(f"{temp:>3} {'Overall':>12} B={budget:<2} | {fmt(overall(nitems)):<63} | {fmt(overall(oitems))}")

    t0 = {budget: overall([with_speed(new, dataset, "0.0", budget) for dataset in DATASETS]) for budget in BUDGETS}
    available = {budget: item for budget, item in t0.items() if item}
    if available:
        winner = max(available, key=lambda budget: available[budget]["tps"])
        print("\nT=0 Overall TPS by budget: " + ", ".join(f"B={budget}: {item['tps']:.1f}" for budget, item in available.items()))
        print(f"T=0 Overall TPS argmax: B={winner} ({available[winner]['tps']:.1f} tok/s)")


if __name__ == "__main__":
    main()
