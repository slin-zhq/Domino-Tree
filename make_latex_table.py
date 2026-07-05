#!/usr/bin/env python3
"""Regenerate DominoTree paper tables from public raw JSONLs.

Expected layout:

    results/raw/dominotree/*.jsonl
    results/raw/dominotree_recollected/*.jsonl
    results/raw/baseline_ddtree_caddtree/*.jsonl
    results/raw/conditioning_ablation/*.jsonl

The official baseline JSONLs are exported summaries of the official CaDDTree
cache rows. They intentionally contain no machine paths or generated text.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


DATASETS = ["gsm8k", "math500", "aime25", "humaneval", "mbpp", "livecodebench", "mt-bench", "alpaca"]
GROUPS = {
    "Math": ["gsm8k", "math500", "aime25"],
    "Code": ["humaneval", "mbpp", "livecodebench"],
    "Chat": ["mt-bench", "alpaca"],
    "Overall": DATASETS,
}
LABELS = {
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
    "aime25": "AIME25",
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
    "livecodebench": "LCB",
    "mt-bench": "MT-Bench",
    "alpaca": "Alpaca",
}
TABLE1_METHODS = [
    ("baseline_ddtree_caddtree", "baseline", "AR"),
    ("baseline_ddtree_caddtree", "dflash", "DFlash"),
    ("baseline_ddtree_caddtree", "ddtree_tb16", "DDTree@16"),
    ("baseline_ddtree_caddtree", "caddtree", "CaDDTree"),
    ("dominotree", "chain", "Domino-chain"),
    ("dominotree", "cond@16", "DominoTree cond@16"),
]


def mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.fmean(vals) if vals else float("nan")


def fmt(value: float, digits: int = 2) -> str:
    return "--" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def temp_token(value: str | float) -> str:
    return f"{float(value):.1f}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def add_exec_idx(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    for rows in grouped.values():
        for idx, row in enumerate(rows):
            row["_exec_idx"] = idx
    return grouped


def group_by_method(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    return add_exec_idx(dict(grouped))


def load_dominotree(raw_dir: Path, dataset: str, temp: str) -> dict[str, list[dict[str, Any]]]:
    override = raw_dir / "dominotree_recollected" / f"{dataset}_T{temp}.jsonl"
    path = override if override.exists() else raw_dir / "dominotree" / f"{dataset}_T{temp}.jsonl"
    grouped = group_by_method(read_jsonl(path))
    # Match the paper script: drop the first execution row per method for DominoTree rows.
    return {method: sorted(rows, key=lambda r: r["_exec_idx"])[1:] for method, rows in grouped.items()}


def load_baseline_ddtree_caddtree(raw_dir: Path, dataset: str, temp: str) -> dict[str, list[dict[str, Any]]]:
    return group_by_method(read_jsonl(raw_dir / "baseline_ddtree_caddtree" / f"{dataset}_T{temp}.jsonl"))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "tps": mean([float(r.get("tps", float("nan"))) for r in rows]),
        "tau": mean([float(r.get("mean_accept", float("nan"))) for r in rows]),
        "ms_draft": mean([float(r.get("ms_draft", float("nan"))) for r in rows]),
        "ms_build": mean([float(r.get("ms_build", float("nan"))) for r in rows]),
        "ms_verify": mean([float(r.get("ms_verify", float("nan"))) for r in rows]),
        "ms_commit": mean([float(r.get("ms_commit", float("nan"))) for r in rows]),
        "ms_chain": mean([float(r.get("ms_chain", float("nan"))) for r in rows]),
        "n": float(len(rows)),
    }


def collect_data(raw_dir: Path, temps: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for temp in temps:
        data[temp] = {}
        for dataset in DATASETS:
            data[temp][dataset] = {
                "dominotree": load_dominotree(raw_dir, dataset, temp),
                "baseline_ddtree_caddtree": load_baseline_ddtree_caddtree(raw_dir, dataset, temp),
            }
    return data


def cell_metric(data: dict[str, Any], temp: str, dataset: str, harness: str, method: str) -> dict[str, float]:
    rows = data[temp][dataset][harness][method]
    out = aggregate(rows)
    ar_method = "baseline" if harness == "baseline_ddtree_caddtree" else "ar"
    ar_tps = aggregate(data[temp][dataset][harness][ar_method])["tps"]
    out["speedup"] = out["tps"] / ar_tps if ar_tps and math.isfinite(ar_tps) else float("nan")
    if harness == "baseline_ddtree_caddtree" and method == "baseline":
        out["speedup"] = 1.0
        out["tau"] = 1.0
    return out


def write_table1(data: dict[str, Any], temps: list[str], out_dir: Path, model_label: str) -> None:
    columns = [LABELS[d] for d in DATASETS] + [f"{group} Avg" for group in GROUPS]
    lines = [
        "# Table 1: Domino-style speedup / tau",
        "",
        "Each cell is `speedup / tau`. Speedup is relative to the method's own harness AR TPS; DominoTree rows use warmup-row exclusion.",
        "",
    ]
    csv_rows = []
    for temp in temps:
        lines += [f"## Temperature = {temp}", ""]
        header = ["Model", "Method"] + columns
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for harness, method, label in TABLE1_METHODS:
            metrics = {ds: cell_metric(data, temp, ds, harness, method) for ds in DATASETS}
            cells = [f"{fmt(metrics[ds]['speedup'])} / {fmt(metrics[ds]['tau'])}" for ds in DATASETS]
            for group_datasets in GROUPS.values():
                cells.append(
                    f"{fmt(mean([metrics[ds]['speedup'] for ds in group_datasets]))} / "
                    f"{fmt(mean([metrics[ds]['tau'] for ds in group_datasets]))}"
                )
            lines.append("| " + " | ".join([model_label, label] + cells) + " |")
            for ds, metric in metrics.items():
                csv_rows.append(
                    {
                        "temp": temp,
                        "dataset": ds,
                        "model": model_label,
                        "harness": harness,
                        "method": label,
                        "speedup": metric["speedup"],
                        "tau": metric["tau"],
                        "tps": metric["tps"],
                        "n": metric["n"],
                    }
                )
        lines.append("")
    (out_dir / "table1.md").write_text("\n".join(lines).rstrip() + "\n")
    with (out_dir / "table1_cells.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)


def write_table2(data: dict[str, Any], temps: list[str], out_dir: Path) -> None:
    lines = [
        "# Table 2: Per-round stage time",
        "",
        "Stage times are mean milliseconds per decoding round from our harness after warmup-row exclusion.",
        "",
        "| Temp | Dataset | Method | draft ms | build ms | verify ms | commit ms | chain ms | n |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    csv_rows = []
    for temp in temps:
        for dataset in DATASETS + ["Overall"]:
            for method, label in [("chain", "Domino-chain"), ("cond@16", "DominoTree cond@16")]:
                rows = []
                if dataset == "Overall":
                    for ds in DATASETS:
                        rows.extend(data[temp][ds]["dominotree"][method])
                else:
                    rows = data[temp][dataset]["dominotree"][method]
                metric = aggregate(rows)
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            temp,
                            LABELS.get(dataset, dataset),
                            label,
                            fmt(metric["ms_draft"]),
                            fmt(metric["ms_build"]),
                            fmt(metric["ms_verify"]),
                            fmt(metric["ms_commit"]),
                            fmt(metric["ms_chain"]),
                            str(int(metric["n"])),
                        ]
                    )
                    + " |"
                )
                csv_rows.append({"temp": temp, "dataset": dataset, "method": label, **metric, "n": int(metric["n"])})
    (out_dir / "table2.md").write_text("\n".join(lines).rstrip() + "\n")
    with (out_dir / "table2_stage_time.csv").open("w", newline="") as f:
        keys = ["temp", "dataset", "method", "ms_draft", "ms_build", "ms_verify", "ms_commit", "ms_chain", "n"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row[k] for k in keys} for row in csv_rows)


def keyed(rows: list[dict[str, Any]], value_key: str) -> dict[int, float]:
    return {int(r["_exec_idx"]): float(r[value_key]) for r in rows if value_key in r and math.isfinite(float(r[value_key]))}


def speedup_by_exec(rows: list[dict[str, Any]], ar_rows: list[dict[str, Any]]) -> dict[int, float]:
    vals, ar = keyed(rows, "tps"), keyed(ar_rows, "tps")
    return {idx: vals[idx] / ar[idx] for idx in sorted(set(vals) & set(ar)) if ar[idx] > 0}


def paired_delta_ci(pairs: list[tuple[float, float]], iters: int, rng: random.Random) -> tuple[float, float, float]:
    if not pairs:
        return float("nan"), float("nan"), float("nan")

    def delta(sample):
        denom = mean([b for _, b in sample])
        return 100.0 * (mean([a for a, _ in sample]) / denom - 1.0) if denom > 0 else float("nan")

    obs = delta(pairs)
    if len(pairs) < 2 or iters <= 0:
        return obs, float("nan"), float("nan")
    vals = []
    for _ in range(iters):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        value = delta(sample)
        if math.isfinite(value):
            vals.append(value)
    vals.sort()
    return obs, vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def pairwise_units(data: dict[str, Any], temp: str, dataset: str, comparison: str) -> list[tuple[float, float]]:
    dominotree = data[temp][dataset]["dominotree"]
    baseline = data[temp][dataset]["baseline_ddtree_caddtree"]
    if comparison == "chain":
        left, right = keyed(dominotree["cond@16"], "tps"), keyed(dominotree["chain"], "tps")
    else:
        left = speedup_by_exec(dominotree["cond@16"], dominotree["ar"])
        baseline_method = "ddtree_tb16" if comparison == "ddtree_tb16" else "caddtree"
        right = speedup_by_exec(baseline[baseline_method], baseline["baseline"])
    return [(left[idx], right[idx]) for idx in sorted(set(left) & set(right))]


def write_pairwise(data: dict[str, Any], temps: list[str], out_dir: Path, bootstrap_iters: int, seed: int) -> None:
    rng = random.Random(seed)
    comparisons = [
        ("chain", "DominoTree cond@16 vs Domino-chain", "raw per-prompt TPS (same harness)"),
        ("ddtree_tb16", "DominoTree cond@16 vs DDTree@16", "speedup-over-own-AR (cross harness)"),
        ("caddtree", "DominoTree cond@16 vs CaDDTree", "speedup-over-own-AR (cross harness)"),
    ]
    row_groups = [(LABELS[d], [d]) for d in DATASETS] + [(name, datasets) for name, datasets in GROUPS.items()]
    lines = [
        "# Pairwise delta with 95% paired bootstrap CI",
        "",
        "Delta is `100 * (mean(DominoTree cond@16 metric) / mean(baseline metric) - 1)`. Bootstrap resamples paired prompt rows.",
        "",
        "| Temp | Dataset/Rollup | Comparison | Metric | N | Delta % | 95% CI |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    csv_rows = []
    for temp in temps:
        for name, datasets in row_groups:
            for key, label, metric in comparisons:
                pairs = []
                for ds in datasets:
                    pairs.extend(pairwise_units(data, temp, ds, key))
                obs, lo, hi = paired_delta_ci(pairs, bootstrap_iters, rng)
                ci = f"[{fmt(lo)}, {fmt(hi)}]" if math.isfinite(lo) else "--"
                lines.append("| " + " | ".join([temp, name, label, metric, str(len(pairs)), fmt(obs), ci]) + " |")
                csv_rows.append({"temp": temp, "dataset_or_rollup": name, "comparison": label, "metric": metric, "n": len(pairs), "delta_pct": obs, "ci_low": lo, "ci_high": hi})
    (out_dir / "pairwise_ci.md").write_text("\n".join(lines).rstrip() + "\n")
    with (out_dir / "pairwise_ci.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)


def write_conditioning_ablation_table(raw_dir: Path, out_dir: Path, bootstrap_iters: int) -> None:
    ablation_dir = raw_dir / "conditioning_ablation"
    if not ablation_dir.exists():
        return

    def by_pair(rows: list[dict[str, Any]], method: str) -> dict[tuple[int, int], dict[str, Any]]:
        return {
            (int(row["sample_idx"]), int(row.get("turn_index", 0))): row
            for row in rows
            if row.get("method") == method
        }

    def load_records(dataset: str) -> list[dict[str, float]]:
        dominotree_rows = read_jsonl(raw_dir / "dominotree" / f"{dataset}_T0.0.jsonl")
        marginal_rows = read_jsonl(ablation_dir / f"{dataset}_T0.0.jsonl")
        ar_map = by_pair(dominotree_rows, "ar")
        cond_map = by_pair(dominotree_rows, "cond@16")
        marg_map = by_pair(marginal_rows, "marg@16")
        keys = sorted(set(ar_map) & set(cond_map) & set(marg_map))
        return [
            {
                "ar_tps": float(ar_map[key]["tps"]),
                "cond_tps": float(cond_map[key]["tps"]),
                "marg_tps": float(marg_map[key]["tps"]),
                "cond_tau": float(cond_map[key]["mean_accept"]),
                "marg_tau": float(marg_map[key]["mean_accept"]),
            }
            for key in keys
        ]

    records_by_dataset = {dataset: load_records(dataset) for dataset in DATASETS}

    def summarize(datasets: list[str], source: dict[str, list[dict[str, float]]] | None = None) -> dict[str, float]:
        source = records_by_dataset if source is None else source
        parts = []
        for dataset in datasets:
            rows = source[dataset]
            ar_tps = mean([row["ar_tps"] for row in rows])
            cond_speed = mean([row["cond_tps"] for row in rows]) / ar_tps
            marg_speed = mean([row["marg_tps"] for row in rows]) / ar_tps
            parts.append(
                {
                    "cond_speed": cond_speed,
                    "marg_speed": marg_speed,
                    "cond_tau": mean([row["cond_tau"] for row in rows]),
                    "marg_tau": mean([row["marg_tau"] for row in rows]),
                    "n": len(rows),
                }
            )
        cond_speed = mean([part["cond_speed"] for part in parts])
        marg_speed = mean([part["marg_speed"] for part in parts])
        return {
            "cond_speed": cond_speed,
            "marg_speed": marg_speed,
            "delta": cond_speed / marg_speed - 1.0,
            "cond_tau": mean([part["cond_tau"] for part in parts]),
            "marg_tau": mean([part["marg_tau"] for part in parts]),
            "n": float(sum(part["n"] for part in parts)),
        }

    def bootstrap_ci(datasets: list[str]) -> tuple[float, float]:
        rng = random.Random(20260705)
        vals = []
        for _ in range(bootstrap_iters):
            sampled = {}
            for dataset in datasets:
                rows = records_by_dataset[dataset]
                sampled[dataset] = [rows[rng.randrange(len(rows))] for _ in rows]
            vals.append(summarize(datasets, sampled)["delta"])
        vals.sort()
        return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]

    rows = []
    for dataset in DATASETS:
        rows.append((LABELS[dataset], [dataset]))
    rows += [(f"{name} Avg", datasets) for name, datasets in GROUPS.items()]

    lines = [
        "# Conditioning Ablation: Cond@16 vs marginal tree (DDTree-analogue)@16",
        "",
        "Matched-budget T=0.0 comparison using public JSONL records. Cond rows come from `raw/dominotree/*_T0.0.jsonl`; marginal-tree rows come from `raw/conditioning_ablation/*_T0.0.jsonl`.",
        "",
        "Speedup is relative to AR rows from the same DominoTree file. Delta is `Cond speedup / marginal-tree speedup - 1`; 95% CIs are paired bootstraps.",
        "",
        "| Dataset / Rollup | Cond speedup | Cond tau | marginal tree (DDTree-analogue) speedup | marginal tree (DDTree-analogue) tau | Delta Cond vs marginal tree (95% CI) | n pairs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    clear = []
    for name, datasets in rows:
        summary = summarize(datasets)
        ci = bootstrap_ci(datasets)
        if name.endswith("Avg") and ci[0] > 0.0:
            clear.append(name)
        lines.append(
            f"| {name} | {fmt(summary['cond_speed'])} | {fmt(summary['cond_tau'])} | "
            f"{fmt(summary['marg_speed'])} | {fmt(summary['marg_tau'])} | "
            f"{100.0 * summary['delta']:+.1f}% [{100.0 * ci[0]:+.1f}, {100.0 * ci[1]:+.1f}] | "
            f"{int(summary['n'])} |"
        )
    lines += ["", "## Readout", "", f"Rollups with CI entirely above 0: {', '.join(clear) if clear else 'none'}.", ""]
    (out_dir / "conditioning_ablation.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--temps", default="0.0,0.5,1.0")
    parser.add_argument("--model-label", default="Qwen3-4B")
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--conditioning-ablation-bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    temps = [temp_token(t.strip()) for t in args.temps.split(",") if t.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = collect_data(args.raw_dir, temps)
    write_table1(data, temps, args.out_dir, args.model_label)
    write_table2(data, temps, args.out_dir)
    write_pairwise(data, temps, args.out_dir, args.bootstrap_iters, args.seed)
    write_conditioning_ablation_table(args.raw_dir, args.out_dir, args.conditioning_ablation_bootstrap_iters)
    manifest = {
        "raw_dir": str(args.raw_dir),
        "temps": temps,
        "warmup_exclusion": "DominoTree rows only: first execution row per dataset/temp/method",
        "baseline_ddtree_caddtree_source": "official CaDDTree repo commit a88f3f3 on native Qwen3-4B-DFlash-b16",
    }
    (args.out_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote tables to {args.out_dir}")


if __name__ == "__main__":
    main()
