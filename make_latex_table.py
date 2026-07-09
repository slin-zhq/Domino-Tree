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
    ("domino_official", "domino", "Domino"),
    ("dominotree", "dominotree@16", "DominoTree (16)"),
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


def load_dominotree(raw_dir: Path, dataset: str, temp: str, drop_first: bool = True) -> dict[str, list[dict[str, Any]]]:
    # dominotree/ holds the DEFAULT results, built with the GPU-native CUDA-graph
    # tree builder. The Python reference-builder results (bit-identical trees, used
    # only for the builder comparison) live in dominotree_python_builder/.
    path = raw_dir / "dominotree" / f"{dataset}_T{temp}.jsonl"
    grouped = group_by_method(read_jsonl(path))
    # Warmup-row exclusion: drop the first execution row per method. These frozen
    # records were collected by the earlier harness, which did NOT run an in-loop
    # warmup prompt, so the first prompt of each method pays the cold-start cost
    # (one-time CUDA kernel compilation, cache/allocator warmup). Excluding it is
    # methodologically equivalent to the warmup prompt the reference DDTree/CaDDTree
    # benchmarks run before timing (and that benchmark.py now also runs): both
    # average over warm prompts only. We keep the exclusion here so these tables
    # reproduce the paper's numbers exactly; a fresh run from the warmup-enabled
    # benchmark.py is already warm from prompt 1 and would not need it.
    trim = slice(1, None) if drop_first else slice(None)  # 4B (no in-loop warmup) drops first row; 8B (warmup) keeps all
    return {method: sorted(rows, key=lambda r: r["_exec_idx"])[trim] for method, rows in grouped.items()}


def load_baseline_ddtree_caddtree(raw_dir: Path, dataset: str, temp: str) -> dict[str, list[dict[str, Any]]]:
    return group_by_method(read_jsonl(raw_dir / "baseline_ddtree_caddtree" / f"{dataset}_T{temp}.jsonl"))


def load_domino_official(raw_dir: Path, model_dir: str, dataset: str, temp: str, warmup: bool) -> dict[str, list[dict[str, Any]]]:
    """Official released Domino: best-of(graph, eager) per dataset, converted to per-row
    records {method:'domino', tps, mean_accept, sample_idx, turn_index}. `warmup=True`
    (4B, in-benchmark warmup) keeps all prompts; `warmup=False` (8B, no warmup) drops the
    first prompt. Returns {'domino': []} if the data is absent (graceful)."""
    base = raw_dir / "domino_official" / model_dir
    def mode_rows(mode: str) -> list[dict[str, Any]] | None:
        path = base / f"T{temp}" / f"{mode}_{dataset}.jsonl"
        if not path.exists():
            return None
        rows: list[dict[str, Any]] = []
        for r in read_jsonl(path):
            qid = int(r["question_id"])
            c = r["choices"][1]  # block-size 16
            accs = c.get("acceptance_lengths", [])
            for ti, (nt, dc) in enumerate(zip(c["new_tokens"], c["decode_times"])):
                if nt and dc > 0:
                    acc = accs[ti] if ti < len(accs) else []
                    rows.append({"method": "domino", "tps": nt / dc,
                                 "mean_accept": statistics.fmean(acc) if acc else float("nan"),
                                 "sample_idx": qid, "turn_index": ti})
        if not warmup:
            rows = [r for r in rows if r["sample_idx"] != 0]
        return rows
    g, e = mode_rows("graph"), mode_rows("eager")
    if g is None:
        return {"domino": []}
    mtps = lambda rs: statistics.fmean([r["tps"] for r in rs]) if rs else 0.0
    best = g if mtps(g) >= mtps(e or []) else e
    return add_exec_idx({"domino": best})


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


def collect_data(raw_dir: Path, temps: list[str], model_dir: str = "qwen3-4b",
                 domino_warmup: bool = True, dt_drop_first: bool = True) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for temp in temps:
        data[temp] = {}
        for dataset in DATASETS:
            data[temp][dataset] = {
                "dominotree": load_dominotree(raw_dir, dataset, temp, dt_drop_first),
                "baseline_ddtree_caddtree": load_baseline_ddtree_caddtree(raw_dir, dataset, temp),
                "domino_official": load_domino_official(raw_dir, model_dir, dataset, temp, domino_warmup),
            }
    return data


def common_ar_tps(data: dict[str, Any], temp: str, dataset: str) -> float:
    """The single lean common AR (our-harness 'ar') used to normalize every method.

    Our AR == the CaDDTree-harness AR (~66 tps, matched dataset-by-dataset); only
    Domino's own AR (spec_generate(block_size=1)) is anomalously slow (~55), which
    would inflate its speedup under own-AR normalization. Normalizing everything by
    one lean AR removes that artifact and barely moves DFlash/DDTree/CaDDTree (their
    AR is already ~66). See docs/domino_tree/domino_ar_overhead_proof_20260708.md.
    """
    return aggregate(data[temp][dataset]["dominotree"]["ar"])["tps"]


# Official Domino methods whose OWN AR (spec_generate(block_size=1)) is anomalously
# heavy (~23% slower than a lean AR); under "surgical" they are normalized by the lean
# common AR instead. Everyone else keeps their own (already-lean ~66 tps) harness AR.
DOMINO_OFFICIAL_METHODS = {"domino_graph", "domino_eager", "domino"}


def cell_metric(data: dict[str, Any], temp: str, dataset: str, harness: str, method: str,
                ar_norm: str = "surgical") -> dict[str, float]:
    rows = data[temp][dataset][harness][method]
    out = aggregate(rows)
    use_common = ar_norm == "common" or (ar_norm == "surgical" and method in DOMINO_OFFICIAL_METHODS)
    if use_common:
        ar_tps = common_ar_tps(data, temp, dataset)
    else:  # own harness AR (reference baselines + DominoTree; and everything under --ar-norm own)
        ar_method = "baseline" if harness == "baseline_ddtree_caddtree" else "ar"
        ar_tps = aggregate(data[temp][dataset][harness][ar_method])["tps"]
    out["speedup"] = out["tps"] / ar_tps if ar_tps and math.isfinite(ar_tps) else float("nan")
    if harness == "baseline_ddtree_caddtree" and method == "baseline":
        out["speedup"] = 1.0
        out["tau"] = 1.0
    return out


def write_table1(data: dict[str, Any], temps: list[str], out_dir: Path, model_label: str,
                 ar_norm: str = "surgical", suffix: str = "") -> None:
    columns = [LABELS[d] for d in DATASETS] + [f"{group} Avg" for group in GROUPS]
    norm_note = {
        "surgical": ("Speedup is relative to each method's own-harness AR, EXCEPT official Domino, "
                     "whose own AR (spec_generate(block_size=1)) is ~23% heavier than a lean AR and "
                     "is instead normalized by the lean common AR (our harness == CaDDTree harness, "
                     "~66 tps). See the repo AR-normalization note."),
        "common": ("Speedup is relative to ONE lean common AR (our-harness AR ~66 tps) for every method."),
        "own": ("Speedup is relative to each method's own harness AR TPS."),
    }[ar_norm]
    lines = [
        "# Table 1: Domino-style speedup / tau",
        "",
        f"Each cell is `speedup / tau`. {norm_note} DominoTree rows use warmup-row exclusion.",
        "",
    ]
    csv_rows = []
    for temp in temps:
        lines += [f"## Temperature = {temp}", ""]
        header = ["Model", "Method"] + columns
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for harness, method, label in TABLE1_METHODS:
            metrics = {ds: cell_metric(data, temp, ds, harness, method, ar_norm) for ds in DATASETS}
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
    (out_dir / f"table1{suffix}.md").write_text("\n".join(lines).rstrip() + "\n")
    with (out_dir / f"table1_cells{suffix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)


def write_table2(data: dict[str, Any], temps: list[str], out_dir: Path, raw_dir: Path) -> None:
    # Both methods are instrumented with identical stage boundaries. DominoTree's split comes from the
    # main sweep (raw/dominotree). Domino-chain's split comes from a dedicated instrumented run
    # (raw/chain_stage_timing), because the chain's original code path timed all post-draft work as one
    # fused block. Both were collected without an in-loop warmup prompt, so below we drop the first row
    # per dataset -- the same warmup-row exclusion as load_dominotree (see the rationale there): it
    # removes the cold-start prompt, equivalent to the reference/benchmark.py warmup prompt.
    def pooled(rows_by_dataset: list[list[dict[str, Any]]]) -> dict[str, float]:
        pooled_rows: list[dict[str, Any]] = []
        for rows in rows_by_dataset:
            pooled_rows.extend(rows)
        m = aggregate(pooled_rows)
        m["ms_total"] = m["ms_draft"] + m["ms_build"] + m["ms_verify"] + m["ms_commit"]
        m["n"] = len(pooled_rows)
        return m

    dtree = pooled([data["0.0"][ds]["dominotree"]["dominotree@16"] for ds in DATASETS])
    chain = pooled([read_jsonl(raw_dir / "chain_stage_timing" / f"{ds}_T0.0.jsonl")[1:] for ds in DATASETS])

    def row(label: str, m: dict[str, float]) -> str:
        return (
            f"| {label} | {fmt(m['ms_draft'])} | {fmt(m['ms_build'])} | {fmt(m['ms_verify'])} "
            f"| {fmt(m['ms_commit'])} | {fmt(m['ms_total'])} | {int(m['n'])} |"
        )

    lines = [
        "# Table 2: Per-round stage time",
        "",
        "Mean milliseconds per decoding round, our harness, Overall across all eight datasets at T=0,",
        "after warmup-row exclusion. Stage times are temperature-invariant (each stage varies <2% across",
        "T in {0, 0.5, 1.0}), so we report T=0. Both methods use identical stage boundaries: `build` is",
        "drafter-side candidate construction (the sequential GRU-correction pass for Domino-chain;",
        "best-first tree construction plus attention-mask assembly for DominoTree), `verify` is the single",
        "target forward, `commit` is the acceptance check plus KV/output write, and `total` is their sum.",
        "Domino-chain's stage split is from a dedicated instrumented run (`results/raw/chain_stage_timing/`),",
        "because the chain's original code path timed all post-draft work as one fused block.",
        "",
        "| Method | draft ms | build ms | verify ms | commit ms | total/round ms | n |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        row("Domino-chain", chain),
        row("DominoTree (16)", dtree),
    ]
    (out_dir / "table2.md").write_text("\n".join(lines).rstrip() + "\n")
    with (out_dir / "table2_stage_time.csv").open("w", newline="") as f:
        keys = ["method", "ms_draft", "ms_build", "ms_verify", "ms_commit", "ms_total", "n"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for label, m in [("Domino-chain", chain), ("DominoTree (16)", dtree)]:
            writer.writerow({"method": label, **{k: m[k] for k in keys if k != "method"}})


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
    if comparison == "domino":
        # DominoTree vs official Domino: raw per-prompt TPS, aligned by (sample_idx, turn_index)
        # across the two harnesses (both share the lean common AR, so raw-TPS ratio == speedup ratio).
        dt = {(int(r["sample_idx"]), int(r["turn_index"])): float(r["tps"])
              for r in dominotree["dominotree@16"] if math.isfinite(float(r.get("tps", float("nan"))))}
        dom = {(int(r["sample_idx"]), int(r["turn_index"])): float(r["tps"])
               for r in data[temp][dataset]["domino_official"]["domino"]}
        keys = sorted(set(dt) & set(dom))
        return [(dt[k], dom[k]) for k in keys]
    else:
        left = speedup_by_exec(dominotree["dominotree@16"], dominotree["ar"])
        baseline_method = comparison  # ddtree_tb16 | caddtree | dflash — all DFlash-harness baselines
        right = speedup_by_exec(baseline[baseline_method], baseline["baseline"])
    return [(left[idx], right[idx]) for idx in sorted(set(left) & set(right))]


def write_pairwise(data: dict[str, Any], temps: list[str], out_dir: Path, bootstrap_iters: int, seed: int, suffix: str = "") -> None:
    rng = random.Random(seed)
    # DFlash was added later; it draws from an independent stream so the original
    # three comparisons consume `rng` in the exact same order as before -> their
    # bootstrap CIs stay byte-identical, and DFlash is purely additive.
    rng_extra = random.Random(seed + 1)
    rng_domino = random.Random(12345)  # official-Domino column (matches the paper's recompute seed)
    comparisons = [
        ("domino", "DominoTree (16) vs Domino", "raw per-prompt TPS (shared lean common AR)"),
        ("ddtree_tb16", "DominoTree (16) vs DDTree@16", "speedup-over-own-AR (cross harness)"),
        ("caddtree", "DominoTree (16) vs CaDDTree", "speedup-over-own-AR (cross harness)"),
        ("dflash", "DominoTree (16) vs DFlash", "speedup-over-own-AR (cross harness)"),
    ]
    row_groups = [(LABELS[d], [d]) for d in DATASETS] + [(name, datasets) for name, datasets in GROUPS.items()]
    lines = [
        "# Pairwise delta with 95% paired bootstrap CI",
        "",
        "Delta is `100 * (mean(DominoTree (16) metric) / mean(baseline metric) - 1)`. Bootstrap resamples paired prompt rows.",
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
                rng_for = {"domino": rng_domino, "dflash": rng_extra}.get(key, rng)
                obs, lo, hi = paired_delta_ci(pairs, bootstrap_iters, rng_for)
                ci = f"[{fmt(lo)}, {fmt(hi)}]" if math.isfinite(lo) else "--"
                lines.append("| " + " | ".join([temp, name, label, metric, str(len(pairs)), fmt(obs), ci]) + " |")
                csv_rows.append({"temp": temp, "dataset_or_rollup": name, "comparison": label, "metric": metric, "n": len(pairs), "delta_pct": obs, "ci_low": lo, "ci_high": hi})
    (out_dir / f"pairwise_ci{suffix}.md").write_text("\n".join(lines).rstrip() + "\n")
    with (out_dir / f"pairwise_ci{suffix}.csv").open("w", newline="") as f:
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
        # Conditioning ablation holds the BUILDER fixed across cond/marg to isolate
        # the scorer, so both come from the Python reference builder (marg has no
        # GPU-native path); this is a controlled scorer comparison, not the default.
        dominotree_rows = read_jsonl(raw_dir / "dominotree_python_builder" / f"{dataset}_T0.0.jsonl")
        marginal_rows = read_jsonl(ablation_dir / f"{dataset}_T0.0.jsonl")
        ar_map = by_pair(dominotree_rows, "ar")
        cond_map = by_pair(dominotree_rows, "dominotree@16")
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
        "# Conditioning Ablation: DominoTree (16) vs marginal tree (DDTree-analogue)@16",
        "",
        "Matched-budget T=0.0 comparison using public JSONL records. DominoTree rows come from `raw/dominotree/*_T0.0.jsonl`; marginal-tree rows come from `raw/conditioning_ablation/*_T0.0.jsonl`.",
        "",
        "Speedup is relative to AR rows from the same DominoTree file. Delta is `DominoTree speedup / marginal-tree speedup - 1`; 95% CIs are paired bootstraps.",
        "",
        "| Dataset / Rollup | DominoTree speedup | DominoTree tau | marginal tree (DDTree-analogue) speedup | marginal tree (DDTree-analogue) tau | Delta DominoTree vs marginal tree (95% CI) | n pairs |",
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
    parser.add_argument("--ar-norm", choices=["surgical", "common", "own"], default="surgical",
                        help="Table 1 speedup normalization. 'surgical' (default): own-harness AR for "
                             "everyone, except official Domino, whose anomalously-heavy own AR is "
                             "replaced by the lean common AR (ours == CaDDTree's ~66). 'common': one "
                             "lean AR for all. 'own': each method over its own harness AR (pre-correction).")
    parser.add_argument("--domino-model-dir", default="qwen3-4b",
                        help="Subdir of results/raw/domino_official/ holding the official Domino JSONLs "
                             "(qwen3-4b or qwen3-8b).")
    parser.add_argument("--domino-no-warmup", action="store_true",
                        help="Set for models whose official Domino was collected WITHOUT an in-benchmark "
                             "warmup (e.g. 8B): drops the first prompt. Default (4B) keeps all prompts.")
    parser.add_argument("--no-warmup-drop", action="store_true",
                        help="Do NOT drop the first DominoTree row per method (for warmup-enabled "
                             "collections such as 8B). Default (4B frozen, no in-loop warmup) drops it.")
    parser.add_argument("--table-suffix", default="",
                        help="Filename suffix, e.g. '_8b' -> table1_8b.md / pairwise_ci_8b.md.")
    args = parser.parse_args()

    temps = [temp_token(t.strip()) for t in args.temps.split(",") if t.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = collect_data(args.raw_dir, temps, model_dir=args.domino_model_dir,
                        domino_warmup=not args.domino_no_warmup, dt_drop_first=not args.no_warmup_drop)

    def safe(name, fn, *fa, **fk):
        """Graceful degradation: build only tables whose inputs exist; report skips loudly."""
        try:
            fn(*fa, **fk)
            print(f"[ok]   {name}")
        except (FileNotFoundError, KeyError, IndexError, ZeroDivisionError) as ex:
            print(f"[skip] {name}: missing/insufficient data ({type(ex).__name__}: {ex})")

    safe("table1", write_table1, data, temps, args.out_dir, args.model_label, args.ar_norm, args.table_suffix)
    safe("pairwise", write_pairwise, data, temps, args.out_dir, args.bootstrap_iters, args.seed, args.table_suffix)
    safe("conditioning_ablation", write_conditioning_ablation_table, args.raw_dir, args.out_dir,
         args.conditioning_ablation_bootstrap_iters)
    # NOTE: the per-round stage-time table (former Table 2) is intentionally not built: official
    # Domino times all post-draft work as one fused block, so there is no DominoTree-vs-Domino
    # stage-level split at parity. Build cost is reported via the Python-vs-GPU-native builder table.
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
