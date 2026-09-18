#!/usr/bin/env python3
"""Check every serving number in the DominoTree v3 (IEEE Access) paper against the raw
data shipped in this directory.

Nothing here trusts a summary table. It reads the per-prompt and per-cell JSONL under
this directory, recomputes every cell of every serving table, and diffs the result
against `PUBLISHED.json` -- which holds the values exactly as printed in the paper. It
exits non-zero if anything disagrees, is missing, or is short (too few prompts/rows).

Run it from anywhere:

    python3 results/serving/verify_published_numbers.py

Expected output ends with: `ALL <N> CELLS REPRODUCE FROM RAW DATA.`

Hardware / protocol this bundle documents (see results/README.md and results/serving/README.md
for the full story): a single RTX 5090/32GB, TP=1, for BOTH Qwen3-4B and Qwen3-8B.

  bs=1        (tab:sglang-bs1, tab:sglang-bs1-cis)   8 datasets x {T=0,T=1}, tree budget 32,
                                                       n=50/dataset (AIME25 n=30, MT-Bench both turns)
  concurrency (tab:sglang-conc, tab:app-conc)         gsm8k/mbpp/mt-bench, c=2..32, T=0, 512-token cap
  long context(tab:sglang-longctx, tab:longctx-budget,
               tab:app-helmet, tab:app-helmet-8b)      HELMET summarization, T=0, n=50/cell,
                                                       cap-1200 generation; DominoTree budget
                                                       16/32(/64 at 8B) swept explicitly

Conventions, stated here so you can check that we describe what we do:

  bs=1
    Per-prompt TPS  = a prompt's total output tokens / its total decode time (MT-Bench sums
                      both turns before dividing -- a turn is not its own sampling unit).
    Per-prompt tau  = MACRO within the prompt: num_output / (num_output / mean_accept) summed
                      over the prompt's turns, i.e. accepted-length is turn-length-weighted
                      within a prompt. A cell's reported tau is the MACRO mean over prompts of
                      this per-prompt value (mean of means, not tokens-over-steps).
    Overall         = unweighted mean over the 8 datasets of the per-dataset cell value.
    CIs             = paired percentile bootstrap over prompts (DominoTree vs. Domino chain),
                      B=5000, seed=0, resampling TPS (paired by sample_idx).

  Concurrency (Table "SGLang goodput under concurrency")
    Goodput at offered concurrency c = UNWEIGHTED mean over the three datasets
    (gsm8k, mbpp, mt-bench). tau = unweighted mean over datasets AND over the full measured
    sweep c = 2,4,8,16,32 (tau is flat in c to <0.1).

  Long context (Table "SGLang long-context single-stream")
    A cell averages the two HELMET summarization tasks. tau = mean over tasks of that task's
    mean accepted length. speedup = mean over tasks of (task mean tps / AR's mean tps on the
    same task) -- mean-of-ratios, not ratio-of-means. Qwen3-4B's main-table DominoTree column
    is tree budget 16; Qwen3-8B's is tree budget 32 (each model's measured long-context
    optimum up to 16K; see tab:longctx-budget and longcontext/<size>/dominotree_b32/).

Requires only the Python standard library.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLISHED = HERE / "PUBLISHED.json"

METHODS = ["ar", "eagle3", "dflash", "domino_chain", "dominotree"]
BS1_DATASETS = ["gsm8k", "math500", "aime25", "humaneval", "mbpp",
                "livecodebench", "mt-bench", "alpaca"]
CONC_DATASETS = ["gsm8k", "mbpp", "mt-bench"]
CONCS = [2, 4, 8, 16, 32]
TASKS = ["infbench_sum", "multi_lexsum"]
BINS = [8192, 16384, 32768]
TEMPS = ["0.0", "1.0"]
SIZES = ["4b", "8b"]
# Which longcontext/<size>/<dir> backs the DominoTree column of the MAIN long-context
# table (tab:sglang-longctx): 4B at tree budget 16 (the default dir), 8B at budget 32
# (its own, separately measured long-context optimum -- see tab:longctx-budget).
LONGCTX_MAIN_ARM = {"4b": "dominotree", "8b": "dominotree_b32"}
BOOT, SEED = 5000, 0

fails: list[str] = []
_count = 0


def note(ok: bool, msg: str) -> None:
    global _count
    _count += 1
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        fails.append(msg)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def require(rows: list, min_n: int, tag: str) -> bool:
    """FAIL LOUDLY on a missing or short cell -- never silently skip it."""
    if len(rows) < min_n:
        note(False, f"{tag}: {len(rows)} rows, need >= {min_n} (missing or truncated file)")
        return False
    return True


# --------------------------------------------------------------------- bs=1 (per-prompt)
def bs1_path(size: str, method: str, dataset: str, temp: str) -> Path:
    return HERE / "bs1" / size / method / f"{dataset}_T{temp}.jsonl"


def bs1_cell(size: str, method: str, dataset: str, temp: str) -> dict:
    by_prompt: dict = {}
    for r in read_jsonl(bs1_path(size, method, dataset, temp)):
        by_prompt.setdefault(r["sample_idx"], []).append(r)
    return by_prompt


def prompt_tps(rows: list) -> float:
    return sum(r["num_output"] for r in rows) / sum(r["decode_time"] for r in rows)


def prompt_tau(rows: list) -> float:
    num = sum(r["num_output"] for r in rows)
    steps = sum(r["num_output"] / r["mean_accept"] for r in rows if r.get("mean_accept"))
    return num / steps if steps > 0 else float("nan")


def expected_n_prompts(dataset: str) -> int:
    return 30 if dataset == "aime25" else 50


def paired_bootstrap(a: dict, b: dict, stat) -> tuple:
    keys = sorted(set(a) & set(b))
    if set(a) ^ set(b):
        raise ValueError(f"UNPAIRED: {len(set(a) ^ set(b))} sample_idx in only one arm")
    av = [stat(a[k]) for k in keys]
    bv = [stat(b[k]) for k in keys]
    point = 100.0 * (st.mean(av) / st.mean(bv) - 1.0)
    rng = random.Random(SEED)
    n = len(keys)
    draws = []
    for _ in range(BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        mb = st.mean(bv[i] for i in idx)
        if mb:
            draws.append(100.0 * (st.mean(av[i] for i in idx) / mb - 1.0))
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return point, lo, hi, n


def check_bs1(published: dict) -> None:
    print("== bs=1 (tab:sglang-bs1) — per-dataset TPS / tau, MACRO ==")
    for size in SIZES:
        for temp in TEMPS:
            overall = {m: ([], []) for m in METHODS}
            for dataset in BS1_DATASETS:
                cells = {m: bs1_cell(size, m, dataset, temp) for m in METHODS}
                ok = True
                for m in METHODS:
                    ok &= require(list(cells[m].values()), expected_n_prompts(dataset),
                                  f"bs1 {size}/T{temp}/{dataset}/{m}")
                if not ok:
                    continue
                for m in METHODS:
                    cell = cells[m]
                    tps = st.mean(prompt_tps(v) for v in cell.values())
                    tau = st.mean(prompt_tau(v) for v in cell.values())
                    overall[m][0].append(tps)
                    overall[m][1].append(tau)
                    key = f"{size}/{temp}/{dataset}/{m}"
                    exp = published.get(key)
                    if exp is None:
                        note(False, f"{key}: no PUBLISHED.json entry")
                        continue
                    note(abs(round(tps) - exp["tps"]) < 0.5,
                         f"{key:<32} tps  data {tps:8.2f} -> {round(tps):.0f}   paper {exp['tps']:.0f}")
                    if m != "ar":
                        note(abs(round(tau, 2) - exp["tau"]) < 5e-3,
                             f"{key:<32} tau  data {tau:8.4f} -> {round(tau, 2):.2f}   paper {exp['tau']:.2f}")
            for m in METHODS:
                tpss, taus = overall[m]
                if len(tpss) != len(BS1_DATASETS):
                    continue
                key = f"{size}/{temp}/{m}"
                exp = published.get("_overall", {}).get(key)
                if exp is None:
                    note(False, f"OVERALL {key}: no PUBLISHED.json entry")
                    continue
                otps = st.mean(tpss)
                note(abs(round(otps) - exp["tps"]) < 0.5,
                     f"OVERALL {key:<24} tps  data {otps:8.2f} -> {round(otps):.0f}   paper {exp['tps']:.0f}")
                if m != "ar":
                    otau = st.mean(taus)
                    note(abs(round(otau, 2) - exp["tau"]) < 5e-3,
                         f"OVERALL {key:<24} tau  data {otau:8.4f} -> {round(otau, 2):.2f}   paper {exp['tau']:.2f}")


def check_bs1_cis(published: dict) -> None:
    print("\n== bs=1 DominoTree vs. Domino chain, paired CI (tab:sglang-bs1-cis) ==")
    for size in SIZES:
        for temp in TEMPS:
            for dataset in BS1_DATASETS:
                tree = bs1_cell(size, "dominotree", dataset, temp)
                chain = bs1_cell(size, "domino_chain", dataset, temp)
                key = f"{size}/{temp}/{dataset}"
                exp = published.get(key)
                if exp is None:
                    note(False, f"{key}: no PUBLISHED.json bs1_cis entry")
                    continue
                if not tree or not chain:
                    note(False, f"{key}: missing DominoTree or Domino-chain bs1 data")
                    continue
                try:
                    p, lo, hi, n = paired_bootstrap(tree, chain, prompt_tps)
                except ValueError as e:
                    note(False, f"{key}: {e}")
                    continue
                note(n == expected_n_prompts(dataset), f"{key}: n={n} prompts (paired)")
                note(abs(round(p, 1) - exp[0]) < 0.15,
                     f"{key:<20} deltaTPS data {p:7.2f} -> {round(p, 1):+.1f}   paper {exp[0]:+.1f}")


# --------------------------------------------------------------------- concurrency
def conc_rows(size: str, method: str) -> list:
    return read_jsonl(HERE / "concurrency" / size / method / f"{method}.jsonl")


def check_conc(published: dict) -> None:
    print("\n== Concurrency goodput, Overall (tab:sglang-conc) ==")
    for size in SIZES:
        for method in METHODS:
            rows = conc_rows(size, method)
            cells = {(r["dataset"], int(r["concurrency"])): r for r in rows}
            key = f"{size}/{method}"
            exp = published.get(key)
            if exp is None:
                note(False, f"{key}: no PUBLISHED.json concurrency entry")
                continue
            missing = {(d, c) for d in CONC_DATASETS for c in CONCS} - set(cells)
            if missing:
                note(False, f"{key}: missing cells {sorted(missing)}")
                continue
            for c in CONCS:
                vals = [cells[(d, c)]["tps"] for d in CONC_DATASETS]
                got = sum(vals) / len(vals)
                want = exp["tps"].get(str(c))
                note(abs(round(got) - want) < 0.5,
                     f"{key:<18} tps c={c:<2} data {got:8.2f} -> {round(got):.0f}   paper {want:.0f}")
            if exp["tau"] is not None:
                acc = [cells[(d, c)]["mean_accept"] for d in CONC_DATASETS for c in CONCS]
                got = sum(acc) / len(acc)
                note(abs(round(got, 2) - exp["tau"]) < 5e-3,
                     f"{key:<18} tau        data {got:8.4f} -> {round(got, 2):.2f}   paper {exp['tau']:.2f}")


def check_conc_appendix(published: dict) -> None:
    print("\n== Concurrency per-dataset detail (tab:app-conc) ==")
    for size in SIZES:
        for method in METHODS:
            rows = conc_rows(size, method)
            cells = {(r["dataset"], int(r["concurrency"])): r for r in rows}
            for d in CONC_DATASETS:
                n_expect = 80 if d == "mt-bench" else 128
                for c in CONCS:
                    r = cells.get((d, c))
                    key = f"{size}/{d}/{method}/{c}"
                    if r is None:
                        note(False, f"{key}: MISSING cell")
                        continue
                    exp = published.get(key)
                    if exp is None:
                        note(False, f"{key}: no PUBLISHED.json app-conc entry")
                        continue
                    note(r.get("n_prompts") == n_expect, f"{key}: n_prompts={r.get('n_prompts')}")
                    note(abs(round(r["tps"]) - exp["tps"]) < 0.5,
                         f"{key:<28} tps {r['tps']:8.2f} -> {round(r['tps']):.0f}   paper {exp['tps']:.0f}")
                    if method != "ar":
                        note(abs(round(r["mean_accept"], 2) - exp["tau"]) < 5e-3,
                             f"{key:<28} tau {r['mean_accept']:8.4f} -> {round(r['mean_accept'], 2):.2f}"
                             f"   paper {exp['tau']:.2f}")


# --------------------------------------------------------------------- long context
def helmet_dir(size: str, arm: str) -> Path:
    return HERE / "longcontext" / size / arm


def helmet_agg(size: str, arm: str) -> dict:
    return {(r["task"], int(r["length_bin"])): r
            for r in read_jsonl(helmet_dir(size, arm) / "helmet.jsonl")}


def helmet_prompts(size: str, arm: str) -> dict:
    out: dict = {}
    for r in read_jsonl(helmet_dir(size, arm) / "helmet.prompts.jsonl"):
        out.setdefault((r["task"], int(r["length_bin"])), {})[r["idx"]] = r
    return out


def helmet_cell_stats(prompts: dict, cell: tuple) -> tuple:
    rows = prompts.get(cell, {})
    tok = sum(r["output_tokens"] for r in rows.values())
    sec = sum(r["decode_time"] for r in rows.values())
    tps = tok / sec if sec else float("nan")
    accs = [r["accept"] for r in rows.values() if r.get("accept")]
    tau = st.mean(accs) if accs else 1.0
    return tps, tau


def check_longctx(published: dict) -> None:
    print("\n== Long context, main table (tab:sglang-longctx) ==")
    for size in SIZES:
        prompts = {m: helmet_prompts(size, m) for m in METHODS}
        prompts[LONGCTX_MAIN_ARM[size]] = helmet_prompts(size, LONGCTX_MAIN_ARM[size])
        for b in BINS:
            ar_tps = {t: helmet_cell_stats(prompts["ar"], (t, b))[0] for t in TASKS}
            for m in METHODS:
                arm = LONGCTX_MAIN_ARM[size] if m == "dominotree" else m
                key = f"{size}/{b}/{m}"
                exp = published.get(key)
                if exp is None:
                    note(False, f"{key}: no PUBLISHED.json longctx entry")
                    continue
                if any(len(prompts[arm].get((t, b), {})) < 50 for t in TASKS):
                    note(False, f"{key}: fewer than 50 prompts in a HELMET cell")
                    continue
                cells = [helmet_cell_stats(prompts[arm], (t, b)) for t in TASKS]
                tau = st.mean(c[1] for c in cells)
                spd = st.mean(c[0] / ar_tps[t] for c, t in zip(cells, TASKS))
                tag = f"{size}/{b // 1024}K {m:<13}"
                note(abs(round(tau, 2) - exp[0]) < 5e-3,
                     f"{tag} tau      data {tau:6.3f} -> {round(tau, 2):.2f}   paper {exp[0]:.2f}")
                note(abs(round(spd, 2) - exp[1]) < 5e-3,
                     f"{tag} speedup  data {spd:6.4f} -> {round(spd, 2):.2f}   paper {exp[1]:.2f}")


def check_longctx_budget(published: dict) -> None:
    print("\n== Long-context tree-budget sweep, DominoTree only (tab:longctx-budget) ==")
    arm_for_budget = {
        "4b": {16: "dominotree", 32: "dominotree_b32"},
        "8b": {16: "dominotree", 32: "dominotree_b32", 64: "dominotree_b64"},
    }
    for size, budgets in arm_for_budget.items():
        for b, arm in budgets.items():
            prompts = helmet_prompts(size, arm)
            for ctx_k, bin_ in [(8, 8192), (16, 16384), (32, 32768)]:
                key = f"{size}/{ctx_k}/{b}"
                exp = published.get(key)
                if any(len(prompts.get((t, bin_), {})) < 50 for t in TASKS):
                    note(False, f"{key}: fewer than 50 prompts (arm={arm})")
                    continue
                if exp is None:
                    note(False, f"{key}: no PUBLISHED.json longctx_budget entry")
                    continue
                cells = [helmet_cell_stats(prompts, (t, bin_)) for t in TASKS]
                tps = st.mean(c[0] for c in cells)
                note(abs(round(tps, 1) - exp) < 0.05,
                     f"{key:<12} (arm={arm:<15}) tps  data {tps:8.3f} -> {round(tps, 1):.1f}"
                     f"   paper {exp:.1f}")


def check_helmet_appendix(published: dict) -> None:
    print("\n== HELMET per-task detail (tab:app-helmet, tab:app-helmet-8b) ==")
    arms = {"4b": METHODS, "8b": METHODS + ["dominotree_b32"]}
    for size, arm_list in arms.items():
        agg = {a: helmet_agg(size, a) for a in arm_list}
        per = {a: helmet_prompts(size, a) for a in arm_list}
        for t in TASKS:
            for b in BINS:
                for a in arm_list:
                    key = f"{size}/{t}/{b}/{a}"
                    exp = published.get(key)
                    row = agg[a].get((t, b))
                    if row is None:
                        note(False, f"{key}: MISSING aggregate row")
                        continue
                    if exp is None:
                        note(False, f"{key}: no PUBLISHED.json helmet_appendix entry")
                        continue
                    tps_raw, tau_raw = helmet_cell_stats(per[a], (t, b))
                    note(abs(round(tps_raw, 1) - exp["tps"]) < 0.05,
                         f"{key:<28} tps {tps_raw:8.3f} -> {round(tps_raw, 1):.1f}   paper {exp['tps']:.1f}")
                    note(abs(round(tau_raw, 2) - exp["tau"]) < 5e-3,
                         f"{key:<28} tau {tau_raw:8.4f} -> {round(tau_raw, 2):.2f}   paper {exp['tau']:.2f}")
                # paired CI: dominotree (and, at 8B, dominotree_b32) vs domino_chain
                for a in [x for x in arm_list if x.startswith("dominotree")]:
                    key = f"{size}/{t}/{b}/{a}"
                    exp_ci = published.get("_ci", {}).get(key)
                    if exp_ci is None:
                        note(False, f"{key}: no PUBLISHED.json helmet CI entry")
                        continue
                    common = sorted(set(per[a].get((t, b), {})) & set(per["domino_chain"].get((t, b), {})))
                    if len(common) != 50:
                        note(False, f"{key}: CI pairing has {len(common)} prompts, need 50")
                        continue
                    for field, metric in (("tps", "tps"), ("accept", "tau")):
                        av = [per[a][(t, b)][i][field] for i in common]
                        bv = [per["domino_chain"][(t, b)][i][field] for i in common]
                        point = 100.0 * (st.mean(av) / st.mean(bv) - 1.0)
                        want = exp_ci[metric][0]
                        note(abs(round(point, 1) - want) < 0.15,
                             f"{key:<28} d{metric}%  data {point:7.2f} -> {round(point, 1):+.1f}"
                             f"   paper {want:+.1f}")


# --------------------------------------------------------------------- --emit support
def emit_all() -> dict:
    """Recompute every published cell straight from raw data (no paper dependency). Used to
    regenerate PUBLISHED.json; the values still need an independent cross-check against the
    paper's LaTeX source, which this public bundle does not ship (done in the working repo's
    scripts/domino_tree_ieee_access/ before a new PUBLISHED.json is committed here)."""
    bs1: dict = {}
    overall: dict = {}
    for size in SIZES:
        for temp in TEMPS:
            acc = {m: ([], []) for m in METHODS}
            for dataset in BS1_DATASETS:
                for m in METHODS:
                    cell = bs1_cell(size, m, dataset, temp)
                    if not cell:
                        continue
                    tps = st.mean(prompt_tps(v) for v in cell.values())
                    tau = st.mean(prompt_tau(v) for v in cell.values())
                    bs1[f"{size}/{temp}/{dataset}/{m}"] = {"tps": round(tps), "tau": round(tau, 2)}
                    acc[m][0].append(tps)
                    acc[m][1].append(tau)
            for m in METHODS:
                if len(acc[m][0]) == len(BS1_DATASETS):
                    overall[f"{size}/{temp}/{m}"] = {"tps": round(st.mean(acc[m][0])),
                                                      "tau": round(st.mean(acc[m][1]), 2)}
    bs1["_overall"] = overall

    bs1_cis: dict = {}
    for size in SIZES:
        for temp in TEMPS:
            for dataset in BS1_DATASETS:
                tree = bs1_cell(size, "dominotree", dataset, temp)
                chain = bs1_cell(size, "domino_chain", dataset, temp)
                if tree and chain:
                    p, lo, hi, n = paired_bootstrap(tree, chain, prompt_tps)
                    bs1_cis[f"{size}/{temp}/{dataset}"] = [round(p, 1), round(lo, 1), round(hi, 1)]

    conc: dict = {}
    conc_ds: dict = {}
    for size in SIZES:
        for method in METHODS:
            rows = conc_rows(size, method)
            cells = {(r["dataset"], int(r["concurrency"])): r for r in rows}
            if len(cells) < len(CONC_DATASETS) * len(CONCS):
                continue
            tps_by_c = {str(c): round(sum(cells[(d, c)]["tps"] for d in CONC_DATASETS) / len(CONC_DATASETS))
                        for c in CONCS}
            tau = None
            if method != "ar":
                acc = [cells[(d, c)]["mean_accept"] for d in CONC_DATASETS for c in CONCS]
                tau = round(sum(acc) / len(acc), 2)
            conc[f"{size}/{method}"] = {"tps": tps_by_c, "tau": tau}
            for d in CONC_DATASETS:
                for c in CONCS:
                    r = cells[(d, c)]
                    conc_ds[f"{size}/{d}/{method}/{c}"] = {
                        "tps": round(r["tps"]),
                        "tau": round(r["mean_accept"], 2) if method != "ar" else None}
    conc["_format"] = "size/method -> {tps: {offered_concurrency: goodput}, tau}"

    longctx: dict = {}
    for size in SIZES:
        prompts = {m: helmet_prompts(size, m) for m in METHODS}
        prompts[LONGCTX_MAIN_ARM[size]] = helmet_prompts(size, LONGCTX_MAIN_ARM[size])
        for b in BINS:
            ar_tps = {t: helmet_cell_stats(prompts["ar"], (t, b))[0] for t in TASKS}
            for m in METHODS:
                arm = LONGCTX_MAIN_ARM[size] if m == "dominotree" else m
                cells = [helmet_cell_stats(prompts[arm], (t, b)) for t in TASKS]
                tau = st.mean(c[1] for c in cells)
                spd = st.mean(c[0] / ar_tps[t] for c, t in zip(cells, TASKS))
                longctx[f"{size}/{b}/{m}"] = [round(tau, 2), round(spd, 2)]

    budget: dict = {}
    arm_for_budget = {"4b": {16: "dominotree", 32: "dominotree_b32"},
                      "8b": {16: "dominotree", 32: "dominotree_b32", 64: "dominotree_b64"}}
    for size, budgets in arm_for_budget.items():
        for b, arm in budgets.items():
            prompts = helmet_prompts(size, arm)
            for ctx_k, bin_ in [(8, 8192), (16, 16384), (32, 32768)]:
                cells = [helmet_cell_stats(prompts, (t, bin_)) for t in TASKS]
                budget[f"{size}/{ctx_k}/{b}"] = round(st.mean(c[0] for c in cells), 1)

    helmet_app: dict = {}
    ci: dict = {}
    arms = {"4b": METHODS, "8b": METHODS + ["dominotree_b32"]}
    for size, arm_list in arms.items():
        per = {a: helmet_prompts(size, a) for a in arm_list}
        for t in TASKS:
            for b in BINS:
                for a in arm_list:
                    tps_raw, tau_raw = helmet_cell_stats(per[a], (t, b))
                    helmet_app[f"{size}/{t}/{b}/{a}"] = {"tps": round(tps_raw, 1), "tau": round(tau_raw, 2)}
                for a in [x for x in arm_list if x.startswith("dominotree")]:
                    common = sorted(set(per[a].get((t, b), {})) & set(per["domino_chain"].get((t, b), {})))
                    if len(common) != 50:
                        continue
                    entry = {}
                    for field, metric in (("tps", "tps"), ("accept", "tau")):
                        av = [per[a][(t, b)][i][field] for i in common]
                        bv = [per["domino_chain"][(t, b)][i][field] for i in common]
                        point = 100.0 * (st.mean(av) / st.mean(bv) - 1.0)
                        entry[metric] = [round(point, 1)]
                    ci[f"{size}/{t}/{b}/{a}"] = entry
    helmet_app["_ci"] = ci

    return {
        "_comment": ("Values recomputed from the raw JSONL shipped in this directory. "
                     "Regenerate with `python3 verify_published_numbers.py --emit PUBLISHED.json`; "
                     "the working repo's scripts/domino_tree_ieee_access/ cross-checks these "
                     "against the paper's LaTeX source before they are committed here. Do not "
                     "hand-edit."),
        "bs1": bs1,
        "bs1_cis": bs1_cis,
        "concurrency": conc,
        "concurrency_per_dataset": conc_ds,
        "longctx": longctx,
        "longctx_budget": budget,
        "helmet_appendix": helmet_app,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", metavar="PATH",
                     help="Write the values recomputed from raw data as PUBLISHED.json-shaped "
                          "JSON to PATH, instead of checking against the existing file. This "
                          "does NOT by itself prove the numbers match the paper -- that "
                          "cross-check is done separately against the LaTeX source, which this "
                          "public bundle does not ship.")
    args = ap.parse_args()

    if args.emit:
        payload = emit_all()
        Path(args.emit).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.emit}")
        return 0

    if not PUBLISHED.is_file():
        print(f"missing {PUBLISHED}", file=sys.stderr)
        return 2
    pub = json.loads(PUBLISHED.read_text())
    print(f"data: {HERE}\n")

    check_bs1(pub["bs1"])
    check_bs1_cis(pub["bs1_cis"])
    check_conc(pub["concurrency"])
    check_conc_appendix(pub["concurrency_per_dataset"])
    check_longctx(pub["longctx"])
    check_longctx_budget(pub["longctx_budget"])
    check_helmet_appendix(pub["helmet_appendix"])

    print("\n" + "=" * 72)
    expected_cells = pub.get("_expected_cell_checks")
    if expected_cells is not None and _count != expected_cells:
        print(f"COVERAGE FAILURE: verified {_count} cells, expected {expected_cells}.")
        print("  Either a raw file grew/shrank or a table's shape changed. Do not lower this")
        print("  number without confirming the table really has fewer cells.")
        return 1
    if fails:
        print(f"{len(fails)} MISMATCH(ES) between the paper and the raw data:")
        for f in fails:
            print("  - " + f)
        return 1
    print(f"ALL {_count} CELLS REPRODUCE FROM RAW DATA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
