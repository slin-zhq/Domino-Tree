#!/usr/bin/env python3
"""Regenerate the Qwen3.6-27B-only tables of the paper (one H200 GPU, without tensor
parallelism) straight from the raw per-prompt / per-cell JSONL in this directory.

    python3 gen_qwen36_27b_tables.py conc       # tab:app-conc-27b (per dataset + Overall)
    python3 gen_qwen36_27b_tables.py conc_main  # the Qwen3.6-27B rows of tab:sglang-conc
    python3 gen_qwen36_27b_tables.py ablation   # tab:app-qwen27b-ablation (budget, timing, width)
    python3 gen_qwen36_27b_tables.py evidence   # the numbers the 27B section quotes in prose
    python3 gen_qwen36_27b_tables.py all        # all of the above

The Qwen3.6-27B rows of the tables it shares with Qwen3-4B/8B are printed by the
generators next to this directory, from the raw data here:

    ../gen_sglang_bs1_table.py      tab:sglang-bs1      (bs=1, T=0 and T=1)
    ../gen_sglang_longctx_table.py  tab:sglang-longctx  (HELMET, mean of the two tasks)
    ../gen_longctx_budget_table.py  tab:longctx-budget  (HELMET tree-budget sweep, helmet_budget/)
    ../gen_appendix_5090.py         tab:app-helmet      (HELMET per task, paired CIs)

verify_qwen36_27b.py re-derives every one of these numbers independently and checks the
printed tables against them.

Five methods were measured; `ar` (plain autoregressive decoding) is the speedup
reference. There is no `eagle3` arm for this target: no EAGLE-3 head exists for
Qwen3.6-27B, so `mtp` (the checkpoint's own multi-token-prediction head, served with
SGLang's NEXTN algorithm) is the draft-head baseline. Bold marks the best printed value,
ties included ("tie" = the values print the same at the column's precision).

Estimands are the same as the Qwen3-4B/8B serving tables':

  bs=1        Per prompt, MT-Bench turns merged (a prompt's TPS is its summed output tokens
              over its summed decode time; its tau is turn-length-weighted). A dataset cell
              is the mean over prompts; Overall is the unweighted mean over the 8 datasets.
  HELMET      A cell's TPS is SUM(output_tokens)/SUM(decode_time) over its 50 prompts; tau is
              the mean per-prompt accepted length. The paired-bootstrap CI against the chain
              resamples the per-prompt `tps`/`accept` fields (mean of per-prompt values), a
              different estimand from the aggregate TPS above it, as in the 4B/8B tables.
  Concurrency Goodput at one (dataset, c) cell is SUM(completion_tokens)/SUM(wall_s), at
              c = 2, 4, 8, 16, 32 (the raw data also holds c=1, which the paper does not
              report). A dataset's tau is the mean of the cell means over c; Overall
              averages the three datasets with equal weight, for goodput and for tau.
  Tuning      20 fixed gate prompts per configuration; tok/s is total completion tokens over
              total wall time (prefill included), tau the mean per-prompt accepted length.
              Every configuration's .verdict.txt must say P9_GATE=PASS.

Requires only the Python standard library.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from functools import lru_cache
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

METHODS = ["mtp", "dflash", "domino_chain", "dominotree"]  # the paper's row order
ALL_METHODS = ["ar"] + METHODS
NAMES = {"dflash": "DFlash", "domino_chain": "Domino", "mtp": "MTP",
         "dominotree": "DominoTree (15)", "ar": "AR"}

DS = ["gsm8k", "math500", "aime25", "humaneval", "mbpp", "livecodebench", "mt-bench", "alpaca"]
DS_NAMES = {"gsm8k": "GSM8K", "math500": "MATH-500", "aime25": "AIME25",
            "humaneval": "HumanEval", "mbpp": "MBPP", "livecodebench": "LCB",
            "mt-bench": "MT-Bench", "alpaca": "Alpaca"}
DS_GROUPS = [("Math", ["gsm8k", "math500", "aime25"]),
             ("Code", ["humaneval", "mbpp", "livecodebench"]),
             ("Chat", ["mt-bench", "alpaca"])]

TASKS = ["infbench_sum", "multi_lexsum"]
TASK_NAMES = {"infbench_sum": r"$\infty$Bench-Sum", "multi_lexsum": "Multi-LexSum"}
BINS = [8192, 16384, 32768]

CONC_DS = ["gsm8k", "mbpp", "mt-bench"]
CONC_DS_NAMES = {"gsm8k": "GSM8K", "mbpp": "MBPP", "mt-bench": "MT-Bench"}
CONCS = [2, 4, 8, 16, 32]  # c=1 rows are in the raw data but not in the paper's tables

END = r" \\"
BOOT, SEED = 5000, 0


def bold(v: float, mx: float, digits: int = 2) -> str:
    """Bold if v prints the same as the column max at `digits` precision (ties all bolded)."""
    s = f"{v:.{digits}f}"
    return r"\textbf{" + s + "}" if s == f"{mx:.{digits}f}" else s


def rows(path: Path) -> list:
    if not path.is_file():
        sys.exit(f"missing raw file: {path}")
    out = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not out:
        sys.exit(f"empty raw file: {path}")
    return out


# --------------------------------------------------------------------------- bs=1
def bs1_groups(method: str, dataset: str, temp: str) -> dict:
    by: dict = {}
    for r in rows(HERE / "bs1" / f"T{temp}" / method / f"{dataset}_T{temp}.jsonl"):
        by.setdefault(r["sample_idx"], []).append(r)
    return by


def _prompt_tps(turns: list) -> float:
    return sum(t["num_output"] for t in turns) / sum(t["decode_time"] for t in turns)


def _prompt_tau(turns: list) -> float:
    n = sum(t["num_output"] for t in turns)
    steps = sum(t["num_output"] / t["mean_accept"] for t in turns)
    return n / steps


@lru_cache(maxsize=None)
def bs1_cell(method: str, dataset: str, temp: str) -> tuple:
    """Returns (tps, tau, n_prompts) for one (method, dataset, temperature) cell.

    PER-PROMPT, turns merged -- exactly ../gen_sglang_bs1_table.py's `cell()` /
    results/serving/verify_published_numbers.py's `bs1_cell()`. See the module
    docstring for why (this is the estimand Tables 5/6 use, and this bundle must not
    put one column on a different one).
    """
    groups = bs1_groups(method, dataset, temp)
    tps = st.fmean(_prompt_tps(v) for v in groups.values())
    tau = st.fmean(_prompt_tau(v) for v in groups.values())
    return tps, tau, len(groups)


def bs1_speedup_overall(method: str, temp: str) -> float:
    return st.fmean(bs1_cell(method, d, temp)[0] / bs1_cell("ar", d, temp)[0] for d in DS)


def bs1_tau_overall(method: str, temp: str) -> float:
    return st.fmean(bs1_cell(method, d, temp)[1] for d in DS)


# --------------------------------------------------------------------------- HELMET
@lru_cache(maxsize=None)
def helmet_prompts(method: str) -> dict:
    out: dict = {}
    for r in rows(HERE / "helmet" / method / "helmet.prompts.jsonl"):
        out.setdefault((r["task"], int(r["length_bin"])), {})[r["idx"]] = r
    return out


def helmet_cell(prompts: dict, cell: tuple) -> tuple:
    rs = list(prompts.get(cell, {}).values())
    if not rs:
        sys.exit(f"empty HELMET cell {cell}")
    tps = sum(r["output_tokens"] for r in rs) / sum(r["decode_time"] for r in rs)
    tau = st.fmean(r.get("accept") or 1.0 for r in rs)
    return tps, tau


def helmet_speedup_overall(prompts: dict, ar_prompts: dict) -> float:
    return st.fmean(helmet_cell(prompts, (t, b))[0] / helmet_cell(ar_prompts, (t, b))[0]
                     for t in TASKS for b in BINS)


def helmet_tau_overall(prompts: dict) -> float:
    return st.fmean(helmet_cell(prompts, (t, b))[1] for t in TASKS for b in BINS)


def paired_ci(a: dict, b: dict, field: str) -> tuple:
    """Paired-bootstrap Delta% for one HELMET cell -- deliberately MACRO (mean of the
    per-prompt `tps`/`accept` field), matching ../gen_appendix_5090.py's `paired_ci`
    and ../verify_r4_longctx.py's `paired()` exactly, including their percentile
    index `int(0.975*len(draws)) - 1` (NOT results/serving/verify_published_numbers.py's
    bs1-CI index `min(len-1, int(0.975*len(draws)))`, which is one draw higher out of
    5000 and belongs to a different, bs=1-specific CI in that file)."""
    keys = sorted(set(a) & set(b))
    if len(keys) != 50 or set(a) != set(b):
        sys.exit(f"paired_ci: expected 50 matched prompts, got {len(keys)}")
    av = [(a[k].get(field) or 1.0) for k in keys]
    bv = [(b[k].get(field) or 1.0) for k in keys]
    point = 100.0 * (st.mean(av) / st.mean(bv) - 1.0)
    rng = random.Random(SEED)
    draws = []
    for _ in range(BOOT):
        idx = [rng.randrange(len(keys)) for _ in keys]
        mb = st.mean(bv[i] for i in idx)
        if mb:
            draws.append(100.0 * (st.mean(av[i] for i in idx) / mb - 1.0))
    draws.sort()
    return point, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def ci_tex(ci: tuple) -> str:
    p, lo, hi = ci
    star = r"$^{*}$" if lo <= 0 <= hi else ""
    return f"{p:+.1f} [{lo:+.1f},{hi:+.1f}]" + star


# --------------------------------------------------------------------------- concurrency
@lru_cache(maxsize=None)
def conc_rows(method: str) -> dict:
    out = {}
    for r in rows(HERE / "concurrency" / method / f"{method}.jsonl"):
        out[(r["dataset"], int(r["concurrency"]))] = r
    missing = {(d, c) for d in CONC_DS for c in CONCS} - set(out)
    if missing:
        sys.exit(f"concurrency/{method}: missing cells {sorted(missing)}")
    return out


def conc_speedup_overall(rows_m: dict, rows_ar: dict) -> float:
    return st.fmean(rows_m[cell]["tps"] / rows_ar[cell]["tps"]
                     for cell in ((d, c) for d in CONC_DS for c in CONCS))


def conc_tau_overall(rows_m: dict) -> float:
    return st.fmean(rows_m[cell]["mean_accept"] for cell in ((d, c) for d in CONC_DS for c in CONCS))


def bold_row(vals, digits):
    """Bold every value that prints the same as the best one (ties included)."""
    best = f"{max(vals):.{digits}f}"
    return [r"\textbf{" + f"{v:.{digits}f}" + "}" if f"{v:.{digits}f}" == best else f"{v:.{digits}f}"
            for v in vals]


# --------------------------------------------------------------------------- concurrency
def conc_values():
    """{block: {method: ([goodput per c], tau or None)}}, blocks = the 3 datasets + Overall."""
    data = {m: conc_rows(m) for m in ALL_METHODS}
    out = {}
    for d in CONC_DS:
        out[d] = {m: ([data[m][d, c]["tps"] for c in CONCS],
                      None if m == "ar" else st.fmean(data[m][d, c]["mean_accept"] for c in CONCS))
                  for m in ALL_METHODS}
    out["Overall"] = {m: ([st.fmean(out[d][m][0][i] for d in CONC_DS) for i in range(len(CONCS))],
                          None if m == "ar" else st.fmean(out[d][m][1] for d in CONC_DS))
                      for m in ALL_METHODS}
    return out


def conc_block(vals, first_label):
    cols = [bold_row([vals[m][0][i] for m in ALL_METHODS], 0) for i in range(len(CONCS))]
    taus = bold_row([vals[m][1] for m in METHODS], 2)
    lines = []
    for j, m in enumerate(ALL_METHODS):
        tau = "--" if m == "ar" else taus[j - 1]
        lines.append((first_label if j == 0 else "") + " & " + NAMES[m] + " & "
                     + " & ".join(cols[i][j] for i in range(len(CONCS))) + " & " + tau + END)
    return lines


def gen_conc():
    v = conc_values()
    head = " & ".join(f"$c{{=}}{c}$" for c in CONCS)
    lines = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccc}", r"\toprule",
             "Dataset & Method & " + head + r" & $\tau$" + END, r"\midrule",
             r"\multicolumn{8}{c}{\textit{Qwen3.6-27B}}" + END]
    for d in CONC_DS:
        lines += [r"\midrule"] + conc_block(v[d], CONC_DS_NAMES[d])
    lines += [r"\midrule\midrule",
              r"\multicolumn{8}{c}{\textit{Qwen3.6-27B -- Overall (mean of the three datasets above)}}" + END,
              r"\midrule"] + conc_block(v["Overall"], "Overall")
    lines += [r"\bottomrule", r"\end{tabular*}"]
    caps = {m: (HERE / "concurrency" / m / "admission_cap.txt").read_text().strip() for m in ALL_METHODS}
    lines.append("% admission caps: " + ", ".join(f"{m}={caps[m]}" for m in ALL_METHODS))
    print("\n".join(lines))


def gen_conc_main():
    """The Qwen3.6-27B block of tab:sglang-conc: the Overall rows of tab:app-conc-27b."""
    lines = [r"\multirow{5}{*}{Qwen3.6-27B}"] + conc_block(conc_values()["Overall"], "")
    print("\n".join(lines))


# --------------------------------------------------------------------------- tuning (ablation)
def _gate_point(path: Path) -> tuple:
    rs = rows(path)
    if len(rs) != 20:
        sys.exit(f"{path}: expected 20 gate prompts, got {len(rs)}")
    verdict = path.with_suffix(".verdict.txt").read_text()
    if not re.search(r"^P9_GATE=PASS$", verdict, re.M):
        sys.exit(f"gate did not pass: {path}")
    return (sum(r["completion_tokens"] for r in rs) / sum(r["wall_s"] for r in rs),
            st.fmean(r["tau"] for r in rs))


LADDER = [8, 12, 15, 32, 64]
TIMING = [8, 15, 32, 64]
GRID_K, GRID_M = [4, 8, 16], [32, 64, 128, 0]
PHASES = ("draft", "build", "prepare", "verify", "accept", "commit")


def ablation_values():
    d = HERE / "ablations"
    # Node-budget sweep: every budget at (k, M) = (8, 128), one session (2026-09-26).
    ladder = {b: _gate_point(d / "budget_ladder_m128" / f"dominotree_b{b}_k8_m128.jsonl") for b in LADDER}
    # Width sweep at B=15, an earlier session; its (k, M) = (8, 64) cell is that session's
    # default-config ladder row.
    grid = {}
    for k in GRID_K:
        for m in GRID_M:
            path = (d / "budget_ladder" / "dominotree_b15.jsonl" if (k, m) == (8, 64)
                    else d / "node_topk_topm_grid" / f"dominotree_b15_k{k}_m{m}.jsonl")
            grid[k, m] = _gate_point(path)
    timing = {}
    for b in TIMING:
        windows = [dict((k, float(v)) for k, v in re.findall(r"(\w+)=([0-9.]+)", line.split("|")[0]))
                   for line in (d / "timing" / f"timing_b{b}.steps.txt").read_text().splitlines()
                   if "ms/step" in line]
        timing[b] = [st.fmean(w[p] for w in windows) for p in PHASES]
    return ladder, grid, timing


def gen_ablation():
    ladder, grid, timing = ablation_values()
    tps = bold_row([ladder[b][0] for b in LADDER], 1)
    tau = bold_row([ladder[b][1] for b in LADDER], 2)
    L = [r"\begin{tabular}{rrr}", r"\toprule", r"Budget $B$ & Tok/s & $\tau$" + END, r"\midrule"]
    L += [f"{b:<2} & {tps[i]} & {tau[i]}" + END for i, b in enumerate(LADDER)]
    L += [r"\bottomrule", r"\end{tabular}", "% ---- per-step timing (ms), k=8, M=128"]
    L += [r"\begin{tabular}{rrrrrrrr}", r"\toprule",
          r"$B$ & Draft & Build & Prepare & Verify & Accept & Commit & Total" + END, r"\midrule"]
    L += [f"{b:<2} & " + " & ".join(f"{v:.3f}" for v in timing[b]) + f" & {sum(timing[b]):.3f}" + END
          for b in TIMING]
    L += [r"\bottomrule", r"\end{tabular}"]
    for k in GRID_K:
        tps = bold_row([grid[k, m][0] for m in GRID_M], 1)
        tau = bold_row([grid[k, m][1] for m in GRID_M], 2)
        L += [f"% ---- width sweep, B=15, k={k}", r"\begin{tabular}{rrr}", r"\toprule",
              r"$M$ & Tok/s & $\tau$" + END, r"\midrule"]
        L += [f"{m:<3} & {tps[i]} & {tau[i]}" + END for i, m in enumerate(GRID_M)]
        L += [r"\bottomrule", r"\end{tabular}"]
    print("\n".join(L))


# --------------------------------------------------------------------------- prose numbers
def evidence():
    """The numbers sec:serving-27b quotes in its text, keyed by what they are."""
    out = {}
    for temp in ("0.0", "1.0"):
        for m in ("domino_chain", "dominotree"):
            out[f"bs1/{temp}/{m}/speedup"] = round(bs1_speedup_overall(m, temp), 2)
    hp = {m: helmet_prompts(m) for m in ALL_METHODS}
    speeds = [helmet_cell(hp["dominotree"], (t, b))[0] / helmet_cell(hp["ar"], (t, b))[0]
              for t in TASKS for b in BINS]
    out["helmet/tree_ar_min"] = round(min(speeds), 2)
    out["helmet/tree_ar_max"] = round(max(speeds), 2)
    out["helmet/tree_chain_ci_above_zero"] = sum(
        paired_ci(hp["dominotree"][t, b], hp["domino_chain"][t, b], "tps")[1] > 0
        for t in TASKS for b in BINS)
    out["helmet/cells"] = len(TASKS) * len(BINS)
    v = conc_values()["Overall"]
    for i, c in enumerate(CONCS):
        out[f"conc/{c}/tree_over_chain_pct"] = round(100 * (v["dominotree"][0][i] / v["domino_chain"][0][i] - 1), 1)
    gate = HERE / "losslessness_gate" / "20260923T171916Z_356641"
    ar = {r["prompt_id"]: r["output_ids"] for r in rows(gate / "ar.jsonl")}
    for m in ("domino", "dominotree", "dominotree_flat"):
        rr = rows(gate / f"{m}.jsonl")
        agree = []
        for r in rr:
            a, b = ar[r["prompt_id"]], r["output_ids"]
            agree.append(next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))))
        out[f"gate/{m}/median_prefix_agreement"] = st.median(agree)
        out[f"gate/{m}/tau"] = round(st.fmean(r["tau"] for r in rr), 2)
    return out


def gen_evidence():
    for key, val in evidence().items():
        print(f"% evidence {key} = {val}")


TABLES = {"conc": gen_conc, "conc_main": gen_conc_main, "ablation": gen_ablation,
          "evidence": gen_evidence}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", choices=list(TABLES) + ["all"])
    args = ap.parse_args()
    if args.table == "all":
        for name, fn in TABLES.items():
            print(f"% ==== {name} ====")
            fn()
            print()
    else:
        TABLES[args.table]()


if __name__ == "__main__":
    main()
