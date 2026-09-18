#!/usr/bin/env python3
"""Paired bootstrap CIs for Qwen3-8B DominoTree-at-budget-32 vs the Domino chain, HELMET
long-context (bs=1, T=0, cap-1200 generation).

Why this script exists: the paper's 8B long-context table (tab:sglang-longctx) reports
DominoTree at tree budget 32 (the measured long-context optimum for 8B up to 16K; see
tab:longctx-budget), collected separately into
results/serving/longcontext/8b/dominotree_b32/helmet.prompts.jsonl. The plain
per-default-budget audit (verify_r4_longctx.py) instead pairs DominoTree at budget 16
(results/serving/longcontext/8b/dominotree/) against the chain, which is the WRONG
DominoTree arm for this table. This script re-runs the same paired-bootstrap logic
(B=5000, resample prompts, seed=0) but pairs the budget-32 DominoTree rows against the
domino_chain rows in results/serving/longcontext/8b/domino_chain/.

Caveat carried into the paper: the budget-32 DominoTree arm (dominotree_b32/) was
collected in a later session on the same pod as the domino_chain arm
(longcontext/8b/domino_chain/). Both are on the same hardware/software stack (same pod),
but not literally the same process invocation as the paired comparison at budget 16.

Usage: python3 results/serving/ci_r4_8b_b32.py [--boot 5000] [--seed 0]
Requires only the Python standard library.
"""
from __future__ import annotations
import argparse, json, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ["infbench_sum", "multi_lexsum"]
BINS = [8192, 16384, 32768]

DOMINOTREE_B32_DIR = os.path.join(HERE, "longcontext", "8b", "dominotree_b32")
DOMINO_CHAIN_DIR = os.path.join(HERE, "longcontext", "8b", "domino_chain")


def load_prompts(d):
    per = [json.loads(l) for l in open(os.path.join(d, "helmet.prompts.jsonl")) if l.strip()]
    P, dupes, bad = {}, [], []
    for r in per:
        k = (r["task"], r["length_bin"])
        cell = P.setdefault(k, {})
        if r["idx"] in cell:
            dupes.append((k, r["idx"]))
        for f in ("output_tokens", "decode_time", "tps"):
            v = r.get(f)
            if v is None or not isinstance(v, (int, float)) or v != v or v <= 0:
                bad.append((k, r["idx"], f, v))
        cell[r["idx"]] = r
    return P, dupes, bad


def load_agg(d):
    agg = [json.loads(l) for l in open(os.path.join(d, "helmet.jsonl")) if l.strip()]
    return {(r["task"], r["length_bin"]): r for r in agg}


def paired(a, b, stat, boot, seed):
    keys = sorted(set(a) & set(b))
    if set(a) ^ set(b):
        raise ValueError(f"UNPAIRED: {len(set(a) ^ set(b))} idx in only one arm")
    av = [stat(a[k]) for k in keys]
    bv = [stat(b[k]) for k in keys]
    point = 100.0 * (st.mean(av) / st.mean(bv) - 1.0)
    rng = random.Random(seed)
    n = len(keys)
    draws = []
    for _ in range(boot):
        idx = [rng.randrange(n) for _ in range(n)]
        mb = st.mean(bv[i] for i in idx)
        if mb > 0:
            draws.append(100.0 * (st.mean(av[i] for i in idx) / mb - 1.0))
    draws.sort()
    return point, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1.0, help="max %% aggregate-vs-raw disagreement")
    a = ap.parse_args()
    problems = []

    P_dt, dup_dt, bad_dt = load_prompts(DOMINOTREE_B32_DIR)
    P_ch, dup_ch, bad_ch = load_prompts(DOMINO_CHAIN_DIR)
    A_dt = load_agg(DOMINOTREE_B32_DIR)
    A_ch = load_agg(DOMINO_CHAIN_DIR)
    if dup_dt:
        problems.append(f"dominotree@32: {len(dup_dt)} duplicate prompt idx (e.g. {dup_dt[0]})")
    if dup_ch:
        problems.append(f"domino_chain: {len(dup_ch)} duplicate prompt idx (e.g. {dup_ch[0]})")
    if bad_dt:
        problems.append(f"dominotree@32: {len(bad_dt)} bad per-prompt fields (e.g. {bad_dt[0]})")
    if bad_ch:
        problems.append(f"domino_chain: {len(bad_ch)} bad per-prompt fields (e.g. {bad_ch[0]})")

    print("\n## R4 8B (DominoTree budget=32) — AGGREGATE AUDIT vs raw rows\n")
    print("| method | task | bin | tps agg | tps raw | tau agg | tau raw | n |")
    print("|---|---|---|---|---|---|---|---|")
    for label, P, A in (("dominotree@32", P_dt, A_dt), ("domino_chain", P_ch, A_ch)):
        for t in TASKS:
            for b in BINS:
                per = P.get((t, b))
                agg = A.get((t, b))
                if not per or not agg:
                    problems.append(f"MISSING {label} {t}/{b}")
                    continue
                tok = sum(r["output_tokens"] for r in per.values())
                sec = sum(r["decode_time"] for r in per.values())
                tps_raw = tok / sec if sec else float("nan")
                accs = [r["accept"] for r in per.values() if r.get("accept")]
                tau_raw = st.mean(accs) if accs else 1.0
                dt = 100 * (agg["tps"] / tps_raw - 1) if tps_raw else float("nan")
                if abs(dt) > a.tol:
                    problems.append(f"{label} {t}/{b}: tps agg {agg['tps']:.2f} vs raw {tps_raw:.2f} ({dt:+.1f}%)")
                print(f"| {label} | {t} | {b} | {agg['tps']:.2f} | {tps_raw:.2f} | "
                      f"{agg.get('mean_accept', float('nan')):.3f} | {tau_raw:.3f} | {len(per)} |")

    print("\n## Qwen3-8B, HELMET long-context — DominoTree@32 vs domino_chain, "
          f"paired bootstrap 95% CI (B={a.boot})\n")
    print("| task | bin | Δtps % [95% CI] | Δtau % [95% CI] | n |")
    print("|---|---|---|---|---|")
    wins = sig = 0
    for t in TASKS:
        for b in BINS:
            tp_, rp_ = P_dt.get((t, b)), P_ch.get((t, b))
            if not tp_ or not rp_:
                print(f"| {t} | {b} | — | — | — |")
                continue
            try:
                p, lo, hi, n = paired(tp_, rp_, lambda r: r["tps"], a.boot, a.seed)
                pa, alo, ahi, _ = paired(tp_, rp_, lambda r: r.get("accept") or 1.0, a.boot, a.seed)
            except ValueError as e:
                problems.append(f"8B@32 {t}/{b}: {e}")
                print(f"| {t} | {b} | **{e}** | | |")
                continue
            mark = "" if (lo > 0 or hi < 0) else " *"
            if p > 0:
                wins += 1
            if lo > 0:
                sig += 1
            print(f"| {t} | {b} | {p:+.1f}% [{lo:+.1f},{hi:+.1f}]{mark} | {pa:+.1f}% [{alo:+.1f},{ahi:+.1f}] | {n} |")
    print(f"\nDominoTree@32 TPS wins **{wins}/6** cells; **{sig}/6** significant. `* = CI straddles 0.`")

    print("\n---\n")
    if problems:
        print(f"**{len(problems)} PROBLEM(S):**")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("**VERIFICATION PASSED** — every aggregate follows from its raw per-prompt rows, "
          "cells present, arms paired by idx.")


if __name__ == "__main__":
    main()
