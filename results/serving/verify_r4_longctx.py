#!/usr/bin/env python3
"""R5 for the long-context axis: re-derive every R4 cell from RAW PER-PROMPT rows and audit the
aggregates against them.

Two independent jobs, because they catch different failures:

  1. AGGREGATE AUDIT. helmet.jsonl carries one summary row per (task, bin). helmet.prompts.jsonl
     carries the 50 underlying per-prompt rows. Recompute the summary from the raw rows and diff.
     A mismatch means the number we would print does not follow from the data we collected.

  2. PAIRED CIs. The aggregate alone cannot say whether a win is distinguishable from noise. The
     bootstrap resamples PROMPTS (paired by idx within a task/bin), which is the sampling unit.

Pairing is VERIFIED, not assumed: both arms must carry the same idx set for a cell.

Usage: python3 results/serving/verify_r4_longctx.py [--boot 5000]
Requires only the Python standard library. Reads results/serving/longcontext/<size>/<method>/
at the DEFAULT tree budget (16); the budget-32/64 sweep arms are audited by
`ci_r4_8b_b32.py` and `verify_published_numbers.py` instead.
"""
from __future__ import annotations
import argparse, json, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ["ar", "eagle3", "dflash", "domino_chain", "dominotree"]
TASKS = ["infbench_sum", "multi_lexsum"]
BINS = [8192, 16384, 32768]
REF = "domino_chain"


def load(model, method):
    d = os.path.join(HERE, "longcontext", model.lower(), method)
    agg = [json.loads(l) for l in open(f"{d}/helmet.jsonl") if l.strip()]
    per = [json.loads(l) for l in open(f"{d}/helmet.prompts.jsonl") if l.strip()]
    A = {(r["task"], r["length_bin"]): r for r in agg}
    # Building a dict keyed on idx SILENTLY OVERWRITES duplicates, so a cell that emitted the same
    # prompt twice (and therefore fewer distinct prompts than claimed) would look complete. Count
    # first, then check. Also enforce the declared n_prompts and per-row field sanity, so a NaN or
    # a zero decode_time cannot flow into an aggregate. -- Codex review 2026-09-15
    P, dupes, bad = {}, [], []
    for r in per:
        k = (r["task"], r["length_bin"])
        cell = P.setdefault(k, {})
        if r["idx"] in cell:
            dupes.append((method, k, r["idx"]))
        for f in ("output_tokens", "decode_time", "tps"):
            v = r.get(f)
            if v is None or not isinstance(v, (int, float)) or v != v or v <= 0:
                bad.append((method, k, r["idx"], f, v))
        cell[r["idx"]] = r
    return A, P, dupes, bad


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

    for model in ("4B", "8B"):
        D = {}
        for m in METHODS:
            A_, P_, dup, badf = load(model, m)
            D[m] = (A_, P_)
            if dup:
                problems.append(f"{model}/{m}: {len(dup)} DUPLICATE prompt idx (e.g. {dup[0]})")
            if badf:
                problems.append(f"{model}/{m}: {len(badf)} bad per-prompt fields (e.g. {badf[0]})")
            for (t_, b_), cell in P_.items():
                n_decl = A_.get((t_, b_), {}).get("n_prompts")
                if n_decl is not None and len(cell) != n_decl:
                    problems.append(f"{model}/{m} {t_}/{b_}: {len(cell)} distinct prompts but "
                                    f"aggregate declares n_prompts={n_decl}")

        print(f"\n## R4 {model} — AGGREGATE AUDIT (published summary vs recomputed from raw rows)\n")
        print("| method | task | bin | tps agg | tps raw | Δ% | tau agg | tau raw | Δ% |")
        print("|---|---|---|---|---|---|---|---|---|")
        for m in METHODS:
            A, P = D[m]
            for t in TASKS:
                for b in BINS:
                    agg, per = A.get((t, b)), P.get((t, b))
                    if not agg or not per:
                        problems.append(f"MISSING {model} {m} {t}/{b}")
                        continue
                    # throughput: total tokens / total decode time across the cell's prompts
                    tok = sum(r["output_tokens"] for r in per.values())
                    sec = sum(r["decode_time"] for r in per.values())
                    tps_raw = tok / sec if sec else float("nan")
                    accs = [r["accept"] for r in per.values() if r.get("accept")]
                    tau_raw = st.mean(accs) if accs else 1.0
                    tau_agg = agg.get("mean_accept") or 1.0
                    dt = 100 * (agg["tps"] / tps_raw - 1) if tps_raw else float("nan")
                    da = 100 * (tau_agg / tau_raw - 1) if tau_raw else float("nan")
                    flag = ""
                    if abs(dt) > a.tol:
                        flag = " **MISMATCH**"
                        problems.append(f"{model} {m} {t}/{b}: tps agg {agg['tps']:.2f} vs raw {tps_raw:.2f} ({dt:+.1f}%)")
                    if abs(da) > a.tol:
                        flag += " **TAU MISMATCH**"
                        problems.append(f"{model} {m} {t}/{b}: tau agg {tau_agg:.3f} vs raw {tau_raw:.3f} ({da:+.1f}%)")
                    print(f"| {m} | {t} | {b} | {agg['tps']:.2f} | {tps_raw:.2f} | {dt:+.2f} "
                          f"| {tau_agg:.2f} | {tau_raw:.2f} | {da:+.2f}{flag} |")

        print(f"\n## R4 {model} — DominoTree vs {REF}, paired bootstrap 95% CI (B={a.boot})\n")
        print("| task | bin | Δtps % [95% CI] | Δtau % [95% CI] | n |")
        print("|---|---|---|---|---|")
        wins = sig = 0
        for t in TASKS:
            for b in BINS:
                tp_, rp_ = D["dominotree"][1].get((t, b)), D[REF][1].get((t, b))
                if not tp_ or not rp_:
                    print(f"| {t} | {b} | — | — | — |"); continue
                try:
                    p, lo, hi, n = paired(tp_, rp_, lambda r: r["tps"], a.boot, a.seed)
                    pa, alo, ahi, _ = paired(tp_, rp_, lambda r: r.get("accept") or 1.0, a.boot, a.seed)
                except ValueError as e:
                    problems.append(f"{model} {t}/{b}: {e}"); print(f"| {t} | {b} | **{e}** | | |"); continue
                mark = "" if (lo > 0 or hi < 0) else " *"
                if p > 0: wins += 1
                if lo > 0: sig += 1
                print(f"| {t} | {b} | {p:+.1f}% [{lo:+.1f},{hi:+.1f}]{mark} | {pa:+.1f}% [{alo:+.1f},{ahi:+.1f}] | {n} |")
        print(f"\nDominoTree TPS wins **{wins}/6** cells; **{sig}/6** significant. `* = CI straddles 0.`")

    print("\n---\n")
    if problems:
        print(f"**{len(problems)} PROBLEM(S):**")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("**VERIFICATION PASSED** — every aggregate follows from its raw per-prompt rows, "
          "all cells present, all arms paired.")


if __name__ == "__main__":
    main()
