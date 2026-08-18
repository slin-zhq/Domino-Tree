#!/usr/bin/env python3
"""Reproduce the paper's three-arm conditioning ladder (Table 13) from public raw data.

    python3 make_conditioning_ladder_table.py

The ladder isolates what DominoTree's conditional score is actually worth by holding the
drafter, the node budget (16), the candidate width (M=64) and the verifier fixed, and
varying ONLY the function that scores a node's candidates:

    marg@16        base drafter logits, no GRU correction; the depth-d menu is computed
                   once and shared by every node at depth d  -> factorized score
    condstatic@16  correction applied, but once per depth along the single greedy Domino
                   trajectory, then shared by every node at that depth -> corrected but
                   still path-INdependent
    dominotree@16  correction recomputed per node from that node's own GRU state
                   -> path-dependent; the property this paper contributes

So marg -> condstatic measures what the correction is worth at all, and
condstatic -> dominotree measures what conditioning on the realized path adds on top,
relative to that fixed greedy-path surrogate.

Two builder configurations are reported, from two independent collections:

    matched/    every arm on the Python builder. Equalizes the builder IMPLEMENTATION,
                so the delta isolates the scoring function. It does NOT equalize
                construction COST, which genuinely differs between arms.
    graphbest/  every arm at its fastest available builder. This is what a deployment
                would observe; it does not attribute the gain to a property, because the
                arms receive unequal optimization effort (Cond-static captures as one
                CUDA graph; DominoTree can only capture the per-node correction).

Acceptance (tau) is builder-invariant by construction -- a builder changes how the tree
is searched for, never which nodes end up in it -- so tau is reported once and should
agree between the two directories. The script checks that.

STATISTICS. Delta% is the paired ratio of speedup-over-AR. Rollups are UNWEIGHTED means
over their constituent datasets, so each benchmark counts equally regardless of its n.
The bootstrap (B=5000) is stratified by dataset and CLUSTERED by conversation: MT-Bench's
second turn is generated from the method's own first-turn answer, so its two turns are
dependent and must be resampled together. For the seven single-turn datasets a cluster is
a single prompt, so clustering is a no-op there.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import statistics as st

ARMS = ["marg@16", "condstatic@16", "dominotree@16"]
LABEL = {
    "gsm8k": "GSM8K", "math500": "MATH-500", "aime25": "AIME25",
    "humaneval": "HumanEval", "mbpp": "MBPP", "livecodebench": "LiveCodeBench",
    "mt-bench": "MT-Bench", "alpaca": "Alpaca",
}
GROUPS = {
    "Math": ["gsm8k", "math500", "aime25"],
    "Code": ["humaneval", "mbpp", "livecodebench"],
    "Chat": ["mt-bench", "alpaca"],
}


def load(root: str):
    """-> (per[ds][unit_key][arm] = tps, arbar[ds], units[ds], clusters[ds])"""
    per: dict = collections.defaultdict(dict)
    ar: dict = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "*_T0.0.jsonl"))):
        ds = os.path.basename(path).replace("_T0.0.jsonl", "")
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["sample_idx"], r["turn_index"])
            if r["method"] == "ar":
                ar[ds].append(r["tps"])
            elif r["method"] in ARMS:
                per[ds].setdefault(key, {})[r["method"]] = (r["tps"], r.get("mean_accept"))
    if not per:
        raise SystemExit(f"no *_T0.0.jsonl records under {root}")
    arbar = {ds: st.fmean(v) for ds, v in ar.items()}
    units = {ds: [k for k, v in d.items() if all(a in v for a in ARMS)] for ds, d in per.items()}
    clusters = {}
    for ds, ks in units.items():
        g = collections.defaultdict(list)
        for k in ks:
            g[k[0]].append(k)           # k = (sample_idx, turn_index); cluster on conversation
        clusters[ds] = list(g.values())
    return per, arbar, units, clusters


def make_stats(per, arbar, units, clusters):
    def speedups(ds, k):
        return {a: per[ds][k][a][0] / arbar[ds] for a in ARMS}

    def tau(dss, arm):
        return st.fmean([st.fmean([per[ds][k][arm][1] for k in units[ds]
                                   if per[ds][k][arm][1] is not None]) for ds in dss])

    def boot(dss, num, den, B, seed):
        rng = random.Random(seed)

        def ratio(sample):
            n = st.fmean([st.fmean([speedups(ds, k)[num] for k in ks]) for ds, ks in sample.items()])
            d = st.fmean([st.fmean([speedups(ds, k)[den] for k in ks]) for ds, ks in sample.items()])
            return 100.0 * (n / d - 1.0)

        obs = ratio({ds: units[ds] for ds in dss})
        reps = []
        for _ in range(B):
            s = {}
            for ds in dss:
                cs = clusters[ds]
                picked = [cs[rng.randrange(len(cs))] for _ in cs]
                s[ds] = [k for c in picked for k in c]
            reps.append(ratio(s))
        reps.sort()
        return obs, reps[int(0.025 * B)], reps[int(0.975 * B)]

    return tau, boot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", default="results/raw/conditioning_ladder/matched")
    ap.add_argument("--graphbest", default="results/raw/conditioning_ladder/graphbest")
    ap.add_argument("--bootstrap-iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-md", default="results/tables/conditioning_ladder.md")
    a = ap.parse_args()

    m = load(a.matched)
    g = load(a.graphbest)
    tau_m, boot_m = make_stats(*m)
    tau_g, boot_g = make_stats(*g)

    rows = [(LABEL[d], [d]) for d in LABEL] + list(GROUPS.items()) + [("Overall", list(LABEL))]

    # tau is builder-invariant; verify that before printing it once.
    worst = max(abs(tau_m(dss, arm) - tau_g(dss, arm)) for _, dss in rows for arm in ARMS)
    note = (f"tau agrees between builder configurations to {worst:.3f} tokens "
            f"(expected: acceptance is builder-invariant).")
    if worst > 0.05:
        note = f"WARNING: tau differs by up to {worst:.3f} between configurations - investigate."

    units_m = m[2]
    out = [
        "# Three-arm conditioning ladder (paper Table 13)",
        "",
        f"Qwen3-4B, T=0, max_new_tokens=2048, budget 16, M=64. B={a.bootstrap_iters}, seed={a.seed}.",
        "Rollups are unweighted means over datasets; bootstrap stratified by dataset and",
        "clustered by conversation. " + note,
        "",
        "| Dataset / Rollup | n | tau Marg | tau Cond-static | tau DominoTree "
        "| matched Cond-static/Marg | matched DominoTree/Cond-static "
        "| best Cond-static/Marg | best DominoTree/Cond-static |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    fmt = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"
    for name, dss in rows:
        n = sum(len(units_m[d]) for d in dss)
        taus = [tau_m(dss, x) for x in ARMS]
        cells = [
            fmt(boot_m(dss, "condstatic@16", "marg@16", a.bootstrap_iters, a.seed)),
            fmt(boot_m(dss, "dominotree@16", "condstatic@16", a.bootstrap_iters, a.seed)),
            fmt(boot_g(dss, "condstatic@16", "marg@16", a.bootstrap_iters, a.seed)),
            fmt(boot_g(dss, "dominotree@16", "condstatic@16", a.bootstrap_iters, a.seed)),
        ]
        out.append("| " + " | ".join([name, str(n)] + [f"{t:.2f}" for t in taus] + cells) + " |")

    text = "\n".join(out)
    print(text)
    os.makedirs(os.path.dirname(a.out_md), exist_ok=True)
    with open(a.out_md, "w") as fh:
        fh.write(text + "\n")
    print(f"\n[written] {a.out_md}")


if __name__ == "__main__":
    main()
