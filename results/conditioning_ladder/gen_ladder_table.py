#!/usr/bin/env python3
"""Rebuild the paper's Table 15 (conditioning ladder): accepted length tau of Marg@16,
Cond-static@16 and DominoTree@16 at Qwen3-4B and Qwen3-8B, T=0, and the two steps
Cond-static/Marg (the correction at all) and DominoTree/Cond-static (path dependence).

    python3 results/conditioning_ladder/gen_ladder_table.py

tau depends only on which nodes a tree contains, not on the code that builds it, so the
table reports tau only; throughput with the shipped builder is Table 1's job. Inputs:
  Qwen3-4B  matched_builder/   (RTX 5080; best_builder/ must give identical tau -- checked)
  Qwen3-8B  qwen3_8b_h100/     (one H100, Python builder for every arm)
Per-dataset tau = mean over prompt/turn rows; rollups are unweighted means over datasets.
Every file must hold exactly the expected (sample_idx, turn_index) rows for every arm.
Stdlib only.
"""
import json, pathlib, statistics as st, sys

HERE = pathlib.Path(__file__).resolve().parent
DS = ["gsm8k", "math500", "aime25", "humaneval", "mbpp", "livecodebench", "mt-bench", "alpaca"]
LABEL = {"gsm8k": "GSM8K", "math500": "MATH-500", "aime25": "AIME25", "humaneval": "HumanEval",
         "mbpp": "MBPP", "livecodebench": "LiveCodeBench", "mt-bench": "MT-Bench", "alpaca": "Alpaca"}
GROUPS = {"Math": DS[:3], "Code": DS[3:6], "Chat": DS[6:]}
ARMS = ["marg@16", "condstatic@16", "dominotree@16"]
BLOCKS = [("Qwen3-4B", HERE / "matched_builder"), ("Qwen3-8B", HERE / "qwen3_8b_h100")]
E = r" \\"


def fail(msg):
    sys.exit(f"FATAL: {msg}")


def expected(ds):
    return sorted((i, t) for i in range(30 if ds == "aime25" else 50)
                  for t in range(2 if ds == "mt-bench" else 1))


def load(d):
    """{ds: {arm: tau}} after checking every arm (and ar) has exactly the expected rows."""
    out = {}
    for ds in DS:
        p = d / f"{ds}_T0.0.jsonl"
        if not p.is_file():
            fail(f"{p} is missing")
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        out[ds] = {}
        for arm in ["ar"] + ARMS:
            rr = [r for r in rows if r["method"] == arm]
            if sorted((r["sample_idx"], r.get("turn_index", 0)) for r in rr) != expected(ds):
                fail(f"{p}: arm {arm} does not have exactly the expected prompt/turn rows")
            if arm != "ar":
                out[ds][arm] = st.fmean(r["mean_accept"] for r in rr)
    return out


def pct(a, b):
    return 100 * (a / b - 1)


def main():
    taus = {name: load(d) for name, d in BLOCKS}
    best = load(HERE / "best_builder")          # 4B, other builder: tau must not move
    for ds in DS:
        for arm in ARMS:
            if f"{best[ds][arm]:.2f}" != f"{taus['Qwen3-4B'][ds][arm]:.2f}":
                fail(f"tau differs between 4B builder configurations: {ds} {arm}")
    L = [r"\begin{tabular}{lc ccccc ccccc}", r"\toprule",
         r"& & \multicolumn{5}{c}{Qwen3-4B} & \multicolumn{5}{c}{Qwen3-8B}" + E,
         r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}",
         r"& & \multicolumn{3}{c}{$\tau$} & \multicolumn{2}{c}{$\Delta\tau$ (\%)}"
         r" & \multicolumn{3}{c}{$\tau$} & \multicolumn{2}{c}{$\Delta\tau$ (\%)}" + E,
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}\cmidrule(lr){11-12}",
         r"Dataset / Rollup & $n$ & Marg & Cond-static & DominoTree & Correction & Path"
         r" & Marg & Cond-static & DominoTree & Correction & Path" + E, r"\midrule"]

    def row(name, n, dsets):
        cells = []
        for block, _ in BLOCKS:
            m, c, t = (st.fmean(taus[block][d][a] for d in dsets) for a in ARMS)
            cells += [f"{m:.2f}", f"{c:.2f}", f"{t:.2f}", f"{pct(c, m):+.1f}", f"{pct(t, c):+.1f}"]
        return f"{name} & {n} & " + " & ".join(cells) + E

    n = {ds: len(expected(ds)) for ds in DS}
    for ds in DS:
        L.append(row(LABEL[ds], n[ds], [ds]))
    L.append(r"\midrule")
    for g, dsets in GROUPS.items():
        L.append(row(g, sum(n[d] for d in dsets), dsets))
    L += [r"\midrule", row(r"\textbf{Overall}", sum(n.values()), DS), r"\bottomrule", r"\end{tabular}"]
    print("\n".join(L))


if __name__ == "__main__":
    main()
