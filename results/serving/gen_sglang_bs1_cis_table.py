#!/usr/bin/env python3
"""Regenerate paper Table 6 (tab:sglang-bs1-cis): DominoTree's single-request throughput
advantage over the Domino chain, per dataset plus an Overall row, with 95% CIs.

    python3 results/serving/gen_sglang_bs1_cis_table.py > table6.tex

Per-dataset rows reuse verify_r2_5090.py's paired_delta verbatim (per-prompt TPS = a prompt's
summed tokens / summed decode time, so MT-Bench's two turns form one unit; percentile
bootstrap over prompts, B=5000, seed 0), so they are the same numbers that script verifies.

Overall = the mean of the eight per-dataset deltas, i.e. the same "mean per-dataset" convention
as the abstract's "1.17x / 1.09x the chain" (mean of per-dataset ratios), so the two agree. Its
CI resamples prompts WITHIN each dataset (stratified), B=5000, seed 0.

A cell is starred when its CI contains zero.
"""
import os, random, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_r2_5090 import load_cell, prompt_tps, paired_delta, DATASETS  # noqa: E402

LABEL = {"gsm8k": "GSM8K", "math500": "MATH-500", "aime25": "AIME25", "humaneval": "HumanEval",
         "mbpp": "MBPP", "livecodebench": "LCB", "mt-bench": "MT-Bench", "alpaca": "Alpaca"}
GROUPS = [("4B", "0.0"), ("4B", "1.0"), ("8B", "0.0"), ("8B", "1.0")]
B, SEED = 5000, 0
E = r" \\"


def overall(model, temp):
    per = []
    for d in DATASETS:
        a, b = load_cell(model, "dominotree", d, temp), load_cell(model, "domino_chain", d, temp)
        keys = sorted(set(a) & set(b))
        if set(a) != set(b):
            raise SystemExit(f"UNPAIRED {model} T{temp} {d}")
        per.append(([prompt_tps(a[k]) for k in keys], [prompt_tps(b[k]) for k in keys]))
    def ratio(samples):
        return st.mean(100 * (st.mean(x) / st.mean(y) - 1) for x, y in samples)
    point = ratio(per)
    rng = random.Random(SEED)
    draws = []
    for _ in range(B):
        s = []
        for av, bv in per:
            idx = [rng.randrange(len(av)) for _ in av]
            s.append(([av[i] for i in idx], [bv[i] for i in idx]))
        draws.append(ratio(s))
    draws.sort()
    return point, draws[int(0.025 * B)], draws[int(0.975 * B) - 1]


def cell(p, lo, hi):
    star = r"$^{*}$" if lo <= 0 <= hi else ""
    return f"{p:+.1f}{star} & $[{lo:+.1f},{hi:+.1f}]$"


def main():
    L = [r"\begin{tabular}{l*{4}{cc}}", r"\toprule",
         r"\multirow{2}{*}{Dataset} & " + " & ".join(
             r"\multicolumn{2}{c}{Qwen3-%s, $T{=}%s$}" % (m, "0" if t == "0.0" else "1") for m, t in GROUPS) + E,
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r" & " + " & ".join([r"$\Delta\%$ & 95\% CI"] * 4) + E, r"\midrule"]
    for gi, members in enumerate((DATASETS[:3], DATASETS[3:6], DATASETS[6:])):
        if gi:
            L.append(r"\cmidrule(lr){1-9}")
        for d in members:
            cells = []
            for m, t in GROUPS:
                p, lo, hi, _n = paired_delta(load_cell(m, "dominotree", d, t),
                                             load_cell(m, "domino_chain", d, t), prompt_tps, B, SEED)
                cells.append(cell(p, lo, hi))
            L.append(LABEL[d] + " & " + " & ".join(cells) + E)
    L.append(r"\midrule")
    L.append(r"\textbf{Overall} & " + " & ".join(cell(*overall(m, t)) for m, t in GROUPS) + E)
    print("\n".join(L + [r"\bottomrule", r"\end{tabular}"]))


if __name__ == "__main__":
    main()
