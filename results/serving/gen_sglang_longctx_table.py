#!/usr/bin/env python3
"""Regenerate paper Table 8 (tab:sglang-longctx): HELMET long-context results, accepted length
tau and speedup over AR as SEPARATE columns, both model sizes.

    python3 results/serving/gen_sglang_longctx_table.py > table8.tex

Values come from PUBLISHED.json's `longctx` block, which verify_published_numbers.py re-derives
from the raw per-prompt HELMET data (each cell averages the two tasks). Bold marks the best
value in each column, for tau and for speedup separately; values that tie at the printed
precision are all bolded, so a tie is never shown as a win. Like Tables 1 and 5 there is no AR
row: its speedup is 1.00 and it has no tau.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = json.load(open(os.path.join(HERE, "PUBLISHED.json")))["longctx"]
METHODS = [("eagle3", "EAGLE-3"), ("dflash", "DFlash"), ("domino_chain", "Domino"),
           ("dominotree", "DominoTree")]
SIZES = [("4b", "Qwen3-4B ($B{=}16$)"), ("8b", "Qwen3-8B ($B{=}32$)")]
BINS = [("8192", "8K"), ("16384", "16K"), ("32768", "32K")]
E = r" \\"

cols = [(s, b, i) for s, _ in SIZES for b, _ in BINS for i in (0, 1)]   # i: 0 = tau, 1 = speedup
fmt = lambda v: f"{v:.2f}"
best = {c: max(fmt(PUB[f"{c[0]}/{c[1]}/{m}"][c[2]]) for m, _ in METHODS) for c in cols}
best = {c: max(PUB[f"{c[0]}/{c[1]}/{m}"][c[2]] for m, _ in METHODS
               if fmt(PUB[f"{c[0]}/{c[1]}/{m}"][c[2]]) == max(fmt(PUB[f"{c[0]}/{c[1]}/{mm}"][c[2]]) for mm, _ in METHODS))
        for c in cols}

L = [r"\begin{tabular}{l*{12}{c}}", r"\toprule",
     " & " + " & ".join(r"\multicolumn{6}{c}{" + lab + "}" for _, lab in SIZES) + E,
     r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}",
     " & " + " & ".join(r"\multicolumn{2}{c}{" + lab + "}" for _ in SIZES for _, lab in BINS) + E,
     "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * k, 3 + 2 * k) for k in range(6)),
     "Method & " + " & ".join([r"$\tau$ & Speedup"] * 6) + E, r"\midrule"]
for m, name in METHODS:
    cells = []
    for c in cols:
        v = PUB[f"{c[0]}/{c[1]}/{m}"][c[2]]
        s = fmt(v) + (r"$\times$" if c[2] == 1 else "")
        cells.append(r"\textbf{" + s + "}" if fmt(v) == fmt(best[c]) else s)
    L.append((r"\textbf{DominoTree}" if m == "dominotree" else name) + " & " + " & ".join(cells) + E)
print("\n".join(L + [r"\bottomrule", r"\end{tabular}"]))
