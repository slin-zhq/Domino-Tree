#!/usr/bin/env python3
"""Regenerate paper Table 9 (tab:longctx-budget): the DominoTree tree-budget sweep at
long context, mean throughput (tok/s) over the two HELMET tasks: Qwen3-4B/8B at budgets
16/32/64, and Qwen3.6-27B at 15/32/64 (qwen36_27b_h200/helmet_budget/, one H200, one session).

    python3 results/serving/gen_longctx_budget_table.py > table9.tex

Values come from PUBLISHED.json's `longctx_budget` block, which verify_published_numbers.py
re-derives from the raw per-prompt HELMET data (each cell averages the two tasks). Bold
marks the best (highest tok/s) budget in each row; a tie at the printed precision bolds
every tied value. Qwen3-4B's row was measured in a single later session on the same
prompts, on a different RTX 5090 host from Table 8 (tab:sglang-longctx) -- see
longcontext/4b/budget_sweep_20260925/PROVENANCE.txt -- so absolute throughput is only
comparable budget-to-budget within a model, not across tables.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = json.load(open(os.path.join(HERE, "PUBLISHED.json")))["longctx_budget"]


def budget_27b(budget, ctx_k):
    """Mean over the two tasks of each task's SUM(output_tokens)/SUM(decode_time), 50 prompts each."""
    path = os.path.join(HERE, "qwen36_27b_h200", "helmet_budget", f"dominotree_b{budget}", "helmet.prompts.jsonl")
    rs = [json.loads(l) for l in open(path) if l.strip()]
    per_task = []
    for task in ("infbench_sum", "multi_lexsum"):
        cell = [r for r in rs if r["task"] == task and int(r["length_bin"]) == ctx_k * 1024]
        assert len(cell) == 50, (path, task, ctx_k, len(cell))
        per_task.append(sum(r["output_tokens"] for r in cell) / sum(r["decode_time"] for r in cell))
    return sum(per_task) / 2

SIZES = [("4b", "Qwen3-4B"), ("8b", "Qwen3-8B"), ("27b", "Qwen3.6-27B")]
BUDGETS = [16, 32, 64]  # the narrowest budget is 15 for Qwen3.6-27B
BINS = [(8, "8K"), (16, "16K"), (32, "32K")]
CTX_WIDTH = 3  # "8K" / "16K" / "32K" left-padded to a common column width
E = r" \\"

fmt = lambda v: f"{v:.1f}"

L = [r"\begin{tabular}{llccc}", r"\toprule",
     r"Model & Context & Narrowest$^{\dagger}$ & $B{=}32$ & $B{=}64$" + E, r"\midrule"]
for i, (size, label) in enumerate(SIZES):
    if i:
        L.append(r"\midrule")
    L.append(r"\multirow{3}{*}{" + label + "}")
    for ctx_k, ctx_lab in BINS:
        if size == "27b":
            raw = [fmt(budget_27b(b, ctx_k)) for b in (15, 32, 64)]
        else:
            raw = [fmt(PUB[f"{size}/{ctx_k}/{b}"]) for b in BUDGETS]
        best = max(raw, key=float)  # numeric, not string, comparison ("98.4" > "105.5" as text)
        cells = [r"\textbf{" + s + "}" if s == best else s for s in raw]
        ctx_col = ctx_lab.ljust(CTX_WIDTH)
        L.append(f" & {ctx_col} & " + " & ".join(cells) + E)
print("\n".join(L + [r"\bottomrule", r"\end{tabular}"]))
