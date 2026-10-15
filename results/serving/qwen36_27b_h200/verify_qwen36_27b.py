#!/usr/bin/env python3
"""Check every Qwen3.6-27B serving number against the raw data shipped in this directory.

Nothing here trusts a summary table. It reads the per-prompt and per-cell JSONL under
this directory, recomputes every cell of every table `gen_qwen36_27b_tables.py` emits,
and diffs the result against `PUBLISHED.json`. It exits non-zero if anything disagrees,
is missing, or is short.

Run it from anywhere:

    python3 results/serving/qwen36_27b_h200/verify_qwen36_27b.py

Expected output ends with: `ALL <N> CELLS REPRODUCE FROM RAW DATA.`

This is a SEPARATE verifier from `results/serving/verify_published_numbers.py` (the
Qwen3-4B/8B audit), but it uses the SAME estimands -- per-prompt bs=1 (turns merged),
aggregate HELMET point estimates with per-prompt-mean paired CIs, unweighted-mean-over-
datasets concurrency (see gen_qwen36_27b_tables.py's module docstring for the exact
cross-references). It is separate because that script's 1112-cell count and
PUBLISHED.json are tightly coupled to the paper's existing serving.tex tables and to an
actively edited paper session, not because this bundle computes anything differently;
this bundle is new data for a different model on different hardware and gets its own
audit trail rather than risking either script's coverage count.

Conventions checked here match gen_qwen36_27b_tables.py exactly (same helper
functions, duplicated rather than imported so this file is a fully independent
check -- a bug shared by both would not be a bug this script could catch, but a bug in
only one of them will surface as a mismatch between their outputs on the same raw data
during development, which is how this file was built).

Requires only the Python standard library.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import difflib
from functools import lru_cache
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLISHED = HERE / "PUBLISHED.json"

METHODS = ["dflash", "domino_chain", "mtp", "dominotree"]
ALL_METHODS = METHODS + ["ar"]
DS = ["gsm8k", "math500", "aime25", "humaneval", "mbpp", "livecodebench", "mt-bench", "alpaca"]
TASKS = ["infbench_sum", "multi_lexsum"]
BINS = [8192, 16384, 32768]
CONC_DS = ["gsm8k", "mbpp", "mt-bench"]
CONCS = [1, 2, 4, 8, 16, 32]
TEMPS = ["0.0", "1.0"]
BOOT, SEED = 5000, 0

fails: list[str] = []
_count = 0


def note(ok: bool, msg: str) -> None:
    global _count
    _count += 1
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        fails.append(msg)


def rows(path: Path) -> list:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def require(n: int, min_n: int, tag: str) -> bool:
    if n < min_n:
        note(False, f"{tag}: {n} prompts, need >= {min_n} (missing or truncated file)")
        return False
    return True


def expected_n_prompts(dataset: str) -> int:
    return 30 if dataset == "aime25" else 50


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
    """PER-PROMPT, turns merged -- exactly ../gen_sglang_bs1_table.py's `cell()` /
    ../verify_published_numbers.py's `bs1_cell()` (the estimand Tables 5/6 use)."""
    groups = bs1_groups(method, dataset, temp)
    if not groups:
        return float("nan"), float("nan"), 0
    tps = st.fmean(_prompt_tps(v) for v in groups.values())
    tau = st.fmean(_prompt_tau(v) for v in groups.values())
    return tps, tau, len(groups)


def check_bs1(published: dict) -> None:
    print("== bs=1 (per dataset, both temperatures) ==")
    for temp in TEMPS:
        overall = {m: ([], []) for m in ALL_METHODS}
        for dataset in DS:
            ok = True
            for m in ALL_METHODS:
                _, _, n = bs1_cell(m, dataset, temp)
                ok &= require(n, expected_n_prompts(dataset), f"bs1 T{temp}/{dataset}/{m}")
            if not ok:
                continue
            for m in ALL_METHODS:
                tps, tau, _ = bs1_cell(m, dataset, temp)
                overall[m][0].append(tps)
                overall[m][1].append(tau)
                key = f"{temp}/{dataset}/{m}"
                exp = published.get(key)
                if exp is None:
                    note(False, f"{key}: no PUBLISHED.json entry")
                    continue
                note(abs(round(tps) - exp["tps"]) < 0.5,
                     f"{key:<28} tps  data {tps:8.2f} -> {round(tps):.0f}   pub {exp['tps']:.0f}")
                if m != "ar":
                    note(abs(round(tau, 2) - exp["tau"]) < 5e-3,
                         f"{key:<28} tau  data {tau:8.4f} -> {round(tau, 2):.2f}   pub {exp['tau']:.2f}")
        for m in ALL_METHODS:
            tpss, taus = overall[m]
            if len(tpss) != len(DS):
                continue
            key = f"{temp}/{m}"
            exp = published.get("_overall", {}).get(key)
            if exp is None:
                note(False, f"OVERALL {key}: no PUBLISHED.json entry")
                continue
            otps = st.mean(tpss)
            note(abs(round(otps) - exp["tps"]) < 0.5,
                 f"OVERALL {key:<20} tps  data {otps:8.2f} -> {round(otps):.0f}   pub {exp['tps']:.0f}")
            if m != "ar":
                otau = st.mean(taus)
                note(abs(round(otau, 2) - exp["tau"]) < 5e-3,
                     f"OVERALL {key:<20} tau  data {otau:8.4f} -> {round(otau, 2):.2f}   pub {exp['tau']:.2f}")


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
        return float("nan"), float("nan")
    tps = sum(r["output_tokens"] for r in rs) / sum(r["decode_time"] for r in rs)
    tau = st.fmean(r.get("accept") or 1.0 for r in rs)
    return tps, tau


def check_helmet(published: dict) -> None:
    print("\n== HELMET (per task and length) ==")
    prompts = {m: helmet_prompts(m) for m in ALL_METHODS}
    for t in TASKS:
        for b in BINS:
            ok = all(require(len(prompts[m].get((t, b), {})), 50, f"helmet {t}/{b}/{m}")
                     for m in ALL_METHODS)
            if not ok:
                continue
            for m in ALL_METHODS:
                tps, tau = helmet_cell(prompts[m], (t, b))
                key = f"{t}/{b}/{m}"
                exp = published.get(key)
                if exp is None:
                    note(False, f"{key}: no PUBLISHED.json entry")
                    continue
                note(abs(round(tps, 1) - exp["tps"]) < 0.05,
                     f"{key:<28} tps {tps:8.3f} -> {round(tps, 1):.1f}   pub {exp['tps']:.1f}")
                if m != "ar":
                    note(abs(round(tau, 2) - exp["tau"]) < 5e-3,
                         f"{key:<28} tau {tau:8.4f} -> {round(tau, 2):.2f}   pub {exp['tau']:.2f}")


def paired_ci(a: dict, b: dict, field: str) -> tuple:
    """MACRO (per-prompt field mean), matching ../gen_appendix_5090.py's `paired_ci` /
    ../verify_r4_longctx.py's `paired()` exactly, including the percentile index
    `int(0.975*len(draws)) - 1`."""
    keys = sorted(set(a) & set(b))
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


def check_helmet_ci(published: dict) -> None:
    print("\n== HELMET DominoTree vs. Domino chain, paired CI ==")
    prompts = {m: helmet_prompts(m) for m in ["dominotree", "domino_chain"]}
    for t in TASKS:
        for b in BINS:
            key = f"{t}/{b}"
            exp = published.get("_ci", {}).get(key)
            if exp is None:
                note(False, f"{key}: no PUBLISHED.json CI entry")
                continue
            for field, metric in [("tps", "tps"), ("accept", "tau")]:
                p, lo, hi = paired_ci(prompts["dominotree"][t, b], prompts["domino_chain"][t, b], field)
                want = exp[metric]
                note(abs(round(p, 1) - want) < 0.15,
                     f"{key:<18} d{metric}%  data {p:7.2f} -> {round(p, 1):+.1f}   pub {want:+.1f}")


# --------------------------------------------------------------------------- concurrency
@lru_cache(maxsize=None)
def conc_rows(method: str) -> dict:
    return {(r["dataset"], int(r["concurrency"])): r for r in rows(HERE / "concurrency" / method / f"{method}.jsonl")}


def check_conc(published: dict) -> None:
    print("\n== Concurrency (per dataset and concurrency, c=1..32 raw cells) ==")
    for m in ALL_METHODS:
        cells = conc_rows(m)
        missing = {(d, c) for d in CONC_DS for c in CONCS} - set(cells)
        if missing:
            note(False, f"{m}: missing cells {sorted(missing)}")
            continue
        for d in CONC_DS:
            n_expect = 80 if d == "mt-bench" else 128
            for c in CONCS:
                r = cells[(d, c)]
                key = f"{d}/{c}/{m}"
                exp = published.get(key)
                if exp is None:
                    note(False, f"{key}: no PUBLISHED.json entry")
                    continue
                note(r.get("n_prompts") == n_expect, f"{key}: n_prompts={r.get('n_prompts')}")
                note(abs(round(r["tps"]) - exp["tps"]) < 0.5,
                     f"{key:<24} tps {r['tps']:8.2f} -> {round(r['tps']):.0f}   pub {exp['tps']:.0f}")
                if m != "ar":
                    note(abs(round(r["mean_accept"], 2) - exp["tau"]) < 5e-3,
                         f"{key:<24} tau {r['mean_accept']:8.4f} -> {round(r['mean_accept'], 2):.2f}"
                         f"   pub {exp['tau']:.2f}")



TCONCS = [2, 4, 8, 16, 32]   # the concurrency levels the paper's tables report
ROW_METHODS = ["mtp", "dflash", "domino_chain", "dominotree"]   # the paper's row order
LONG_METHODS = ["ar"] + ROW_METHODS


def printed_evidence():
    """The numbers the paper's 27B section quotes in prose (gen_qwen36_27b_tables.py evidence)."""
    result = {}
    for temp in TEMPS:
        for m in ("domino_chain", "dominotree"):
            result[f"bs1/{temp}/{m}/speedup"] = round(st.mean(bs1_cell(m,d,temp)[0]/bs1_cell("ar",d,temp)[0] for d in DS),2)
    hp = {m:helmet_prompts(m) for m in ALL_METHODS}
    sp = [helmet_cell(hp["dominotree"],(t,b))[0]/helmet_cell(hp["ar"],(t,b))[0] for t in TASKS for b in BINS]
    result["helmet/tree_ar_min"],result["helmet/tree_ar_max"] = round(min(sp),2),round(max(sp),2)
    result["helmet/tree_chain_ci_above_zero"] = sum(paired_ci(hp["dominotree"][t,b],hp["domino_chain"][t,b],"tps")[1]>0 for t in TASKS for b in BINS)
    result["helmet/cells"] = len(TASKS)*len(BINS)
    for c in TCONCS:
        tree = st.mean(conc_rows("dominotree")[d,c]["tps"] for d in CONC_DS)
        chain = st.mean(conc_rows("domino_chain")[d,c]["tps"] for d in CONC_DS)
        result[f"conc/{c}/tree_over_chain_pct"] = round(100*(tree/chain-1),1)
    gate = HERE/"losslessness_gate"/"20260923T171916Z_356641"
    ar = {r["prompt_id"]:r["output_ids"] for r in rows(gate/"ar.jsonl")}
    for m in ("domino","dominotree","dominotree_flat"):
        rr = rows(gate/f"{m}.jsonl")
        assert len(rr)==20 and {r["prompt_id"] for r in rr}==set(ar)
        agree=[]
        for r in rr:
            n=0
            for a,b in zip(ar[r["prompt_id"]],r["output_ids"]):
                if a!=b: break
                n+=1
            agree.append(n)
        result[f"gate/{m}/median_prefix_agreement"] = st.median(agree)
        result[f"gate/{m}/tau"] = round(st.mean(r["tau"] for r in rr),2)
    return result


def _budget_tps(budget, b):
    rs = rows(HERE/"helmet_budget"/f"dominotree_b{budget}"/"helmet.prompts.jsonl")
    per = []
    for t in TASKS:
        cell = [r for r in rs if r["task"]==t and int(r["length_bin"])==b]
        assert len(cell)==50, (budget, t, b)
        per.append(sum(r["output_tokens"] for r in cell)/sum(r["decode_time"] for r in cell))
    return st.mean(per)


def _gate_point(path):
    rr=rows(path)
    assert len(rr)==20 and len({r['prompt_id'] for r in rr})==20, path
    assert re.search(r'^P9_GATE=PASS$',path.with_suffix('.verdict.txt').read_text(),re.M),path
    return [f"{sum(r['completion_tokens'] for r in rr)/sum(r['wall_s'] for r in rr):.1f}", f"{st.mean(r['tau'] for r in rr):.2f}"]


def printed_rows():
    """Independently recompute the Qwen3.6-27B rows of every paper table, as printed strings
    (in the paper's row and column order, '--' where the paper prints a dash)."""
    f2 = lambda v: f"{v:.2f}"
    result = {}
    out = []
    for temp in TEMPS:
        for m in ROW_METHODS:
            sp=[bs1_cell(m,d,temp)[0]/bs1_cell("ar",d,temp)[0] for d in DS]
            ta=[bs1_cell(m,d,temp)[1] for d in DS]
            out.append([f2(v) for pair in zip(sp+[st.mean(sp)],ta+[st.mean(ta)]) for v in pair])
    result["tab:sglang-bs1"] = out
    cc = {m:conc_rows(m) for m in ALL_METHODS}
    def conc_row(m, ds):
        vals=[f"{st.mean(cc[m][d,c]['tps'] for d in ds):.0f}" for c in TCONCS]
        vals.append("--" if m=="ar" else f2(st.mean(st.mean(cc[m][d,c]["mean_accept"] for c in TCONCS) for d in ds)))
        return vals
    result["tab:sglang-conc"] = [conc_row(m, CONC_DS) for m in LONG_METHODS]
    result["tab:app-conc-27b"] = [conc_row(m, ds) for ds in [[d] for d in CONC_DS]+[CONC_DS] for m in LONG_METHODS]
    hp = {m:helmet_prompts(m) for m in ALL_METHODS}
    out = []
    for m in ROW_METHODS:
        vals=[]
        for b in BINS:
            vals.append(f2(st.mean(helmet_cell(hp[m],(t,b))[1] for t in TASKS)))
            vals.append(f2(st.mean(helmet_cell(hp[m],(t,b))[0]/helmet_cell(hp["ar"],(t,b))[0] for t in TASKS)))
        out.append(vals)
    result["tab:sglang-longctx"] = out
    result["tab:longctx-budget"] = [[f"{_budget_tps(B,b):.1f}" for B in (15,32,64)] for b in BINS]
    out = []
    for t in TASKS:
        for m in LONG_METHODS:
            vals=[]
            for b in BINS:
                tp,ta=helmet_cell(hp[m],(t,b)); vals += [f"{tp:.1f}", f2(ta)]
            out.append(vals)
        for field in ("tps","accept"):
            out.append([f"{v:+.1f}" for b in BINS for v in paired_ci(hp["dominotree"][t,b],hp["domino_chain"][t,b],field)])
    result["tab:app-helmet"] = out
    base=HERE/'ablations'
    out = [_gate_point(base/'budget_ladder_m128'/f'dominotree_b{b}_k8_m128.jsonl') for b in (8,12,15,32,64)]
    for b in (8,15,32,64):
        txt=(base/'timing'/f'timing_b{b}.steps.txt').read_text()
        means=[st.mean(float(v) for v in re.findall(r'\b'+k+r'=([\d.]+)',txt)) for k in ('draft','build','prepare','verify','accept','commit')]
        out.append([f"{v:.3f}" for v in means]+[f"{sum(means):.3f}"])
    for k in (4,8,16):
        for m in (32,64,128,0):
            path=base/'node_topk_topm_grid'/f'dominotree_b15_k{k}_m{m}.jsonl'
            if (k,m)==(8,64): path=base/'budget_ladder'/'dominotree_b15.jsonl'
            out.append(_gate_point(path))
    result['tab:app-qwen27b-ablation'] = out
    return result


# --------------------------------------------------------------------------- printed tables
SERVING = HERE.parent
GENERATORS = {   # label -> (command, text marker the 27B part starts after)
    "tab:sglang-bs1": ([SERVING/"gen_sglang_bs1_table.py"], None),
    "tab:sglang-conc": ([HERE/"gen_qwen36_27b_tables.py", "conc_main"], None),
    "tab:sglang-longctx": ([SERVING/"gen_sglang_longctx_table.py"], None),
    "tab:longctx-budget": ([SERVING/"gen_longctx_budget_table.py"], None),
    "tab:app-helmet": ([SERVING/"gen_appendix_5090.py"], r"\label{tab:app-helmet}"),
    "tab:app-conc-27b": ([HERE/"gen_qwen36_27b_tables.py", "conc"], None),
    "tab:app-qwen27b-ablation": ([HERE/"gen_qwen36_27b_tables.py", "ablation"], None),
}
PAPER_FILES = {label: label.replace(":", "-") + ".tex" for label in GENERATORS}
METHOD_RE = re.compile(r"^(AR|MTP|DFlash|Domino|DominoTree|EAGLE-3)\b|vs\.")


def _strip_comments(text):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in text.splitlines())


def _cells(line):
    return [c.strip() for c in line.split("\\\\")[0].split("&")]


def _nums(cells):
    out=[]
    for c in cells:
        c=re.sub(r"\\textbf\{([^{}]*)\}",r"\1",c).replace(r"$\times$","").replace(r"$^{*}$","")
        c=re.sub(r"\\multicolumn\{\d+\}\{c\}\{(.*)\}$",r"\1",c)
        out += re.findall(r"--|[-+]?\d+(?:\.\d+)?",c)
    return out


def extract_27b(label, text):
    """Pull the Qwen3.6-27B numbers out of a table's LaTeX (generator output or paper file)."""
    text = _strip_comments(text)
    data = [l for l in text.splitlines() if "&" in l and "\\\\" in l]
    if label == "tab:sglang-longctx":
        rowsout=[]
        for l in data:
            c=_cells(l); name=re.sub(r"\\textbf\{([^{}]*)\}",r"\1",c[0]).strip()
            if len(c)==19 and name in ("MTP","DFlash","Domino","DominoTree"):
                rowsout.append(_nums(c[13:19]))
        return rowsout
    if label in ("tab:app-conc-27b",):
        seg = data
    elif label == "tab:app-qwen27b-ablation":
        return [_nums(_cells(l)[1:]) for l in data if re.match(r"^\s*\d+\s*&", l)]
    else:
        start = {"tab:sglang-bs1": r"\multirow{4}{*}{Qwen3.6-27B}", "tab:sglang-conc": r"\multirow{5}{*}{Qwen3.6-27B}",
                 "tab:longctx-budget": r"\multirow{3}{*}{Qwen3.6-27B}", "tab:app-helmet": r"\textbf{Qwen3.6-27B}"}[label]
        seg, on = [], False
        for l in text.splitlines():
            if start in l: on = True; continue
            if on and (l.strip().startswith((r"\midrule", r"\bottomrule")) and not seg_continues(label, l)):
                on = False
            if on and "&" in l and "\\\\" in l: seg.append(l)
    rowsout=[]
    for l in seg:
        c=_cells(l)
        if label == "tab:longctx-budget":
            rowsout.append(_nums(c[2:])); continue
        name=re.sub(r"\\textbf\{([^{}]*)\}",r"\1",c[1]).strip()
        if METHOD_RE.search(name): rowsout.append(_nums(c[2:]))
    return rowsout


def seg_continues(label, line):
    # The HELMET block spans both tasks (separated by \cmidrule, not \midrule) and ends at \bottomrule.
    return label == "tab:app-helmet" and not line.strip().startswith(r"\bottomrule")


def check_printed(pub, paper_dir):
    print("\n== Printed tables: generator output and stored values vs. raw data ==")
    expected=printed_rows()
    for label,(cmd,marker) in GENERATORS.items():
        out=subprocess.run([sys.executable,*map(str,cmd)],capture_output=True,text=True,check=True).stdout
        if marker: out=out.split(marker,1)[1]
        got=extract_27b(label,out)
        want=expected[label]; stored=pub["printed_rows"].get(label,[])
        note(len(got)==len(want)==len(stored), f"{label}: {len(want)} rows (generator {len(got)}, stored {len(stored)})")
        for i,row in enumerate(want):
            g=got[i] if i<len(got) else []; p_=stored[i] if i<len(stored) else []
            note(g==row==p_, f"{label} row {i}: raw {row}" + ("" if g==row else f" | generator {g}") + ("" if p_==row else f" | stored {p_}"))
    gen_ev=dict(re.findall(r"^% evidence (\S+) = (\S+)$",subprocess.run([sys.executable,str(HERE/"gen_qwen36_27b_tables.py"),"evidence"],capture_output=True,text=True,check=True).stdout,re.M))
    for key,v in printed_evidence().items():
        note(key in gen_ev and float(gen_ev[key])==v==pub["prose_evidence"].get(key), f"prose {key} = {v}")
    if paper_dir:
        print(f"\n== Paper tables in {paper_dir}/tables (report only) ==")
        bad=0
        for label,fname in PAPER_FILES.items():
            got=extract_27b(label,(Path(paper_dir)/"tables"/fname).read_text())
            want=expected[label]
            if len(got)!=len(want): print(f"  ROWS {label}: paper {len(got)} vs raw {len(want)}"); bad+=1; continue
            for i,(g,w) in enumerate(zip(got,want)):
                for j,(a,b) in enumerate(zip(g,w)):
                    if a!=b: print(f"  DIFF {label} row {i} col {j}: paper {a}, raw {b}"); bad+=1
        print(f"  paper vs raw: {bad} cell(s) differ")


# --------------------------------------------------------------------------- --emit support
def emit_all() -> dict:
    bs1: dict = {}
    overall: dict = {}
    for temp in TEMPS:
        acc = {m: ([], []) for m in ALL_METHODS}
        for dataset in DS:
            for m in ALL_METHODS:
                tps, tau, n = bs1_cell(m, dataset, temp)
                if n == 0:
                    continue
                bs1[f"{temp}/{dataset}/{m}"] = {"tps": round(tps), "tau": round(tau, 2)}
                acc[m][0].append(tps)
                acc[m][1].append(tau)
        for m in ALL_METHODS:
            if len(acc[m][0]) == len(DS):
                overall[f"{temp}/{m}"] = {"tps": round(st.mean(acc[m][0])), "tau": round(st.mean(acc[m][1]), 2)}
    bs1["_overall"] = overall

    helmet: dict = {}
    prompts = {m: helmet_prompts(m) for m in ALL_METHODS}
    for t in TASKS:
        for b in BINS:
            for m in ALL_METHODS:
                tps, tau = helmet_cell(prompts[m], (t, b))
                helmet[f"{t}/{b}/{m}"] = {"tps": round(tps, 1), "tau": round(tau, 2)}
    ci: dict = {}
    for t in TASKS:
        for b in BINS:
            entry = {}
            for field, metric in [("tps", "tps"), ("accept", "tau")]:
                p, _, _ = paired_ci(prompts["dominotree"][t, b], prompts["domino_chain"][t, b], field)
                entry[metric] = round(p, 1)
            ci[f"{t}/{b}"] = entry
    helmet["_ci"] = ci

    conc: dict = {}
    for m in ALL_METHODS:
        cells = conc_rows(m)
        for (d, c), r in cells.items():
            conc[f"{d}/{c}/{m}"] = {"tps": round(r["tps"]),
                                     "tau": round(r["mean_accept"], 2) if m != "ar" else None}

    return {
        "_comment": ("Values recomputed from the raw JSONL shipped in this directory. "
                     "Regenerate with `python3 verify_qwen36_27b.py --emit PUBLISHED.json`. "
                     "Do not hand-edit."),
        "_expected_cell_checks": None,  # filled in below
        "printed_rows": printed_rows(),
        "prose_evidence": printed_evidence(),
        "bs1": bs1,
        "helmet_appendix": helmet,
        "concurrency_per_dataset": conc,
    }


def _count_checks(payload) -> int:
    """How many note() calls a full run makes, so a silently skipped check is caught."""
    n = 0
    for k, v in payload["bs1"].items():
        n += 0 if k == "_overall" else (1 if k.endswith("/ar") else 2)
    n += sum(1 if k.endswith("/ar") else 2 for k in payload["bs1"]["_overall"])
    n += sum(1 if k.endswith("/ar") else 2 for k in payload["helmet_appendix"] if not k.startswith("_"))
    n += 2 * len(payload["helmet_appendix"]["_ci"])
    n += sum(2 if k.endswith("/ar") else 3 for k in payload["concurrency_per_dataset"])
    n += sum(1 + len(v) for v in payload["printed_rows"].values()) + len(payload["prose_evidence"])
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", metavar="PATH", help="Write recomputed values as PUBLISHED.json-shaped JSON.")
    ap.add_argument("--paper-dir", help="Also compare the Qwen3.6-27B cells of the paper's tables/ files (report only).")
    args = ap.parse_args()

    if args.emit:
        payload = emit_all()
        payload["_expected_cell_checks"] = _count_checks(payload)
        Path(args.emit).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.emit}")
        return 0

    if not PUBLISHED.is_file():
        print(f"missing {PUBLISHED}", file=sys.stderr)
        return 2
    pub = json.loads(PUBLISHED.read_text())
    print(f"data: {HERE}\n")

    check_bs1(pub["bs1"])
    check_helmet(pub["helmet_appendix"])
    check_helmet_ci(pub["helmet_appendix"])
    check_conc(pub["concurrency_per_dataset"])
    check_printed(pub, args.paper_dir)

    print("\n" + "=" * 72)
    expected = pub.get("_expected_cell_checks")
    if expected is not None and _count != expected:
        print(f"COVERAGE FAILURE: verified {_count} cells, expected {expected}.")
        return 1
    if fails:
        print(f"{len(fails)} MISMATCH(ES) between PUBLISHED.json and the raw data:")
        for f in fails:
            print("  - " + f)
        return 1
    print(f"ALL {_count} CELLS REPRODUCE FROM RAW DATA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
