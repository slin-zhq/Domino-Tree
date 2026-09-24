#!/usr/bin/env python3
"""Regenerate paper Table 5 (tab:sglang-bs1) in Table 1's layout.

The published version crammed "TPS/tau" into one cell and labelled the rollup
"(MACRO)". This emits Table 1's structure instead -- Math/Code/Chat column groups,
Speedup and tau as SEPARATE columns -- so the two headline tables read alike.

Like Table 1 there is no AR row. AR's absolute throughput range per model and temperature
is printed as trailing comments, for the caption.

Speedup is computed per dataset from RAW per-prompt rows (mean method TPS / mean AR
TPS in the same cell), never from the rounded values in PUBLISHED.json. Overall is
the unweighted mean of the eight per-dataset values, matching Table 1.

    python3 results/serving/gen_sglang_bs1_table.py > table5.tex
"""
import argparse, json, pathlib, statistics, sys

DS = [("gsm8k","GSM8K"),("math500","MATH-500"),("aime25","AIME25"),
      ("humaneval","HumanEval"),("mbpp","MBPP"),("livecodebench","LCB"),
      ("mt-bench","MT-Bench"),("alpaca","Alpaca")]
METHODS = [("eagle3","EAGLE-3"),("dflash","DFlash"),
           ("domino_chain","Domino"),("dominotree","DominoTree")]
END = r" \\"

def rows(root, size, method, ds, temp):
    p = root/"results"/"serving"/"bs1"/size/method/f"{ds}_T{temp}.jsonl"
    if not p.exists(): sys.exit(f"missing raw cell: {p}")
    out=[json.loads(l) for l in p.open() if l.strip()]
    if not out: sys.exit(f"empty raw cell: {p}")
    return out

def cell(root,size,method,ds,temp):
    """Per-prompt aggregation, matching results/serving/verify_published_numbers.py.

    MT-Bench is two TURNS of one conversation, and a turn is not its own sampling
    unit: a prompt's TPS is its summed output tokens over its summed decode time,
    and its tau is turn-length-weighted within the prompt. Taking a plain mean over
    rows instead (which is Table 1's convention, where the reference harness defines
    the unit differently) shifts MT-Bench by ~1.5% and would put one column of this
    table on a different estimand from every other serving table. The cell value is
    then the unweighted mean over prompts.
    """
    by = {}
    for x in rows(root,size,method,ds,temp):
        by.setdefault(x["sample_idx"], []).append(x)
    tps, tau = [], []
    for turns in by.values():
        n = sum(t["num_output"] for t in turns)
        tps.append(n / sum(t["decode_time"] for t in turns))
        steps = sum(t["num_output"] / t["mean_accept"] for t in turns)
        tau.append(n / steps)
    return statistics.fmean(tps), statistics.fmean(tau)

AR_TPS={}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[2],
                    help="Domino-Tree checkout (default: the one containing this script)")
    a=ap.parse_args()
    root=a.repo
    L=[r"\begin{tabular}{ll*{9}{cc}}", r"\toprule",
       r"Model & Method & \multicolumn{6}{c}{Math} & \multicolumn{6}{c}{Code} & "
       r"\multicolumn{4}{c}{Chat} & \multicolumn{2}{c}{Overall}"+END,
       r"\cmidrule(lr){3-8}\cmidrule(lr){9-14}\cmidrule(lr){15-18}\cmidrule(lr){19-20}",
       r" & & "+" & ".join(r"\multicolumn{2}{c}{"+n+"}" for _,n in DS)
       +r" & \multicolumn{2}{c}{Avg.}"+END]
    for temp in ["0.0","1.0"]:
        L += [r"\midrule",
              r"\multicolumn{2}{c}{Temperature = "+f"{float(temp):g}"+"} & "
              + " & ".join([r"Speedup & $\tau$"]*9)+END]
        for size,label in [("4b","Qwen3-4B"),("8b","Qwen3-8B")]:
            L += [r"\midrule", r"\multirow{4}{*}{"+label+"}"]
            ar={d:cell(root,size,"ar",d,temp)[0] for d,_ in DS}
            AR_TPS[size,temp]=(min(ar.values()), max(ar.values()))
            best={}
            for d,_ in DS+[("Overall","")]:
                sp={};ta={}
                for m,_n in METHODS:
                    if d=="Overall":
                        sp[m]=statistics.fmean(cell(root,size,m,x,temp)[0]/ar[x] for x,_ in DS)
                        ta[m]=statistics.fmean(cell(root,size,m,x,temp)[1] for x,_ in DS)
                    else:
                        t,u=cell(root,size,m,d,temp); sp[m]=t/ar[d]; ta[m]=u
                best[d]=(max(sp,key=sp.get),max(ta,key=ta.get),sp,ta)
            for m,name in METHODS:
                vs=[]
                for d,_ in DS+[("Overall","")]:
                    bs,bt,sp,ta=best[d]
                    f=lambda v,is_best: (r"\textbf{%.2f}" if is_best else "%.2f")%v
                    vs += [f(sp[m],m==bs), f(ta[m],m==bt)]
                nm = r"\textbf{DominoTree}" if m=="dominotree" else name
                L.append(" & "+nm+" & "+" & ".join(vs)+END)
    L += [r"\bottomrule", r"\end{tabular}"]
    print("\n".join(L))
    # AR has no row (as in Table 1); its absolute throughput range goes in the caption.
    for (size,temp),(lo,hi) in sorted(AR_TPS.items()):
        print(f"% AR tok/s range {size} T={temp}: {lo:.0f}-{hi:.0f}")

main()
