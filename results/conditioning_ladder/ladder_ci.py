#!/usr/bin/env python3
"""Three-point conditioning ladder with paired, stratified bootstrap CIs.

Convention matches tab:c3: Delta% = (A speedup / B speedup - 1), paired per evaluated
unit (prompt, or MT-Bench turn), stratified by dataset, B=5000 resamples.
"""
import json, glob, random, statistics, collections, sys

ROOT = __import__("sys").argv[1] if len(__import__("sys").argv)>1 else "results/domino_tree/conditioning_ladder_20260816"
ARMS = ["marg@16", "condstatic@16", "dominotree@16"]
GROUPS = {"Math": ["gsm8k","math500","aime25"], "Code": ["humaneval","mbpp","livecodebench"],
          "Chat": ["mt-bench","alpaca"]}
LABEL = {"gsm8k":"GSM8K","math500":"MATH-500","aime25":"AIME25","humaneval":"HumanEval",
         "mbpp":"MBPP","livecodebench":"LiveCodeBench","mt-bench":"MT-Bench","alpaca":"Alpaca"}

per = collections.defaultdict(dict)   # (ds, key) -> {arm: (tps, tau)}
ar  = collections.defaultdict(list)
for f in sorted(glob.glob(f"{ROOT}/*_T0.0.jsonl")):
    ds = f.split("/")[-1].replace("_T0.0.jsonl","")
    for l in open(f):
        if not l.strip(): continue
        r = json.loads(l)
        key = (r["sample_idx"], r["turn_index"])
        if r["method"] == "ar":
            ar[ds].append(r["tps"]); continue
        if r["method"] in ARMS:
            per[ds].setdefault(key, {})[r["method"]] = (r["tps"], r.get("mean_accept"))

arbar = {ds: statistics.fmean(v) for ds, v in ar.items()}
units = {ds: [k for k, v in d.items() if all(a in v for a in ARMS)] for ds, d in per.items()}

def speedups(ds, key):
    return {a: per[ds][key][a][0] / arbar[ds] for a in ARMS}

def boot(dss, num, den, B=5000, seed=0):
    """Paper convention: a rollup is the UNWEIGHTED mean over its datasets, so the
    ratio is (mean_ds mean_unit num) / (mean_ds mean_unit den). Resampling is paired
    (same units for both arms) and stratified within each dataset."""
    rng = random.Random(seed)
    def ratio(sample_by_ds):
        n = statistics.fmean([statistics.fmean([speedups(ds,k)[num] for k in ks]) for ds, ks in sample_by_ds.items()])
        d = statistics.fmean([statistics.fmean([speedups(ds,k)[den] for k in ks]) for ds, ks in sample_by_ds.items()])
        return 100*(n/d - 1)
    obs = ratio({ds: units[ds] for ds in dss})
    reps = []
    for _ in range(B):
        s = {ds: [units[ds][rng.randrange(len(units[ds]))] for _ in units[ds]] for ds in dss}
        reps.append(ratio(s))
    reps.sort()
    return obs, reps[int(0.025*B)], reps[int(0.975*B)]

def tau(dss, arm):   # unweighted over datasets, per paper convention
    return statistics.fmean([statistics.fmean([per[ds][k][arm][1] for k in units[ds]
                                               if per[ds][k][arm][1] is not None]) for ds in dss])

def spd(dss, arm):   # unweighted over datasets, per paper convention
    return statistics.fmean([statistics.fmean([speedups(ds,k)[arm] for k in units[ds]]) for ds in dss])

rows = [(LABEL[d], [d]) for d in LABEL] + [(g, v) for g, v in GROUPS.items()] + [("Overall", list(LABEL))]
print(f"{'row':<14}{'marg tau':>9}{'stat tau':>9}{'DT tau':>8}{'marg sp':>9}{'stat sp':>9}{'DT sp':>8}"
      f"{'stat/marg %':>26}{'DT/stat %':>26}{'DT/marg %':>26}")
for name, dss in rows:
    n = sum(len(units[d]) for d in dss)
    a = [tau(dss,x) for x in ARMS]; s = [spd(dss,x) for x in ARMS]
    d1 = boot(dss,"condstatic@16","marg@16"); d2 = boot(dss,"dominotree@16","condstatic@16")
    d3 = boot(dss,"dominotree@16","marg@16")
    fmt = lambda t: f"{t[0]:+6.2f} [{t[1]:+6.2f},{t[2]:+6.2f}]"
    print(f"{name:<14}{a[0]:9.2f}{a[1]:9.2f}{a[2]:8.2f}{s[0]:9.2f}{s[1]:9.2f}{s[2]:8.2f}"
          f"{fmt(d1):>26}{fmt(d2):>26}{fmt(d3):>26}   n={n}")
