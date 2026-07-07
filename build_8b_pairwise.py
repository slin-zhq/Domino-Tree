#!/usr/bin/env python3
"""8B per-prompt paired-bootstrap CIs, mirroring make_latex_table.write_pairwise.

Cross-harness (vs DFlash/DDTree/CaDDTree): each side's per-prompt speedup =
method_tps[i]/own_AR_tps[i]; pair by prompt index; Delta% = 100*(mean(ours)/
mean(ref)-1) with paired bootstrap. Same-harness (vs Domino-chain): raw per-prompt
TPS, no normalization. No warmup trim (warmup-enabled runs). AR is the T=0 per-prompt
AR on each side (temperature-independent to <1%).
"""
import json, glob, statistics as st, random, math

R = "results/raw/8b"
OUR_T0_GPU = R + "/recollect_gpunative_8b_t0_20260707"     # dtree+chain, T=0 gpu-native
OUR_TGT0   = R + "/recollect_gpunative_8b_2048_20260707"   # dtree+chain, T>0 gpu-native
OUR_AR     = R + "/collect_8b_2048_20260704/our"           # ar, T=0
REF        = R + "/ref8b_perprompt_jsonl"                  # dflash/ddtree_tb16/caddtree(+baseline@T0)
DS = ['gsm8k','math500','aime25','humaneval','mbpp','livecodebench','mt-bench','alpaca']
GROUPS = {'Math':['gsm8k','math500','aime25'], 'Code':['humaneval','mbpp','livecodebench'],
          'Chat':['mt-bench','alpaca'], 'Overall':DS}
B, SEED = 10000, 12345

def rows(d, ds, temp):
    fs = glob.glob(f"{d}/*{ds}_T{temp}.jsonl")
    return [json.loads(l) for l in open(fs[0]) if l.strip()] if fs else []

def by_idx(rs, method, key='tps'):
    return {int(r['sample_idx']): float(r[key]) for r in rs
            if r.get('method') == method and r.get(key) is not None}

def our_units(ds, temp):
    """returns dict: dtree_tps, chain_tps, ar_tps, dtree_speedup keyed by prompt idx."""
    odir = OUR_T0_GPU if temp == '0.0' else OUR_TGT0
    o = rows(odir, ds, temp)
    ar = by_idx(rows(OUR_AR, ds, '0.0'), 'ar')
    dt = by_idx(o, 'dominotree@16'); ch = by_idx(o, 'chain')
    sp = {i: dt[i]/ar[i] for i in dt if i in ar and ar[i] > 0}
    return dt, ch, ar, sp

def ref_speedup(ds, temp, method):
    rmeth = by_idx(rows(REF, ds, temp), method)
    rbase = by_idx(rows(REF, ds, '0.0'), 'baseline')   # T=0 ref AR (temp-independent)
    return {i: rmeth[i]/rbase[i] for i in rmeth if i in rbase and rbase[i] > 0}

def paired_delta_ci(pairs, iters, rng):
    if not pairs: return float('nan'), float('nan'), float('nan')
    def delta(s):
        den = st.fmean([b for _, b in s])
        return 100.0*(st.fmean([a for a, _ in s])/den - 1.0) if den > 0 else float('nan')
    obs = delta(pairs)
    if len(pairs) < 2: return obs, float('nan'), float('nan')
    vals = []
    for _ in range(iters):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        v = delta(s)
        if math.isfinite(v): vals.append(v)
    vals.sort()
    return obs, vals[int(0.025*(len(vals)-1))], vals[int(0.975*(len(vals)-1))]

def pairs_for(dss, temp, comparison):
    out = []
    for ds in dss:
        dt, ch, ar, sp = our_units(ds, temp)
        if comparison == 'chain':
            for i in sorted(set(dt) & set(ch)): out.append((dt[i], ch[i]))
        else:
            rsp = ref_speedup(ds, temp, comparison)
            for i in sorted(set(sp) & set(rsp)): out.append((sp[i], rsp[i]))
    return out

LABELS = [('dflash','vs DFlash'), ('ddtree_tb16','vs DDTree@16'),
          ('caddtree','vs CaDDTree'), ('chain','vs Domino-chain')]
for temp in ['0.0', '0.5', '1.0']:
    print(f"\n===== Qwen3-8B  T={temp}  (Delta% [95% CI], n) =====")
    for key, lbl in LABELS:
        rng = random.Random(SEED)
        line = f"  {lbl:16}"
        for gname in ['Math','Code','Chat','Overall']:
            pr = pairs_for(GROUPS[gname], temp, key)
            obs, lo, hi = paired_delta_ci(pr, B, rng)
            line += f"  {gname}={obs:+.2f}[{lo:+.2f},{hi:+.2f}](n{len(pr)})"
        print(line)
