#!/usr/bin/env python3
"""Build 8B Table 1 blocks, GPU-native DominoTree at every temperature.

T=0 our-side DominoTree+chain now come from the GPU-native recollection
(recollect_gpunative_8b_t0_20260707), replacing the python-builder cond@16.
AR (builder-independent, temp-independent to <1%) is reused from the T=0
collection for every temperature, on both harness sides. Emits paste-ready
LaTeX rows with per-column bold on the max speedup and max tau.
"""
import json, statistics as st, os, glob
from collections import defaultdict

R = "results/raw/8b"
REF_T0   = f"{R}/collect_8b_2048_20260704/baseline_official"   # dflash/ddtree/caddtree @ T=0
OUR_T0   = f"{R}/collect_8b_2048_20260704/our"                 # ar (+ python-builder cond@16) @ T=0
REF_TGT0 = f"{R}/reference_8b_tgt0_20260707"                   # reference @ T>0
OUR_TGT0 = f"{R}/recollect_gpunative_8b_2048_20260707"         # gpu-native chain+dominotree @ T>0
OUR_T0_GPU = f"{R}/recollect_gpunative_8b_t0_20260707"         # gpu-native chain+dominotree @ T=0

DS = ['gsm8k','math500','aime25','humaneval','mbpp','livecodebench','mt-bench','alpaca']

def ref_rows(d, ds, temp):
    f = f"{d}/{ds}_T{temp}.pt.summary.json"
    if not os.path.exists(f): return None
    return {r['method']: r for r in json.load(open(f))['rows']}

def ref_ar_t0(ds):
    rows = ref_rows(REF_T0, ds, '0.0')
    if rows is None: return None
    if 'baseline' in rows: return rows['baseline']['tps_total']
    for m in ['dflash','ddtree_tb16','caddtree']:
        if m in rows and rows[m].get('speedup_vs_baseline',0) > 0:
            return rows[m]['tps_total'] / rows[m]['speedup_vs_baseline']
    return None

def our_stats(d, ds, temp, method):
    fs = glob.glob(f"{d}/*{ds}_T{temp}.jsonl")
    if not fs: return None, None
    rows = [json.loads(l) for l in open(fs[0]) if l.strip()]
    # No warmup-row trim: these 8B runs use the warmup-enabled benchmark.py
    # (in-loop "Warmup" prompt heats kernels/caches before timing), so every
    # measured prompt is warm and we take a plain mean over all of them --
    # matching the DDTree/CaDDTree/DFlash reference SOP and the reference rows
    # here (which are all-prompt aggregates).
    rr = [r for r in rows if r.get('method') == method]
    if not rr: return None, None
    return st.fmean(r['tps'] for r in rr), st.fmean(r['mean_accept'] for r in rr)

def our_ar_t0(ds):
    tps, _ = our_stats(OUR_T0, ds, '0.0', 'ar'); return tps

HDR = ['DFlash','DDTree (16)','CaDDTree','Domino','DominoTree (16)']

def compute(temp, our_dir, dtree_method):
    out = {h: [] for h in HDR}
    for ds in DS:
        rar, oar = ref_ar_t0(ds), our_ar_t0(ds)
        rows = ref_rows(REF_TGT0 if temp != '0.0' else REF_T0, ds, temp)
        if rows is None or rar is None or oar is None:
            return None
        for lbl, m in [('DFlash','dflash'),('DDTree (16)','ddtree_tb16'),('CaDDTree','caddtree')]:
            r = rows[m]; out[lbl].append((r['tps_total']/rar, r['accept_len_round_weighted']))
        ctps, ctau = our_stats(our_dir, ds, temp, 'chain')
        dtps, dtau = our_stats(our_dir, ds, temp, dtree_method)
        if None in (ctps, dtps): return None
        out['Domino'].append((ctps/oar, ctau))
        out['DominoTree (16)'].append((dtps/oar, dtau))
    return out

def latex_block(temp, out):
    # per-column maxima (to 2 d.p.) for bolding
    ncol = len(DS)
    max_sp = [max(out[h][j][0] for h in HDR) for j in range(ncol)]
    max_ta = [max(out[h][j][1] for h in HDR) for j in range(ncol)]
    # add overall column
    def b(v, mx): return f"\\textbf{{{v:.2f}}}" if f"{v:.2f}" == f"{mx:.2f}" else f"{v:.2f}"
    print(f"% ---- Qwen3-8B, T={temp} ----")
    for h in HDR:
        cells = []
        for j in range(ncol):
            sp, ta = out[h][j]
            cells += [b(sp, max_sp[j]), b(ta, max_ta[j])]
        ov_sp = st.fmean(sp for sp,_ in out[h]); ov_ta = st.fmean(ta for _,ta in out[h])
        ov_max_sp = max(st.fmean(s for s,_ in out[k]) for k in HDR)
        ov_max_ta = max(st.fmean(t for _,t in out[k]) for k in HDR)
        cells += [b(ov_sp, ov_max_sp), b(ov_ta, ov_max_ta)]
        name = f"\\textbf{{{h}}}" if h == 'DominoTree (16)' else h
        print(f"& {name} & " + " & ".join(cells) + r" \\")
    print()

for temp, odir, dm in [('0.0', OUR_T0_GPU, 'dominotree@16'),
                       ('0.5', OUR_TGT0, 'dominotree@16'),
                       ('1.0', OUR_TGT0, 'dominotree@16')]:
    out = compute(temp, odir, dm)
    if out is None:
        print(f"% T={temp}: DATA NOT READY (our_dir={odir})\n")
        continue
    latex_block(temp, out)

# python-builder T=0 (validation vs current paper: DominoTree 6.30/10.53, Domino 6.77/10.13)
print("% --- python-builder T=0 (sanity vs current paper 8B T=0) ---")
b0 = compute('0.0', OUR_T0, 'cond@16')
if b0:
    for h in ['Domino','DominoTree (16)']:
        ov_sp = st.fmean(s for s,_ in b0[h]); ov_ta = st.fmean(t for _,t in b0[h])
        print(f"%   {h}: Overall {ov_sp:.2f}/{ov_ta:.2f}")
