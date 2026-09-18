#!/usr/bin/env python3
"""Rebuild Table 1, paired CIs, and the Domino-variant appendix table from per-prompt raw data.

Requires numpy only. Torch and the DDTree harness are required only when an explicit
private-pickle fallback is requested. No summaries/CSV/table literals are inputs:
every number is recomputed from per-prompt/per-turn raw records. Missing, duplicate,
nonfinite or short cells are fatal (the script raises rather than silently skipping a
cell). All source files are SHA256-recorded in provenance.json in the output directory.

Usage:
    python gen_table1.py [--out-dir results/table1_audit] [--bootstrap-iters 10000]

Means are over prompt/turn rows; Overall is the unweighted dataset mean. DominoTree uses
same-session T=0 AR, reused at T>0. Reference baselines (DFlash/CaDDTree) use their own
harness's AR; DDTree uses its own same-cell AR. Official Domino uses the common lean AR
from the DominoTree collection, best of eager/graph per cell. Its legacy 8B run excludes
prompt 0 for both metrics. Paired CIs: ratio of pooled mean per-row TPS (vs. Domino) or
per-row own-AR speedups (vs. reference baselines); dataset-stratified, conversation-
clustered bootstrap, 10,000 draws, seed 12345. No implicit intersection/drop of unpaired
records -- an unpaired cell is a fatal error, not a silent skip.

Headline node budget in the current (IEEE Access) revision of Table 1 is 32 at Qwen3-4B
and 128 at Qwen3-8B (both DDTree and DominoTree; see the paper's tab:main). Raw sources:

    results/raw/tab1f_4b/                  DominoTree (+ AR), Qwen3-4B, fused GPU-native
                                            builder, budgets 16/32/64 (headline: 32)
    results/raw/tab1_8b/                   DominoTree (+ AR), Qwen3-8B, budget 128
    results/raw/baseline_ddtree_caddtree/  AR/DFlash/CaDDTree, Qwen3-4B (own harness)
    results/raw/8b/ref8b_perprompt_jsonl/  AR/DFlash/CaDDTree, Qwen3-8B (own harness)
    results/raw/domino_official/qwen3-4b/  official Domino decoder, graph+eager, Qwen3-4B
    results/raw/8b/domino_official/qwen3-8b/  official Domino decoder, graph+eager, Qwen3-8B
    results/raw/conditioning_ladder/matched/  marg@16 / condstatic@16 / dominotree@16
                                               (the conditioning-decomposition appendix)

DDTree's budget-32 (4B) and budget-128 (8B) arms are exported as dependency-free
per-prompt JSONL in `results/raw/ddtree_b32_4b/` and
`results/raw/ddtree_b128_8b/`.  The JSONL preserves acceptance lengths, timing, and
the protocol fields checked below; it intentionally omits the original tensors and
harness-local paths. A private torch-pickle fallback exists only through the explicit
`--ddtree-pickle-root4b`/`--ddtree-pickle-root8b` options.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, re
from pathlib import Path
from statistics import fmean
import numpy as np

ROOT = Path(__file__).resolve().parent
DS = ['gsm8k','math500','aime25','humaneval','mbpp','livecodebench','mt-bench','alpaca']
TEMPS = ['0.0','0.5','1.0']
METHODS = ['DFlash','DDTree','CaDDTree','Domino','DominoTree']
GROUPS = {'Math':DS[:3], 'Code':DS[3:6], 'Chat':DS[6:], 'Overall':DS}
SOURCES = {}
END = r' \\'

def require(ok, message):
    if not ok: raise ValueError(message)

def source(p):
    require(p.is_file(), f'MISSING RAW FILE: {p}')
    SOURCES[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def jsonl(p):
    return [json.loads(l) for l in source(p).read_text().splitlines() if l.strip()]

def keys(ds):
    return {(i,t) for i in range(30 if ds=='aime25' else 50) for t in range(2 if ds=='mt-bench' else 1)}

def check(rows, ds, label, expected=None):
    expected = keys(ds) if expected is None else expected
    require(len(rows)==len(expected), f'SHORT/EXTRA CELL {label}: {len(rows)} != {len(expected)}')
    out={}
    for r in rows:
        k=(r['sample_idx'],r['turn_index'])
        require(k not in out, f'DUPLICATE {label}: {k}')
        require(all(math.isfinite(r[v]) and r[v]>0 for v in ['tps','mean_accept']), f'INVALID METRIC {label}: {r}')
        out[k]=r
    require(set(out)==expected, f'IDENTITY MISMATCH {label}: missing={expected-set(out)}, extra={set(out)-expected}')
    return out

def arm(p, ds, method):
    rows=[]
    for r in jsonl(p):
        if r['method'] != method: continue
        r=dict(r);r['source']=str(p)
        # Legacy reference8b sample_idx enumerates execution rows, not conversations.
        if 'turn_index' not in r:
            i=int(r['sample_idx']);r['sample_idx']=i//2 if ds=='mt-bench' else i;r['turn_index']=i%2 if ds=='mt-bench' else 0
        if 'num_output' in r: require(0<r['num_output']<=2048, f'INVALID LENGTH {p}: {r["num_output"]}')
        rows.append(r)
    return check(rows, ds, f'{p}:{method}')

def ddtree_jsonl_arms(p, ds, temp, size, budget):
    method=f'ddtree_tb{budget}'
    common=dict(dataset=ds,temperature=float(temp),budget=budget,max_new_tokens=2048,max_samples=50,
                flash_attn=False,skip_baseline=False)
    out={}
    for m in ['baseline',method]:
        rr=[]
        for row in jsonl(p):
            if row.get('method') != m: continue
            require(all(row.get(k)==v for k,v in common.items()), f'WRONG PROTOCOL {p}: {row}')
            require(f'Qwen3-{size}' in row.get('model','') and 'DFlash-b16' in row.get('draft',''), f'WRONG MODEL {p}: {row}')
            require(str(budget) in str(row.get('tree_budget','')).split(','), f'WRONG BUDGET {p}: {row}')
            acc=row.get('acceptance_lengths')
            require(isinstance(acc,list) and len(acc)>0 and len(acc)==row.get('decode_rounds'), f'INVALID ACCEPTANCES {p}: {row}')
            require(all(isinstance(x,int) and x>0 for x in acc), f'INVALID ACCEPTANCES {p}: {row}')
            require(0<row.get('num_output',0)<=2048 and row.get('time_per_output_token',0)>0, f'INVALID LENGTH/TIME {p}: {row}')
            r=dict(row);r['tps']=1/float(r['time_per_output_token']);r['mean_accept']=fmean(acc);r['source']=str(p)
            rr.append(r)
        out[m]=check(rr, ds, f'{p}:{m}')
    return out[method],out['baseline']

def ddtree_pickle_arms(p, ds, temp, size, budget):
    import torch
    x=torch.load(source(p), map_location='cpu', weights_only=False)
    a=x['args'];method=f'ddtree_tb{budget}'
    require(a['dataset']==ds and float(a['temperature'])==float(temp), f'WRONG CELL {p}')
    require(a['max_new_tokens']==2048 and a['max_samples']==50, f'WRONG PROTOCOL {p}')
    require(f'Qwen3-{size}' in a['model_name_or_path'] and 'DFlash-b16' in a['draft_name_or_path'], f'WRONG MODEL {p}')
    require(str(budget) in a['tree_budget'].split(',') and method in x['methods'], f'WRONG BUDGET {p}')
    require(not a['flash_attn'] and not a['skip_baseline'], f'WRONG HARNESS {p}')
    out={}
    for m in ['baseline',method]:
        rr=[]
        for i,r in enumerate(x['responses']):
            require(m in r, f'MISSING ARM {p}, response {i}: {m}')
            v=r[m];acc=v.acceptance_lengths
            require(len(acc)>0 and len(acc)==v.decode_rounds, f'INVALID ACCEPTANCES {p}:{i}:{m}')
            rr.append(dict(sample_idx=i//2 if ds=='mt-bench' else i,turn_index=i%2 if ds=='mt-bench' else 0,
                           tps=1/float(v.time_per_output_token),mean_accept=fmean(acc),source=str(p)))
        out[m]=check(rr, ds, f'{p}:{m}')
    return out[method],out['baseline']

def ddarms(p, ds, temp, size, budget, pickle_path=None):
    if p.is_file(): return ddtree_jsonl_arms(p, ds, temp, size, budget)
    require(pickle_path is not None, f'MISSING RAW FILE: {p} (pickle fallback requires an explicit --ddtree-pickle-root{size.lower()})')
    return ddtree_pickle_arms(pickle_path, ds, temp, size, budget)

def official(root, ds, temp, size, mode):
    p=root/f'qwen3-{size.lower()}'/f'T{temp}'/f'{mode}_{ds}.jsonl'
    rr=[]
    records=jsonl(p)
    require(len(records)==(30 if ds=='aime25' else 50),f'SHORT OFFICIAL CELL {p}: {len(records)}')
    for row in records:
        c=row['choices'][1]
        require(len(c['new_tokens'])==len(c['decode_times'])==len(c['acceptance_lengths'])==(2 if ds=='mt-bench' else 1),f'MISSING TURN {p}')
        for t,(n,d,a) in enumerate(zip(c['new_tokens'],c['decode_times'],c['acceptance_lengths'])):
            require(0<n<=2048 and d>0 and len(a)>0,f'INVALID OFFICIAL ROW {p}')
            rr.append(dict(sample_idx=int(row['question_id']),turn_index=t,tps=n/d,mean_accept=fmean(a),source=str(p)))
    allrows=check(rr,ds,str(p))
    if size=='8B': return {k:v for k,v in allrows.items() if k[0]!=0}
    return allrows

def mean(rows, field): return fmean(r[field] for r in rows.values())

def cell(rows, ar):
    return dict(tps=mean(rows,'tps'),tau=mean(rows,'mean_accept'),speedup=mean(rows,'tps')/mean(ar,'tps'),n=len(rows),ar_tps=mean(ar,'tps'))

def boot(data, size, temp, comp, datasets, iters):
    rng=np.random.default_rng(12345)
    sums=np.zeros((iters,2));observed=np.zeros(2);count=0
    for ds in datasets:
        own,oa=data[size,temp,ds,'DominoTree'];other,ba=data[size,temp,ds,comp]
        expect=keys(ds)
        if comp=='Domino' and size=='8B': expect={k for k in expect if k[0]!=0}
        require(set(other)==expect and expect<=set(own),f'UNPAIRED {size}/{temp}/{ds}/{comp}')
        clusters=[]
        for i in sorted({k[0] for k in expect}):
            pairs=[]
            for k in sorted(k for k in expect if k[0]==i):
                if comp=='Domino': pairs.append([own[k]['tps'],other[k]['tps']])
                else:
                    require(k in oa and k in ba,f'MISSING PAIRED AR {size}/{temp}/{ds}/{k}')
                    pairs.append([own[k]['tps']/oa[k]['tps'],other[k]['tps']/ba[k]['tps']])
            clusters.append(np.sum(pairs,axis=0));count+=len(pairs)
        a=np.asarray(clusters);observed+=a.sum(axis=0)
        sums+=a[rng.integers(0,len(a),size=(iters,len(a)))].sum(axis=1)
    reps=100*(sums[:,0]/sums[:,1]-1)
    lo,hi=np.quantile(reps,[.025,.975])
    return dict(delta=100*(observed[0]/observed[1]-1),lo=float(lo),hi=float(hi),n=count)

def label(size,m): return f'{m} ({32 if size=="4B" else 128})' if m in ['DDTree','DominoTree'] else m

def bold(v,mx):
    s=f'{v:.2f}';return r'\textbf{'+s+'}' if s==f'{mx:.2f}' else s

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir',type=Path,default=ROOT/'results/table1_audit')
    ap.add_argument('--bootstrap-iters',type=int,default=10000)
    ap.add_argument('--ddtree-pickle-root4b',type=Path,help='trusted private pickle directory; opt-in fallback only')
    ap.add_argument('--ddtree-pickle-root8b',type=Path,help='trusted private pickle directory; opt-in fallback only')
    args=ap.parse_args();out=args.out_dir;out.mkdir(parents=True,exist_ok=True)
    data={};cells={};variants={};budget_check={}
    for size in ['4B','8B']:
        budget=32 if size=='4B' else 128
        dtroot=ROOT/'results/raw'/('tab1f_4b' if size=='4B' else 'tab1_8b')
        refroot=ROOT/'results/raw/baseline_ddtree_caddtree' if size=='4B' else ROOT/'results/raw/8b/ref8b_perprompt_jsonl'
        ddroot=ROOT/'results/raw'/('ddtree_b32_4b' if size=='4B' else 'ddtree_b128_8b')
        pickleroot=args.ddtree_pickle_root4b if size=='4B' else args.ddtree_pickle_root8b
        officialroot=ROOT/'results/raw/domino_official' if size=='4B' else ROOT/'results/raw/8b/domino_official'
        for temp in TEMPS:
            for ds in DS:
                oa=arm(dtroot/f'{ds}_T0.0.jsonl',ds,'ar')
                dr=arm(dtroot/f'{ds}_T{temp}.jsonl',ds,f'dominotree@{budget}')
                data[size,temp,ds,'DominoTree']=(dr,oa)
                ba=arm(refroot/f'{ds}_T{temp if size=="4B" else "0.0"}.jsonl',ds,'baseline')
                for m,raw in [('DFlash','dflash'),('CaDDTree','caddtree')]: data[size,temp,ds,m]=(arm(refroot/f'{ds}_T{temp}.jsonl',ds,raw),ba)
                data[size,temp,ds,'DDTree']=ddarms(ddroot/f'{ds}_T{temp}.jsonl',ds,temp,size,budget,
                                                    pickleroot/f'{ds}_T{temp}.json' if pickleroot else None)
                modes={mode:official(officialroot,ds,temp,size,mode) for mode in ['graph','eager']}
                winner=max(modes,key=lambda m:mean(modes[m],'tps'))
                data[size,temp,ds,'Domino']=(modes[winner],oa)
                for mode,rr in modes.items(): variants[size,temp,ds,mode]=cell(rr,oa)
                for m in METHODS:
                    cells[size,temp,ds,m]=cell(*data[size,temp,ds,m])
                    if m=='Domino': cells[size,temp,ds,m]['mode']=winner
                if size=='4B' and temp=='0.0':
                    for b in [16,32,64]: budget_check[temp,ds,b]=cell(arm(dtroot/f'{ds}_T{temp}.jsonl',ds,f'dominotree@{b}'),oa)
            for m in METHODS:
                cells[size,temp,'Overall',m]={k:fmean(cells[size,temp,d,m][k] for d in DS) for k in ['tps','tau','speedup','ar_tps']}
                cells[size,temp,'Overall',m]['n']=sum(cells[size,temp,d,m]['n'] for d in DS)
    with (out/'rows.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['model','temperature','dataset','method','sample_idx','turn_index','tps','tau','source','ar_source','ar_mean_tps'])
        for (size,temp,ds,m),(rr,ar) in data.items():
            for k,r in sorted(rr.items()): w.writerow([size,temp,ds,m,*k,r['tps'],r['mean_accept'],r['source'],next(iter(ar.values()))['source'],mean(ar,'tps')])
    serial=[dict(model=s,temp=t,dataset=d,method=m,**v) for (s,t,d,m),v in cells.items()]
    (out/'cells.json').write_text(json.dumps(serial,indent=2)+'\n')
    with (out/'cells.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['model','temp','dataset','method','tps','tau','speedup','n','ar_tps','mode']);w.writeheader();w.writerows(serial)
    tex=[r'\begin{tabular}{ll*{9}{cc}}',r'\toprule',r'Model & Method & \multicolumn{6}{c}{Math} & \multicolumn{6}{c}{Code} & \multicolumn{4}{c}{Chat} & \multicolumn{2}{c}{Overall}'+END,r'\cmidrule(lr){3-8}\cmidrule(lr){9-14}\cmidrule(lr){15-18}\cmidrule(lr){19-20}',r' & & '+' & '.join(r'\multicolumn{2}{c}{'+d+'}' for d in ['GSM8K','MATH-500','AIME25','HumanEval','MBPP','LCB','MT-Bench','Alpaca','Avg.'])+END]
    for temp in TEMPS:
        tex += [r'\midrule',r'\multicolumn{2}{c}{Temperature = '+f'{float(temp):g}'+'} & '+' & '.join([r'Speedup & $\tau$']*9)+END]
        for size in ['4B','8B']:
            tex += [r'\midrule',r'\multirow{5}{*}{Qwen3-'+size+'}']
            for m in METHODS:
                vs=[]
                for ds in DS+['Overall']:
                    for k in ['speedup','tau']: vs.append(bold(cells[size,temp,ds,m][k],max(cells[size,temp,ds,x][k] for x in METHODS)))
                name=label(size,m);name=r'\textbf{'+name+'}' if m=='DominoTree' else name
                tex.append('& '+name+' & '+' & '.join(vs)+END)
    tex += [r'\bottomrule',r'\end{tabular}'];table='\n'.join(tex)
    (out/'table1.tex').write_text(table+'\n')
    md=[]
    for size in ['4B','8B']:
        md += [f'### Qwen3-{size}', '', 'Each cell is speedup / tau. Overall is dataset-macro.', '', '| T | Method | '+' | '.join(DS+['Overall'])+' |','|---|---|'+'---:|'*9]
        for temp in TEMPS:
            for m in METHODS:
                vals=[f'{cells[size,temp,d,m]["speedup"]:.2f} / {cells[size,temp,d,m]["tau"]:.2f}' for d in DS+['Overall']]
                md.append(f'| {float(temp):g} | {label(size,m)} | '+' | '.join(vals)+' |')
        md.append('')
    (out/'table1.md').write_text('\n'.join(md))
    pairs=[];pt=[r'\begin{tabular}{l*{3}{cc}}',r'\toprule',r'Category & \multicolumn{2}{c}{vs.\ Domino} & \multicolumn{2}{c}{vs.\ DDTree} & \multicolumn{2}{c}{vs.\ CaDDTree}'+END,r' & $\Delta\%$ & 95\% CI & $\Delta\%$ & 95\% CI & $\Delta\%$ & 95\% CI'+END]
    for size in ['4B','8B']:
        pt += [r'\midrule',r'\multicolumn{7}{c}{\textbf{Qwen3-'+size+'}}'+END]
        for temp in TEMPS:
            pt += [r'\midrule',r'\multicolumn{7}{l}{\textit{Temperature = '+f'{float(temp):g}'+'}}'+END]
            for group,datasets in GROUPS.items():
                vs=[]
                for comp in ['Domino','DDTree','CaDDTree']:
                    c=boot(data,size,temp,comp,datasets,args.bootstrap_iters);pairs.append(dict(model=size,temp=temp,group=group,comparison=comp,**c))
                    a=f'{c["delta"]:.2f}';ci=f'[{c["lo"]:.2f},\\ {c["hi"]:.2f}]'
                    vs += [r'\textbf{'+a+'}' if c['lo']>0 else a, r'$\mathbf{'+ci+'}$' if c['lo']>0 else '$'+ci+'$']
                pt.append((r'\textbf{Overall}' if group=='Overall' else group)+' & '+' & '.join(vs)+END)
    pt += [r'\bottomrule',r'\end{tabular}'];pairtable='\n'.join(pt)
    (out/'pairwise.tex').write_text(pairtable+'\n');(out/'pairwise.json').write_text(json.dumps(pairs,indent=2)+'\n')
    vt=[r'\begin{tabular}{llcccc}',r'\toprule',r'Model & $T$ & DominoTree & Domino-graph & Domino-eager & $\Delta\%$ vs.\ graph / eager'+END]
    for size in ['4B','8B']:
        vt += [r'\midrule',r'\multirow{3}{*}{Qwen3-'+size+'}']
        for temp in TEMPS:
            dt=cells[size,temp,'Overall','DominoTree']['speedup'];g,e=[fmean(variants[size,temp,d,mode]['speedup'] for d in DS) for mode in ['graph','eager']]
            vt.append(f' & {temp} & {dt:.2f} & {g:.2f} & {e:.2f} & '+rf'\textbf{{{100*(dt/g-1):+.1f}}} / \textbf{{{100*(dt/e-1):+.1f}}}'+END)
    vt += [r'\bottomrule',r'\end{tabular}'];varianttable='\n'.join(vt)
    (out/'domino_variants.tex').write_text(varianttable+'\n')
    claims={}
    for size in ['4B','8B']:
        loss=[(t,d) for t in TEMPS for d in DS if cells[size,t,d,'DominoTree']['tau']<max(cells[size,t,d,m]['tau'] for m in METHODS if m!='DominoTree')]
        speedloss=[(t,d) for t in TEMPS for d in DS if cells[size,t,d,'DominoTree']['speedup']<max(cells[size,t,d,m]['speedup'] for m in METHODS if m!='DominoTree')]
        gains=[(100*(cells[size,t,d,'DominoTree']['tps']/cells[size,t,d,'Domino']['tps']-1),t,d) for t in TEMPS for d in DS]
        claims[size]=dict(tau_lead_cells=24-len(loss),tau_losses=loss,speedup_losses=speedloss,
                          domino_raw_gain_range=[min(gains),max(gains)],domino_wins=sum(v[0]>0 for v in gains),
                          max_speedup=max((cells[size,t,d,'DominoTree']['speedup'],t,d) for t in TEMPS for d in DS),
                          overall=[dict(temp=t,**cells[size,t,'Overall','DominoTree']) for t in TEMPS])
    ladder={}
    for method in ['marg@16','condstatic@16','dominotree@16']:
        ladder[method]=fmean(mean(arm(ROOT/'results/raw/conditioning_ladder/matched'/f'{d}_T0.0.jsonl',d,method),'mean_accept') for d in DS)
    claims['controlled_decomposition_B16']=dict(tau=ladder,correction_pct=100*(ladder['condstatic@16']/ladder['marg@16']-1),path_pct=100*(ladder['dominotree@16']/ladder['condstatic@16']-1),total_pct=100*(ladder['dominotree@16']/ladder['marg@16']-1))
    claims['4B_budget_t0']={b:{k:fmean(budget_check['0.0',d,b][k] for d in DS) for k in ['tps','tau','speedup']} for b in [16,32,64]}
    (out/'claims.json').write_text(json.dumps(claims,indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(sources=SOURCES,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),bootstrap_iters=args.bootstrap_iters,seed=12345,notes=__doc__),indent=2)+'\n')
    print(json.dumps(claims,indent=2));print(f'PASS: {len(serial)} cells (including macro rollups), {len(pairs)} paired comparisons; {len(SOURCES)} raw sources')

if __name__=='__main__': main()
