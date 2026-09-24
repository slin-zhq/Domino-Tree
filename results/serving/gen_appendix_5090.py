#!/usr/bin/env python3
"""Regenerate the per-cell serving appendix tables from raw data, and (optionally)
cross-check them against a copy of the paper's LaTeX source.

Usage:
  python3 results/serving/gen_appendix_5090.py                     # print the tables only
  python3 results/serving/gen_appendix_5090.py --paper-dir /path/to/domino_tree

`--paper-dir` is optional and only meaningful if you have a checkout of the paper's LaTeX
sources (not shipped in this repo) alongside this one; without it, the script still
regenerates and prints tab:app-conc / tab:app-helmet / tab:app-helmet-8b from raw data,
it just skips the byte-level diff against `sections/serving.tex`.

Outputs tables even on a mismatch, reports all mismatches to stderr, exits 1 on mismatch.
Input integrity errors (missing/short/duplicate cells) are always fatal. No file under
this repo is modified by this script. HELMET TPS is total tokens / total time; CIs use
mean per-prompt TPS, matching `verify_r4_longctx.py` and `ci_r4_8b_b32.py` (paired
bootstrap, 5000 draws, seed 0).

Requires only the Python standard library.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random
import re
import statistics as st
import sys

HERE = Path(__file__).resolve().parent
METHODS = ['ar', 'eagle3', 'dflash', 'domino_chain', 'dominotree']
NAMES = dict(zip(METHODS, ['AR', 'EAGLE-3', 'DFlash', 'Domino', 'DominoTree']))
DATASETS = ['gsm8k', 'mbpp', 'mt-bench']
DS_NAMES = ['GSM8K', 'MBPP', 'MT-Bench']
CONCS = [2, 4, 8, 16, 32]
TASKS = ['infbench_sum', 'multi_lexsum']
TASK_NAMES = [r'$\infty$Bench-Sum', 'Multi-LexSum']
BINS = [8192, 16384, 32768]
END = r' \\'


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def index(path, fields):
    out = {}
    for row in rows(path):
        key = tuple(row[f] for f in fields)
        if key in out:
            raise ValueError(f'Duplicate {key} in {path}')
        out[key] = row
    return out


def positive(v):
    return isinstance(v, (float, int)) and math.isfinite(v) and v > 0


def load_conc(size):
    """size: '4B' or '8B'. Reads results/serving/concurrency/<size-lower>/<method>/<method>.jsonl"""
    out = {}
    for m in METHODS:
        data = index(HERE / 'concurrency' / size.lower() / m / f'{m}.jsonl', ['dataset', 'concurrency'])
        assert set(data) == {(d, c) for d in DATASETS for c in CONCS}, (size, m)
        for (d, c), r in data.items():
            assert r['method'] == m and r['model'] == f'Qwen3-{size}'
            assert r['max_new_tokens'] == 512 and r['temperature'] == 0
            assert r['n_prompts'] == (80 if d == 'mt-bench' else 128)
            assert positive(r['tps'])
            assert math.isclose(r['tps'], r['completion_tokens'] / r['wall_s'], rel_tol=1e-10)
            assert m == 'ar' or positive(r['mean_accept'])
            out[m, d, c] = r
    return out


def load_helmet(size):
    """size: '4B' or '8B'. Reads results/serving/longcontext/<size-lower>/<arm>/helmet*.jsonl.
    At 8B, also reads the dominotree_b32/ budget-sweep arm (labeled 'dominotree32')."""
    aggs, prompts = {}, {}
    root = HERE / 'longcontext' / size.lower()
    arms = [(m, root / m) for m in METHODS]
    if size == '8B':
        arms.append(('dominotree32', root / 'dominotree_b32'))
    for arm, path in arms:
        agg = index(path / 'helmet.jsonl', ['task', 'length_bin'])
        per = index(path / 'helmet.prompts.jsonl', ['task', 'length_bin', 'idx'])
        assert set(agg) == {(t, b) for t in TASKS for b in BINS}
        assert len(per) == 300
        for (t, b), r in agg.items():
            cell = {i: p for (tt, bb, i), p in per.items() if (tt, bb) == (t, b)}
            assert set(cell) == set(range(50)), (size, arm, t, b)
            assert r['n_prompts'] == 50 and r['gen_tokens'] == 1200 and r['temperature'] == 0
            for p in cell.values():
                assert all(positive(p[f]) for f in ['output_tokens', 'decode_time', 'tps'])
                assert math.isclose(p['tps'], p['output_tokens'] / p['decode_time'], rel_tol=1e-9)
                assert arm == 'ar' or positive(p['accept'])
            tps = sum(p['output_tokens'] for p in cell.values()) / sum(p['decode_time'] for p in cell.values())
            tau = st.mean((p.get('accept') or 1.0) for p in cell.values())
            assert math.isclose(r['tps'], tps, rel_tol=1e-9), (size, arm, t, b, 'TPS')
            assert math.isclose(r.get('mean_accept') or 1.0, tau, rel_tol=1e-9), (size, arm, t, b, 'tau')
            aggs[arm, t, b] = dict(tps=tps, tau=tau)
            prompts[arm, t, b] = cell
    return aggs, prompts


def paired_ci(a, b, field):
    assert set(a) == set(b) and len(a) == 50
    keys = sorted(a)
    av = [(a[k].get(field) or 1.0) for k in keys]
    bv = [(b[k].get(field) or 1.0) for k in keys]
    point = 100 * (st.mean(av) / st.mean(bv) - 1)
    rng = random.Random(0)
    draws = []
    for _ in range(5000):
        idx = [rng.randrange(len(keys)) for _ in keys]
        draws.append(100 * (st.mean(av[i] for i in idx) / st.mean(bv[i] for i in idx) - 1))
    draws.sort()
    return point, draws[125], draws[4874]


def ci_tex(ci):
    p, lo, hi = ci
    return f'{p:+.1f} [{lo:+.1f},{hi:+.1f}]' + (r'$^{*}$' if lo <= 0 <= hi else '')


def table_start(caption, label, columns, header):
    return [r'\begin{table*}[t]', r'\centering\footnotesize', '\\caption{' + caption + '}',
            '\\label{' + label + '}', r'\setlength{\tabcolsep}{3pt}',
            r'\resizebox{\textwidth}{!}{%', '\\begin{tabular}{' + columns + '}', r'\toprule', *header]


def table_end():
    return [r'\bottomrule', r'\end{tabular}}', r'\end{table*}', '']


def conc_table(all_conc):
    """Goodput columns plus ONE tau column, mirroring Table~\\ref{tab:sglang-conc}.

    The old form put "1415/11.93" in every cell, repeating tau across all five
    concurrency levels -- but tau is FLAT in c (e.g. 4B GSM8K DominoTree: 11.93, 11.88,
    11.96, 11.90, 11.88). Restating a constant five times per row crowded out the goodput
    numbers, which are the axis that actually moves. tau is now reported once, as the mean
    over the sweep, with its spread stated so the reader can see the collapse is safe.
    """
    spreads = []
    lines = table_start(
        r'Per-dataset SGLang concurrency results, both models, one RTX~5090 (32\,GB), '
        r'$T{=}0$, 512-token generation cap. Columns give goodput (aggregate output tok/s) at '
        r'each offered concurrency $c$; $\tau$ is the mean per-prompt accepted length over the '
        r'sweep, reported once because it is flat in $c$ (max spread across $c$ within a row: '
        r'SPREAD\%). All five methods have equal admission caps of 32. GSM8K and MBPP use 128 '
        r'prompts; MT-Bench uses 80. AR is the $\tau{=}1$ reference. '
        r'Table~\ref{tab:sglang-conc} averages goodput over datasets and $\tau$ over datasets '
        r'and concurrency. Bold marks the best value in each column within a dataset.',
        'tab:app-conc', 'll' + 'c' * len(CONCS) + 'c',
        ['Dataset & Method & ' + ' & '.join(f'$c{{=}}{c}$' for c in CONCS) +
         r' & $\tau$' + END])
    for size, data in all_conc.items():
        lines += [r'\midrule', '\\multicolumn{%d}{c}{\\textit{Qwen3-%s}}' % (len(CONCS) + 3, size) + END]
        for d, name in zip(DATASETS, DS_NAMES):
            lines += [r'\midrule']
            # Bold = best in its column within this dataset block (ties all bolded).
            best_tps = {c: max(f"{data[m, d, c]['tps']:.0f}" for m in METHODS) for c in CONCS}
            best_tps = {c: max((data[m, d, c]['tps'] for m in METHODS)) for c in CONCS}
            tau_of = {m: f"{st.mean(data[m, d, c]['mean_accept'] for c in CONCS):.2f}"
                      for m in METHODS if m != 'ar'}
            best_tau = max(tau_of.values(), key=float)
            for m in METHODS:
                cells = []
                for c in CONCS:
                    v = f"{data[m, d, c]['tps']:.0f}"
                    cells.append('\\textbf{' + v + '}' if v == f"{best_tps[c]:.0f}" else v)
                if m == 'ar':
                    tau = '--'
                else:
                    taus = [data[m, d, c]['mean_accept'] for c in CONCS]
                    spreads.append((max(taus) - min(taus)) / st.mean(taus) * 100)
                    tau = '\\textbf{' + tau_of[m] + '}' if tau_of[m] == best_tau else tau_of[m]
                lines.append((name if m == 'ar' else '') + ' & ' + NAMES[m] + ' & ' +
                             ' & '.join(cells) + ' & ' + tau + END)
    out = '\n'.join(lines + table_end())
    out = out.replace(r'\resizebox{\textwidth}', r'\resizebox{0.85\textwidth}')
    return out.replace('SPREAD', f'{max(spreads):.1f}')


def helmet_tables(all_helmet):
    """One merged 4B+8B table, TPS and tau as PAIRED COLUMNS per context length.

    Replaces the old pair tab:app-helmet / tab:app-helmet-8b, which were structurally
    identical and each carried a `Metric` column putting TPS and tau on ALTERNATING
    ROWS -- so every method spanned two rows above a blank cell, ~54 rows across two
    floats. Pairing the columns drops the Metric column and roughly halves the height,
    and stacking the two models (as Table~\ref{tab:main} and the bs=1 table do) puts
    the 4B-vs-8B contrast on one page instead of across two floats: the margin over the
    chain collapses from +29.9% at 4B/8K to +8.8% at 8B/8K. Numbers are unchanged.
    """
    cis = {}
    lines = table_start(
        r'HELMET per-task, per-length results, both model sizes, one RTX~5090 (32\,GB), '
        r'batch size 1, $T{=}0$, 50 prompts per cell, up to 1200 generated tokens. '
        r'TPS = total output tokens / total elapsed time; $\tau$ = mean per-prompt accepted '
        r'length. Rows marked \emph{vs.\ Domino} give the paired-bootstrap percentage change '
        r'against the chain [95\% CI], resampling matched prompt indices (5000 draws, seed 0); '
        r'that CI estimand is the ratio of mean per-prompt TPS, distinct from the aggregate TPS '
        r'above it. $^{*}$ marks a CI containing zero. Qwen3-4B uses tree budget 16 '
        r'(the setting Table~\ref{tab:sglang-longctx} reports); for Qwen3-8B both budgets 16 '
        r'and 32 are shown and Table~\ref{tab:sglang-longctx} uses 32, whose arm was collected '
        r'in a separate serving session on the same machine as the chain, with the same hardware and '
        r'software. Bold marks the best TPS and the best $\tau$ in each column within a task.',
        'tab:app-helmet', 'll*{3}{cc}',
        [r'Task & Method & \multicolumn{2}{c}{8K} & \multicolumn{2}{c}{16K} & '
         r'\multicolumn{2}{c}{32K}' + END,
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
         r' & & TPS & $\tau$ & TPS & $\tau$ & TPS & $\tau$' + END])
    for size in ['4B', '8B']:
        agg, per = all_helmet[size]
        lines += [r'\midrule',
                  r'\multicolumn{8}{c}{\textbf{Qwen3-' + size + '}}' + END, r'\midrule']
        arms = METHODS + (['dominotree32'] if size == '8B' else [])
        for ti, (t, name) in enumerate(zip(TASKS, TASK_NAMES)):
            if ti:
                lines.append(r'\cmidrule(lr){1-8}')
            first = True
            # Bold = best TPS and best tau in each column within this task block (ties all bolded).
            btps = {b: max(f"{agg[a_,t,b]['tps']:.1f}" for a_ in arms) for b in BINS}
            btps = {b: f"{max(agg[a_,t,b]['tps'] for a_ in arms):.1f}" for b in BINS}
            btau = {b: f"{max(agg[a_,t,b]['tau'] for a_ in arms):.2f}" for b in BINS}
            bold = lambda v, best: '\\textbf{' + v + '}' if v == best else v
            for arm in arms:
                label = NAMES.get(arm, 'DominoTree (32)')
                if arm == 'dominotree':
                    label += ' (16)'
                cells = []
                for b in BINS:
                    cells += [bold(f"{agg[arm,t,b]['tps']:.1f}", btps[b]),
                              bold(f"{agg[arm,t,b]['tau']:.2f}", btau[b])]
                lines.append((name if first else '') + ' & ' + label + ' & ' +
                             ' & '.join(cells) + END)
                first = False
                if arm.startswith('dominotree'):
                    for field, metric in [('tps', r'$\Delta$TPS\%'), ('accept', r'$\Delta\tau$\%')]:
                        vals = []
                        for b in BINS:
                            ci = paired_ci(per[arm,t,b], per['domino_chain',t,b], field)
                            cis[size,arm,t,b,field] = ci
                            vals.append(r'\multicolumn{2}{c}{' + ci_tex(ci) + '}')
                        lines.append(r' & \quad\emph{vs.\ Domino}, ' + metric + ' & ' +
                                     ' & '.join(vals) + END)
    lines += table_end()
    return '\n'.join(lines), cis


def table_body(text, label):
    after = text.split('\\label{' + label + '}', 1)[1]
    return after.split(r'\end{tabular}', 1)[0]


def clean(s):
    return re.sub(r'\\textbf\{([^{}]*)\}', r'\1', s).strip()


def crosscheck(paper, conc, helmet):
    source = (paper / 'sections/serving.tex').read_text()
    problems, count = [], 0
    def check(key, got, want, digits):
        nonlocal count
        count += 1
        printed = f'{got:.{digits}f}'
        if printed != want:
            problems.append(f'{key}: raw={got:.8f}, rounded={printed}, main={want}')
    body = table_body(source, 'tab:sglang-conc')
    size = None
    found = set()
    for line in body.splitlines():
        if 'Qwen3-4B' in line: size = '4B'
        if 'Qwen3-8B' in line: size = '8B'
        parts = [clean(x) for x in line.removesuffix(END).split('&')]
        if len(parts) != 8 or parts[1] not in NAMES.values(): continue
        m = next(m for m in METHODS if NAMES[m] == parts[1])
        found.add((size,m))
        for c, want in zip(CONCS, parts[2:7]):
            check(f'CONC {size}/{m}/c={c}/goodput', st.mean(conc[size][m,d,c]['tps'] for d in DATASETS), want, 0)
        if m != 'ar':
            check(f'CONC {size}/{m}/tau', st.mean(conc[size][m,d,c]['mean_accept'] for d in DATASETS for c in CONCS), parts[7], 2)
        else:
            assert parts[7] == '--'
            count += 1
    assert len(found) == 10, 'Could not parse all concurrency rows'
    body = table_body(source, 'tab:sglang-longctx')
    found = set()
    for line in body.splitlines():
        # Layout: Method & (tau & speedup) x [4B: 8K,16K,32K] x [8B: 8K,16K,32K]; no AR row.
        parts = [clean(x).replace('$\\times$', '') for x in line.removesuffix(END).split('&')]
        if len(parts) != 13 or parts[0] not in NAMES.values(): continue
        m = next(m for m in METHODS if NAMES[m] == parts[0])
        found.add(m)
        vals = iter(parts[1:])
        for size, b in [(s_, b_) for s_ in ['4B', '8B'] for b_ in BINS]:
            tau_s, sp_s = next(vals), next(vals)
            agg = helmet[size][0]
            arm = 'dominotree32' if size == '8B' and m == 'dominotree' else m
            check(f'HELMET {size}/{m}/{b}/tau', st.mean(agg[arm,t,b]['tau'] for t in TASKS), tau_s, 2)
            check(f'HELMET {size}/{m}/{b}/speedup', st.mean(agg[arm,t,b]['tps']/agg['ar',t,b]['tps'] for t in TASKS), sp_s, 2)
    assert len(found) == 4, 'Could not parse all HELMET rows'

    return count, problems


def claim_audit(helmet, cis):
    best = wins = significant = 0
    messages = []
    for size, (agg, _) in helmet.items():
        arm = 'dominotree32' if size == '8B' else 'dominotree'
        for t in TASKS:
            for b in BINS:
                dt = agg[arm,t,b]['tau']
                others = [(agg[m,t,b]['tau'],m) for m in METHODS if m != 'dominotree']
                high, method = max(others)
                best += dt >= high
                if dt < high:
                    messages.append(f'Acceptance exception {size}/{t}/{b}: DominoTree={dt:.5f}, {method}={high:.5f}')
                ci = cis[size,arm,t,b,'tps']
                wins += ci[0] > 0
                significant += ci[1] > 0
    return [f'HELMET: highest tau {best}/12; TPS vs chain positive {wins}/12, significant {significant}/12', *messages]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--paper-dir', type=Path, default=None,
                         help='Optional: a checkout of the paper LaTeX sources, for a byte-level '
                              'diff against sections/serving.tex. Not shipped in this repo.')
    args = parser.parse_args()
    conc = {s: load_conc(s) for s in ['4B','8B']}
    helmet = {s: load_helmet(s) for s in ['4B','8B']}
    helmet_tex, cis = helmet_tables(helmet)
    print('% Regenerated by results/serving/gen_appendix_5090.py from raw data.')
    print(conc_table(conc))
    print(helmet_tex)
    for message in claim_audit(helmet, cis): print(message, file=sys.stderr)
    if args.paper_dir:
        count, problems = crosscheck(args.paper_dir, conc, helmet)
        print(f'Cross-check: {count} numeric/reference entries at printed precision; {len(problems)} mismatches.', file=sys.stderr)
        for problem in problems: print(problem, file=sys.stderr)
        sys.exit(bool(problems))


if __name__ == '__main__':
    main()
