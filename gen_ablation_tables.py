#!/usr/bin/env python3
"""Rebuild the paper's builder and ablation tables from per-prompt raw data.

    python3 gen_ablation_tables.py [--out-dir results/ablation_tables]

Requires numpy only. Every number is recomputed from the per-prompt JSONL under
results/raw/; nothing is read from a summary. Missing or unpaired cells are fatal.

  Table 3  (tab:builder-stagetime)  per-round stage cost, Python reference vs frontier builder
  Table 4  (tab:builder-compare)    DominoTree throughput delta vs Marg and vs chain, per builder
  Table 10 (tab:budget-ablation)    node budget 16/32/64 vs chain, frontier builder
  Table 11 (tab:topm-ablation)      candidate width M=32/64/128 at budgets 16 and 32
      -> results/raw/tab_refresh/   Qwen3-4B, one RTX 5080, T in {0,0.5,1}, 8 datasets,
                                    every comparison arm collected in the SAME process
  Table 12 (tab:topm-saturation)    acceptance vs M up to the full vocabulary
      -> results/raw/candidate_width_saturation/   (m0 = full vocabulary)
  Table 13 (tab:draftsample)        sampled vs deterministic draft, chain and DominoTree(16)
      -> results/raw/draft_sampling_ablation/
  Limitations: Qwen3-8B verify cost vs tree budget on the A6000 (83.4 / 76.0 / 84.1 ms)
      -> results/raw/budget8b/
  Section "Builder cost", Qwen3-8B prose: heap-era build saving (Python -> GPU-native heap)
      and the budget-16 Python-builder control (accepted-length lead vs throughput delta)
      -> results/raw/8b/collect_8b_2048_20260704/our/  (Python builder; arm named cond@16)
         results/raw/8b/dominotree/                    (GPU-native heap builder)

Paired deltas (Tables 4, 10, 11) are the ratio of mean per-row TPS with a 95% percentile
bootstrap CLUSTERED by prompt (MT-Bench's two turns are one conversation), 10,000 draws,
seed 12345. Tables 12 and 13 are ratios of means over all rows, as printed.
"""
import argparse, collections, json, pathlib, statistics as st
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
RAW = ROOT / "results" / "raw"
DS = ["gsm8k", "math500", "aime25", "humaneval", "mbpp", "livecodebench", "mt-bench", "alpaca"]
E = r" \\"


def jsonl(p):
    if not p.exists():
        raise SystemExit(f"missing raw file: {p}")
    return [r for r in (json.loads(l) for l in p.open() if l.strip()) if isinstance(r, dict)]


def by_method(p):
    out = collections.defaultdict(dict)
    for r in jsonl(p):
        out[r["method"]][(r["sample_idx"], r["turn_index"])] = r
    return out


def refresh(kind, ds, temp, x):
    """kind='builder' -> x is the builder name; kind='topm' -> x is M."""
    name = f"builder_{x}_{ds}_T{temp}.jsonl" if kind == "builder" else f"topm_M{x}_{ds}_T{temp}.jsonl"
    return by_method(RAW / "tab_refresh" / name)


def boot(pairs, iters=10000, seed=12345):
    clusters = collections.defaultdict(list)
    for (ds, idx, _turn), (a, b) in pairs.items():
        clusters[ds, idx].append((a, b))
    sums = np.array([[sum(a for a, _ in c), sum(b for _, b in c)] for c in clusters.values()])
    obs = (sums[:, 0].sum() / sums[:, 1].sum() - 1) * 100
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(iters, len(sums)))
    s = sums[idx].sum(1)
    lo, hi = np.percentile((s[:, 0] / s[:, 1] - 1) * 100, [2.5, 97.5])
    return obs, lo, hi


def paired(kind, x, temp, arm, comp):
    pairs = {}
    for ds in DS:
        r = refresh(kind, ds, temp, x)
        if set(r[arm]) != set(r[comp]):
            raise SystemExit(f"UNPAIRED {kind}/{x}/{ds}/T{temp}: {arm} vs {comp}")
        for k in r[arm]:
            pairs[(ds, *k)] = (r[arm][k]["tps"], r[comp][k]["tps"])
    return boot(pairs)


def stage(kind, x, temp, arm):
    acc = collections.defaultdict(list)
    for ds in DS:
        for v in refresh(kind, ds, temp, x)[arm].values():
            for k in ("ms_draft", "ms_build", "ms_verify", "ms_commit", "tps", "mean_accept"):
                acc[k].append(v[k])
    m = {k: st.fmean(v) for k, v in acc.items()}
    m["total"] = m["ms_draft"] + m["ms_build"] + m["ms_verify"] + m["ms_commit"]
    return m


def fci(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


def table3():
    """Single-column layout: build cost and throughput under each builder, per budget."""
    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"\multirow{2}{*}{Budget} & \multicolumn{3}{c}{Build (ms per round)} & \multicolumn{2}{c}{Throughput (tok/s)}" + E,
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
         r" & Python & Frontier & Ratio & Python & Frontier" + E, r"\midrule"]
    for b in (16, 32, 64):
        p, f = stage("builder", "python", "0.0", f"dominotree@{b}"), stage("builder", "frontier", "0.0", f"dominotree@{b}")
        L.append(f"{b} & {p['ms_build']:.2f} & {f['ms_build']:.2f} & {p['ms_build']/f['ms_build']:.1f}$\\times$ & "
                 f"{p['tps']:.1f} & {f['tps']:.1f}" + E)
    return L + [r"\bottomrule", r"\end{tabular}"]


def table4():
    """Single-column layout: one row per (temperature, builder)."""
    L = [r"\begin{tabular}{llcc}", r"\toprule",
         r"$T$ & Builder & vs.\ Marg & vs.\ Domino chain" + E, r"\midrule"]
    for ti, temp in enumerate(("0.0", "0.5", "1.0")):
        if ti:
            L.append(r"\midrule")
        for bi, b in enumerate(("python", "frontier")):
            c = [fci(paired("builder", b, temp, "dominotree@16", comp)) for comp in ("marg@16", "chain")]
            if b == "frontier":
                c = [r"\textbf{" + v + "}" for v in c]
            lbl = (r"\multirow{2}{*}{$" + f"{float(temp):g}" + "$}") if bi == 0 else ""
            L.append(f"{lbl} & {'Python' if b == 'python' else 'Frontier'} & " + " & ".join(c) + E)
    return L + [r"\bottomrule", r"\end{tabular}"]


def table10():
    rows = {b: stage("builder", "frontier", "0.0", f"dominotree@{b}") for b in (16, 32, 64)}
    best = max(rows, key=lambda b: rows[b]["tps"])
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Node budget & Build (ms) & $\tau$ & Throughput (tok/s) & $\Delta\%$ vs.\ chain (95\% CI)" + E, r"\midrule"]
    for b, m in rows.items():
        bold = (lambda s: r"\textbf{" + s + "}") if b == best else (lambda s: s)
        v = fci(paired("builder", "frontier", "0.0", f"dominotree@{b}", "chain"))
        L.append(" & ".join([bold(str(b)), f"{m['ms_build']:.2f}", bold(f"{m['mean_accept']:.2f}"),
                             bold(f"{m['tps']:.1f}"), bold(v)]) + E)
    return L + [r"\bottomrule", r"\end{tabular}"]


def table11():
    L = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"Budget & Width $M$ & Build (ms) & $\tau$ & Throughput (tok/s) & $\Delta\%$ vs.\ chain (95\% CI)" + E]
    for b in (16, 32):
        L.append(r"\midrule")
        rows = {M: (stage("topm", M, "0.0", f"dominotree@{b}"),
                    paired("topm", M, "0.0", f"dominotree@{b}", "chain")) for M in (32, 64, 128)}
        # Bold = best tau, throughput and delta within this budget (ties at printed precision all bolded).
        bt = max(f"{r[0]['mean_accept']:.2f}" for r in rows.values())
        bp = max(f"{r[0]['tps']:.1f}" for r in rows.values())
        bd = max(f"{r[1][0]:+.2f}" for r in rows.values())
        bf = lambda v, best: r"\textbf{" + v + "}" if v == best else v
        for i, (M, (m, d)) in enumerate(rows.items()):
            lbl = r"\multirow{3}{*}{" + str(b) + "}" if i == 0 else ""
            delta = fci(d)
            if f"{d[0]:+.2f}" == bd:
                delta = r"\textbf{" + delta + "}"
            tau_s, tps_s = format(m['mean_accept'], '.2f'), format(m['tps'], '.1f')
            L.append(f"{lbl} & {M} & {m['ms_build']:.2f} & {bf(tau_s, bt)} & {bf(tps_s, bp)} & {delta}" + E)
    return L + [r"\bottomrule", r"\end{tabular}"]


def table12():
    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Dataset & $M{=}16$ & $M{=}64$ & $M{=}128$ & $M{=}256$ & full vocab" + E, r"\midrule"]
    for ds, nm in (("gsm8k", "GSM8K"), ("humaneval", "HumanEval"), ("alpaca", "Alpaca")):
        vals = []
        for m in ("16", "64", "128", "256", "0"):
            r = [x for x in jsonl(RAW / "candidate_width_saturation" / f"fullvocab_{ds}_m{m}.jsonl")
                 if x["method"] == "dominotree@16"]
            vals.append(f"{st.fmean(x['mean_accept'] for x in r):.2f}")
        L.append(f"{nm} & " + " & ".join(vals) + E)
    return L + [r"\bottomrule", r"\end{tabular}"]


def table13():
    L = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"& & \multicolumn{2}{c}{Domino-chain} & \multicolumn{2}{c}{DominoTree (16)}" + E,
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Dataset & $T$ & $\Delta\tau\%$ & $\Delta$TPS\% & $\Delta\tau\%$ & $\Delta$TPS\%" + E, r"\midrule"]
    for di, (ds, nm) in enumerate((("gsm8k", "GSM8K"), ("humaneval", "HumanEval"), ("alpaca", "Alpaca"))):
        if di:
            L.append(r"\cmidrule(lr){1-6}")
        for ti, T in enumerate(("0.5", "1.0")):
            on = jsonl(RAW / "draft_sampling_ablation" / f"draftsample_{ds}_T{T}_on.jsonl")
            off = jsonl(RAW / "draft_sampling_ablation" / f"draftsample_{ds}_T{T}_off.jsonl")
            cells = []
            for meth in ("chain", "dominotree@16"):
                a = [x for x in on if x["method"] == meth]
                b = [x for x in off if x["method"] == meth]
                for f in ("mean_accept", "tps"):
                    d = 100 * (st.fmean(x[f] for x in a) / st.fmean(x[f] for x in b) - 1)
                    cells.append(f"${d:+.1f}$")
            lbl = (r"\multirow{2}{*}{" + nm + "}") if ti == 0 else ""
            L.append(f"{lbl} & {T} & " + " & ".join(cells) + E)
    return L + [r"\bottomrule", r"\end{tabular}"]


def budget8b():
    acc = collections.defaultdict(list)
    for p in sorted((RAW / "budget8b").glob("*_T0.0.jsonl")):
        for r in jsonl(p):
            acc[r["method"]].append(r["ms_verify"])
    return [f"{m}: mean verify {st.fmean(v):.1f} ms (n={len(v)})"
            for m, v in sorted(acc.items(), key=lambda kv: (len(kv[0]), kv[0])) if m != "ar"]


def prose_8b():
    def load(pat, strip=""):
        a = collections.defaultdict(dict)
        for p in sorted(RAW.glob(pat)):
            ds = p.name.replace(strip, "").split("_T")[0]
            for r in jsonl(p):
                a[r["method"]][(ds, r["sample_idx"], r["turn_index"])] = r
        return a
    py = load("8b/collect_8b_2048_20260704/our/*_T0.0.jsonl")
    gn = load("8b/dominotree/*_T0.0.jsonl")
    b_py = st.fmean(x["ms_build"] for x in py["cond@16"].values())
    b_gn = st.fmean(x["ms_build"] for x in gn["dominotree@16"].values())
    # Qwen3-4B, same builder pair (Python -> GPU-native heap), budget 16, T=0
    p4 = st.fmean(r["ms_build"] for q in sorted(RAW.glob("dominotree_python_builder/*_T0.0.jsonl"))
                  for r in jsonl(q) if r["method"] == "dominotree@16")
    g4 = st.fmean(r["ms_build"] for q in sorted(RAW.glob("dominotree/*_T0.0.jsonl"))
                  for r in jsonl(q) if r["method"] == "dominotree@16")
    t, c = py["cond@16"], py["chain"]
    if set(t) != set(c):
        raise SystemExit("UNPAIRED 8B budget-16 control")
    lead = 100 * (st.fmean(t[k]["mean_accept"] for k in t) / st.fmean(c[k]["mean_accept"] for k in t) - 1)
    o, lo, hi = boot({k: (t[k]["tps"], c[k]["tps"]) for k in t})
    return [f"4B build: Python {p4:.2f} ms -> GPU-native heap {g4:.2f} ms, saving {p4-g4:.2f} ms",
            f"8B build: Python {b_py:.2f} ms -> GPU-native heap {b_gn:.2f} ms, saving {b_py-b_gn:.2f} ms"
            f" ({(b_py-b_gn)/(p4-g4):.1f}x the 4B saving)",
            f"8B budget-16 control (Python builder): accepted-length lead over chain {lead:+.2f}%",
            f"   throughput delta vs chain {o:+.2f}% [95% CI {lo:+.2f}, {hi:+.2f}]"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=pathlib.Path, default=ROOT / "results" / "ablation_tables")
    out = ap.parse_args().out_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("table3_builder_stagetime", table3), ("table4_builder_compare", table4),
                     ("table10_budget_ablation", table10), ("table11_topm_ablation", table11),
                     ("table12_topm_saturation", table12), ("table13_draftsample", table13)):
        text = "\n".join(fn()) + "\n"
        (out / f"{name}.tex").write_text(text)
        print(f"%% ---- {name} ----\n{text}")
    lines = budget8b()
    (out / "limitations_budget8b_verify.txt").write_text("\n".join(lines) + "\n")
    print("%% ---- Limitations: Qwen3-8B verify cost vs budget (A6000) ----")
    print("\n".join("%  " + l for l in lines))
    p8 = prose_8b()
    (out / "prose_8b_builder.txt").write_text("\n".join(p8) + "\n")
    print("%% ---- Builder-cost section, Qwen3-8B prose numbers ----")
    print("\n".join("%  " + l for l in p8))
    print(f"%% wrote {out}")


if __name__ == "__main__":
    main()
