#!/usr/bin/env python3
"""R5 for the IEEE Access resubmission: re-derive every bs=1 5090 number from RAW
per-prompt JSONL, with paired bootstrap CIs.

Why this exists separately from `verify_published_numbers.py`: that script diffs every
recomputed cell against `PUBLISHED.json` and gives a single pass/fail line. This one
prints the full derivation as markdown tables (per-dataset TPS, MACRO tau, MICRO tau,
and DominoTree-vs-reference paired CIs) -- useful when you want to see the numbers
themselves, not just confirm they match what is already recorded.

What it enforces, rather than assumes:
  * PAIRING IS VERIFIED. A paired delta is only meaningful if both arms ran the same
    prompts. Every sample_idx in one arm must appear in the other, or the cell is
    refused -- not silently intersected.
  * MT-Bench turns of one prompt are ONE resampling unit. The two turns are dependent;
    treating them as independent units understates the interval.
  * Per-prompt throughput pools the prompt's turns: sum(tokens)/sum(decode_time).
    Averaging per-turn TPS would weight a 12-token turn like a 900-token one.
  * The bootstrap resamples PROMPTS, not rows.

Usage:
  python3 results/serving/verify_r2_5090.py            # both models
  python3 ... --model 8B --ref domino_chain --boot 5000

Requires only the Python standard library. Reads results/serving/bs1/<size>/<method>/.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ["ar", "eagle3", "dflash", "domino_chain", "dominotree"]
DATASETS = ["gsm8k", "math500", "aime25", "humaneval", "mbpp",
            "livecodebench", "mt-bench", "alpaca"]


def load_cell(model, method, dataset, temp):
    """-> {sample_idx: [row, ...]}  (a prompt's turns grouped together)"""
    path = os.path.join(HERE, "bs1", model.lower(), method, f"{dataset}_T{temp}.jsonl")
    if not os.path.isfile(path):
        return None
    by_prompt = collections.defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                by_prompt[r["sample_idx"]].append(r)
    return dict(by_prompt) or None


def prompt_tps(rows):
    tok = sum(r["num_output"] for r in rows)
    sec = sum(r["decode_time"] for r in rows)
    return tok / sec if sec > 0 else float("nan")


def prompt_tau(rows):
    """One PROMPT's tau: its accepted tokens divided by its verify steps.

    Within a prompt this is token-weighted, which is what we want -- a 900-token turn carries
    more verify steps than a 12-token one, so averaging the two turns' tau equally would be
    wrong. (num_output / mean_accept reconstructs a turn's verify-step count.)
    """
    num = sum(r["num_output"] for r in rows)
    steps = sum(r["num_output"] / r["mean_accept"] for r in rows if r.get("mean_accept"))
    return num / steps if steps > 0 else float("nan")


def prompt_steps(rows):
    """A prompt's verify-step count -- the denominator of the MICRO estimand."""
    return sum(r["num_output"] / r["mean_accept"] for r in rows if r.get("mean_accept"))


def prompt_tokens(rows):
    return sum(r["num_output"] for r in rows)


# TWO estimands, and the paper must say WHICH. They answer different questions and the code
# must not blur them (Codex review 2026-09-14 -- the earlier version computed MACRO while the
# docstring claimed token weighting, which is how a wrong number reaches a table):
#   MACRO  mean over prompts of per-prompt tau. Equal weight per prompt. Matches how our earlier
#          tables were built, and is the right unit when the PROMPT is the sampling unit.
#   MICRO  sum(tokens) / sum(steps) over all prompts. Equal weight per TOKEN. This is the literal
#          reading of "accepted tokens per verify step".
# We report MACRO as primary (it matches the bootstrap's resampling unit -- prompts) and print
# MICRO beside it so any divergence is visible rather than hidden.
def dataset_tau_macro(cell):
    return st.mean(prompt_tau(v) for v in cell.values())


def dataset_tau_micro(cell):
    tok = sum(prompt_tokens(v) for v in cell.values())
    stp = sum(prompt_steps(v) for v in cell.values())
    return tok / stp if stp > 0 else float("nan")


def paired_delta(a, b, stat, boot, seed):
    """Percent delta of a over b on shared prompts, with a percentile bootstrap CI."""
    keys = sorted(set(a) & set(b))
    missing = (set(a) ^ set(b))
    if missing:
        raise ValueError(f"UNPAIRED: {len(missing)} sample_idx present in only one arm")
    # sample_idx agreement is NOT enough. If one arm dropped a single MT-Bench turn, both arms
    # still carry that sample_idx and the pairing looks fine while the per-prompt statistic is
    # computed over different content -- a silent bias in a PAIRED comparison. Check the turns
    # and the field sanity too.  -- Codex review 2026-09-14
    for k in keys:
        ta = sorted(r.get("turn_index") for r in a[k])
        tb = sorted(r.get("turn_index") for r in b[k])
        if ta != tb:
            raise ValueError(f"UNPAIRED TURNS at sample_idx={k}: {ta} vs {tb}")
        if len(set(ta)) != len(ta):
            raise ValueError(f"DUPLICATE TURNS at sample_idx={k}: {ta}")
        for r in a[k] + b[k]:
            for f in ("num_output", "decode_time", "mean_accept"):
                v = r.get(f)
                if v is None or not isinstance(v, (int, float)) or v != v or v <= 0:
                    raise ValueError(f"BAD {f}={v!r} at sample_idx={k}")
    av = [stat(a[k]) for k in keys]
    bv = [stat(b[k]) for k in keys]
    point = 100.0 * (st.mean(av) / st.mean(bv) - 1.0)
    rng = random.Random(seed)
    n = len(keys)
    draws = []
    for _ in range(boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ma = st.mean(av[i] for i in idx)
        mb = st.mean(bv[i] for i in idx)
        if mb > 0:
            draws.append(100.0 * (ma / mb - 1.0))
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws)) - 1]
    return point, lo, hi, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="both", choices=["4B", "8B", "both"])
    ap.add_argument("--ref", default="domino_chain")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    models = ["4B", "8B"] if a.model == "both" else [a.model]
    problems = []

    for model in models:
        for temp in ("0.0", "1.0"):
            cells = {m: {d: load_cell(model, m, d, temp) for d in DATASETS}
                     for m in METHODS}
            for m in METHODS:
                for d in DATASETS:
                    if cells[m][d] is None:
                        problems.append(f"MISSING {model} T{temp} {m}/{d}")

            print(f"\n## {model}, T={temp} — TPS (tok/s), mean over prompts\n")
            print("| dataset | " + " | ".join(METHODS) + " |")
            print("|---" * (len(METHODS) + 1) + "|")
            for d in DATASETS:
                row = []
                for m in METHODS:
                    c = cells[m][d]
                    row.append(f"{st.mean(prompt_tps(v) for v in c.values()):.2f}"
                               if c else "—")
                print(f"| {d} | " + " | ".join(row) + " |")

            print(f"\n## {model}, T={temp} — tau, MACRO (mean over prompts of per-prompt tau)\n")
            print("| dataset | " + " | ".join(METHODS) + " |")
            print("|---" * (len(METHODS) + 1) + "|")
            for d in DATASETS:
                row = [f"{dataset_tau_macro(cells[m][d]):.2f}" if cells[m][d] else "—"
                       for m in METHODS]
                print(f"| {d} | " + " | ".join(row) + " |")

            print(f"\n## {model}, T={temp} — tau, MICRO (total accepted tokens / total verify "
                  f"steps)\n")
            print("*Printed so any divergence from MACRO is visible. The paper must state which "
                  "it reports; the CIs below resample PROMPTS and so pair with MACRO.*\n")
            print("| dataset | " + " | ".join(METHODS) + " |")
            print("|---" * (len(METHODS) + 1) + "|")
            for d in DATASETS:
                row = [f"{dataset_tau_micro(cells[m][d]):.2f}" if cells[m][d] else "—"
                       for m in METHODS]
                print(f"| {d} | " + " | ".join(row) + " |")

            print(f"\n## {model}, T={temp} — DominoTree vs {a.ref}, "
                  f"paired bootstrap 95% CI (B={a.boot})\n")
            print("| dataset | ΔTPS % [95% CI] | Δtau % [95% CI] | n |")
            print("|---|---|---|---|")
            wins = 0
            for d in DATASETS:
                tree, ref = cells["dominotree"][d], cells[a.ref][d]
                if not tree or not ref:
                    print(f"| {d} | — | — | — |")
                    continue
                try:
                    tp, tlo, thi, n = paired_delta(tree, ref, prompt_tps,
                                                   a.boot, a.seed)
                    ap_, alo, ahi, _ = paired_delta(tree, ref, prompt_tau,
                                                    a.boot, a.seed)
                except ValueError as e:
                    problems.append(f"{model} T{temp} {d}: {e}")
                    print(f"| {d} | **{e}** | | |")
                    continue
                sig = "" if tlo > 0 or thi < 0 else " *"
                if tp > 0:
                    wins += 1
                print(f"| {d} | {tp:+.1f}% [{tlo:+.1f},{thi:+.1f}]{sig} "
                      f"| {ap_:+.1f}% [{alo:+.1f},{ahi:+.1f}] | {n} |")
            print(f"\n`* = CI straddles zero (not significant).` "
                  f"DominoTree TPS wins **{wins}/{len(DATASETS)}** datasets.")

    print("\n---\n")
    if problems:
        print(f"**{len(problems)} PROBLEM(S):**")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("**All cells present and paired. No problems.**")


if __name__ == "__main__":
    main()
