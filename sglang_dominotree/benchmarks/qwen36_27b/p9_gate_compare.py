#!/usr/bin/env python3
"""P9 losslessness gate — verdict. Compares each speculative method's T=0 outputs to
plain autoregressive decoding (AR), token by token.

Byte-identity to AR is NOT the bar (docs/domino_tree_sglang_integration/
bit_identity_wrong_bar.md): verify kernels reorder bf16 reductions, so a near-tie can
resolve differently and a valid method may leave AR's path. The bar is RELATIVE:

  reference  = domino (our chain, upstream's hybrid chain verify + state commit)
  candidate  = dominotree (P9)
  control    = dominotree_flat (DOMINOTREE_DIAG_FLAT_GDN=1: the bug P9 fixes)

Decision rule, fixed BEFORE any data was seen (2026-09-24):
  PASS          candidate median agreement >= 0.5 x reference's, AND candidate
                full-match rate >= reference's - 0.20, AND the control is clearly
                worse (control median < 0.5 x candidate's) -- the test has power.
  INCONCLUSIVE  candidate looks fine but the control is NOT clearly worse: the prompt
                set cannot distinguish the bug, so a pass would mean nothing.
  FAIL          otherwise.
  ERROR         missing, empty, duplicate, malformed, or incomplete arm (exit 2).
"agreement" = tokens matching AR before the first divergence, capped at the
shorter output (an early EOS that AR also emits counts as full agreement).

  python p9_gate_compare.py gate/ar.jsonl gate/domino.jsonl gate/dominotree.jsonl \
      gate/dominotree_flat.jsonl
"""

import argparse
import json
import math
import statistics
import sys


class GateArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, f"P9_GATE=ERROR ({message})\n")


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["prompt_id"] in rows:
                raise ValueError(f"{path}: duplicate prompt id {r['prompt_id']}")
            ids = r["output_ids"]
            if not isinstance(ids, list) or not ids or any(type(x) is not int for x in ids):
                raise ValueError(f"{path}: empty or invalid output_ids for {r['prompt_id']}")
            if not math.isfinite(float(r["tau"])):
                raise ValueError(f"{path}: invalid tau for {r['prompt_id']}")
            rows[r["prompt_id"]] = r
    if not rows:
        raise ValueError(f"{path}: empty arm")
    return rows


def agreement(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, False
    return n, len(a) == len(b)


def summarize(ar, rows):
    agree, full, taus = [], 0, []
    per = {}
    for pid, r in ar.items():
        k, same = agreement(r["output_ids"], rows[pid]["output_ids"])
        agree.append(k)
        full += same
        taus.append(rows[pid]["tau"])
        per[pid] = (k, same, len(rows[pid]["output_ids"]))
    n = len(agree)
    return {
        "n": n,
        "median_agree": statistics.median(agree),
        "mean_agree": statistics.mean(agree),
        "full_rate": full / n,
        "mean_tau": statistics.mean(taus),
        "per": per,
    }


def main():
    ap = GateArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs=4, metavar="ARM")
    ap.add_argument("--per-dataset", type=int, default=5)
    args = ap.parse_args()
    labels = ["domino", "dominotree", "dominotree_flat"]
    try:
        if args.per_dataset <= 0:
            raise ValueError("--per-dataset must be positive")
        # Same fixed ids as p9_gate_client.py; require completeness even if ALL
        # four files were truncated at the same prompt.
        expected = {f"{ds}/{i}" for ds in ("gsm8k", "humaneval", "mt-bench", "math500")
                    for i in range(args.per_dataset)}
        arms = [load(p) for p in args.arms]
        for path, rows in zip(args.arms, arms):
            if set(rows) != expected:
                raise ValueError(f"{path}: prompt ids differ: missing={sorted(expected - set(rows))}, "
                                 f"unexpected={sorted(set(rows) - expected)}")
        ar = arms[0]
        s = {lab: summarize(ar, rows) for lab, rows in zip(labels, arms[1:])}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"P9_GATE=ERROR ({exc})")
        return 2

    print(f"{'prompt':14s} " + " ".join(f"{lab:>18s}" for lab in labels))
    for pid in ar:
        cells = []
        for lab in labels:
            k, same, n = s[lab]["per"][pid]
            cells.append(f"{'=' if same else str(k)+'/'+str(n):>18s}")
        print(f"{pid:14s} " + " ".join(cells))
    print()
    for lab in labels:
        x = s[lab]
        print(f"{lab:16s} n={x['n']:2d} median_agree={x['median_agree']:7.1f} "
              f"mean_agree={x['mean_agree']:7.1f} full_match={x['full_rate']:.2f} "
              f"mean_tau={x['mean_tau']:.2f}")

    ref, cand, ctl = s["domino"], s["dominotree"], s["dominotree_flat"]
    cand_ok = (cand["median_agree"] >= 0.5 * ref["median_agree"]
               and cand["full_rate"] >= ref["full_rate"] - 0.20)
    has_power = ctl["median_agree"] < 0.5 * cand["median_agree"]
    if cand_ok and has_power:
        verdict = "P9_GATE=PASS"
    elif cand_ok:
        verdict = "P9_GATE=INCONCLUSIVE (control not clearly worse: no power)"
    else:
        verdict = "P9_GATE=FAIL"
    print(verdict)
    return 0 if verdict == "P9_GATE=PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
