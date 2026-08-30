"""Reproduce the budget-tuned SGLang bs=1 numbers with no GPU and no model.

Reads the raw per-prompt JSONL under bs1/4b/{dominotree_b32,ar_b32session}/ and
prints speedup over AR and mean accepted length tau, per dataset and temperature
-- the DominoTree rows of the paper's per-dataset bs=1 appendix table.

Two conventions matter and are deliberate:

* AR comes from ``ar_b32session/``, collected in the SAME session as the
  DominoTree run it normalises. The benchmark host is shared, so absolute tok/s
  drifts between sessions; a same-session ratio cancels that drift. ``ar/`` holds
  the AR from the earlier budget-16 collection and is NOT interchangeable here.
* ``dominotree/`` (tree budget 15, i.e. 16 verify slots) is the previously
  published configuration and is kept alongside for the audit trail;
  ``dominotree_b32/`` is the tuned budget-32 configuration.

Stdlib only.  Usage:  python3 aggregate_bs1_budget.py [--dir .]
"""
import argparse
import glob
import json
import os
import statistics

DATASETS = ["gsm8k", "math500", "aime25", "humaneval", "mbpp",
            "livecodebench", "mt-bench", "alpaca"]
TEMPS = ["0.0", "0.5", "1.0"]


def read(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect(base, sub):
    out = {}
    for path in glob.glob(os.path.join(base, sub, "*.jsonl")):
        name = os.path.basename(path)
        for temp in TEMPS:
            suffix = "_T%s.jsonl" % temp
            if name.endswith(suffix):
                rows = read(path)
                if rows:
                    out[(name[: -len(suffix)], temp)] = (
                        statistics.fmean(r["tps"] for r in rows),
                        statistics.fmean(r["mean_accept"] for r in rows),
                        len(rows),
                    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    base = os.path.join(args.dir, "bs1", "4b")

    tree = collect(base, "dominotree_b32")
    ar = collect(base, "ar_b32session")
    if not tree or not ar:
        raise SystemExit("missing bs1/4b/{dominotree_b32,ar_b32session}/ under %s" % base)

    print("DominoTree, tree budget 32, SGLang bs=1, Qwen3-4B")
    print("speedup over same-session AR / mean accepted length tau\n")
    header = "%-15s" % "dataset"
    for temp in TEMPS:
        header += "%18s" % ("T=%s" % temp)
    print(header)

    overall = {t: ([], []) for t in TEMPS}
    for ds in DATASETS:
        line = "%-15s" % ds
        for temp in TEMPS:
            t = tree.get((ds, temp))
            a = ar.get((ds, temp))
            if not t or not a:
                line += "%18s" % "--"
                continue
            speedup = t[0] / a[0]
            overall[temp][0].append(speedup)
            overall[temp][1].append(t[1])
            line += "%18s" % ("%.2fx / %.2f" % (speedup, t[1]))
        print(line)

    line = "%-15s" % "OVERALL"
    for temp in TEMPS:
        sp, tau = overall[temp]
        line += "%18s" % ("%.2fx / %.2f" % (statistics.fmean(sp), statistics.fmean(tau))
                          if sp else "--")
    print(line)

    n = {v[2] for v in tree.values()}
    print("\nprompts per cell: %s" % (sorted(n) if len(n) > 1 else n.pop()))


if __name__ == "__main__":
    main()
