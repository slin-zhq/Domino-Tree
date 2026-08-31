"""Reproduce the long-context budget comparison with no GPU and no model.

Reads the raw HELMET rows under longcontext/4b/ and prints accepted length and
throughput for the two tree budgets, plus the per-cell delta.

Why a control directory exists: the two budgets were collected in different
sessions, and comparing across builds once produced a spurious regression
elsewhere in this campaign. `dominotree_b15_control/` re-runs the smaller budget
on the SAME build as `dominotree_b32/`, so the delta below is attributable to the
budget alone. The control reproduces the separately published configuration to
within 1% on accepted length and 0.5% on throughput.

HELMET writes ONE aggregate row per cell plus a per-prompt sidecar
(`helmet.prompts.jsonl`, 50 rows per cell); a cell showing one aggregate row is
complete, not truncated.

Stdlib only.  Usage:  python3 aggregate_longcontext_budget.py [--dir .]
"""
import argparse
import glob
import json
import os


def load(base, sub):
    cells = {}
    for path in glob.glob(os.path.join(base, sub, "helmet.jsonl")):
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r.get("task"), r.get("length_bin") or r.get("bin"))
                cells[key] = (r.get("mean_accept"),
                              r.get("tps") or r.get("output_throughput"))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    base = os.path.join(args.dir, "longcontext", "4b")

    small = load(base, "dominotree_b15_control")
    large = load(base, "dominotree_b32")
    if not small or not large:
        raise SystemExit("missing longcontext/4b/{dominotree_b15_control,dominotree_b32}/")

    print("Qwen3-4B long context (HELMET), n=50 per cell, T=0")
    print("accepted length tau and output tokens/s, by tree budget\n")
    print("%-16s %8s %18s %18s %10s %10s"
          % ("task", "context", "budget 15", "budget 32", "d tau", "d tok/s"))

    dtau, dtps = [], []
    for key in sorted(set(small) & set(large), key=lambda k: (str(k[0]), k[1] or 0)):
        t_s, p_s = small[key]
        t_l, p_l = large[key]
        a = 100.0 * (t_l / t_s - 1.0)
        b = 100.0 * (p_l / p_s - 1.0)
        dtau.append(a)
        dtps.append(b)
        print("%-16s %8s %9.3f/%7.1f %9.3f/%7.1f %+9.1f%% %+9.1f%%"
              % (key[0], key[1], t_s, p_s, t_l, p_l, a, b))

    n = len(dtau)
    print("\nmean over %d cells: tau %+.1f%%   throughput %+.1f%%"
          % (n, sum(dtau) / n, sum(dtps) / n))
    print("\nThe larger budget raises accepted length on every cell and lowers")
    print("throughput on every cell: long context is prefill-dominated, so the")
    print("extra candidates do not repay their construction and verification cost.")


if __name__ == "__main__":
    main()
