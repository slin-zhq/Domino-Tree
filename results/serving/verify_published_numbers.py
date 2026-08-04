#!/usr/bin/env python3
"""Check every serving number in the DominoTree paper against the raw data shipped here.

Nothing in this script trusts a summary table. It reads the per-prompt and per-cell
JSONL under this directory, recomputes each published cell, and diffs the result
against `PUBLISHED.json` -- which holds the values exactly as printed in the paper's
serving tables. It exits non-zero if anything disagrees.

Run it from anywhere:

    python3 results/serving/verify_published_numbers.py

Expected output: `ALL CELLS REPRODUCE FROM RAW DATA.`

Conventions, stated here so you can check that we describe what we do:

  Long context (Table "SGLang long-context single-stream")
    A cell averages the two HELMET summarization tasks.
      tau     = mean over tasks of (that task's mean accepted length)
      speedup = mean over tasks of (that task's mean throughput / AR's mean
                throughput on the same task)   -- mean-of-ratios, not ratio-of-means

  Concurrency (Table "SGLang concurrency goodput")
    Goodput at offered concurrency c = UNWEIGHTED mean over the three datasets
    (gsm8k, mbpp, mt-bench). tau = unweighted mean over datasets AND over the full
    measured sweep c = 1,2,4,8,16,32 (tau is flat in c; the table shows only the
    matched-admission columns c <= 8, where every method admits the full offered load).

Requires only the Python standard library.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLISHED = HERE / "PUBLISHED.json"

TASKS = ["infbench_sum", "multi_lexsum"]
DATASETS = ["gsm8k", "mbpp", "mt-bench"]
CONCS_SHOWN = [1, 2, 4, 8]
CONCS_TAU = [1, 2, 4, 8, 16, 32]

fails: list[str] = []


def note(ok: bool, msg: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        fails.append(msg)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ------------------------------------------------------------- long context
def helmet_cell(size: str, method: str, task: str, bin_: int, key: str) -> float:
    rows = read_jsonl(HERE / "longcontext" / size / method / f"helmet_{size}.prompts.jsonl")
    xs = [r[key] for r in rows if r["task"] == task and int(r["length_bin"]) == bin_]
    if not xs:
        raise KeyError(f"no rows for {size}/{method}/{task}/{bin_}")
    return sum(xs) / len(xs)


def check_longctx(published: dict) -> None:
    print("== Long context (HELMET summarization, bs=1) ==")
    for key, value in sorted(published.items()):
        if key.startswith("_"):
            continue
        ptau, pspd = value
        size, b, method = key.split("/")
        b = int(b)
        if method == "ar":
            continue  # AR is the normalizer: tau == 1.00, speedup == 1.00 by definition
        ar = {t: helmet_cell(size, "ar", t, b, "tps") for t in TASKS}
        tau = sum(helmet_cell(size, method, t, b, "accept") for t in TASKS) / len(TASKS)
        spd = sum(helmet_cell(size, method, t, b, "tps") / ar[t] for t in TASKS) / len(TASKS)
        tag = f"{size}/{b // 1024}K {method:<13}"
        note(abs(round(tau, 2) - ptau) < 5e-3,
             f"{tag} tau      data {tau:6.3f} -> {round(tau, 2):.2f}   paper {ptau:.2f}")
        note(abs(round(spd, 2) - pspd) < 5e-3,
             f"{tag} speedup  data {spd:6.4f} -> {round(spd, 2):.2f}   paper {pspd:.2f}")


# -------------------------------------------------------------- concurrency
def check_conc(published: dict) -> None:
    print("\n== Concurrency goodput (matched admission, c <= 8) ==")
    for key, exp in sorted(published.items()):
        if key.startswith("_"):
            continue
        size, method = key.split("/")
        rows = read_jsonl(HERE / "concurrency" / size / method / f"{method}.jsonl")
        cells = {(r["dataset"], int(r["concurrency"])): r for r in rows}

        for c in CONCS_SHOWN:
            want = exp["tps"].get(str(c))
            if want is None:
                continue
            vals = [cells[(d, c)]["tps"] for d in DATASETS if (d, c) in cells]
            if len(vals) != len(DATASETS):
                note(False, f"{size} {method:<13} tps c={c}: only {len(vals)}/3 datasets present")
                continue
            got = sum(vals) / len(vals)
            note(abs(round(got) - want) < 0.5,
                 f"{size} {method:<13} tps c={c:<2} data {got:8.2f} -> {round(got):.0f}"
                 f"   paper {want:.0f}")

        if exp["tau"] is not None:
            taus = []
            for c in CONCS_TAU:
                acc = [cells[(d, c)]["mean_accept"] for d in DATASETS
                       if (d, c) in cells and cells[(d, c)].get("mean_accept")]
                if len(acc) == len(DATASETS):
                    taus.append(sum(acc) / len(acc))
            got = sum(taus) / len(taus)
            note(abs(round(got, 2) - exp["tau"]) < 5e-3,
                 f"{size} {method:<13} tau       data {got:8.4f} -> {round(got, 2):.2f}"
                 f"   paper {exp['tau']:.2f}")


# --------------------------------------------------------------- admission caps
def report_caps() -> None:
    """The caps are not a claim to check -- they are context for reading the table."""
    print("\n== Admission caps recorded alongside each concurrency run ==")
    for size in ("4b", "8b"):
        parts = []
        for d in sorted((HERE / "concurrency" / size).iterdir()):
            f = d / f"status_{d.name}.done"
            if f.is_file():
                kv = dict(t.split("=", 1) for t in f.read_text().split() if "=" in t)
                parts.append(f"{d.name}={kv.get('cap', '?')}")
        print(f"  {size}: " + "  ".join(parts))
    print("  (A goodput column is like-for-like only where the offered concurrency c is\n"
          "   at or below EVERY method's cap. That is why the paper's main table stops\n"
          "   at c=8 and the full c=32 sweep sits in the appendix.)")


def main() -> int:
    if not PUBLISHED.is_file():
        print(f"missing {PUBLISHED}", file=sys.stderr)
        return 2
    pub = json.loads(PUBLISHED.read_text())
    print(f"data: {HERE}\n")
    check_longctx(pub["longctx"])
    check_conc(pub["concurrency"])
    report_caps()

    print("\n" + "=" * 72)
    if fails:
        print(f"{len(fails)} MISMATCH(ES) between the paper and the raw data:")
        for f in fails:
            print("  - " + f)
        return 1
    print("ALL CELLS REPRODUCE FROM RAW DATA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
