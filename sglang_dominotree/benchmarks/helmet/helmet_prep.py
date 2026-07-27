#!/usr/bin/env python3
"""Materialise HELMET prompts into flat per-cell jsonl for helmet_longctx.py.

WHY THIS EXISTS (and why it is SEPARATE from the driver)
--------------------------------------------------------
Project hard rule: NEVER re-implement official code. HELMET's prompts (context
assembly, per-task instruction template, truncation to a length bin) are HELMET's
artifact. This script uses HELMET's OWN materials to produce the exact prompt
strings, then writes them flat so the measurement driver can replay them verbatim.

It runs on the GPU box (MIRLab), where the HELMET repo + data live and the served
model's tokenizer is available. It writes:

    <out-dir>/<task>/<length_bin>.jsonl        # one prompt per line:
      {"task","length_bin","idx","text","gen_cap","input_tokens_nominal"}

`text` is the FINAL string to POST as /generate `text`: HELMET's task template
applied to the (already length-truncated) example, then the served model's chat
template (add_generation_prompt=True) — i.e. exactly what HELMET would feed the
model. Order is HELMET's dataset order (NOT shuffled), so the driver's first-N
prefix rule (n=50 nests in n=100) holds.

THE ONE HELMET SEAM — verify on MIRLab before the big run
---------------------------------------------------------
Two facts come from HELMET, not from us; both are parameterised, never guessed:
  1. WHERE the pre-built, length-binned examples live (the HF snapshot jsonl for a
     given task+bin). Pass via --helmet-data-glob (default matches HELMET's
     `data/<task>*<bin>*.jsonl` filename-encodes-length convention).
  2. HELMET's per-task instruction TEMPLATE. Either the example jsonl already holds
     a ready prompt field (--prompt-field), OR we format a template string you copy
     verbatim from HELMET's config for that task (--template / --template-file).
Smoke-test with --limit 2 and eyeball one `text` before scaling (RUNBOOK Step 2b).

OFFLINE FIXTURE (works now, no HELMET / no tokenizer / no GPU)
-------------------------------------------------------------
  python helmet_prep.py --selftest-fixture --out-dir /tmp/helmet_fixture \\
      --tasks demo --length-bins 1024 --limit 4
emits synthetic HELMET-shaped cells so helmet_longctx.py --dry-run and
the driver run end-to-end without any HELMET install (fixture mode).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Optional

# HELMET per-task generation caps (repo configs/*.yaml). Written into each cell so
# the driver uses HELMET's faithful cap per task without re-deriving it.
HELMET_GEN_CAP = {
    "infbench_sum": 1200,
    "multi_lexsum": 1200,
    "alce_asqa": 300,
    "alce_qampari": 300,
    "narrativeqa": 100,
}


# --------------------------------------------------------------------------
# HELMET example loading (the SEAM). Reads HELMET's pre-built, length-binned
# jsonl examples for one (task, length_bin).
# --------------------------------------------------------------------------
def find_helmet_file(data_glob: str, task: str, length_bin: int) -> Optional[str]:
    """Resolve the HELMET example file for (task, length_bin) from a glob template.

    `data_glob` may contain {task} and {bin}; we also try a tolerant fallback that
    just requires both the task name and the bin number to appear in the filename
    (HELMET encodes the token length in the filename, e.g. infbench_sum_eng_32752).
    """
    templated = data_glob.format(task=task, bin=length_bin)
    hits = sorted(glob.glob(templated))
    if hits:
        return hits[0]
    # tolerant fallback: any file under the glob's dir mentioning task AND ~bin.
    root = Path(data_glob.split("{")[0]).parent if "{" in data_glob else Path(data_glob).parent
    for p in sorted(root.rglob("*.jsonl")):
        name = p.name.lower()
        if task.lower() in name and str(length_bin) in name:
            return str(p)
    return None


def load_helmet_examples(path: str, limit: int) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if 0 < limit <= len(rows):
                break
    return rows


def build_prompt_text(example: dict, *, prompt_field: Optional[str],
                      template: Optional[str], tokenizer) -> str:
    """Produce the final POST-able string for ONE example.

    - If --prompt-field is given and present, that field IS HELMET's ready prompt
      (context+instruction already merged) -> use verbatim.
    - Else format HELMET's `template` (copied verbatim from HELMET's config for this
      task) over the example fields.
    Then wrap with the served model's chat template (HELMET evaluates chat models
    with add_generation_prompt=True).
    """
    if prompt_field and prompt_field in example and example[prompt_field]:
        user_content = str(example[prompt_field])
    elif template is not None:
        try:
            user_content = template.format(**example)
        except KeyError as exc:
            raise SystemExit(
                f"template references {exc} but the example lacks it. Example keys: "
                f"{sorted(example.keys())}. Fix --template (copy HELMET's verbatim) "
                f"or use --prompt-field <name-of-ready-prompt-field>."
            )
    else:
        raise SystemExit(
            "need either --prompt-field <field> (example already holds the prompt) "
            "or --template/--template-file (HELMET's task template)."
        )

    if tokenizer is None:  # fixture path only
        return f"<|user|>\n{user_content}\n<|assistant|>\n"
    msgs = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# --------------------------------------------------------------------------
def write_cell(out_dir: str, task: str, length_bin: int, prompts: list[dict]) -> str:
    d = Path(out_dir) / task
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{length_bin}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in prompts:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(path)


def emit_fixture(out_dir: str, tasks: list[str], length_bins: list[int], limit: int) -> None:
    """Synthetic HELMET-shaped cells for offline testing (no HELMET/tokenizer)."""
    n = max(1, limit)
    for task in tasks:
        cap = HELMET_GEN_CAP.get(task, 512)
        for L in length_bins:
            prompts = []
            for i in range(n):
                body = " ".join(f"clause{i}-{j}" for j in range(max(1, L // 8)))
                text = (f"<|user|>\n[FIXTURE task={task} bin={L}] Summarise the following. "
                        f"{body}\n<|assistant|>\n")
                prompts.append({"task": task, "length_bin": L, "idx": i,
                                "text": text, "gen_cap": cap,
                                "input_tokens_nominal": L, "fixture": True})
            p = write_cell(out_dir, task, L, prompts)
            print(f"[fixture] {p}  ({n} prompts, gen_cap={cap})")


# --------------------------------------------------------------------------
def parse_int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def parse_str_list(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, help="Root for <task>/<length_bin>.jsonl cells.")
    p.add_argument("--tasks", default="infbench_sum,multi_lexsum",
                   help="HELMET tasks to materialise.")
    p.add_argument("--length-bins", default="8192,16384,32768",
                   help="Input length bins (Llama-2 tokens in HELMET's filenames).")
    p.add_argument("--limit", type=int, default=100,
                   help="Max prompts per cell (HELMET ships ~100/task). n=100 superset of n=50.")
    # ---- real HELMET path ----
    p.add_argument("--model-path", default=None,
                   help="HF path for the served model's tokenizer + chat template (required for a "
                        "real run; must match the SGLang server's model).")
    p.add_argument("--helmet-data-glob", default=None,
                   help="Glob for HELMET example files; may use {task}/{bin}. "
                        "e.g. ~/HELMET/data/{task}*{bin}*.jsonl")
    p.add_argument("--prompt-field", default=None,
                   help="Field in HELMET's jsonl that already holds the ready prompt "
                        "(context+instruction). If set, --template is not needed.")
    p.add_argument("--template", default=None,
                   help="HELMET's per-task template string (copy VERBATIM from HELMET config), "
                        "with {field} placeholders matching the example jsonl keys.")
    p.add_argument("--template-file", default=None,
                   help="Path to a file whose contents are the template (alternative to --template).")
    # ---- offline fixture ----
    p.add_argument("--selftest-fixture", action="store_true",
                   help="Emit synthetic HELMET-shaped cells (no HELMET/tokenizer/GPU).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tasks = parse_str_list(args.tasks)
    length_bins = parse_int_list(args.length_bins)
    if not tasks or not length_bins:
        raise SystemExit("need at least one --tasks and one --length-bins value.")

    if args.selftest_fixture:
        emit_fixture(args.out_dir, tasks, length_bins, args.limit)
        print(f"\n[fixture done] -> {args.out_dir}")
        return 0

    # ---- real HELMET materialisation ----
    if not args.helmet_data_glob:
        raise SystemExit("--helmet-data-glob is required for a real run (points at HELMET's "
                         "length-binned example jsonl). See README.md Step 2.")
    template = args.template
    if args.template_file:
        template = Path(args.template_file).read_text(encoding="utf-8")

    tokenizer = None
    if args.model_path:
        from transformers import AutoTokenizer  # lazy; only on the GPU box
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        print(f"[tokenizer] {args.model_path}")
    else:
        print("[warn] no --model-path: emitting prompts WITHOUT the served model's chat template "
              "(fixture-style wrap). Provide --model-path for a faithful run.")

    total = 0
    for task in tasks:
        cap = HELMET_GEN_CAP.get(task)
        if cap is None:
            print(f"[warn] task={task}: no known HELMET gen cap; defaulting to 512. "
                  f"Set the right cap in HELMET_GEN_CAP if this is a long-output task.")
            cap = 512
        for L in length_bins:
            src = find_helmet_file(args.helmet_data_glob, task, L)
            if not src:
                print(f"[skip] task={task} bin={L}: no HELMET file matched "
                      f"'{args.helmet_data_glob.format(task=task, bin=L)}'. Check the glob.")
                continue
            examples = load_helmet_examples(src, args.limit)
            prompts = []
            for i, ex in enumerate(examples):
                text = build_prompt_text(ex, prompt_field=args.prompt_field,
                                         template=template, tokenizer=tokenizer)
                prompts.append({
                    "task": task, "length_bin": L, "idx": i,
                    "text": text, "gen_cap": cap,
                    "input_tokens_nominal": L,
                    "helmet_src": os.path.basename(src),
                })
            path = write_cell(args.out_dir, task, L, prompts)
            total += len(prompts)
            print(f"[cell] task={task:<14} bin={L:>6} src={os.path.basename(src):<32} "
                  f"n={len(prompts):>3} gen_cap={cap} -> {path}")

    print(f"\n[prep done] {total} prompts across {len(tasks)} task(s) x {len(length_bins)} bin(s) "
          f"-> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
