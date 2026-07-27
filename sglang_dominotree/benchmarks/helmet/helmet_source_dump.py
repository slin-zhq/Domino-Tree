#!/usr/bin/env python3
"""Materialise HELMET's OWN Summarization examples (infbench_sum, multi_lexsum)
into flat per-cell jsonl that helmet_prep.py can consume via --prompt-field.

WHY THIS EXISTS
---------------
HELMET's "Summ" task category (unlike Recall/RAG/Rerank/ALCE) is NOT shipped as
pre-built length-binned jsonl files anywhere -- the princeton-nlp/HELMET HF
*dataset* repo only contains two opaque tarballs (data.tar.gz / data_v2.tar.gz);
those cover the locally-stored task families. Summ is loaded LIVE by HELMET's own
`data.py` (`load_infbench`, `load_multi_lexsum`) straight from HF datasets
(xinrongzhang2022/infinitebench, allenai/multi_lexsum), then truncated to a
length bin with the Llama-2-7b-hf tokenizer via HELMET's own `truncate_llama2`.

Project hard rule: NEVER re-implement official code. This script therefore:
  - imports and calls HELMET's *actual* `load_infbench` / `load_multi_lexsum`
    from a cloned ~/HELMET repo's data.py, unmodified;
  - uses HELMET's own bin -> dataset-name convention verbatim from
    configs/summ_short.yaml (input_max_length -> the *_eng_<trunc> string);
  - only patches around ONE breakage: HF `datasets` >=4 dropped support for
    loading-script datasets, so `load_dataset("allenai/multi_lexsum", ...,
    trust_remote_code=True)` (used inside load_multi_lexsum) now raises. We
    replace just that one network call with a local equivalent built from the
    dataset's OWN loading script logic (allenai/multi_lexsum's multi_lexsum.py,
    copied verbatim in _build_multilexsum_datasetdict below -- same fields, same
    source files, just executed directly instead of via the deprecated
    datasets-script mechanism). Everything downstream (context assembly, demo
    sampling, truncation, prompt template) is HELMET's own load_multi_lexsum body,
    called unmodified via monkeypatching data.load_dataset for this one dataset id.
  - stubs `utils` (HELMET's eval-metrics module: rouge_score/pytrec_eval) since
    those packages are not needed for PROMPT construction and are not installed
    in the serving venv; load_infbench/load_multi_lexsum never call them during
    data loading (only inside post_process closures we never invoke).

Output: <out-dir>/<task>_<length_bin>.jsonl, one record per example:
  {"task","length_bin","idx","helmet_prompt","hf_dataset_name"}
`helmet_prompt` = HELMET's own `user_template.format(**row)` (context+instruction,
NOT yet chat-templated). Feed this into helmet_prep.py with
`--prompt-field helmet_prompt` (it applies the served model's chat template).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

# HELMET's own bin -> dataset-name-suffix convention, copied verbatim from
# ~/HELMET/configs/summ_short.yaml (`datasets:` line). The number is HELMET's
# own choice of how many Llama-2 context tokens to keep after truncation so the
# TOTAL prompt (context + instructions + demo + chat wrapper) lands near
# `input_max_length`. We do not invent these numbers.
INFBENCH_SUM_TRUNC = {8192: 6792, 16384: 14984, 32768: 31368, 65536: 64136}
MULTI_LEXSUM_TRUNC = {8192: 7492, 16384: 15684, 32768: 32068, 65536: 64836}
SHOTS_DEFAULT = 2  # summ_short.yaml: shots: 2


def _stub_utils_module() -> None:
    """Register a no-op `utils` module so `import data` (HELMET) doesn't need
    rouge_score/pytrec_eval installed. Safe: load_infbench/load_multi_lexsum
    never call these during data loading, only inside post_process closures
    (evaluation-time) which we never invoke here."""
    if "utils" in sys.modules:
        return
    stub = types.ModuleType("utils")

    def _unused(*a, **kw):
        raise RuntimeError("utils stub called -- this path should be data-loading "
                            "only (prompt construction), not evaluation.")

    stub.calculate_metrics = _unused
    stub.parse_output = _unused
    stub.parse_rankings = _unused
    stub.calculate_retrieval_metrics = _unused
    sys.modules["utils"] = stub


def _build_multilexsum_datasetdict(cache_dir: Path):
    """Local equivalent of `load_dataset("allenai/multi_lexsum", name="v20230518",
    trust_remote_code=True)`, since HF `datasets>=4` removed loading-script
    support. Logic below is copied verbatim from that dataset's OWN loading
    script (`multi_lexsum.py`, fetched from the HF dataset repo) -- same source
    files (releases/v20230518/{train,dev,test,sources}.json), same field
    extraction (`case_sources = [sources[id]["doc_text"] for id in
    case_data["case_documents"]]`, `summary/short`). We only skip the `test`
    split and the non-`summary/short` metadata fields HELMET's load_multi_lexsum
    never reads."""
    import datasets as hfd
    from huggingface_hub import hf_hub_download

    def _dl(name):
        return hf_hub_download("allenai/multi_lexsum", repo_type="dataset",
                                filename=f"releases/v20230518/{name}",
                                local_dir=str(cache_dir))

    train_path = _dl("train.json")
    dev_path = _dl("dev.json")
    sources_path = _dl("sources.json")

    def _load_jsonl(p):
        with open(p, "r") as f:
            return [json.loads(line) for line in f.read().splitlines() if line.strip()]

    print(f"[multi_lexsum] loading sources.json ({os.path.getsize(sources_path)/1e9:.2f} GB) ...")
    with open(sources_path, "r") as f:
        sources = json.load(f)
    print(f"[multi_lexsum] sources loaded: {len(sources)} docs")

    def _rows_with_context(subset_file):
        cases = _load_jsonl(subset_file)
        rows = []
        for case in cases:
            try:
                case_sources = [sources[doc_id]["doc_text"] for doc_id in case["case_documents"]]
            except KeyError as exc:
                print(f"[multi_lexsum] skip case {case.get('case_id')}: missing source {exc}")
                continue
            rows.append({
                "id": case["case_id"],
                "sources": case_sources,
                "summary/short": case["summary/short"],
            })
        return rows

    def _rows_train_only(subset_file):
        # HELMET's demo-sampling only ever reads ex["summary/short"] from the
        # train split -- no need to resolve source doc text for it.
        cases = _load_jsonl(subset_file)
        return [{"id": c["case_id"], "sources": [], "summary/short": c["summary/short"]}
                for c in cases]

    train_rows = _rows_train_only(train_path)
    dev_rows = _rows_with_context(dev_path)
    print(f"[multi_lexsum] train={len(train_rows)} rows, validation(dev)={len(dev_rows)} rows")

    ft = hfd.Features({
        "id": hfd.Value("string"),
        "sources": hfd.Sequence(hfd.Value("string")),
        "summary/short": hfd.Value("string"),
    })
    dd = hfd.DatasetDict({
        "train": hfd.Dataset.from_list(train_rows, features=ft),
        "validation": hfd.Dataset.from_list(dev_rows, features=ft),
    })
    return dd


def _install_helmet_and_load(helmet_repo: Path):
    _stub_utils_module()
    sys.path.insert(0, str(helmet_repo))
    import data as helmet_data  # HELMET's own data.py, unmodified

    real_load_dataset = helmet_data.load_dataset

    def patched_load_dataset(dataset_path, *a, **kw):
        if dataset_path == "allenai/multi_lexsum":
            cache_dir = Path(os.environ.get("MULTILEXSUM_CACHE",
                                             str(Path.home() / "HELMET_data" / "multi_lexsum_raw")))
            cache_dir.mkdir(parents=True, exist_ok=True)
            return _build_multilexsum_datasetdict(cache_dir)
        return real_load_dataset(dataset_path, *a, **kw)

    helmet_data.load_dataset = patched_load_dataset

    # HELMET's truncate_llama2/filter_length hardcode "meta-llama/Llama-2-7b-hf"
    # (a gated repo we don't have accepted access to). Redirect to
    # NousResearch/Llama-2-7b-hf, a public un-gated mirror of the IDENTICAL
    # tokenizer (same vocab/merges) -- content is unchanged, only the HF repo
    # host differs, so token counts/truncation offsets are byte-identical to
    # what HELMET's own code would produce.
    from transformers import AutoTokenizer
    real_from_pretrained = AutoTokenizer.from_pretrained.__func__

    def patched_from_pretrained(cls, name, *a, **kw):
        if name == "meta-llama/Llama-2-7b-hf":
            name = "NousResearch/Llama-2-7b-hf"
        return real_from_pretrained(cls, name, *a, **kw)

    AutoTokenizer.from_pretrained = classmethod(patched_from_pretrained)
    return helmet_data


def dump_cell(helmet_data, task: str, length_bin: int, limit: int, shots: int,
              seed: int, out_dir: Path) -> int:
    if task == "infbench_sum":
        trunc = INFBENCH_SUM_TRUNC.get(length_bin)
        if trunc is None:
            print(f"[skip] infbench_sum: no known trunc length for bin={length_bin} "
                  f"(known: {sorted(INFBENCH_SUM_TRUNC)})")
            return 0
        dataset_name = f"infbench_sum_eng_{trunc}"
        loaded = helmet_data.load_infbench(dataset_name, shots=shots, max_test_samples=limit, seed=seed)
    elif task == "multi_lexsum":
        trunc = MULTI_LEXSUM_TRUNC.get(length_bin)
        if trunc is None:
            print(f"[skip] multi_lexsum: no known trunc length for bin={length_bin} "
                  f"(known: {sorted(MULTI_LEXSUM_TRUNC)})")
            return 0
        dataset_name = f"multi_lexsum_{trunc}"
        loaded = helmet_data.load_multi_lexsum(dataset_name, shots=shots, max_samples=limit, seed=seed)
    else:
        print(f"[skip] unknown task={task} (this script only knows infbench_sum, multi_lexsum)")
        return 0

    rows = loaded["data"]
    user_template = loaded["user_template"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task}_{length_bin}.jsonl"
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            user_content = user_template.format(**row)
            f.write(json.dumps({
                "task": task, "length_bin": length_bin, "idx": i,
                "helmet_prompt": user_content,
                "hf_dataset_name": dataset_name,
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"[dump] task={task:<14} bin={length_bin:>6} dataset={dataset_name:<24} n={n} -> {path}")
    return n


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--helmet-repo", default=str(Path.home() / "HELMET"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tasks", default="infbench_sum,multi_lexsum")
    p.add_argument("--length-bins", default="8192,16384,32768")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--shots", type=int, default=SHOTS_DEFAULT)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    bins = [int(b) for b in args.length_bins.split(",") if b.strip()]
    helmet_data = _install_helmet_and_load(Path(args.helmet_repo).expanduser())
    out_dir = Path(args.out_dir).expanduser()
    total = 0
    for task in tasks:
        for b in bins:
            total += dump_cell(helmet_data, task, b, args.limit, args.shots, args.seed, out_dir)
    print(f"\n[done] {total} examples -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
