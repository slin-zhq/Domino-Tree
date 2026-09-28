#!/usr/bin/env python3
"""Export every arm of a CaDDTree-harness torch cache (.pt) as portable per-prompt JSONL.

Same row schema as convert_ddtree_raw.py, but all arms present (baseline only at T=0,
dflash, ddtree_tbB, caddtree), so readers need no torch. Usage:
  python export_reference_pt.py IN_DIR OUT_DIR --harness-path /path/to/CaDDTree
"""
import argparse, hashlib, json, math, sys
from pathlib import Path
from statistics import fmean

ap = argparse.ArgumentParser()
ap.add_argument("in_dir", type=Path); ap.add_argument("out_dir", type=Path)
ap.add_argument("--harness-path", type=Path, required=True)
a = ap.parse_args()
sys.path.insert(0, str(a.harness_path.resolve()))
import torch

a.out_dir.mkdir(parents=True, exist_ok=True)
for p in sorted(a.in_dir.glob("*.pt")):
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    run = torch.load(p, map_location="cpu", weights_only=False)
    cfg = run["args"]
    methods = (["baseline"] if "baseline" in run["responses"][0] else []) + list(run["methods"])
    ds = str(cfg["dataset"])
    rows = []
    for i, resp in enumerate(run["responses"]):
        for m in methods:
            if m not in resp: raise SystemExit(f"{p}: response {i} lacks {m}")
            v = resp[m]
            acc = [int(x) for x in v.acceptance_lengths]; rounds = int(v.decode_rounds)
            tpot = float(v.time_per_output_token)
            n_out = int(v.output_ids.shape[-1]) - int(v.num_input_tokens)
            assert acc and len(acc) == rounds and tpot > 0 and math.isfinite(tpot) and 0 < n_out <= 2048, (p, i, m)
            rows.append(dict(schema_version=1, method=m, dataset=ds, temperature=float(cfg["temperature"]),
                sample_idx=i // 2 if ds == "mt-bench" else i, turn_index=i % 2 if ds == "mt-bench" else 0,
                num_output=n_out, time_per_output_token=tpot, tps=1.0 / tpot, decode_rounds=rounds,
                acceptance_lengths=acc, mean_accept=fmean(acc),
                model=Path(str(cfg["model_name_or_path"])).name, draft=Path(str(cfg["draft_name_or_path"])).name,
                max_new_tokens=int(cfg["max_new_tokens"]), max_samples=int(cfg["max_samples"]),
                tree_budget=str(cfg["tree_budget"]), flash_attn=bool(cfg.get("flash_attn")),
                skip_baseline=bool(cfg.get("skip_baseline")), source_sha256=sha))
    out = a.out_dir / (p.stem + ".jsonl")
    out.write_text("".join(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n" for r in rows))
    print(f"{out.name}: methods={methods} responses={len(run['responses'])} rows={len(rows)}")
