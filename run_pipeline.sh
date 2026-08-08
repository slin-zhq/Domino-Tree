#!/usr/bin/env bash
# =============================================================================
# DominoTree reproduction pipeline -- ONE command to collect data on your GPU
# and build the paper tables.
#
#   MODEL_PATH=/path/to/Qwen3-4B \
#   DRAFT_PATH=/path/to/Qwen3-4B-Domino-b16 \
#   DOMINO_CODE=/path/to/Domino/code \
#   bash run_pipeline.sh
#
# It collects, on a SINGLE GPU (clean, one job at a time -> fair ratios):
#   1. DominoTree + our lean AR (our harness, benchmark.py, GPU-native builder)
#   2. Official Domino, eager and --use-graph (Domino's own benchmark, warmup-patched
#      in the open -- see the patch block below; Domino's source is never modified)
# then builds the tables with make_latex_table.py. The DFlash/DDTree/CaDDTree
# reference rows ship frozen under results/raw/baseline_ddtree_caddtree/ (collected
# on the authors' GPU; clearly marked in the tables). Re-run any time -- finished
# cells are skipped, so it resumes.
#
# WHY the odd bits are handled for you (so you don't hit the traps we did):
#   * Speedup normalization is "surgical": every method is divided by a lean AR
#     (~66 tok/s on our 5080). Domino's OWN AR is spec_generate(block_size=1) -- its
#     speculative loop with drafting off -- which carries per-token bookkeeping and
#     runs ~23% slower, so we do NOT use it as Domino's denominator. make_latex_table
#     records the AR denominator per row.
#   * AR is temperature-independent (<1%), so we measure it once at T=0 and reuse it.
#   * Domino's released benchmark has no warmup prompt of its own, so this pipeline
#     inserts one into a copy (see the patch block below) -- matching how the published
#     4B baseline was collected, and what make_latex_table's 4B convention assumes.
#   * Single GPU by default: concurrent jobs on a shared node perturb absolute TPS.
#     Set CUDA_VISIBLE_DEVICES to pick the GPU; multi-GPU is intentionally not automated.
# =============================================================================
set -uo pipefail

# ---- config (env-overridable; paper defaults) -------------------------------
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the target model (e.g. Qwen3-4B)}"
DRAFT_PATH="${DRAFT_PATH:?set DRAFT_PATH to the Domino drafter (e.g. Qwen3-4B-Domino-b16)}"
# NB: no apostrophes inside ${VAR:?...} -- bash parses one as an opening quote and the
# whole script then fails to parse. (That bug shipped here once; keep it plain.)
DOMINO_CODE="${DOMINO_CODE:?set DOMINO_CODE to the code/ dir of the released Domino repo}"
PY="${PYTHON:-python}"
OUT="${OUT:-results/raw}"
read -r -a DATASETS <<< "${DATASETS:-gsm8k math500 aime25 humaneval mbpp livecodebench mt-bench alpaca}"
read -r -a TEMPS    <<< "${TEMPS:-0.0 0.5 1.0}"
N="${N:-50}"; MNT="${MNT:-2048}"; BUDGET="${BUDGET:-16}"; CORR_TOPM="${CORR_TOPM:-64}"
WITH_ABLATION="${WITH_ABLATION:-0}"   # 1 => also collect the conditioning ablation (marg + python-builder cond, T=0)
FAST_DOMINO="${FAST_DOMINO:-0}"       # 1 => skip Domino's b=1 AR via a 1-line-patched copy (faster; default runs it unmodified)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- GPU auto-detect (single GPU) -------------------------------------------
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
  export CUDA_VISIBLE_DEVICES=0
  [ "${ngpu:-1}" -gt 1 ] && echo "[pipeline] ${ngpu} GPUs found; using GPU 0 (set CUDA_VISIBLE_DEVICES to change). Single-GPU is intentional for clean timing."
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null | head -1)
ts() { date -Is; }
mkdir -p "${OUT}/dominotree" "${OUT}/domino_official"

# ---- run manifest (provenance: env, versions, policies) ---------------------
"${PY}" - "$OUT" "$MODEL_PATH" "$DRAFT_PATH" "$GPU_NAME" "$N" "$MNT" "$BUDGET" "$CORR_TOPM" <<'PYEOF'
import json, sys, platform, subprocess
out, model, draft, gpu, n, mnt, budget, corrm = sys.argv[1:9]
def ver(m):
    try: return __import__(m).__version__
    except Exception: return None
try: commit = subprocess.check_output(["git","rev-parse","--short","HEAD"], text=True).strip()
except Exception: commit = None
json.dump({
  "config": {"model": model, "draft": draft, "n": int(n), "max_new_tokens": int(mnt),
             "budget": int(budget), "corr_topm": int(corrm)},
  "env": {"gpu": gpu, "python": platform.python_version(),
          "torch": ver("torch"), "transformers": ver("transformers")},
  "policy": {"normalization": "surgical: every method / lean AR; Domino NOT / its own spec-loop AR",
             "ar": "measured at T=0 only, reused at T>0 (temperature-independent <1%)",
             "warmup": "Domino run through a warmup-patched copy of its own benchmark (4B convention); all prompts kept",
             "regime": "single-GPU, one job at a time"},
  "domino_tree_commit": commit,
}, open(f"{out}/run_manifest.json","w"), indent=2)
print(f"[pipeline] wrote {out}/run_manifest.json")
PYEOF
echo "[pipeline] $(ts) GPU=${CUDA_VISIBLE_DEVICES} (${GPU_NAME}) datasets=(${DATASETS[*]}) temps=(${TEMPS[*]}) n=${N}"

# ---- build the patched copy of Domino's benchmark ---------------------------
# The published 4B Domino baseline was collected with a WARMUP prompt inserted into
# Domino's own benchmark, so Domino is warmed exactly like every other method (the
# DDTree/CaDDTree SOP) rather than corrected for afterwards. This is NOT an
# optimization: make_latex_table's 4B convention keeps every Domino prompt (it does
# not drop a cold first row), so collecting without the warmup would fold one cold
# prompt into the published Domino row and understate the baseline.
#
# The patch is applied here, in the open, to a COPY -- Domino's own benchmark.py is
# never modified. Both edits assert, so a silent no-op aborts the run instead of
# quietly producing a differently-measured baseline.
#
# FAST_DOMINO=1 additionally drops Domino's b=1 AR arm. That one IS purely a speed
# optimization: we normalize by our lean AR and never by Domino's, and the b=16
# result is unaffected.
DOMINO_BENCH="benchmark_noar.py"
"${PY}" - "${DOMINO_CODE}/benchmark.py" "${DOMINO_CODE}/${DOMINO_BENCH}" "${FAST_DOMINO}" <<'PATCHEOF'
import sys
src_path, dst_path, fast = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
src = open(src_path).read()

# (1) insert a warmup prompt immediately before the timed loop
warmup = (
    '    # WARMUP (added by run_pipeline.sh): match the DDTree/CaDDTree SOP -- one\n'
    '    # spec_generate on a short "Warmup" prompt so kernels/graph/allocator are hot\n'
    '    # before any timed prompt, instead of dropping a cold row during analysis.\n'
    '    _wu = tokenizer.apply_chat_template([{"role": "user", "content": "Warmup"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)\n'
    '    _wu_ids = tokenizer.encode(_wu, return_tensors="pt").to(target.device)\n'
    '    draft_model.spec_generate(target=target, input_ids=_wu_ids, max_new_tokens=min(args.max_new_tokens, 16), block_size=block_size, stop_token_ids=[tokenizer.eos_token_id], temperature=args.temperature, graph_runner=graph_runner, use_bias=args.use_bias, return_dict=True)\n'
    '    logger.info("warmup done")\n'
)
anchor = "    answers = []\n"
assert anchor in src, (
    f"warmup anchor {anchor!r} not found in {src_path} -- the released Domino "
    "benchmark has changed; re-check this patch before trusting the baseline")
src = src.replace(anchor, warmup + anchor, 1)
assert src.count("warmup done") == 1, "warmup insert did not apply exactly once"

# (2) optionally drop Domino's b=1 AR arm
if fast:
    before = src
    src = src.replace("for bs in [1, block_size]:", "for bs in [block_size]:")
    src = src.replace("for choice, bs in [(choice_b1, 1), (choice_bk, block_size)]:",
                      "for choice, bs in [(choice_bk, block_size)]:")
    assert src != before and "for bs in [1, block_size]" not in src, "b=1 removal did not apply"

open(dst_path, "w").write(src)
print("[pipeline] patched Domino benchmark -> " + dst_path + " (warmup inserted"
      + (", b=1 AR dropped)" if fast else ")"))
PATCHEOF
[ -f "${DOMINO_CODE}/${DOMINO_BENCH}" ] || {
  echo "[FATAL] could not patch Domino's benchmark; aborting rather than collecting a baseline measured differently from the published one."
  exit 1
}

run_ours() {  # $1=methods $2=extra-flags $3=out.jsonl
  local out="$3"
  [ -f "$out" ] && { echo "[skip] ours $out"; return; }
  echo "[$(ts)] OURS $(basename "$out") methods=$1"
  "${PY}" "${HERE}/benchmark.py" --model-name-or-path "${MODEL_PATH}" --draft-name-or-path "${DRAFT_PATH}" \
    --domino-code "${DOMINO_CODE}" --methods "$1" $2 --budgets "${BUDGET}" --corr-topm "${CORR_TOPM}" \
    --node-topk 8 --max-samples "${N}" --max-new-tokens "${MNT}" --out "$out" \
    || { echo "[FAIL] ours $out"; rm -f "$out"; }
}
run_domino() {  # $1=temp $2=mode(graph|eager) $3=dataset
  local gf=""; [ "$2" = graph ] && gf="--use-graph"
  local ans="${OUT}/domino_official/$2_$3_T$1.jsonl"
  [ -f "$ans" ] && { echo "[skip] domino $ans"; return; }
  echo "[$(ts)] DOMINO $3 T$1 $2"
  ( cd "${DOMINO_CODE}" && "${PY}" "${DOMINO_BENCH}" --model-name-or-path "${MODEL_PATH}" \
      --draft-name-or-path "${DRAFT_PATH}" --dataset "$3" --max-samples "${N}" --max-new-tokens "${MNT}" \
      --temperature "$1" --block-size "${BUDGET}" --use-bias ${gf} --attn-implementation sdpa \
      --answer-file "$ans" ) || { echo "[FAIL] domino $ans"; rm -f "$ans"; }
}

# ---- Stage 1: ours (DominoTree + AR). AR only at T=0 (temperature-independent) ----
for T in "${TEMPS[@]}"; do
  methods="dominotree"; [ "$T" = "0.0" ] && methods="ar,dominotree"
  for ds in "${DATASETS[@]}"; do
    # temperature is set inside benchmark.py via its own default; pass it through --temperature
    run_ours "$methods" "--gpu-native-build --temperature $T" "${OUT}/dominotree/${ds}_T${T}.jsonl"
  done
done

# ---- Stage 1b (optional): conditioning ablation (marg + python-builder cond, T=0) ----
if [ "${WITH_ABLATION}" = "1" ]; then
  mkdir -p "${OUT}/conditioning_ablation" "${OUT}/dominotree_python_builder"
  for ds in "${DATASETS[@]}"; do
    run_ours "ar,marg"     "--python-builder --temperature 0.0" "${OUT}/conditioning_ablation/${ds}_T0.0.jsonl"
    run_ours "ar,dominotree" "--python-builder --temperature 0.0" "${OUT}/dominotree_python_builder/${ds}_T0.0.jsonl"
  done
fi

# ---- Stage 2: official Domino (eager + --use-graph), all temps --------------
for T in "${TEMPS[@]}"; do
  for mode in graph eager; do
    for ds in "${DATASETS[@]}"; do run_domino "$T" "$mode" "$ds"; done
  done
done

# ---- Stage 3: build whatever tables the collected data supports -------------
echo "[$(ts)] building tables (surgical AR normalization)"
"${PY}" "${HERE}/make_latex_table.py" --raw-dir "${OUT}" --out-dir "results/tables" --ar-norm surgical \
  --temps "$(IFS=,; echo "${TEMPS[*]}")" || echo "[warn] table build reported issues (see above)"
echo "[$(ts)] done. Tables in results/tables/ ; provenance in ${OUT}/run_manifest.json"
