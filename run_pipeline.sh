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
#   2. Official Domino, eager and --use-graph (Domino's own benchmark, unmodified)
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
#   * Domino's benchmark has no warmup; make_latex_table warmup-excludes its 1st prompt.
#   * Single GPU by default: concurrent jobs on a shared node perturb absolute TPS.
#     Set CUDA_VISIBLE_DEVICES to pick the GPU; multi-GPU is intentionally not automated.
# =============================================================================
set -uo pipefail

# ---- config (env-overridable; paper defaults) -------------------------------
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the target model (e.g. Qwen3-4B)}"
DRAFT_PATH="${DRAFT_PATH:?set DRAFT_PATH to the Domino drafter (e.g. Qwen3-4B-Domino-b16)}"
DOMINO_CODE="${DOMINO_CODE:?set DOMINO_CODE to the released Domino repo's code/ dir}"
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
             "warmup": "Domino first prompt warmup-excluded (its benchmark has no warmup)",
             "regime": "single-GPU, one job at a time"},
  "domino_tree_commit": commit,
}, open(f"{out}/run_manifest.json","w"), indent=2)
print(f"[pipeline] wrote {out}/run_manifest.json")
PYEOF
echo "[pipeline] $(ts) GPU=${CUDA_VISIBLE_DEVICES} (${GPU_NAME}) datasets=(${DATASETS[*]}) temps=(${TEMPS[*]}) n=${N}"

# ---- optional: build a b=1-skipping copy of Domino's benchmark (--fast) -----
DOMINO_BENCH="benchmark.py"
if [ "${FAST_DOMINO}" = "1" ]; then
  DOMINO_BENCH="benchmark_noar.py"
  cp "${DOMINO_CODE}/benchmark.py" "${DOMINO_CODE}/${DOMINO_BENCH}"
  sed -i.bak "s/for bs in \[1, block_size\]:/for bs in [block_size]:/; s/for choice, bs in \[(choice_b1, 1), (choice_bk, block_size)\]:/for choice, bs in [(choice_bk, block_size)]:/" "${DOMINO_CODE}/${DOMINO_BENCH}"
  echo "[pipeline] FAST_DOMINO: using a b=1-skipping copy of Domino's benchmark (${DOMINO_BENCH}); the b=16 result is unchanged."
fi

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
