#!/usr/bin/env bash
# Refresh the three builder/top-M ablations on the SHIPPED fused frontier builder.
# Replaces paper Tables 3, 4 and 11, all of which were collected on builders that
# no longer ship (Python reference / GPU-native heap).
#
# BUDGETS 16 AND 32, both emitted from ONE process so they are same-session.
#   32 is the SHIPPED operating point for Qwen3-4B (paper Table 1), so the builder
#   tables must be measured there. 16 is kept because the heap builder's cost is
#   LINEAR in budget while the frontier builder's tracks tree DEPTH -- so the gap
#   WIDENS with budget (6.4x at B=16 -> 9.1x at B=32 on archived data). Reporting
#   both turns a formatting fix into evidence for the cost model.
#
# JOB B (Tables 3+4+10): ar,chain,marg,dominotree at budgets 16,32 under BOTH builders
#                     (frontier = shipped, python = reference), 8 datasets, 3 temps.
#                     Gives per-round stage times AND the DDTree-analogue sign-flip,
#                     in-harness and same-session (stronger than the cross-harness
#                     DDTree comparison the old Table 4 used).
# JOB A (Table 11):   top-M in {32,64,128} at budget 16, 8 datasets, T=0, fused only.
#                     `chain` is re-collected inside EVERY M pass so each Delta% vs
#                     chain is a within-session pairing -- never across runs.
set -Eeuo pipefail

RUN_ROOT="$HOME/tab_refresh"
OUT="$RUN_ROOT/out"
HARNESS="$HOME/tab1f/Domino-Tree"
PY="$HOME/SpecDec-Optimize/.venv/bin/python"
GPU="${GPU:-1}"
FRONTIER="$HARNESS/sglang_dominotree/src/dominotree_sglang/tree/frontier.py"
DATASETS=(gsm8k math500 aime25 humaneval mbpp livecodebench mt-bench alpaca)

mkdir -p "$OUT"
rm -f "$RUN_ROOT/DONE" "$RUN_ROOT/FAILED"
exec >>"$RUN_ROOT/run.log" 2>&1

on_error(){ rc=$?; echo "[$(date -Is)] FAILED (exit=$rc)"; touch "$RUN_ROOT/FAILED"; exit "$rc"; }
trap on_error ERR

apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk '$1 ~ /^[0-9]+$/')
if [ -n "$apps" ]; then echo "[$(date -Is)] GPU $GPU busy: $apps"; exit 2; fi

echo "[$(date -Is)] === tab_refresh START (GPU $GPU) ==="

validate(){ # file, expected methods...
  local f=$1; shift
  local got exp n base
  exp=$(IFS=,; echo "$*")
  got=$(jq -s -r '[.[]|select(type=="object")|.method]|unique|sort|join(",")' "$f")
  [ "$got" = "$exp" ] || { echo "[$(date -Is)] bad labels in $f: got=$got want=$exp"; return 1; }
  base=""
  for m in "$@"; do
    n=$(jq -s -r --arg m "$m" '[.[]|select(type=="object")|select(.method==$m)|.sample_idx]|unique|length' "$f")
    [ "$n" -ge 25 ] || { echo "[$(date -Is)] short cell $f/$m: $n"; return 1; }
    if [ -z "$base" ]; then base=$n; elif [ "$n" -ne "$base" ]; then
      echo "[$(date -Is)] UNPAIRED $f: $m=$n others=$base"; return 1; fi
  done
  echo "[$(date -Is)] ok $f ($base samples/arm)"
}

run_case(){ # file, methods, temp, dataset, builder(frontier|python), topm, expected...
  local f=$1 methods=$2 temp=$3 ds=$4 builder=$5 topm=$6; shift 6
  if [ -s "$f" ] && validate "$f" "$@"; then echo "[$(date -Is)] [skip] $f"; return; fi
  rm -f "$f"
  echo "[$(date -Is)] START $(basename "$f") methods=$methods T=$temp builder=$builder M=$topm"
  local bargs
  if [ "$builder" = frontier ]; then bargs="--builder frontier"; else bargs="--builder heap --python-builder"; fi
  (
    cd "$HARNESS"
    if [ "$builder" = frontier ]; then
      export DOMINOTREE_FRONTIER_SRC="$FRONTIER" DOMINOTREE_BUILDER_GRU_TABLE=1
      unset DOMINOTREE_BUILDER_FUSION DOMINOTREE_BUILDER_FUSED_DEPTH
    else
      unset DOMINOTREE_FRONTIER_SRC DOMINOTREE_BUILDER_GRU_TABLE
    fi
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u benchmark.py \
      --model-name-or-path "$HOME/models/Qwen3-4B" \
      --draft-name-or-path "$HOME/models/Qwen3-4B-Domino-b16" \
      --domino-code "$HOME/SpecDec-Optimize/ref_repo/Domino/code" \
      --methods "$methods" --budgets 16,32,64 --corr-topm "$topm" --node-topk 8 \
      $bargs --max-samples 50 --max-new-tokens 2048 \
      --temperature "$temp" --dataset "$ds" --out "$f"
  )
  validate "$f" "$@"
}

# ---- JOB B: builder comparison (Tables 3 + 4) -------------------------------
# AR is 50 of every 85 GPU-minutes (measured on tab1f_4b) and is independent of BOTH
# the builder and the temperature, so it is collected exactly once, in the frontier
# T=0 pass. Nothing here needs it anywhere else: Tables 3/4/10/11 all compare against
# `chain` and `marg`, which are re-collected inside EVERY pass, so every delta stays a
# within-session pairing. Skipping the 5 redundant AR passes saves ~4 h.
for builder in frontier python; do
  for temp in 0.0 0.5 1.0; do
    if [ "$builder" = frontier ] && [ "$temp" = 0.0 ]; then
      M_LIST=ar,chain,marg,dominotree
      EXPECT=(ar chain dominotree@16 dominotree@32 dominotree@64 marg@16 marg@32 marg@64)
    else
      M_LIST=chain,marg,dominotree
      EXPECT=(chain dominotree@16 dominotree@32 dominotree@64 marg@16 marg@32 marg@64)
    fi
    for ds in "${DATASETS[@]}"; do
      run_case "$OUT/builder_${builder}_${ds}_T${temp}.jsonl" "$M_LIST" \
        "$temp" "$ds" "$builder" 64 "${EXPECT[@]}"
    done
  done
done
echo "[$(date -Is)] === JOB B done ==="

# ---- JOB A: top-M sweep on the shipped builder (Table 11) -------------------
for M in 32 64 128; do
  for ds in "${DATASETS[@]}"; do
    run_case "$OUT/topm_M${M}_${ds}_T0.0.jsonl" chain,dominotree \
      0.0 "$ds" frontier "$M" chain dominotree@16 dominotree@32 dominotree@64
  done
done
echo "[$(date -Is)] === JOB A done ==="

touch "$RUN_ROOT/DONE"
trap - ERR
echo "[$(date -Is)] === tab_refresh DONE ==="
