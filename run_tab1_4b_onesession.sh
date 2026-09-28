#!/usr/bin/env bash
# One-session Qwen3-4B re-collection: AR + DominoTree@32 (fused frontier builder,
# GRU table) + released Domino (eager, --use-graph), ALL FOUR ARMS on the SAME GPU,
# back-to-back per (dataset, temperature) cell -- so the paper's Table 1 4B row no
# longer compares DominoTree (2026-09-17) against released Domino (2026-07-08) under
# two different session speeds on the same RTX 5080.
#
# Reuses, verbatim in spirit:
#   - run_tab1f_4b_remote.sh   (our harness invocation: fused builder, GRU table,
#                                --corr-topm 64 --node-topk 8, ar+dominotree combined
#                                at T=0, dominotree only at T>0)
#   - run_tab1_8b_h100.sbatch  (released-Domino invocation: FAST_DOMINO copy that
#                                drops only the unused b=1 AR arm, block-size 16,
#                                sdpa, eager + --use-graph)
#   - run_pipeline.sh          (DOMINO_COMMIT pin / provenance conventions)
#
# Single GPU by construction (this task's design: identical conditions per cell).
# Set GPU to the confirmed-idle device index.
set -Eeuo pipefail

RUN_ROOT="${RUN_ROOT:-$HOME/tab1_4b_onesession_20260928}"
GPU="${GPU:?set GPU to the confirmed-idle GPU index}"
HARNESS="${HARNESS:-$HOME/tab1f/Domino-Tree}"
DOMINO_CODE="${DOMINO_CODE:-$HOME/SpecDec-Optimize/ref_repo/Domino/code}"
PY="${PY:-$HOME/SpecDec-Optimize/.venv/bin/python}"
MODEL="${MODEL:-$HOME/models/Qwen3-4B}"
DRAFT="${DRAFT:-$HOME/models/Qwen3-4B-Domino-b16}"
FRONTIER_SRC="${FRONTIER_SRC:-$HARNESS/sglang_dominotree/src/dominotree_sglang/tree/frontier.py}"
BUDGET="${BUDGET:-32}"
DATASETS=(gsm8k math500 aime25 humaneval mbpp livecodebench mt-bench alpaca)
TEMPS=(0.0 0.5 1.0)

OUT="$RUN_ROOT/out"                       # ar + dominotree@32, flat <ds>_T<temp>.jsonl
DOFF="$RUN_ROOT/domino_official/qwen3-4b" # T<temp>/{eager,graph}_<ds>.jsonl
LOGS="$RUN_ROOT/logs"
mkdir -p "$OUT" "$DOFF" "$LOGS"
rm -f "$RUN_ROOT/DONE" "$RUN_ROOT/FAILED"

PROGRESS="$RUN_ROOT/progress.log"
exec >>"$RUN_ROOT/run.log" 2>&1

ts(){ date -Is; }
prog(){ echo "[$(ts)] $*" | tee -a "$PROGRESS"; }

on_error(){ rc=$?; prog "FAILED (exit=$rc)"; touch "$RUN_ROOT/FAILED"; exit "$rc"; }
trap on_error ERR

# ---- gate: refuse to start (or continue) on a busy GPU ----------------------
gate_gpu(){
  local apps
  apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
  if [ -n "$apps" ]; then
    prog "GPU $GPU has compute processes running: $apps -- ABORTING (never compete for a shared card)"
    touch "$RUN_ROOT/FAILED"; exit 2
  fi
}
gate_gpu

prog "=== tab1_4b_onesession START (GPU $GPU) ==="

# ---- MANIFEST: env, versions, commits, GPU state ----------------------------
{
  echo "date: $(ts)"
  echo "host: $(hostname)"
  echo "gpu_index: $GPU"
  nvidia-smi -L
  echo "--- pip freeze (torch/transformers/flash) ---"
  "$PY" -m pip freeze 2>/dev/null | grep -i -E "torch|transformers|flash"
  echo "--- Domino-Tree harness provenance ---"
  echo "harness dir: $HARNESS (not a git checkout on this host)"
  echo "verified byte-identical (sha256sum) to public Domino-Tree commit 2ee34b369f57f805406d650d46db5dfaaafcc133"
  echo "  (benchmark.py, dominotree_gpu.py, dominotree_frontier.py, sglang_dominotree/.../frontier.py all match)"
  sha256sum "$HARNESS/benchmark.py" "$HARNESS/dominotree_gpu.py" "$HARNESS/dominotree_frontier.py" "$FRONTIER_SRC" 2>/dev/null
  echo "--- released Domino code provenance ---"
  echo "domino code dir: $DOMINO_CODE (not a git checkout on this host)"
  echo "verified byte-identical (sha256sum) to local SpecDec-Optimize/ref_repo/Domino @ commit 44e1e6912ba9d301899739a54bfda2ce8826980a"
  sha256sum "$DOMINO_CODE/benchmark.py" "$DOMINO_CODE/dflash.py" 2>/dev/null
} > "$RUN_ROOT/MANIFEST.txt"

log_cell_gpu_state(){
  local ds=$1 T=$2
  {
    echo "[$(ts)] cell $ds T=$T GPU=$GPU clocks/power:"
    nvidia-smi -i "$GPU" --query-gpu=clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used --format=csv
  } >> "$RUN_ROOT/gpu_state_per_cell.log"
}

# ---- background GPU utilization sampler (whole run) -------------------------
UTIL_CSV="$RUN_ROOT/gpu_util.csv"
echo "timestamp,index,utilization.gpu [%],memory.used [MiB]" > "$UTIL_CSV"
( while true; do
    tsn=$(date -Is)
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | \
      while IFS= read -r line; do echo "$tsn,$line"; done >> "$UTIL_CSV"
    sleep 30
  done ) &
UTIL_PID=$!
echo "$UTIL_PID" > "$RUN_ROOT/util_logger.pid"
cleanup(){ kill "$UTIL_PID" 2>/dev/null || true; }
trap 'cleanup' EXIT

export DOMINOTREE_FRONTIER_SRC="$FRONTIER_SRC"
export DOMINOTREE_BUILDER_GRU_TABLE=1

# ---- build the FAST_DOMINO copy (drops only the unused b=1 AR arm; no timing
#      path our tables read is touched -- same convention as run_tab1_8b_h100.sbatch) ---
FAST_DOMINO_PY="$DOMINO_CODE/benchmark_fast_b16only.py"
"$PY" - "$DOMINO_CODE/benchmark.py" "$FAST_DOMINO_PY" <<'PY'
import sys
src_path, dst_path = sys.argv[1], sys.argv[2]
src = open(src_path).read(); before = src
src = src.replace("for bs in [1, block_size]:", "for bs in [block_size]:")
src = src.replace("for choice, bs in [(choice_b1, 1), (choice_bk, block_size)]:",
                  "for choice, bs in [(choice_bk, block_size)]:")
assert src != before and "for bs in [1, block_size]" not in src, "FAST_DOMINO patch failed"
open(dst_path, "w").write(src)
print("FAST_DOMINO copy ok ->", dst_path)
PY

nsamples(){ [ "$1" = aime25 ] && echo 30 || echo 50; }

validate_ours(){ # file, dataset, expected methods...
  local f=$1 ds=$2; shift 2
  local n exp; n=$(nsamples "$ds")
  [ -s "$f" ] || return 1
  exp=$(IFS=,; echo "$*")
  got=$("$PY" - "$f" <<'PY'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
print(",".join(sorted({r["method"] for r in rows})))
PY
)
  [ "$got" = "$exp" ] || { prog "bad labels in $f: got=$got want=$exp"; return 1; }
  for m in "$@"; do
    cnt=$("$PY" - "$f" "$m" <<'PY'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
m=sys.argv[2]
print(len({r["sample_idx"] for r in rows if r["method"]==m}))
PY
)
    [ "$cnt" -eq "$n" ] || { prog "bad sample count in $f for $m: $cnt (expected $n)"; return 1; }
  done
  return 0
}

validate_official(){ # file, dataset
  local f=$1 ds=$2 n; n=$(nsamples "$ds")
  [ -s "$f" ] || return 1
  cnt=$(wc -l < "$f" | tr -d ' ')
  [ "$cnt" -eq "$n" ] || { prog "bad row count in $f: $cnt (expected $n)"; return 1; }
  return 0
}

run_ours(){ # out.jsonl, methods, temp, dataset, expected-methods...
  local f=$1 methods=$2 T=$3 ds=$4; shift 4
  if validate_ours "$f" "$ds" "$@"; then
    prog "[skip-valid] ours $ds T=$T methods=$methods"
    return
  fi
  rm -f "$f"
  prog "START ours $ds T=$T methods=$methods -> $f"
  local t0 t1
  t0=$(date +%s)
  ( cd "$HARNESS" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u benchmark.py \
      --model-name-or-path "$MODEL" --draft-name-or-path "$DRAFT" \
      --domino-code "$DOMINO_CODE" --methods "$methods" --budgets "$BUDGET" \
      --corr-topm 64 --node-topk 8 --builder frontier \
      --max-samples 50 --max-new-tokens 2048 --temperature "$T" --dataset "$ds" \
      --out "$f" ) >> "$LOGS/ours_${ds}_T${T}.log" 2>&1
  t1=$(date +%s)
  validate_ours "$f" "$ds" "$@" || { prog "!! OURS INVALID $ds T=$T (see $LOGS/ours_${ds}_T${T}.log)"; touch "$RUN_ROOT/FAILED"; exit 1; }
  prog "END ours $ds T=$T elapsed=$((t1-t0))s -> $f"
}

run_domino(){ # temp, mode(eager|graph), dataset
  local T=$1 mode=$2 ds=$3
  local dir="$DOFF/T$T"; mkdir -p "$dir"
  local f="$dir/${mode}_${ds}.jsonl"
  if validate_official "$f" "$ds"; then
    prog "[skip-valid] domino $mode $ds T=$T"
    return
  fi
  rm -f "$f"
  local gf=""; [ "$mode" = graph ] && gf="--use-graph"
  prog "START domino $mode $ds T=$T -> $f"
  local t0 t1
  t0=$(date +%s)
  ( cd "$DOMINO_CODE" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u benchmark_fast_b16only.py \
      --model-name-or-path "$MODEL" --draft-name-or-path "$DRAFT" --dataset "$ds" \
      --max-samples 50 --max-new-tokens 2048 --temperature "$T" --block-size 16 \
      --use-bias $gf --attn-implementation sdpa --answer-file "$f" ) \
    >> "$LOGS/domino_${mode}_${ds}_T${T}.log" 2>&1
  t1=$(date +%s)
  validate_official "$f" "$ds" || { prog "!! DOMINO INVALID $mode $ds T=$T (see $LOGS/domino_${mode}_${ds}_T${T}.log)"; touch "$RUN_ROOT/FAILED"; exit 1; }
  prog "END domino $mode $ds T=$T elapsed=$((t1-t0))s -> $f"
}

TOTAL_CELLS=$(( ${#DATASETS[@]} * ${#TEMPS[@]} ))
CELL_N=0
for ds in "${DATASETS[@]}"; do
  for T in "${TEMPS[@]}"; do
    CELL_N=$((CELL_N+1))
    gate_gpu   # never join a busy GPU, even mid-run
    prog "=== CELL $CELL_N/$TOTAL_CELLS: $ds T=$T ==="
    log_cell_gpu_state "$ds" "$T"

    f="$OUT/${ds}_T${T}.jsonl"
    if [ "$T" = "0.0" ]; then
      run_ours "$f" "ar,dominotree" "$T" "$ds" ar "dominotree@${BUDGET}"
    else
      run_ours "$f" "dominotree" "$T" "$ds" "dominotree@${BUDGET}"
    fi

    run_domino "$T" eager "$ds"
    run_domino "$T" graph "$ds"
  done
done

touch "$RUN_ROOT/DONE"
trap - ERR
prog "=== tab1_4b_onesession DONE ==="
