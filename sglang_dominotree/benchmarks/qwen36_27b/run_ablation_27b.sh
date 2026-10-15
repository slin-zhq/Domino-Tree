#!/usr/bin/env bash
# H200/TP=1 reproduction of nano5_collect stage 1/1b + nano5_followup A/B.
# Required: MODEL, DRAFT_DOMINO. Optional: PY, GPU, OUT (parent directory).
# Fresh output follows results/serving/qwen36_27b_h200/{ablations,losslessness_gate}.
# Selection: highest rounded aggregate tokens/s among P9_GATE=PASS ONLY;
# coordinate descent (budget, then k/M, then lower budgets), original tie order.
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset DOMINOTREE_DIAG_FLAT_GDN DOMINOTREE_STEP_TIMING
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_serverlib_27b.sh"
MODEL="${MODEL:?set MODEL}"
DRAFT_DOMINO="${DRAFT_DOMINO:?set DRAFT_DOMINO}"
PY="${PY:-python}"
OUT_ROOT="${OUT:-$HERE/ablation_out}"
mkdir -p "$OUT_ROOT"
OUT=$(mktemp -d "$OUT_ROOT/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
LADDER="$OUT/ablations/budget_ladder"
GRID="$OUT/ablations/node_topk_topm_grid"
TIMING="$OUT/ablations/timing"
REF="$OUT/losslessness_gate/gate_ref"
mkdir -p "$LADDER" "$GRID" "$TIMING" "$REF" "$OUT/logs"
CONFIG=(MODEL="$MODEL" DRAFT_DOMINO="$DRAFT_DOMINO" PY="$PY" GPU="${GPU:-0}"
        TP=1 MEMFRAC=0.80 MAXRUN=2 CGMAXBS=2 DOMINOTREE_BUILDER_GRU_TABLE=1)
printf '%s\n' "${CONFIG[@]}" > "$OUT/config.txt"
log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] ablation_27b: $*"; }
log "fresh run: $OUT"

collect() {  # method label output budget k M [timing interval] [prompts/dataset]
  local method="$1" label="$2" file="$3" b="$4" k="$5" m="$6"
  start_server_27b "$method" "$OUT/logs/server_$label.log" "${CONFIG[@]}" \
    DOMINOTREE_TREE_BUDGET="$b" DOMINOTREE_NODE_TOPK="$k" DOMINOTREE_CORR_TOPM="$m" \
    DOMINOTREE_STEP_TIMING="${7:-0}" || return 2
  run_client_27b "$PY" -u "$HERE/p9_gate_client.py" --port "$PORT27B" --label "$label" \
    --model-path "$MODEL" --bench-dir "$HERE/../bs1" --per-dataset "${8:-5}" \
    --max-new-tokens 1024 --out "$file.part" > "$OUT/logs/client_$label.log" 2>&1 || return 2
  mv "$file.part" "$file" || return 2
  stop_server_27b
}

# References AND negative control are freshly measured on this hardware/config.
collect ar ar "$REF/ar.jsonl" 15 8 64
collect domino_chain domino "$REF/domino.jsonl" 15 8 64
collect dominotree_flat dominotree_flat "$REF/dominotree_flat.jsonl" 15 8 64

measure() {  # budget k M directory label; sets VERDICT, TPS, TAU
  local b="$1" k="$2" m="$3" dir="$4" label="$5" rc=0
  collect dominotree "$label" "$dir/$label.jsonl" "$b" "$k" "$m"
  "$PY" "$HERE/p9_gate_compare.py" "$REF/ar.jsonl" "$REF/domino.jsonl" \
    "$dir/$label.jsonl" "$REF/dominotree_flat.jsonl" > "$dir/$label.verdict.txt" || rc=$?
  if [ "$rc" -gt 1 ]; then cat "$dir/$label.verdict.txt" >&2; exit "$rc"; fi
  VERDICT=$(tail -1 "$dir/$label.verdict.txt")
  local metrics
  metrics=$("$PY" - "$dir/$label.jsonl" <<'PY'
import json, math, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
assert rows and all(math.isfinite(r['wall_s']) and r['wall_s'] > 0 for r in rows)
print(round(sum(r['completion_tokens'] for r in rows) / sum(r['wall_s'] for r in rows), 1),
      round(sum(r['tau'] for r in rows) / len(rows), 2))
PY
  )
  read -r TPS TAU <<< "$metrics"
  log "$label: $VERDICT tok/s=$TPS tau=$TAU"
}

is_better() {
  [ "$VERDICT" = P9_GATE=PASS ] && "$PY" -c \
    'import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)' "$TPS" "$BEST"
}

# Stage 1: original ladder at k=8, M=64, GRU table on.
BEST=0; CHOSEN=""
for B in 15 32 64; do
  measure "$B" 8 64 "$LADDER" "dominotree_b$B"
  echo "$B $TPS $VERDICT" >> "$LADDER/summary_default_kM.txt"
  if is_better; then BEST="$TPS"; CHOSEN="$B"; fi
done
if [ -z "$CHOSEN" ]; then
  log 'No budget passed the gate; no configuration selected.'
  exit 1
fi
echo "$CHOSEN" > "$OUT/chosen_budget.txt"

# Stage 1b: 3 x 4 grid; (8,64) reuses the selected ladder observation.
# M=0 means full vocabulary, not zero corrections.
K_CHOSEN=8; M_CHOSEN=64
for K in 4 8 16; do
  for M in 32 64 128 0; do
    [ "$K" = 8 ] && [ "$M" = 64 ] && continue
    measure "$CHOSEN" "$K" "$M" "$GRID" "dominotree_b${CHOSEN}_k${K}_m${M}"
    echo "$CHOSEN $K $M $TPS $TAU $VERDICT" >> "$GRID/grid_summary.txt"
    if is_better; then BEST="$TPS"; K_CHOSEN="$K"; M_CHOSEN="$M"; fi
  done
done
echo "$CHOSEN $K_CHOSEN $M_CHOSEN" > "$OUT/chosen_config.txt"

# Followup A: lower budgets at the selected k/M; retain the PASS-qualified baseline.
FINAL="$CHOSEN"
for B in 8 12; do
  measure "$B" "$K_CHOSEN" "$M_CHOSEN" "$LADDER" "dominotree_b${B}_k${K_CHOSEN}_m${M_CHOSEN}"
  echo "$B $K_CHOSEN $M_CHOSEN $TPS $VERDICT" >> "$LADDER/summary_followup_k${K_CHOSEN}m${M_CHOSEN}.txt"
  if is_better; then BEST="$TPS"; FINAL="$B"; fi
done
echo "$FINAL $K_CHOSEN $M_CHOSEN" > "$OUT/final_config.txt"
log "FINAL CONFIG: budget=$FINAL k=$K_CHOSEN M=$M_CHOSEN tok/s=$BEST (PASS only)"

# Followup B: synchronized per-phase timings, excluded from selection/throughput.
for B in 8 15 32 64; do
  L="timing_b$B"
  collect dominotree "$L" "$TIMING/$L.jsonl" "$B" "$K_CHOSEN" "$M_CHOSEN" 100 2
  if ! grep 'step timing over' "$OUT/logs/server_$L.log" > "$TIMING/$L.steps.txt"; then
    log "ERROR: no per-phase timing reports for $L"; exit 2
  fi
done
log "complete: $OUT/final_config.txt; phase timings: $TIMING"
