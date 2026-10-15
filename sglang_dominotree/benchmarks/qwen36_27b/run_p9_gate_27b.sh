#!/usr/bin/env bash
# P9 losslessness gate for Qwen3.6-27B (or any hybrid Gated-DeltaNet target): must PASS
# before any DominoTree benchmark on this target's numbers are trusted -- see
# ../../../results/serving/qwen36_27b_h200/README.md "Losslessness gate" for why.
#
# Four arms, T=0, token-by-token comparison against AR:
#   ar                plain autoregressive decoding -- the reference every arm is diffed against
#   domino            the chain drafter (reference: known-correct hybrid chain verify)
#   dominotree        the tree, P9 ON (the candidate)
#   dominotree_flat   the tree, P9 OFF (DOMINOTREE_DIAG_FLAT_GDN=1) -- the NEGATIVE CONTROL,
#                     i.e. the exact bug P9 fixes. Diagnostic only; never publish a
#                     throughput number from this arm.
#
# Required env: MODEL, DRAFT_DOMINO.
# Optional env: PY, OUT (parent directory, default ./gate_out), PER_DATASET (prompts per
#   dataset, default 5 -- the paper's gate used 5 x 4 datasets = 20 prompts),
#   GATE_CONFIG=h200 (default) or headline-h100 (archived two-H100/TP=2 flags).
#   MAX_NEW_TOKENS (default 1024 -- long generations needed; see p9_gate_client.py's
#   docstring for why short ones would hide the bug).
#
# Usage:
#   MODEL=/path/Qwen3.6-27B DRAFT_DOMINO=/path/qwen3.6-27B-domino \
#     bash run_p9_gate_27b.sh
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset DOMINOTREE_DIAG_FLAT_GDN DOMINOTREE_STEP_TIMING
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_serverlib_27b.sh"
MODEL="${MODEL:?set MODEL (target model)}"
DRAFT_DOMINO="${DRAFT_DOMINO:?set DRAFT_DOMINO}"
PY="${PY:-python}"
# OUT is a parent directory. Every invocation gets a new, exclusive child.
OUT_ROOT="${OUT:-$HERE/gate_out}"
mkdir -p "$OUT_ROOT"
OUT=$(mktemp -d "$OUT_ROOT/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
PER_DATASET="${PER_DATASET:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

# Fixed headline profile: run inside an allocation with two H100 GPUs.
# H200 remains the default; tree overrides are allowed for its local checks.
GATE_CONFIG="${GATE_CONFIG:-h200}"
case "$GATE_CONFIG" in
  h200)
    CONFIG=(TP=1 GPU="${GPU:-0}" MEMFRAC=0.80 MAXRUN=2 CGMAXBS=2
            DOMINOTREE_BUILDER_GRU_TABLE=1
            DOMINOTREE_TREE_BUDGET="${DOMINOTREE_TREE_BUDGET:-15}"
            DOMINOTREE_NODE_TOPK="${DOMINOTREE_NODE_TOPK:-8}"
            DOMINOTREE_CORR_TOPM="${DOMINOTREE_CORR_TOPM:-128}") ;;
  headline-h100)
    CONFIG=(TP=2 GPU="${GPU:-0,1}" MEMFRAC=0.80 MAXRUN=8 CGMAXBS=8
            DOMINOTREE_BUILDER_GRU_TABLE=0 DOMINOTREE_TREE_BUDGET=15
            DOMINOTREE_NODE_TOPK=8 DOMINOTREE_CORR_TOPM=64) ;;
  *) echo "unknown GATE_CONFIG=$GATE_CONFIG (h200|headline-h100)" >&2; exit 2 ;;
esac
printf '%s\n' "GATE_CONFIG=$GATE_CONFIG" "${CONFIG[@]}" > "$OUT/config.txt"
log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] p9_gate_27b: $*"; }
log "fresh run: $OUT"

run_arm() {
  local server_method="$1" client_label="$2"
  log "================ ARM=$client_label (METHOD=$server_method) ================"
  if ! start_server_27b "$server_method" "$OUT/server_${client_label}.log" \
      "${CONFIG[@]}" MODEL="$MODEL" PY="$PY" DRAFT_DOMINO="$DRAFT_DOMINO"; then
    echo "P9_GATE=ERROR ($client_label failed to boot)" | tee -a "$OUT/gate.log"
    return 2
  fi
  if ! run_client_27b "$PY" -u "$HERE/p9_gate_client.py" --port "$PORT27B" --label "$client_label" \
      --model-path "$MODEL" --bench-dir "$HERE/../bs1" \
      --per-dataset "$PER_DATASET" --max-new-tokens "$MAX_NEW_TOKENS" \
      --out "$OUT/${client_label}.jsonl.part" >> "$OUT/gate.log" 2>&1; then
    echo "P9_GATE=ERROR ($client_label client or server failed)" | tee -a "$OUT/gate.log"
    return 2
  fi
  mv "$OUT/${client_label}.jsonl.part" "$OUT/${client_label}.jsonl" || return 2
  stop_server_27b
}

run_arm ar ar || exit 2
run_arm domino_chain domino || exit 2
run_arm dominotree dominotree || exit 2
# Only this explicit server method enables the diagnostic switch.
run_arm dominotree_flat dominotree_flat || exit 2

log "================ COMPARING ================"
verdict_rc=0
"$PY" "$HERE/p9_gate_compare.py" "$OUT/ar.jsonl" "$OUT/domino.jsonl" \
    "$OUT/dominotree.jsonl" "$OUT/dominotree_flat.jsonl" --per-dataset "$PER_DATASET" \
    2>&1 | tee -a "$OUT/gate.log" || verdict_rc=$?
log "gate output: $OUT/gate.log ; verdict: $(tail -1 "$OUT/gate.log")"
exit "$verdict_rc"
