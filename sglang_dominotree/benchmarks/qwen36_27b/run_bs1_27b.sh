#!/usr/bin/env bash
# Qwen3.6-27B bs=1 serving sweep: 5 methods x 8 datasets x {T=0.0, T=1.0}.
# Reuses ../bs1/bench_bs1.py unmodified (it only needs --port/--model-path, so it works
# for any target); only the launcher (launch_server_27b.sh, via _serverlib_27b.sh) and
# this orchestrator are specific to this target.
#
# Required env:
#   MODEL          target model path or HF id (Qwen/Qwen3.6-27B)
#   DRAFT_DOMINO   Domino draft (domino_chain + dominotree)
#   DRAFT_DFLASH   DFlash draft
# Optional env:
#   PY, OUT (default ./out/<basename MODEL>), DATASETS, TEMPS (default 0.0,1.0),
#   NSAMPLES (default 50), MAXNEW (default 2048), METHODS (default
#   "dominotree domino_chain dflash mtp ar")
#
# Usage (detach; monitor with tail -F orch_bs1_27b.log):
#   MODEL=/path/Qwen3.6-27B DRAFT_DOMINO=/path/qwen3.6-27B-domino \
#   DRAFT_DFLASH=/path/Qwen3.6-27B-DFlash \
#     setsid bash run_bs1_27b.sh > orch_bs1_27b.log 2>&1 < /dev/null &
set -uo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset DOMINOTREE_DIAG_FLAT_GDN
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_serverlib_27b.sh"
MODEL="${MODEL:?set MODEL (target model)}"
PY="${PY:-python}"
LABEL="$(basename "$MODEL")"
OUT="${OUT:-$HERE/out/$LABEL}"
DATASETS="${DATASETS:-gsm8k,math500,aime25,humaneval,mbpp,livecodebench,mt-bench,alpaca}"
TEMPS="${TEMPS:-0.0,1.0}"
NSAMPLES="${NSAMPLES:-50}"
MAXNEW="${MAXNEW:-2048}"
METHODS="${METHODS:-dominotree domino_chain dflash mtp ar}"

log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] bs1_27b: $*"; }
mkdir -p "$OUT/logs"
log "MODEL=$MODEL DATASETS=$DATASETS TEMPS=$TEMPS N=$NSAMPLES MAXNEW=$MAXNEW"

for M in $METHODS; do
  log "================ METHOD=$M ================"
  ENVS=()
  if [ "$M" = domino_chain ] || [ "$M" = dominotree ]; then
    ENVS+=(SGLANG_PLUGINS=dominotree DRAFT_DOMINO="${DRAFT_DOMINO:-}")
  fi
  [ "$M" = dflash ] && ENVS+=(DRAFT_DFLASH="${DRAFT_DFLASH:-}")
  start_server_27b "$M" "$OUT/logs/server_${M}.log" "${ENVS[@]}" MODEL="$MODEL" PY="$PY" || continue

  for DS in ${DATASETS//,/ }; do
    for T in ${TEMPS//,/ }; do
      OUTF="$OUT/$M/${DS}_T${T}.jsonl"
      mkdir -p "$(dirname "$OUTF")"
      if [ -s "$OUTF" ]; then log "skip $M/$DS/T$T (exists)"; continue; fi
      log "run $M $DS T=$T -> $OUTF"
      run_client_27b env -u HF_DATASETS_OFFLINE -u HF_HUB_OFFLINE "$PY" -u "$HERE/../bs1/bench_bs1.py" \
          --host 127.0.0.1 --port "$PORT27B" --method "$M" \
          --dataset "$DS" --temperature "$T" --max-samples "$NSAMPLES" \
          --max-new-tokens "$MAXNEW" --model-path "$MODEL" \
          --request-timeout 1800 --out "$OUTF.part" \
          >> "$OUT/logs/run_${M}.log" 2>&1 \
        && mv "$OUTF.part" "$OUTF" \
        || log "!! $M $DS T=$T exited nonzero (see logs/run_${M}.log)"
    done
  done
  stop_server_27b
done
log "================ ALL DONE ================"
