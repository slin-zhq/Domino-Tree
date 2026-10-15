#!/usr/bin/env bash
# Launch ONE SGLang server for the Qwen3.6-27B benchmarks (a hybrid Gated-DeltaNet +
# attention target -- needs P9, `--linear-attn-backend triton`, and its own flags,
# hence a launcher separate from ../launch_server.sh, which is for the plain-attention
# Qwen3-4B/8B targets).
#
# All five methods share IDENTICAL serving flags -- only the speculative algorithm and
# the draft model differ. Do not add a flag for one method only.
#
# Required env:
#   METHOD   ar | dflash | domino_chain | dominotree | mtp
#   MODEL    target model path or HF id (Qwen/Qwen3.6-27B)
#   DRAFT_DOMINO / DRAFT_DFLASH   draft model for the chosen METHOD (mtp and ar need none)
#
# Optional env:
#   PORT           server port                      default: 20000 + PID % 20000 (per-run;
#                  a FIXED port shared by two concurrent runs on one host silently
#                  contaminates both -- see PROVENANCE.txt's incident record. Pass your
#                  own PORT only if you can guarantee no other run on this host uses it.)
#   PY             python executable                default python
#   GPU            CUDA_VISIBLE_DEVICES              default 0
#   MEMFRAC        --mem-fraction-static             default 0.80 (bs=1/HELMET value; use
#                  0.85 for concurrency -- see run_conc_27b.sh)
#   MAXRUN         --max-running-requests            default 2   (bs=1/HELMET value; the
#                  admission cap -- see run_conc_27b.sh for the concurrency sweep)
#   CGMAXBS        decode cuda-graph max batch       default 2   (8 for concurrency)
#   DOMINOTREE_TREE_BUDGET / _NODE_TOPK / _CORR_TOPM   DominoTree config, default 15/8/128
#                  (this device's measured optimum -- see ../../../results/serving/
#                  qwen36_27b_h200/ablations/)
#   DOMINOTREE_BUILDER_GRU_TABLE  opt-in GRU input table    default 1
#   TP             tensor parallelism (default 1; use GRU table 0 at TP>1)
#   METHOD=dominotree_flat is diagnostic ONLY, explicitly labeled by gate callers.
#
# THE ADMISSION CAP (MAXRUN), MEMFRAC, and everything else about why these flags matter
# is explained in ../launch_server.sh and ../concurrency/README.md; that reasoning is
# unchanged here. What IS specific to this target:
#
#   --linear-attn-backend triton   the Gated-DeltaNet layers' recurrent-state kernel.
#   P9 (tree verify on hybrid GDN targets, this branch's sglang_dominotree code) is
#   installed automatically whenever SGLANG_PLUGINS=dominotree and the target is
#   detected as hybrid -- there is no separate flag to turn it on. There IS a env var
#   to turn it OFF for the losslessness gate's negative control: see
#   run_p9_gate_27b.sh / DOMINOTREE_DIAG_FLAT_GDN. Never set that for a benchmark run.
#
# PID-scoped kill + refuse-if-busy: unlike ../launch_server.sh's "pkill by port
# pattern" (safe only because that script always fully owns its port), this script
# tracks its OWN server's PID and refuses to launch if something else already answers
# on PORT -- the exact fix for the port-collision incident recorded in
# ../../../results/serving/qwen36_27b_h200/PROVENANCE.txt (two jobs shared port 31300,
# and each one's pkill-by-port killed and each health check accepted the OTHER job's
# server, silently contaminating ~2h of data). Callers (run_*_27b.sh) source
# _serverlib_27b.sh for PID/identity checks and EXIT/INT/TERM cleanup.
#
# Usage (foreground; run_*_27b.sh backgrounds it for you with the right PORT):
#   METHOD=dominotree MODEL=/path/Qwen3.6-27B DRAFT_DOMINO=/path/qwen3.6-27B-domino \
#     bash launch_server_27b.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${METHOD:?set METHOD (ar|dflash|domino_chain|dominotree|mtp)}"
MODEL="${MODEL:?set MODEL (target model path or HF id)}"
PY="${PY:-python}"
GPU="${GPU:-0}"
TP="${TP:-1}"
SERVED_MODEL_NAME27B="${SERVED_MODEL_NAME27B:-qwen27b-$("$PY" -c 'import uuid; print(uuid.uuid4().hex)')}"
PORT="${PORT:-$((20000 + $$ % 20000))}"
MEMFRAC="${MEMFRAC:-0.80}"
MAXRUN="${MAXRUN:-2}"
CGMAXBS="${CGMAXBS:-2}"
DOMINOTREE_TREE_BUDGET="${DOMINOTREE_TREE_BUDGET:-15}"
DOMINOTREE_NODE_TOPK="${DOMINOTREE_NODE_TOPK:-8}"
DOMINOTREE_CORR_TOPM="${DOMINOTREE_CORR_TOPM:-128}"
DOMINOTREE_BUILDER_GRU_TABLE="${DOMINOTREE_BUILDER_GRU_TABLE:-1}"

# An inherited diagnostic switch must never contaminate a normal/positive run.
unset DOMINOTREE_DIAG_FLAT_GDN SGLANG_PLUGINS
if [ "$METHOD" = dominotree_flat ]; then
  export DOMINOTREE_DIAG_FLAT_GDN=1
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] launch_server_27b[$METHOD]: $*"; }

if curl --connect-timeout 2 --max-time 5 -sS -o /dev/null "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  log "!! port $PORT already answers -- refusing to launch (see the port-collision"
  log "!! incident this refusal exists to prevent, in the qwen36_27b_h200 PROVENANCE.txt)."
  exit 1
fi

COMMON=(--model-path "$MODEL" --tp-size "$TP" --served-model-name "$SERVED_MODEL_NAME27B" --trust-remote-code
        --attention-backend flashinfer --page-size 1 --linear-attn-backend triton
        --mem-fraction-static "$MEMFRAC" --max-running-requests "$MAXRUN"
        --host 127.0.0.1 --port "$PORT")
if [ "$TP" -gt 1 ]; then
  COMMON+=(--cuda-graph-max-bs-decode "$CGMAXBS" --disable-custom-all-reduce)
else
  COMMON+=(--cuda-graph-max-bs "$CGMAXBS")
fi
log "served model identity: $SERVED_MODEL_NAME27B"
printf 'server flags: '; printf '%q ' "${COMMON[@]}"; printf '\n'


case "$METHOD" in
  ar)
    exec "$PY" -m sglang.launch_server "${COMMON[@]}" ;;
  dflash)
    exec "$PY" -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm DFLASH \
      --speculative-draft-model-path "${DRAFT_DFLASH:?set DRAFT_DFLASH}" \
      --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 16 ;;
  mtp)
    # The model card's own recommended NEXTN config for this checkpoint's MTP head --
    # no draft checkpoint needed, it is served from the target's own weights.
    exec "$PY" -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm NEXTN --speculative-num-steps 3 \
      --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 ;;
  domino_chain)
    export SGLANG_PLUGINS=dominotree
    exec "$PY" -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm DOMINO \
      --speculative-draft-model-path "${DRAFT_DOMINO:?set DRAFT_DOMINO}" \
      --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 16 ;;
  dominotree|dominotree_flat)
    export SGLANG_PLUGINS=dominotree
    export DOMINOTREE_BUILDER_GRU_TABLE DOMINOTREE_TREE_BUDGET DOMINOTREE_NODE_TOPK DOMINOTREE_CORR_TOPM
    exec "$PY" -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm DOMINOTREE \
      --speculative-draft-model-path "${DRAFT_DOMINO:?set DRAFT_DOMINO}" \
      --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 16 ;;
  *)
    echo "unknown METHOD=$METHOD" >&2; exit 2 ;;
esac
