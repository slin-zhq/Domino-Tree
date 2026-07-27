#!/usr/bin/env bash
# Launch one SGLang server for the HELMET long-context benchmark, for any of the 5
# methods. bs=1 long-context KV recipe: large pool (MEMFRAC high) so a long prompt
# fits, tiny batch caps (bs=1 needs no batch capacity).
#
# All methods share IDENTICAL serving flags except the speculative algorithm + draft
# model, so any tau/tps difference reflects the METHOD, not the launch. DominoTree +
# Domino require the plugin (SGLANG_PLUGINS=dominotree); DFLASH/EAGLE3 are native.
#
# Fully parameterised — no site-specific paths. Set these env vars:
#   METHOD   ar|dflash|eagle3|domino_chain|dominotree     (required)
#   PORT     server port                                  (required)
#   MODEL    path/name of the TARGET model                (required)
#   TP       tensor-parallel size (1, or 2 for a big model on 2 GPUs)   default 1
#   DRAFT    path/name of the draft model  (required for spec methods; ignored for ar)
#   PY       python executable             default: python
#   MEMFRAC  --mem-fraction-static         default 0.85  (large KV pool for long prompts)
#   MAXRUN   --max-running-requests        default 4     (bs=1: small on purpose)
#   CGMAXBS  decode cuda-graph max bs      default 4
#   GPUS     CUDA_VISIBLE_DEVICES          default 0 (TP=1) / 0,1 (TP=2)
#   EXTRA_ARGS  any extra sglang flags (string, appended)
#
# Example (DominoTree, 8B, TP=2):
#   METHOD=dominotree PORT=31600 TP=2 MODEL=Qwen/Qwen3-8B DRAFT=./Qwen3-8B-Domino-b16 \
#     setsid bash launch_helmet_server.sh > server.log 2>&1 < /dev/null &
set -u
METHOD="${METHOD:?set METHOD (ar|dflash|eagle3|domino_chain|dominotree)}"
PORT="${PORT:?set PORT}"
MODEL="${MODEL:?set MODEL (target model path/name)}"
TP="${TP:-1}"
DRAFT="${DRAFT:-}"
PY="${PY:-python}"
MEMFRAC="${MEMFRAC:-0.85}"
MAXRUN="${MAXRUN:-4}"
CGMAXBS="${CGMAXBS:-4}"
# Some drafters (the Domino chain at long context on tight VRAM) spike activation
# memory during prefill CUDA-graph replay and OOM. PREFILL_GRAPH=disabled skips the
# prefill graph (decode graph + acceptance untouched). Keep it IDENTICAL across all
# methods in a comparison for fairness.
PREFILL_GRAPH="${PREFILL_GRAPH:-on}"
if [ "$TP" -ge 2 ]; then export CUDA_VISIBLE_DEVICES="${GPUS:-0,1}"; else export CUDA_VISIBLE_DEVICES="${GPUS:-0}"; fi
EXTRA_ARGS="${EXTRA_ARGS:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Port-scoped bracket pkill: "[s]" avoids self-kill; ${PORT} scope spares a co-tenant.
pkill -9 -f "[s]glang.launch_server.*${PORT}" 2>/dev/null || true
sleep 3

COMMON=(--model-path "$MODEL" --tp-size "$TP" --trust-remote-code
        --attention-backend flashinfer --page-size 1 --mem-fraction-static "$MEMFRAC"
        --max-running-requests "$MAXRUN" --port "$PORT")
if [ "$TP" -ge 2 ]; then
  # TP>1: use the decode-specific cuda-graph knob and disable custom all-reduce
  # (portable across GPUs that lack a fast custom kernel).
  COMMON+=(--cuda-graph-max-bs-decode "$CGMAXBS" --disable-custom-all-reduce)
else
  COMMON+=(--cuda-graph-max-bs "$CGMAXBS")
fi
[ "$PREFILL_GRAPH" = "disabled" ] && COMMON+=(--cuda-graph-backend-prefill disabled)
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && COMMON+=($EXTRA_ARGS)

SPEC=(--speculative-num-draft-tokens 16)
case "$METHOD" in
  ar)
    exec $PY -m sglang.launch_server "${COMMON[@]}" ;;
  dflash)
    exec $PY -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm DFLASH --speculative-draft-model-path "${DRAFT:?set DRAFT}" \
      --speculative-num-steps 1 --speculative-eagle-topk 1 "${SPEC[@]}" ;;
  eagle3)
    exec $PY -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm EAGLE3 --speculative-draft-model-path "${DRAFT:?set DRAFT}" \
      --speculative-num-steps 7 --speculative-eagle-topk 10 "${SPEC[@]}" ;;
  domino_chain)
    export SGLANG_PLUGINS=dominotree
    exec $PY -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm DOMINO --speculative-draft-model-path "${DRAFT:?set DRAFT}" \
      --speculative-num-steps 1 --speculative-eagle-topk 1 "${SPEC[@]}" ;;
  dominotree)
    # frontier+graph builder is the DEFAULT — no env knobs needed.
    export SGLANG_PLUGINS=dominotree
    exec $PY -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm DOMINOTREE --speculative-draft-model-path "${DRAFT:?set DRAFT}" \
      --speculative-num-steps 1 --speculative-eagle-topk 1 "${SPEC[@]}" ;;
  *) echo "unknown METHOD=$METHOD" >&2; exit 2;;
esac
