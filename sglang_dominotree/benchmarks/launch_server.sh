#!/usr/bin/env bash
# Launch ONE SGLang server for the serving benchmarks (bs1/ and concurrency/).
#
# All five methods share IDENTICAL serving flags -- only the speculative algorithm
# and the draft model differ. That is the fairness rule behind every serving number
# in the paper: any difference between methods reflects the speculative method, not
# the launch configuration. Do not add a flag for one method only.
#
# (The long-context benchmark uses its own launcher, benchmarks/helmet/
# launch_helmet_server.sh, because it needs a different KV/batch recipe: a huge KV
# pool for a 32K prompt and a tiny batch cap.)
#
# Required env:
#   METHOD   ar | dflash | eagle3 | domino_chain | dominotree
#   PORT     server port
#   MODEL    target model path or HF id
#   DRAFT_DOMINO / DRAFT_DFLASH / DRAFT_EAGLE3   draft model for the chosen METHOD
#
# Optional env:
#   TP             tensor-parallel size            default 1   (use 2 for a big model on 2 GPUs)
#   PY             python executable               default python
#   GPU            CUDA_VISIBLE_DEVICES (TP=1)     default 0   (TP>1 spans 0..TP-1)
#   MEMFRAC        --mem-fraction-static           default 0.7   (see note below)
#   MAXRUN         --max-running-requests          default 32  (the admission cap; see below)
#   CGMAXBS        decode cuda-graph max batch     default = MAXRUN
#   PREFILL_GRAPH  on | disabled                   default on
#
# THE ADMISSION CAP (MAXRUN) matters for how results are read. It bounds how many
# requests the server runs *simultaneously*; offered requests beyond it queue. Each
# method can sustain a different cap on a given GPU, because a draft model costs
# weights plus its own KV and tree verify additionally materializes a
# batch x draft x vocab logits buffer. Pick each method's highest crash-free cap
# (see concurrency/README.md), record it, and compare methods only at concurrencies
# within every compared method's cap -- past its cap a method is running its cap,
# not the offered load.
#
# MEMFRAC trades KV-pool size against transient headroom. 0.7 is what the paper's
# single-stream and 4B concurrency runs used on 16 GB cards: DominoTree's tree
# KV-move kernel needs transient workspace at high concurrency and OOMs above that
# on a small card. Long context wants the opposite (a big KV pool for a 32K prompt),
# so benchmarks/helmet/ uses 0.85 with a tiny batch cap; the 8B/TP=2 concurrency runs
# also used 0.85 because sharded weights leave more of each card for KV. If you OOM,
# lower it -- for ALL methods, so the comparison stays fair.
#
# PREFILL_GRAPH=disabled skips only the *prefill* CUDA graph (decode graph and
# acceptance are untouched). Some methods spike activation memory during prefill
# graph replay at long context; if one OOMs, set it for ALL methods so the
# comparison stays fair.
#
# Usage:
#   METHOD=dominotree PORT=31600 MODEL=Qwen/Qwen3-8B TP=2 MAXRUN=8 \
#   DRAFT_DOMINO=./Qwen3-8B-Domino-b16 \
#     setsid bash launch_server.sh > server_dominotree.log 2>&1 < /dev/null &
set -u
METHOD="${METHOD:?set METHOD (ar|dflash|eagle3|domino_chain|dominotree)}"
PORT="${PORT:?set PORT}"
MODEL="${MODEL:?set MODEL (target model path or HF id)}"
TP="${TP:-1}"
PY="${PY:-python}"
GPU="${GPU:-0}"
MEMFRAC="${MEMFRAC:-0.7}"
MAXRUN="${MAXRUN:-32}"
CGMAXBS="${CGMAXBS:-$MAXRUN}"
PREFILL_GRAPH="${PREFILL_GRAPH:-on}"

if [ "$TP" -gt 1 ]; then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((TP - 1)))"
else
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Port-scoped bracket pkill: "[s]" avoids matching this command itself; the ${PORT}
# scope avoids killing a co-tenant server on another GPU/port.
pkill -9 -f "[s]glang.launch_server.*${PORT}" 2>/dev/null || true
sleep 3

COMMON=(--model-path "$MODEL" --tp-size "$TP" --trust-remote-code
        --attention-backend flashinfer --page-size 1
        --mem-fraction-static "$MEMFRAC"
        --max-running-requests "$MAXRUN" --port "$PORT")
if [ "$TP" -gt 1 ]; then
  COMMON+=(--cuda-graph-max-bs-decode "$CGMAXBS" --disable-custom-all-reduce)
else
  COMMON+=(--cuda-graph-max-bs "$CGMAXBS")
fi
[ "$PREFILL_GRAPH" = "disabled" ] && COMMON+=(--cuda-graph-backend-prefill disabled)

# Every speculative method gets the same 16-token draft budget.
case "$METHOD" in
  ar)
    exec $PY -m sglang.launch_server "${COMMON[@]}" ;;
  dflash)
    exec $PY -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm DFLASH \
      --speculative-draft-model-path "${DRAFT_DFLASH:?set DRAFT_DFLASH}" \
      --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 16 ;;
  eagle3)
    exec $PY -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm EAGLE3 \
      --speculative-draft-model-path "${DRAFT_EAGLE3:?set DRAFT_EAGLE3}" \
      --speculative-num-steps 7 --speculative-eagle-topk 10 --speculative-num-draft-tokens 16 ;;
  domino_chain)
    export SGLANG_PLUGINS=dominotree
    exec $PY -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm DOMINO \
      --speculative-draft-model-path "${DRAFT_DOMINO:?set DRAFT_DOMINO}" \
      --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 16 ;;
  dominotree)
    # The frontier builder with CUDA-graph capture is the DEFAULT -- no env knobs.
    export SGLANG_PLUGINS=dominotree
    exec $PY -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm DOMINOTREE \
      --speculative-draft-model-path "${DRAFT_DOMINO:?set DRAFT_DOMINO}" \
      --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 16 ;;
  *)
    echo "unknown METHOD=$METHOD" >&2; exit 2 ;;
esac
