#!/usr/bin/env bash
# Find one method's highest crash-free admission cap (`--max-running-requests`)
# on your GPU, by booting the server at each candidate cap and driving a short
# burst at that concurrency.
#
# You need this because the cap is hardware-dependent and differs per method: a
# draft model costs weights plus its own KV, and tree verify additionally
# materializes a batch x draft x vocab logits buffer, so on a small card the tree
# admits fewer concurrent requests than a chain. The cap is part of the result --
# record it, and compare methods only at concurrencies within every compared
# method's cap.
#
# Usage:
#   METHOD=dominotree MODEL=Qwen/Qwen3-8B TP=2 DRAFT_DOMINO=./Qwen3-8B-Domino-b16 \
#     bash find_caps.sh 32 16 12 8 4
# Prints the first candidate that boots and completes a burst without OOM.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${METHOD:?set METHOD}"
MODEL="${MODEL:?set MODEL}"
TP="${TP:-1}"
PY="${PY:-python}"
PORT="${PORT:-31690}"
MEMFRAC="${MEMFRAC:-0.85}"
CANDIDATES=("$@")
[ ${#CANDIDATES[@]} -eq 0 ] && CANDIDATES=(32 16 12 8 4)

log(){ echo "[$(date '+%F %T')] find_caps[$METHOD]: $*"; }
kill_server(){ pkill -9 -f "[s]glang.launch_server.*${PORT}" 2>/dev/null || true; sleep 5; }
trap kill_server EXIT

for CAP in "${CANDIDATES[@]}"; do
  log "trying cap=$CAP"
  kill_server
  setsid env METHOD="$METHOD" PORT="$PORT" MODEL="$MODEL" TP="$TP" PY="$PY" \
      MEMFRAC="$MEMFRAC" MAXRUN="$CAP" CGMAXBS="$CAP" \
      DRAFT_DOMINO="${DRAFT_DOMINO:-}" DRAFT_DFLASH="${DRAFT_DFLASH:-}" \
      DRAFT_EAGLE3="${DRAFT_EAGLE3:-}" \
      bash "$HERE/../launch_server.sh" > "$HERE/capfind_${METHOD}_${CAP}.log" 2>&1 < /dev/null &

  ok=0
  for _ in $(seq 1 180); do
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  if [ "$ok" -ne 1 ]; then log "cap=$CAP: server did not boot"; continue; fi

  # Burst CAP concurrent requests; any OOM shows up as a failed request or an
  # error in the server log.
  fails=0
  for i in $(seq 1 "$CAP"); do
    curl -fsS "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' \
      -d '{"text":"Write a detailed explanation of how a binary search works.","sampling_params":{"temperature":0,"max_new_tokens":512}}' \
      > /dev/null 2>&1 || fails=$((fails+1)) &
  done
  wait
  if grep -qiE "out of memory|CUDA error|OutOfMemory" "$HERE/capfind_${METHOD}_${CAP}.log"; then
    log "cap=$CAP: OOM in server log"; continue
  fi
  if [ "$fails" -gt 0 ]; then log "cap=$CAP: $fails/$CAP requests failed"; continue; fi
  if ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    log "cap=$CAP: server unhealthy after burst"; continue
  fi

  log "cap=$CAP: OK  <-- highest crash-free cap for $METHOD"
  echo "$CAP"
  exit 0
done

log "no candidate cap succeeded"
exit 1
