#!/usr/bin/env bash
# Shared lifecycle for benchmark servers. All cleanup is scoped to our child PIDs.
set -u
SERVERLIB27B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT27B="${PORT27B:-$((20000 + $$ % 20000))}"
SERVER_PID27B=""
CLIENT_PID27B=""
SERVER_NAME27B=""

curl_27b() { curl --connect-timeout 2 --max-time 5 "$@"; }

assert_server_27b() {
  # Check before AND after HTTP: another listener must not rescue a dead child.
  [ -n "$SERVER_PID27B" ] && kill -0 "$SERVER_PID27B" 2>/dev/null || return 1
  curl_27b -fsS "http://127.0.0.1:$PORT27B/health" >/dev/null 2>&1 || return 1
  local info
  info=$(curl_27b -fsS "http://127.0.0.1:$PORT27B/v1/models") || return 1
  "${PY:-python}" -c 'import json,sys
info=json.loads(sys.argv[1])
sys.exit(0 if [m["id"] for m in info["data"]] == [sys.argv[2]] else 1)' \
    "$info" "$SERVER_NAME27B" || return 1
  kill -0 "$SERVER_PID27B" 2>/dev/null
}

start_server_27b() {  # method logfile [ENV=value ...]
  local method="$1" logfile="$2"
  stop_server_27b
  # No -f: even an HTTP error response means the port belongs to someone else.
  if curl_27b -sS -o /dev/null "http://127.0.0.1:$PORT27B/health" 2>/dev/null; then
    echo "!! port $PORT27B already answers; refusing to launch" >&2
    return 1
  fi
  mkdir -p "$(dirname "$logfile")" || return 1
  SERVER_NAME27B="qwen27b-$method-$("${PY:-python}" -c 'import uuid; print(uuid.uuid4().hex)')" || return 1
  # With job control disabled the background child's PID is not already a group
  # leader, so setsid execs in place and $! remains the server's session/group PID.
  set +m
  setsid env "${@:3}" METHOD="$method" PORT="$PORT27B" \
    SERVED_MODEL_NAME27B="$SERVER_NAME27B" \
    bash "$SERVERLIB27B/launch_server_27b.sh" > "$logfile" 2>&1 < /dev/null &
  SERVER_PID27B=$!
  local t0=$SECONDS
  while kill -0 "$SERVER_PID27B" 2>/dev/null && (( SECONDS - t0 < 1800 )); do
    if assert_server_27b; then
      echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $method verified as $SERVER_NAME27B on port $PORT27B"
      return 0
    fi
    sleep 2
  done
  echo "!! $method failed PID/health/identity checks -- see $logfile" >&2
  tail -12 "$logfile" >&2
  stop_server_27b
  return 1
}

run_client_27b() {
  assert_server_27b || { echo '!! server PID/identity check failed before collection' >&2; return 1; }
  "$@" &
  CLIENT_PID27B=$!
  local rc=0
  wait "$CLIENT_PID27B" || rc=$?
  CLIENT_PID27B=""
  assert_server_27b || { echo '!! server PID/identity changed during collection' >&2; return 1; }
  return "$rc"
}

stop_server_27b() {
  [ -n "$CLIENT_PID27B$SERVER_PID27B" ] || return 0
  if [ -n "$CLIENT_PID27B" ]; then
    kill "$CLIENT_PID27B" 2>/dev/null || true
  fi
  if [ -n "$SERVER_PID27B" ]; then
    kill -TERM -- "-$SERVER_PID27B" 2>/dev/null || kill -TERM "$SERVER_PID27B" 2>/dev/null || true
  fi
  sleep 2
  if [ -n "$CLIENT_PID27B" ]; then
    kill -KILL "$CLIENT_PID27B" 2>/dev/null || true
    wait "$CLIENT_PID27B" 2>/dev/null || true
    CLIENT_PID27B=""
  fi
  if [ -n "$SERVER_PID27B" ]; then
    kill -KILL -- "-$SERVER_PID27B" 2>/dev/null || kill -KILL "$SERVER_PID27B" 2>/dev/null || true
    wait "$SERVER_PID27B" 2>/dev/null || true
    SERVER_PID27B=""
  fi
}

trap stop_server_27b EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
