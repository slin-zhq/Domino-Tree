#!/usr/bin/env bash
# Copy a completed run_tab1f_4b_remote.sh collection to results/raw_repro/tab1f_4b/,
# then print a summary comparing it against the headline-16 data already in this repo
# (results/raw/dominotree/, the pre-fused-builder collection).
#
# SRC is an rsync source you provide (e.g. `user@host:path/to/tab1f_4b/` if the
# collection ran on a remote box, or a local path if it ran here) -- there is no
# built-in remote host in this script.
set -euo pipefail

SRC="${SRC:?set SRC to the rsync source, e.g. user@host:path/to/tab1f_4b/ or a local path}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$ROOT/results/raw_repro/tab1f_4b"
mkdir -p "$DEST"
rsync -az "$SRC" "$DEST/"
"${PYTHON:-python3}" "$ROOT/agg_tab1f_4b.py" \
    "$DEST" "$ROOT/results/raw/dominotree"
