#!/usr/bin/env bash
# =============================================================================
# DominoTree reproduction pipeline -- ONE command to collect data on your GPU
# and build the paper tables.
#
#   MODEL_PATH=/path/to/Qwen3-4B \
#   DRAFT_PATH=/path/to/Qwen3-4B-Domino-b16 \
#   DOMINO_CODE=/path/to/Domino/code \
#   bash run_pipeline.sh
#
# It collects, on a SINGLE GPU (clean, one job at a time -> fair ratios):
#   1. DominoTree + our lean AR (our harness, benchmark.py, GPU-native builder)
#   2. Official Domino, eager and --use-graph -- Domino's own benchmark, run AS RELEASED
#      at a pinned commit (DOMINO_COMMIT below). We do not patch their code.
# then builds the tables with make_latex_table.py. The DFlash/DDTree/CaDDTree
# reference rows ship frozen under results/raw/baseline_ddtree_caddtree/ (collected
# on the authors' GPU; clearly marked in the tables). Re-run any time -- finished
# cells are skipped, so it resumes.
#
# WHY the odd bits are handled for you (so you don't hit the traps we did):
#   * Speedup normalization is "surgical": every method is divided by a lean AR
#     (~66 tok/s on our 5080). Domino's OWN AR is spec_generate(block_size=1) -- its
#     speculative loop with drafting off -- which carries per-token bookkeeping and
#     runs ~23% slower, so we do NOT use it as Domino's denominator. make_latex_table
#     records the AR denominator per row.
#   * AR is temperature-independent (<1%), so we measure it once at T=0 and reuse it.
#   * Domino's released benchmark runs no warmup, so its FIRST prompt is cold. We do not
#     patch a warmup in; we run their benchmark as released and drop that cold prompt in
#     ANALYSIS (--domino-no-warmup) -- the same convention the published 8B baseline used.
#     Either way every prompt that reaches a table is warm.
#     Heads-up on exact reproduction: the published *4B* baseline was collected the other
#     way (a warmup patched into Domino's benchmark, all 50 prompts kept). Both are valid;
#     they differ only in sample composition (49 warm prompts vs 50), so a 4B Domino row
#     you collect here lands within sampling noise of the published one, not identical to
#     it. Nothing needs re-collecting -- see the README section on this.
#   * Single GPU by default: concurrent jobs on a shared node perturb absolute TPS.
#     Set CUDA_VISIBLE_DEVICES to pick the GPU; multi-GPU is intentionally not automated.
# =============================================================================
set -uo pipefail

# ---- config (env-overridable; paper defaults) -------------------------------
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the target model (e.g. Qwen3-4B)}"
DRAFT_PATH="${DRAFT_PATH:?set DRAFT_PATH to the Domino drafter (e.g. Qwen3-4B-Domino-b16)}"
# NB: no apostrophes inside ${VAR:?...} -- bash parses one as an opening quote and the
# whole script then fails to parse. (That bug shipped here once; keep it plain.)
DOMINO_CODE="${DOMINO_CODE:?set DOMINO_CODE to the code/ dir of the released Domino repo}"
# The Domino commit the published baseline was measured at. We do not check it out for you
# (your DOMINO_CODE may be a tarball, or deliberately newer) -- we verify and say so, which
# is what stops "which Domino was that?" from being unanswerable later.
DOMINO_COMMIT="${DOMINO_COMMIT:-e4aad4851}"
PY="${PYTHON:-python}"
# Collect into a SEPARATE tree, never into results/raw. results/raw holds the frozen
# published data, and every cell this pipeline would write already exists there -- so
# defaulting to it made the run skip DominoTree entirely and rebuild the tables from our
# numbers while reporting success. Your results and ours stay in different directories.
OUT="${OUT:-results/raw_repro}"
read -r -a DATASETS <<< "${DATASETS:-gsm8k math500 aime25 humaneval mbpp livecodebench mt-bench alpaca}"
read -r -a TEMPS    <<< "${TEMPS:-0.0 0.5 1.0}"
N="${N:-50}"; MNT="${MNT:-2048}"; BUDGET="${BUDGET:-16}"; CORR_TOPM="${CORR_TOPM:-64}"
WITH_ABLATION="${WITH_ABLATION:-0}"   # 1 => also collect the conditioning ablation (marg + python-builder cond, T=0)
# 0 (default) => run Domino's benchmark.py EXACTLY as released. No patching at all.
# 1           => run a COPY with only the b=1 AR arm removed. That arm runs at roughly AR
#                speed (~7-9x slower than the b=16 arm we report) and dominates Domino's
#                runtime, yet NO published number reads it: we normalize by our own lean
#                AR, never Domino's (see the normalization note above). So removing it
#                skips work without changing a result. Warmup is never patched, in either
#                mode -- methodology always comes from Domino's code as released.
FAST_DOMINO="${FAST_DOMINO:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- GPU auto-detect (single GPU) -------------------------------------------
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
  export CUDA_VISIBLE_DEVICES=0
  [ "${ngpu:-1}" -gt 1 ] && echo "[pipeline] ${ngpu} GPUs found; using GPU 0 (set CUDA_VISIBLE_DEVICES to change). Single-GPU is intentional for clean timing."
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null | head -1)
# `date -Is` is GNU-only; BSD/macOS date rejects it and every log line loses its
# timestamp. Build the same format portably.
ts() { date +%Y-%m-%dT%H:%M:%S%z; }

# ---- preflight: fail with an instruction, not a traceback 40 lines deep -----
missing=""
"${PY}" -c "import torch" 2>/dev/null || missing="${missing}\n  - torch (ours): pip install -r ${HERE}/requirements.txt, plus a CUDA-matched torch build"
"${PY}" -c "import loguru" 2>/dev/null || missing="${missing}\n  - loguru (Domino's): pip install -r \"\$(dirname ${DOMINO_CODE})/requirements.txt\" — the released Domino repo has its own deps"
if [ -n "$missing" ]; then
  printf "[FATAL] missing Python dependencies:%b\n" "$missing"
  echo "Both this repo's harness and the released Domino benchmark run in the SAME interpreter (${PY}); install both sets before collecting."
  exit 1
fi
[ -f "${DOMINO_CODE}/benchmark.py" ] || { echo "[FATAL] DOMINO_CODE=${DOMINO_CODE} has no benchmark.py -- point it at the code/ dir of the released Domino repo."; exit 1; }

# ---- which Domino are we measuring? -----------------------------------------
# A different commit is allowed -- it is simply no longer the one the published numbers
# came from, and that should be visible in the log and recorded in the manifest rather
# than discovered later.
DOMINO_HEAD="$(git -C "${DOMINO_CODE}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ "${DOMINO_HEAD}" = "unknown" ]; then
  echo "[pipeline] Domino is not a git checkout here, so the commit pin cannot be verified (published baseline used ${DOMINO_COMMIT})."
elif git -C "${DOMINO_CODE}" merge-base --is-ancestor "${DOMINO_COMMIT}" HEAD 2>/dev/null; then
  echo "[pipeline] Domino at ${DOMINO_HEAD} (contains the pinned ${DOMINO_COMMIT})."
else
  echo "[pipeline] NOTE: Domino is at ${DOMINO_HEAD}, which does NOT contain the pinned ${DOMINO_COMMIT}."
  echo "[pipeline]       The published baseline used ${DOMINO_COMMIT}; check that commit out for the closest match."
fi

# Does THIS Domino warm itself up before timing? At the pinned commit it does NOT, so its
# first prompt is cold and we drop it in analysis.
#
# We deliberately do NOT let a source grep decide this. A lexical match is unreliable in
# precisely the direction that would hurt: a stray "warmup" in a comment would make us KEEP
# a cold prompt, understating the baseline and flattering our own result. A real warmup
# spelled `warm_up`/`preheat`, or living in an imported helper, would be missed entirely.
# So the automatic behaviour is always the SAFE one -- drop the first prompt, whose worst
# case is losing one warm row -- and the grep only raises a flag for a human to act on.
# Declare the truth explicitly with DOMINO_HAS_WARMUP=0|1.
DOMINO_HAS_WARMUP="${DOMINO_HAS_WARMUP:-0}"
if grep -qiE "warm.?up|preheat" "${DOMINO_CODE}/benchmark.py"; then
  echo "[pipeline] NOTE: benchmark.py mentions a warmup, which the pinned ${DOMINO_COMMIT} did not."
  echo "[pipeline]       If this Domino now warms itself, re-run with DOMINO_HAS_WARMUP=1 so its first prompt is KEPT."
fi
if [ "${DOMINO_HAS_WARMUP}" = "0" ]; then
  DOMINO_WARMUP_FLAG="--domino-no-warmup"
  echo "[pipeline] Domino warmup: assumed absent -> its cold first prompt is dropped in analysis."
  echo "[pipeline]   (For a multi-turn dataset such as mt-bench this drops every turn of prompt 0, not just the first;"
  echo "[pipeline]    measured effect is under 0.35%, and it is the same convention the published 8B baseline used.)"
else
  DOMINO_WARMUP_FLAG=""
  echo "[pipeline] Domino warmup: declared present (DOMINO_HAS_WARMUP=1) -> all prompts kept."
fi

# Domino's raw files are laid out per target model, which is also how make_latex_table
# reads them back (domino_official/<model-dir>/T<temp>/<mode>_<dataset>.jsonl).
DOMINO_MODEL_DIR="${DOMINO_MODEL_DIR:-$(basename "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]')}"
mkdir -p "${OUT}/dominotree" "${OUT}/domino_official/${DOMINO_MODEL_DIR}"

# The DFlash / DDTree / CaDDTree reference rows are NOT collected here -- they ship
# frozen (collected on the authors' GPU) and are marked as such in the tables. Seed them
# into the collection tree by reference so the table build can produce complete tables
# from what you actually ran plus those frozen rows.
if [ -d "${HERE}/results/raw/baseline_ddtree_caddtree" ] && [ ! -e "${OUT}/baseline_ddtree_caddtree" ]; then
  ln -s "${HERE}/results/raw/baseline_ddtree_caddtree" "${OUT}/baseline_ddtree_caddtree" 2>/dev/null \
    || cp -R "${HERE}/results/raw/baseline_ddtree_caddtree" "${OUT}/baseline_ddtree_caddtree"
  echo "[pipeline] reference rows (DFlash/DDTree/CaDDTree) linked in from the shipped frozen set; everything else below is collected by you."
fi

# ---- run manifest (provenance: env, versions, policies) ---------------------
"${PY}" - "$OUT" "$MODEL_PATH" "$DRAFT_PATH" "$GPU_NAME" "$N" "$MNT" "$BUDGET" "$CORR_TOPM" "$DOMINO_HEAD" "$DOMINO_HAS_WARMUP" "$FAST_DOMINO" <<'PYEOF'
import json, sys, platform, subprocess
out, model, draft, gpu, n, mnt, budget, corrm, dhead, dwarm, dfast = sys.argv[1:12]
def ver(m):
    try: return __import__(m).__version__
    except Exception: return None
try:  # stderr silenced: running from a tarball rather than a clone is fine, not an error
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                     text=True, stderr=subprocess.DEVNULL).strip()
except Exception: commit = None
json.dump({
  "config": {"model": model, "draft": draft, "n": int(n), "max_new_tokens": int(mnt),
             "budget": int(budget), "corr_topm": int(corrm)},
  "env": {"gpu": gpu, "python": platform.python_version(),
          "torch": ver("torch"), "transformers": ver("transformers")},
  "policy": {"normalization": "surgical: every method / lean AR; Domino NOT / its own spec-loop AR",
             "ar": "measured at T=0 only, reused at T>0 (temperature-independent <1%)",
             "warmup": ("Domino benchmark run as released; its cold first prompt dropped in analysis"
                        if dwarm == "0" else
                        "Domino benchmark reports a warmup of its own; all prompts kept"),
             "domino_patched": ("b=1 AR arm removed in a copy (FAST_DOMINO=1); no timing path altered"
                                if dfast == "1" else "no -- benchmark.py run exactly as released"),
             "regime": "single-GPU, one job at a time"},
  "domino_tree_commit": commit,
  "domino_commit": dhead,
}, open(f"{out}/run_manifest.json","w"), indent=2)
print(f"[pipeline] wrote {out}/run_manifest.json")
PYEOF
echo "[pipeline] $(ts) GPU=${CUDA_VISIBLE_DEVICES} (${GPU_NAME}) datasets=(${DATASETS[*]}) temps=(${TEMPS[*]}) n=${N}"

# ---- Domino's benchmark: run it as released ---------------------------------
# No methodology is patched into Domino's benchmark. It runs exactly as released, and its
# cold first prompt is handled in ANALYSIS (DOMINO_WARMUP_FLAG above). That keeps the
# baseline provably the authors' own code and retires a brittle source-text patch that
# would have broken the moment they edited that file.
#
# FAST_DOMINO=1 is the one optional exception, and it is deliberately narrow: it copies
# benchmark.py and removes ONLY the b=1 AR arm -- work whose output no published number
# reads. It never touches warmup or any timing path that reaches a table.
DOMINO_BENCH="benchmark.py"
if [ "${FAST_DOMINO}" = "1" ]; then
  DOMINO_BENCH="benchmark_nob1.py"
  # Delete any copy left by an earlier run BEFORE regenerating. Without this, a patch that
  # fails today would leave yesterday's file in place, the existence check below would pass,
  # and we would silently benchmark stale code while recording today's Domino commit.
  rm -f "${DOMINO_CODE}/${DOMINO_BENCH}"
  "${PY}" - "${DOMINO_CODE}/benchmark.py" "${DOMINO_CODE}/${DOMINO_BENCH}" <<'PATCHEOF'
import sys
src_path, dst_path = sys.argv[1], sys.argv[2]
src = open(src_path).read()
before = src
src = src.replace("for bs in [1, block_size]:", "for bs in [block_size]:")
src = src.replace("for choice, bs in [(choice_b1, 1), (choice_bk, block_size)]:",
                  "for choice, bs in [(choice_bk, block_size)]:")
assert src != before and "for bs in [1, block_size]" not in src, (
    "FAST_DOMINO=1: could not remove the b=1 arm from " + src_path + " -- the released "
    "benchmark has changed shape. Re-run WITHOUT FAST_DOMINO=1 to collect it as released.")
open(dst_path, "w").write(src)
print("[pipeline] FAST_DOMINO: b=1 AR arm removed in a copy -> " + dst_path
      + "  (b=16 results unaffected; Domino's own source untouched)")
PATCHEOF
  # Check the patch's EXIT STATUS, not just that some file exists: the script runs under
  # `set -uo pipefail` without `-e`, so a failed patch would otherwise be stepped over.
  patch_rc=$?
  if [ "${patch_rc}" -ne 0 ] || [ ! -f "${DOMINO_CODE}/${DOMINO_BENCH}" ]; then
    echo "[FATAL] FAST_DOMINO=1 but the b=1 removal did not succeed (exit ${patch_rc})."
    echo "        Re-run without FAST_DOMINO=1 to collect Domino's benchmark exactly as released."
    exit 1
  fi
else
  echo "[pipeline] Domino benchmark: running benchmark.py AS RELEASED, unpatched."
  echo "[pipeline]   Its unused b=1 AR arm is ~7-9x the cost of the b=16 arm we report; set FAST_DOMINO=1 to skip it."
fi

run_ours() {  # $1=methods $2=extra-flags $3=out.jsonl
  local out="$3"
  [ -f "$out" ] && { echo "[skip] ours $out"; return; }
  echo "[$(ts)] OURS $(basename "$out") methods=$1"
  "${PY}" "${HERE}/benchmark.py" --model-name-or-path "${MODEL_PATH}" --draft-name-or-path "${DRAFT_PATH}" \
    --domino-code "${DOMINO_CODE}" --methods "$1" $2 --budgets "${BUDGET}" --corr-topm "${CORR_TOPM}" \
    --node-topk 8 --max-samples "${N}" --max-new-tokens "${MNT}" --out "$out" \
    || { echo "[FAIL] ours $out"; rm -f "$out"; }
}
run_domino() {  # $1=temp $2=mode(graph|eager) $3=dataset
  local gf=""; [ "$2" = graph ] && gf="--use-graph"
  # layout must match make_latex_table.load_domino_official
  local ans="${OUT}/domino_official/${DOMINO_MODEL_DIR}/T$1/$2_$3.jsonl"
  mkdir -p "$(dirname "$ans")"
  [ -f "$ans" ] && { echo "[skip] domino $ans"; return; }
  echo "[$(ts)] DOMINO $3 T$1 $2"
  ( cd "${DOMINO_CODE}" && "${PY}" "${DOMINO_BENCH}" --model-name-or-path "${MODEL_PATH}" \
      --draft-name-or-path "${DRAFT_PATH}" --dataset "$3" --max-samples "${N}" --max-new-tokens "${MNT}" \
      --temperature "$1" --block-size "${BUDGET}" --use-bias ${gf} --attn-implementation sdpa \
      --answer-file "$ans" ) || { echo "[FAIL] domino $ans"; rm -f "$ans"; }
}

# ---- Stage 1: ours (DominoTree + AR). AR only at T=0 (temperature-independent) ----
for T in "${TEMPS[@]}"; do
  methods="dominotree"; [ "$T" = "0.0" ] && methods="ar,dominotree"
  for ds in "${DATASETS[@]}"; do
    # temperature is set inside benchmark.py via its own default; pass it through --temperature
    run_ours "$methods" "--gpu-native-build --temperature $T" "${OUT}/dominotree/${ds}_T${T}.jsonl"
  done
done

# ---- Stage 1b (optional): conditioning ablation (marg + python-builder cond, T=0) ----
if [ "${WITH_ABLATION}" = "1" ]; then
  mkdir -p "${OUT}/conditioning_ablation" "${OUT}/dominotree_python_builder"
  for ds in "${DATASETS[@]}"; do
    run_ours "ar,marg"     "--python-builder --temperature 0.0" "${OUT}/conditioning_ablation/${ds}_T0.0.jsonl"
    run_ours "ar,dominotree" "--python-builder --temperature 0.0" "${OUT}/dominotree_python_builder/${ds}_T0.0.jsonl"
  done
fi

# ---- Stage 2: official Domino (eager + --use-graph), all temps --------------
for T in "${TEMPS[@]}"; do
  for mode in graph eager; do
    for ds in "${DATASETS[@]}"; do run_domino "$T" "$mode" "$ds"; done
  done
done

# ---- Stage 3: build whatever tables the collected data supports -------------
# make_latex_table expects a complete grid and raises on the first missing cell, which
# after a partial collection reads as a crash rather than "you have not collected that
# yet". Say which cells are missing, then let it run.
missing_cells=""
for T in "${TEMPS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    [ -f "${OUT}/dominotree/${ds}_T${T}.jsonl" ] || missing_cells="${missing_cells} ${ds}_T${T}"
  done
done
if [ -n "${missing_cells}" ]; then
  echo "[warn] not all cells were collected; the table build will stop at the first gap."
  echo "[warn] missing DominoTree cells:${missing_cells}"
  echo "[warn] re-run this script to retry them -- finished cells are skipped, so it resumes."
fi

TABLES_OUT="${TABLES_OUT:-${HERE}/results/tables_repro}"
echo "[$(ts)] building tables (surgical AR normalization) -> ${TABLES_OUT}"
"${PY}" "${HERE}/make_latex_table.py" --raw-dir "${OUT}" --out-dir "${TABLES_OUT}" --ar-norm surgical \
  --domino-model-dir "${DOMINO_MODEL_DIR}" ${DOMINO_WARMUP_FLAG} \
  --temps "$(IFS=,; echo "${TEMPS[*]}")" || echo "[warn] table build reported issues (see above)"
echo "[$(ts)] done."
echo "  your tables : ${TABLES_OUT}/       (ours ship in results/tables_gpunative/ -- compare, do not overwrite)"
echo "  your raw    : ${OUT}/"
echo "  provenance  : ${OUT}/run_manifest.json"
