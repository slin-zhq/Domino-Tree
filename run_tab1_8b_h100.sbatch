#!/bin/bash
# Qwen3-8B research-harness re-collection for paper Table 1 (8B half; Tables 2 and 14 derive from it)
# on ONE Nano5 H100, all methods and all temperatures in ONE job, so every comparison (including the
# AR normalization) comes from the same GPU. Replaces the A6000 8B rows. 2026-09-25.
#
# Protocol = the public repo's run_pipeline.sh (8 datasets x T{0,0.5,1}, n=50 [AIME25 = 30],
# 2048 new tokens), with the builder of the 4B fused Table-1 recollection (run_tab1f_4b_remote.sh:
# --builder frontier + GRU table). Budget 128 = the paper's 8B setting, re-confirmed on this H100 by
# the budget ablation (jobs 361290/361296). Baselines as in the paper:
#   ours:           AR (T=0 only; temperature-independent, as in run_pipeline.sh) + DominoTree@128
#   official Domino: eager and --use-graph, block 16, sdpa (matches our harness's attention), with
#                    run_pipeline.sh's FAST_DOMINO copy (drops only its unused b=1 AR arm)
#   reference harness (CaDDTree a88f3f3): DFlash + DDTree@128 + CaDDTree; its own AR at T=0 only
#                    (the July reference runs used --skip-baseline at T>0).
# Resumable: a cell whose output already exists and validates is skipped.
#SBATCH --job-name=rh8b_tab1
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=collect-%j.out
set -uo pipefail
# Site paths are parameters (the job that produced the published data used one working dir
# holding venv/, src/{Domino-Tree,Domino,CaDDTree}/ and models/; site module loads omitted).
ROOT=${ROOT:?set ROOT to the working dir holding venv/, src/ and models/}
export DOMINOTREE_FRONTIER_SRC=${DOMINOTREE_FRONTIER_SRC:-$ROOT/src/Domino-Tree/sglang_dominotree/src/dominotree_sglang/tree/frontier.py}
export DOMINOTREE_BUILDER_GRU_TABLE=1
V=$ROOT/venv/bin/python
DT=$ROOT/src/Domino-Tree
DOMINO_CODE=$ROOT/src/Domino/code
CADDTREE=$ROOT/src/CaDDTree
MODEL=$ROOT/models/Qwen3-8B
DRAFT_DOMINO=$ROOT/models/Qwen3-8B-Domino-b16
DRAFT_DFLASH=$ROOT/models/Qwen3-8B-DFlash-b16
OUT=$ROOT/collect/tab1_8b_h100          # fixed dir (not per-job) so a resubmission resumes
mkdir -p "$OUT"/{ours,domino_official,reference}
DATASETS=(gsm8k math500 aime25 humaneval mbpp livecodebench mt-bench alpaca)
TEMPS=(0.0 0.5 1.0)
B=128
ts(){ date -u +%FT%TZ; }
nvidia-smi -L | tee -a "$OUT/gpu.txt"; hostname | tee -a "$OUT/gpu.txt"
{ echo "job=$SLURM_JOB_ID start=$(ts)"; git -C "$DT" rev-parse HEAD; git -C "$DOMINO_CODE/.." rev-parse HEAD;
  git -C "$CADDTREE" rev-parse HEAD; "$V" -c "import torch,transformers,flash_attn;print(torch.__version__,transformers.__version__,flash_attn.__version__)"; } >> "$OUT/provenance.txt"

nsamples(){ [ "$1" = aime25 ] && echo 30 || echo 50; }
ours_ok(){  # $1=file, $2..=methods; each method must have the expected number of unique prompts
  local f=$1 ds=$2; shift 2; [ -s "$f" ] || return 1
  "$V" - "$f" "$(nsamples "$ds")" "$@" <<'PY' || return 1
import json, sys
f, n, methods = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
rows = [json.loads(l) for l in open(f)]
got = {m: len({r["sample_idx"] for r in rows if r["method"] == m}) for m in methods}
sys.exit(0 if all(v == n for v in got.values()) and {r["method"] for r in rows} == set(methods) else 1)
PY
}

# FAST_DOMINO copy, exactly as run_pipeline.sh does it (Domino's own file untouched).
"$V" - "$DOMINO_CODE/benchmark.py" "$DOMINO_CODE/benchmark_fast_b16only.py" <<'PY' || exit 1
import sys
src = open(sys.argv[1]).read(); before = src
src = src.replace("for bs in [1, block_size]:", "for bs in [block_size]:")
src = src.replace("for choice, bs in [(choice_b1, 1), (choice_bk, block_size)]:",
                  "for choice, bs in [(choice_bk, block_size)]:")
assert src != before and "for bs in [1, block_size]" not in src, "FAST_DOMINO patch failed"
open(sys.argv[2], "w").write(src); print("FAST_DOMINO copy ok")
PY

# ---- Stage 1: ours ----
for T in "${TEMPS[@]}"; do
  if [ "$T" = 0.0 ]; then methods=ar,dominotree; labels=(ar "dominotree@$B"); else methods=dominotree; labels=("dominotree@$B"); fi
  for ds in "${DATASETS[@]}"; do
    f="$OUT/ours/${ds}_T${T}.jsonl"
    ours_ok "$f" "$ds" "${labels[@]}" && { echo "[$(ts)] skip ours $ds T$T"; continue; }
    rm -f "$f"; echo "[$(ts)] OURS $ds T$T $methods"
    ( cd "$DT" && "$V" -u benchmark.py --model-name-or-path "$MODEL" --draft-name-or-path "$DRAFT_DOMINO" \
        --domino-code "$DOMINO_CODE" --methods "$methods" --budgets "$B" --corr-topm 64 --node-topk 8 \
        --builder frontier --max-samples 50 --max-new-tokens 2048 --temperature "$T" --dataset "$ds" --out "$f" ) \
      > "$OUT/ours/${ds}_T${T}.log" 2>&1
    ours_ok "$f" "$ds" "${labels[@]}" || echo "[$(ts)] !! OURS INVALID $ds T$T (see log)"
  done
done

# ---- Stage 2: official Domino (graph + eager) ----
for T in "${TEMPS[@]}"; do for mode in graph eager; do for ds in "${DATASETS[@]}"; do
  a="$OUT/domino_official/T$T/${mode}_${ds}.jsonl"; mkdir -p "$(dirname "$a")"
  [ -s "$a" ] && { echo "[$(ts)] skip domino $mode $ds T$T"; continue; }
  gf=""; [ "$mode" = graph ] && gf=--use-graph
  echo "[$(ts)] DOMINO $mode $ds T$T"
  ( cd "$DOMINO_CODE" && "$V" -u benchmark_fast_b16only.py --model-name-or-path "$MODEL" \
      --draft-name-or-path "$DRAFT_DOMINO" --dataset "$ds" --max-samples 50 --max-new-tokens 2048 \
      --temperature "$T" --block-size 16 --use-bias $gf --attn-implementation sdpa --answer-file "$a" ) \
    > "$OUT/domino_official/T$T/${mode}_${ds}.log" 2>&1 || { echo "[$(ts)] !! DOMINO FAILED $mode $ds T$T"; rm -f "$a"; }
done; done; done

# ---- Stage 3: reference harness (DFlash, DDTree@128, CaDDTree; its AR at T=0 only) ----
for T in "${TEMPS[@]}"; do for ds in "${DATASETS[@]}"; do
  p="$OUT/reference/${ds}_T${T}.pt"
  [ -s "$p" ] && { echo "[$(ts)] skip reference $ds T$T"; continue; }
  sb=""; [ "$T" != 0.0 ] && sb=--skip-baseline
  echo "[$(ts)] REFERENCE $ds T$T"
  ( cd "$CADDTREE" && "$V" -u -m torch.distributed.run --nproc_per_node=1 --standalone benchmark_caddtree.py \
      --model-name-or-path "$MODEL" --draft-name-or-path "$DRAFT_DFLASH" --dataset "$ds" --max-samples 50 \
      --max-new-tokens 2048 --temperature "$T" --tree-budget "$B" \
      --cost-model-path cost_models/Qwen3-8B_convex.pt $sb --save-path "$p" ) \
    > "$OUT/reference/${ds}_T${T}.log" 2>&1 || { echo "[$(ts)] !! REFERENCE FAILED $ds T$T"; rm -f "$p"; }
done; done

echo "job=$SLURM_JOB_ID end=$(ts)" >> "$OUT/provenance.txt"
echo "[$(ts)] COLLECTION PASS COMPLETE (check for '!!' lines above)"
