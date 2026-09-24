#!/usr/bin/env bash
# Re-derive every table and figure of the IEEE Access paper from the raw data in this
# repository. No GPU, no model weights, no paper source needed. Needs numpy + matplotlib.
#
#   bash reproduce_paper.sh            # writes everything under results/reproduced/
#
# Each step either prints a PASS line or exits non-zero. What each step covers:
#   1  Tables 7, 9 (+ the values behind Tables 6 and 8): every serving cell vs the printed value
#   2  Tables 16, 17: per-dataset concurrency and per-task HELMET appendix tables
#   3  Tables 5, 6, 8: batch-size-1 serving (speedup + tau), its per-dataset CIs + Overall,
#      and the long-context table
#   4  Tables 1, 2, 14 + the abstract's Table-1 claims
#   5  Figure 1 (drawn from step 4's cells.json)
#   6  Tables 3, 4, 10, 11, 12, 13, and the Qwen3-8B numbers quoted in prose
#   7  Table 15: conditioning ladder, both builder configurations
set -euo pipefail
cd "$(dirname "$0")"
OUT=results/reproduced
mkdir -p "$OUT"

echo "== 1. serving cells ==";          python3 results/serving/verify_published_numbers.py | tail -1
echo "== 2. serving appendix ==";       python3 results/serving/gen_appendix_5090.py > "$OUT/appendix_serving.tex" && echo "wrote $OUT/appendix_serving.tex"
echo "== 3. Tables 5, 6, 8 ==";
python3 results/serving/gen_sglang_bs1_table.py     > "$OUT/table5_sglang_bs1.tex"     && echo "wrote $OUT/table5_sglang_bs1.tex"
python3 results/serving/gen_sglang_bs1_cis_table.py > "$OUT/table6_sglang_bs1_cis.tex" && echo "wrote $OUT/table6_sglang_bs1_cis.tex"
python3 results/serving/gen_sglang_longctx_table.py > "$OUT/table8_sglang_longctx.tex" && echo "wrote $OUT/table8_sglang_longctx.tex"
echo "== 4. Tables 1, 2, 14 ==";        python3 gen_table1.py --out-dir "$OUT/table1" | tail -1
echo "== 5. Figure 1 ==";               python3 gen_figure1.py "$OUT/table1/cells.json" "$OUT/figure1_dual.pdf"
echo "== 6. builder + ablation tables =="; python3 gen_ablation_tables.py --out-dir "$OUT/ablations" | tail -1
echo "== 7. Table 15 ==";
for cfg in matched_builder best_builder; do
  python3 results/conditioning_ladder/ladder_ci.py "results/conditioning_ladder/$cfg" > "$OUT/table15_$cfg.txt"
  echo "wrote $OUT/table15_$cfg.txt"
done
echo "ALL STEPS COMPLETED -- compare $OUT/ against the paper's tables."
