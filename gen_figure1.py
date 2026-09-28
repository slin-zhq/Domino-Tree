#!/usr/bin/env python3
"""Draw the paper's Figure 1 (per-dataset speedup over AR, Qwen3-4B and Qwen3-8B, T=0)
from the cells.json that gen_table1.py writes, so the figure and Table 1 share one source.

    python3 gen_table1.py --out-dir results/table1_audit
    python3 gen_figure1.py results/table1_audit/cells.json figure1_dual.pdf

Requires numpy and matplotlib.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DS = ['gsm8k','math500','aime25','humaneval','mbpp','livecodebench','mt-bench','alpaca']
METHODS = ['DFlash','DDTree','CaDDTree','Domino','DominoTree']
BUDGET = {'4B': 32, '8B': 128}  # DDTree / DominoTree node budget per model (Table 1)

cells = {(c['model'],c['temp'],c['dataset'],c['method']): c for c in json.load(open(sys.argv[1]))}
fig, axes = plt.subplots(1, 2, figsize=(12, 3.7), layout='constrained', sharey=True)
colors = ['#7d8791','#e8a347','#55a58a','#578ac0','#bf4d58']
for ax, size in zip(axes, ['4B', '8B']):
    for i, m in enumerate(METHODS):
        ax.bar(np.arange(8)+(i-2)*.16, [cells[size,'0.0',d,m]['speedup'] for d in DS],
               width=.15, label=m, color=colors[i])
    ax.set_xticks(range(8), ['GSM8K','MATH','AIME','HEval','MBPP','LCB','MTB','Alpaca'], rotation=35, ha='right')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    ax.set_title(f'Qwen3-{size} (tree budget {BUDGET[size]})', fontsize=10, loc='left')
axes[0].set_ylabel('Speedup over own AR')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='outside upper center', ncol=5, frameon=False)
fig.savefig(sys.argv[2]); plt.close(fig)
print("wrote", sys.argv[2])
