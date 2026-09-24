#!/usr/bin/env python3
"""Draw the paper's Figure 1 (per-dataset accepted length and speedup, Qwen3-4B, T=0)
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
def label(m): return f'{m} (32)' if m in ['DDTree','DominoTree'] else m

cells = {(c['model'],c['temp'],c['dataset'],c['method']): c for c in json.load(open(sys.argv[1]))}
fig, axes = plt.subplots(1, 2, figsize=(12, 3.7), layout='constrained')
colors = ['#7d8791','#e8a347','#55a58a','#578ac0','#bf4d58']
for ax, metric, ylabel in zip(axes, ['tau','speedup'], ['Mean accepted length','Speedup over own AR']):
    for i, m in enumerate(METHODS):
        ax.bar(np.arange(8)+(i-2)*.16, [cells['4B','0.0',d,m][metric] for d in DS],
               width=.15, label=label(m), color=colors[i])
    ax.set_xticks(range(8), ['GSM8K','MATH','AIME','HEval','MBPP','LCB','MTB','Alpaca'], rotation=35, ha='right')
    ax.set_ylabel(ylabel); ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='outside upper center', ncol=5, frameon=False)
fig.savefig(sys.argv[2]); plt.close(fig)
print("wrote", sys.argv[2])
