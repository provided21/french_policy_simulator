# -*- coding: utf-8 -*-
"""Quick clean chart from pre-computed data — no re-encoding needed."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── Font ───────────────────────────
for f in ['Microsoft YaHei', 'SimHei']:
    matches = [x for x in fm.findSystemFonts() if f.lower() in x.lower().replace(' ','')]
    if matches:
        zh = fm.FontProperties(fname=matches[0])
        break
else:
    zh = fm.FontProperties()

plt.rcParams.update({'font.family': zh.get_name(), 'axes.unicode_minus': False,
                     'figure.dpi': 200, 'savefig.bbox': 'tight'})

# ── Data (from run output) ─────────
intra = {'支持': 0.5399, '中立': 0.5508, '反对': 0.5568}
cross = {'支持-中立': 0.5480, '支持-反对': 0.5509, '中立-反对': 0.5569}
avg_intra = np.mean(list(intra.values()))
avg_cross = np.mean(list(cross.values()))

# ── Plot ───────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4))

x = [0, 1]
bars = ax.bar(x, [avg_intra, avg_cross], width=0.35,
              color=['#2E86C1', '#E67E22'], edgecolor='white', linewidth=1.2)

# Big value labels inside bars
for bar, val in zip(bars, [avg_intra, avg_cross]):
    ax.text(bar.get_x()+bar.get_width()/2., val/2, f'{val:.4f}',
            ha='center', va='center', fontsize=24, fontweight='bold', color='white')

# Individual dots
for v in intra.values():
    ax.scatter(0, v, color='#1a5276', s=70, zorder=5, edgecolor='white', linewidth=0.8)
for v in cross.values():
    ax.scatter(1, v, color='#a04000', s=70, zorder=5, edgecolor='white', linewidth=0.8)

# Delta
y_top = max(avg_intra, avg_cross) + 0.01
ax.plot([0, 0, 1, 1], [y_top-0.003, y_top, y_top, y_top-0.003],
        color='#555', lw=1)
ax.text(0.5, y_top+0.002, f'Δ = {abs(avg_intra-avg_cross):.4f}',
        ha='center', fontsize=13, fontweight='bold', color='#C0392B')

# Labels
ax.set_xticks(x)
ax.set_xticklabels(['类内相似度\n(同立场内)', '跨类相似度\n(不同立场间)'],
                   fontsize=12, fontproperties=zh)
ax.set_ylabel('余弦相似度', fontsize=12, fontproperties=zh)
ax.set_ylim(0.50, y_top+0.018)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#ccc')
ax.spines['bottom'].set_color('#ccc')
ax.tick_params(colors='#555', bottom=False)

plt.tight_layout()
fig.savefig('D:/Sophomore Year/Spring/Data Science and Data Analysis/french_policy_simulator/output/figures/semantic_overlap.png',
            facecolor='white', dpi=200)
plt.close()
print('Done')
