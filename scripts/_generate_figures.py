# -*- coding: utf-8 -*-
"""Generate figures for PPT: EDA chart + sampling comparison chart."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Find Chinese font
font_candidates = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei']
zh_font = None
for f in font_candidates:
    matches = [x for x in fm.findSystemFonts() if f.lower() in x.lower().replace(' ', '')]
    if matches:
        zh_font = fm.FontProperties(fname=matches[0])
        break
if zh_font is None:
    zh_font = fm.FontProperties()

plt.rcParams.update({'font.family': zh_font.get_name() if zh_font else 'sans-serif',
                     'axes.unicode_minus': False,
                     'figure.dpi': 150,
                     'savefig.bbox': 'tight',
                     'savefig.pad_inches': 0.1})

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'figures')
os.makedirs(OUT, exist_ok=True)

CLR_DARK  = '#0B0E14'
CLR_CARD  = '#141822'
CLR_WHITE = '#E8ECF0'
CLR_GREY  = '#808894'
CLR_ACC   = '#4E94E8'
CLR_GREEN = '#34D399'
CLR_ORANGE = '#F0A030'
CLR_RED   = '#F05050'

# ════════════════════════════════
# 1. EDA - Age distribution
# ════════════════════════════════
print('Generating age distribution...')
import pandas as pd
from src.utils import Config
df = pd.read_parquet(Config.DATA_PATH, columns=['age'])

fig, ax = plt.subplots(figsize=(6, 3.5))
fig.patch.set_facecolor(CLR_DARK)
ax.set_facecolor(CLR_DARK)
ax.hist(df['age'].dropna(), bins=range(18, 107), color=CLR_ACC, edgecolor=CLR_CARD, alpha=0.85)
ax.set_title('法国合成画像 - 年龄分布', color=CLR_WHITE, fontsize=14, fontproperties=zh_font, pad=10)
ax.set_xlabel('年龄', color=CLR_GREY, fontsize=10, fontproperties=zh_font)
ax.set_ylabel('人数', color=CLR_GREY, fontsize=10, fontproperties=zh_font)
ax.tick_params(colors=CLR_GREY, labelsize=8)
for spine in ax.spines.values(): spine.set_color(CLR_CARD)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'eda_age.png'), facecolor=CLR_DARK, dpi=150)
plt.close()
print('  -> eda_age.png')

# ════════════════════════════════
# 2. EDA - Occupation distribution (top 10)
# ════════════════════════════════
print('Generating occupation distribution...')
df_occ = pd.read_parquet(Config.DATA_PATH, columns=['occupation'])
occ_counts = df_occ['occupation'].value_counts().head(10).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(6, 3.5))
fig.patch.set_facecolor(CLR_DARK)
ax.set_facecolor(CLR_DARK)
colors_bar = [CLR_ACC] * len(occ_counts)
ax.barh(range(len(occ_counts)), occ_counts.values, color=colors_bar, height=0.65)
ax.set_yticks(range(len(occ_counts)))
ax.set_yticklabels(occ_counts.index, fontsize=8, color=CLR_WHITE)
ax.set_title('法国合成画像 - 职业分布 (Top 10)', color=CLR_WHITE, fontsize=14, fontproperties=zh_font, pad=10)
ax.set_xlabel('人数', color=CLR_GREY, fontsize=10, fontproperties=zh_font)
ax.tick_params(colors=CLR_GREY, labelsize=8)
for spine in ax.spines.values(): spine.set_color(CLR_CARD)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'eda_occupation.png'), facecolor=CLR_DARK, dpi=150)
plt.close()
print('  -> eda_occupation.png')

# ════════════════════════════════
# 3. Sampling comparison bar chart
# ════════════════════════════════
print('Generating sampling comparison chart...')
categories = ['支持\n(回到62岁)', '中立\n(维持64岁)', '反对\n(进一步提高)']
ifop   = [61, 34, 5]
sim    = [72.8, 17.9, 9.3]
random = [43.4, 37.2, 19.5]

x = np.arange(len(categories))
w = 0.25

fig, ax = plt.subplots(figsize=(7, 3.8))
fig.patch.set_facecolor(CLR_DARK)
ax.set_facecolor(CLR_DARK)

bars1 = ax.bar(x - w, ifop, w, label='IFOP 真实调查', color=CLR_GREY, alpha=0.8)
bars2 = ax.bar(x, sim, w, label='语义检索', color=CLR_GREEN, alpha=0.85)
bars3 = ax.bar(x + w, random, w, label='随机抽样', color=CLR_RED, alpha=0.7)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.8, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=9, color=CLR_WHITE)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10, color=CLR_WHITE)
ax.set_ylabel('百分比 (%)', color=CLR_GREY, fontsize=10, fontproperties=zh_font)
ax.set_title('抽样策略对比：LLM 输出 vs IFOP 真实调查', color=CLR_WHITE, fontsize=14, fontproperties=zh_font, pad=10)
ax.legend(fontsize=9, loc='upper right', facecolor=CLR_CARD, edgecolor=CLR_CARD, labelcolor=CLR_WHITE)
ax.set_ylim(0, 85)
ax.tick_params(colors=CLR_GREY, labelsize=9)
for spine in ax.spines.values(): spine.set_color(CLR_CARD)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'sampling_comparison.png'), facecolor=CLR_DARK, dpi=150)
plt.close()
print('  -> sampling_comparison.png')

# ════════════════════════════════
# 4. Prompt iteration comparison bar chart
# ════════════════════════════════
print('Generating prompt comparison chart...')
runs = ['Run 1\n(强制表态)', 'Run 2\n(法国语境)', 'Run 3\n(纯画像)', 'IFOP\n(真实调查)']
support = [89, 72.8, 28.9, 61]
colors_run = [CLR_RED, CLR_GREEN, CLR_ORANGE, CLR_GREY]

fig, ax = plt.subplots(figsize=(7, 3.5))
fig.patch.set_facecolor(CLR_DARK)
ax.set_facecolor(CLR_DARK)
bars = ax.bar(runs, support, color=colors_run, width=0.5, alpha=0.85)
for bar, val in zip(bars, support):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5, f'{val}%',
            ha='center', fontsize=12, color=CLR_WHITE, fontweight='bold')
# Add deviation labels
deviations = ['+28pp', '+11.8pp', '-32.1pp', '基准']
for i, (bar, d) in enumerate(zip(bars, deviations)):
    clr = CLR_GREEN if i == 1 else (CLR_RED if d.startswith('+') or d.startswith('-') and i != 3 else CLR_GREY)
    if i != 3:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2., d,
                ha='center', fontsize=10, color=clr, fontweight='bold')

ax.set_title('Prompt 三轮迭代：支持「回到 62 岁」的比例', color=CLR_WHITE, fontsize=14, fontproperties=zh_font, pad=10)
ax.set_ylabel('支持率 (%)', color=CLR_GREY, fontsize=10, fontproperties=zh_font)
ax.set_ylim(0, 100)
ax.tick_params(colors=CLR_GREY, labelsize=10)
for spine in ax.spines.values(): spine.set_color(CLR_CARD)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'prompt_comparison.png'), facecolor=CLR_DARK, dpi=150)
plt.close()
print('  -> prompt_comparison.png')

print(f'\nAll figures saved to: {OUT}')
