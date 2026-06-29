# -*- coding: utf-8 -*-
"""Intra-class vs cross-class cosine similarity — PPT page 9 figure."""
import os, sys, numpy as np, pandas as pd, sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Chinese font ───────────────────
font_candidates = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']
zh_font = None
for f in font_candidates:
    matches = [x for x in fm.findSystemFonts() if f.lower() in x.lower().replace(' ','')]
    if matches:
        zh_font = fm.FontProperties(fname=matches[0])
        break
if zh_font is None:
    zh_font = fm.FontProperties()
plt.rcParams.update({'font.family': zh_font.get_name() if zh_font else 'sans-serif',
                     'axes.unicode_minus': False, 'figure.dpi': 150,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1})

CLR_DARK, CLR_WHITE, CLR_GREY = '#0B0E14', '#E8ECF0', '#808894'
CLR_GREEN, CLR_RED, CLR_ORANGE = '#34D399', '#F05050', '#F0A030'

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, 'output', 'figures')
os.makedirs(OUT, exist_ok=True)

# ── Load ───────────────────────────
print('Loading responses...')
conn = sqlite3.connect(os.path.join(PROJECT, 'data', 'results.db'))
df_resp = pd.read_sql_query("""
    SELECT persona_id, stance FROM responses
    WHERE query_id = '20260511_022424' AND stance IS NOT NULL
""", conn)
conn.close()
print(f'  {len(df_resp)} responses')

print('Loading persona_text (2 cols only)...')
df_full = pd.read_parquet(os.path.join(PROJECT, 'data', 'processed', 'df_full.parquet'),
                           columns=['persona_id', 'persona_text'])
df = df_resp.merge(df_full, on='persona_id', how='inner')
print(f'  {len(df)} merged')

# ── Embed ──────────────────────────
print('Encoding...')
MODEL_DIR = os.path.join(PROJECT, 'all-MiniLM-L6-v2')
model = SentenceTransformer(MODEL_DIR)
N = 200

results = {}
for stance in ['support', 'neutral', 'oppose']:
    texts = df[df['stance'] == stance]['persona_text'].dropna().head(N).tolist()
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    sim = emb @ emb.T
    np.fill_diagonal(sim, 0)
    results[stance] = {'emb': emb, 'intra': float(sim.mean()), 'n': len(texts)}
    print(f'  {stance}: n={len(texts)}, intra={results[stance]["intra"]:.4f}')

for s1, s2 in [('support','neutral'),('support','oppose'),('neutral','oppose')]:
    cross = float((results[s1]['emb'] @ results[s2]['emb'].T).mean())
    results[f'{s1}_vs_{s2}'] = cross
    print(f'  {s1} vs {s2}: cross={cross:.4f}')

# ── Plot: 2 bars only ──────────────
intra_vals = [results['support']['intra'], results['neutral']['intra'],
              results['oppose']['intra']]
cross_vals = [results['support_vs_neutral'], results['support_vs_oppose'],
              results['neutral_vs_oppose']]
avg_intra = np.mean(intra_vals)
avg_cross = np.mean(cross_vals)

fig, ax = plt.subplots(figsize=(6, 4.5))
fig.patch.set_facecolor(CLR_DARK); ax.set_facecolor(CLR_DARK)

# Two clean bars
bars = ax.bar(['类内相似度', '跨类相似度'], [avg_intra, avg_cross],
              color=[CLR_GREEN, CLR_ORANGE], width=0.42, alpha=0.88, edgecolor=None)

# Big value labels inside bars
for bar, val in zip(bars, [avg_intra, avg_cross]):
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()/2.,
            f'{val:.4f}', ha='center', va='center', fontsize=22, color=CLR_DARK,
            fontweight='bold')

# Scatter individual values on top
for i, v in enumerate(intra_vals):
    ax.scatter(bars[0].get_x()+bars[0].get_width()/2., v, color=CLR_WHITE, s=50, zorder=5)
for i, v in enumerate(cross_vals):
    ax.scatter(bars[1].get_x()+bars[1].get_width()/2., v, color=CLR_WHITE, s=50, zorder=5)

# Delta annotation
diff = avg_intra - avg_cross
y_mid = max(avg_intra, avg_cross) + 0.015
ax.annotate('', xy=(0, y_mid-0.002), xytext=(1, y_mid-0.002),
            arrowprops=dict(arrowstyle='<->', color=CLR_GREY, lw=1.2))
ax.text(0.5, y_mid+0.002, f'Δ = {diff:.4f}', ha='center', fontsize=13,
        color=CLR_WHITE, fontweight='bold')

# One-line conclusion below
ax.text(0.5, -0.08, '支持者、中立者、反对者的 persona_text 在语义空间中无法区分',
        ha='center', fontsize=12, color=CLR_RED, fontproperties=zh_font,
        transform=ax.transAxes)

ax.set_ylim(0.50, y_mid+0.02)
ax.set_ylabel('余弦相似度', color=CLR_GREY, fontsize=12, fontproperties=zh_font)
ax.set_title('三类立场人物的 persona_text 嵌入相似度', color=CLR_WHITE,
             fontsize=15, fontproperties=zh_font, pad=14)
ax.tick_params(colors=CLR_GREY, labelsize=12)
for spine in ax.spines.values(): spine.set_visible(False)
ax.tick_params(bottom=False)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'semantic_overlap.png'), facecolor=CLR_DARK, dpi=150)
plt.close()
print(f'\nDone: {os.path.join(OUT, "semantic_overlap.png")}')
