"""
深度诊断：为什么 R² 卡在 0.27？
"""
import sqlite3, pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# 1. 检查 parquet 中有哪些列
# ============================================================
print("=" * 60)
print("1. Parquet 中的列")
print("=" * 60)
df_full = pd.read_parquet("data/processed/df_full.parquet")
print(f"总列数: {len(df_full.columns)}")
print(f"列名: {list(df_full.columns)}")

# ============================================================
# 2. 检查 persona_text 包含哪些信息
# ============================================================
print("\n" + "=" * 60)
print("2. persona_text 内容分析")
print("=" * 60)
sample_text = df_full["persona_text"].iloc[0]
print(f"persona_text 长度: {len(sample_text)} 字符")
print(f"前 500 字符:\n{sample_text[:500]}")
print(f"\n...后 200 字符:\n{sample_text[-200:]}")

# ============================================================
# 3. 检查文本字段是否被单独存储
# ============================================================
print("\n" + "=" * 60)
print("3. 各文本字段的可用性")
print("=" * 60)
text_fields = ["persona", "cultural_background", "professional_persona",
               "sports_persona", "arts_persona", "travel_persona",
               "culinary_persona", "hobbies_and_interests",
               "career_goals_and_ambitions", "skills_and_expertise"]
for f in text_fields:
    if f in df_full.columns:
        non_null = df_full[f].notna().sum()
        sample = df_full[f].dropna().iloc[0] if non_null > 0 else "N/A"
        sample_str = str(sample)[:80]
        print(f"  ✓ {f}: {non_null}/{len(df_full)} non-null, 示例: {sample_str}...")
    else:
        print(f"  ✗ {f}: 不存在于 parquet")

# ============================================================
# 4. 加载 DB 数据，分析特征与 stance 的关系
# ============================================================
print("\n" + "=" * 60)
print("4. 合并数据，分析每类特征的解释力")
print("=" * 60)
conn = sqlite3.connect("data/results.db")
df_resp = pd.read_sql_query("""
    SELECT persona_id, support_score, stance, age, sex, occupation,
           education_level, departement
    FROM responses WHERE query_id = '20260511_022424' AND stance IS NOT NULL
""", conn)
conn.close()

df = df_resp.merge(df_full, on="persona_id", how="left", suffixes=("_db", ""))
print(f"合并后: {len(df)} 条, 列数: {len(df.columns)}")

# 修复列名冲突
for c in ["age", "sex", "occupation", "education_level", "departement"]:
    if f"{c}_db" in df.columns:
        df[c] = df[f"{c}_db"]
        df.drop(columns=[f"{c}_db"], inplace=True)

# ============================================================
# 5. 检查 LLM 随机性 — 有相似画像的人得分是否一致？
# ============================================================
print("\n" + "=" * 60)
print("5. 相似画像的一致性检查")
print("=" * 60)

# 按完全相同的特征组合分组
demo_cols = ["age", "sex", "occupation", "education_level", "departement",
             "marital_status", "household_type", "commune"]
for c in demo_cols:
    if c in df.columns:
        df[c] = df[c].astype(str)

groups = df.groupby(demo_cols)
group_sizes = groups.size()
print(f"唯一个人画像组合数: {len(groups)} / {len(df)} 总样本")
print(f"最大重复组: {group_sizes.max()} 人")

# 对重复组，检查 support_score 的方差
multi_groups = group_sizes[group_sizes >= 3]
print(f"\n≥3人的重复组: {len(multi_groups)} 个")
if len(multi_groups) > 0:
    within_group_stds = []
    for name, _ in multi_groups.items():
        group_scores = groups.get_group(name)["support_score"]
        within_group_stds.append(group_scores.std())
    avg_within_std = np.mean(within_group_stds)
    print(f"重复组内 support_score 平均标准差: {avg_within_std:.4f}")
    print(f"（总标准差: {df['support_score'].std():.4f}）")
    print(f"→ 组内方差占比: {avg_within_std**2 / df['support_score'].var():.4f}")

# ============================================================
# 6. 检查文本字段是否比人口统计字段更有区分力
# ============================================================
print("\n" + "=" * 60)
print("6. 按 stance 分组的文本字段差异")
print("=" * 60)

# 分析不同 stance 的人在 persona_text 上是否有明显差异
# 用 SentenceTransformer 编码后看聚类
from sentence_transformers import SentenceTransformer
import os
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "all-MiniLM-L6-v2")
model = SentenceTransformer(MODEL_DIR)

# 每类 stance 取 50 个样本来比较
for stance_name in ["support", "neutral", "oppose"]:
    subset = df[df["stance"] == stance_name]["persona_text"].dropna().head(50)
    if len(subset) > 0:
        emb = model.encode(subset.tolist(), normalize_embeddings=True)
        # 计算类内平均相似度（类内紧密度）
        sim_matrix = emb @ emb.T
        np.fill_diagonal(sim_matrix, 0)
        avg_sim = sim_matrix.mean()
        print(f"  {stance_name}: 类内平均余弦相似度 = {avg_sim:.4f}, 样本数={len(subset)}")

# 跨类相似度
for s1, s2 in [("support", "neutral"), ("support", "oppose"), ("neutral", "oppose")]:
    texts1 = df[df["stance"] == s1]["persona_text"].dropna().head(50).tolist()
    texts2 = df[df["stance"] == s2]["persona_text"].dropna().head(50).tolist()
    if texts1 and texts2:
        emb1 = model.encode(texts1, normalize_embeddings=True)
        emb2 = model.encode(texts2, normalize_embeddings=True)
        cross_sim = (emb1 @ emb2.T).mean()
        print(f"  {s1} vs {s2}: 跨类平均余弦相似度 = {cross_sim:.4f}")

# ============================================================
# 7. 检查 temperature 的影响
# ============================================================
print("\n" + "=" * 60)
print("7. 关于 LLM temperature=0.7 的影响")
print("=" * 60)
print("temperature=0.7 → LLM 输出有随机性")
print("同一 persona 两次调用可能得到不同 score")
print("这是 R² 上不去的重要原因之一")
print(f"\n如果 LLM 的随机噪声标准差为 σ_noise，")
print(f"总方差 = 信号方差 + σ_noise²")
print(f"当前 total_var = {df['support_score'].var():.4f}")
print(f"R²上限 = 信号方差 / 总方差")
print(f"如果 σ_noise 占比大，R² 上限就被锁死了")

# ============================================================
# 8. 总结
# ============================================================
print("\n" + "=" * 60)
print("8. 初步诊断")
print("=" * 60)
print("""
可能原因（按可能性排序）:

A. LLM 随机性 (temperature=0.7)
   同一画像 → 不同回答 → 无法被特征捕捉的方差

B. 特征信息不足
   10 个文本字段（persona, cultural_background等）未被单独利用
   它们可能比人口统计特征更有区分力

C. 问题是分类不是回归
   support_score 本质是 3 个簇，不是连续变量
   回归模型被大量"支持"样本主导

D. 样本极度不平衡
   72.8% 支持 vs 9.3% 反对
   模型学会"全猜支持"就能拿 73% accuracy
""")
