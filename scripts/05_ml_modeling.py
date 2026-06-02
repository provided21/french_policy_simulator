"""
ML 建模脚本 — 4 种特征方案对比（回归 + 分类）

方案 A: 仅 8 个人口特征
方案 B: 人口特征 + persona_text 嵌入 384d（当前方案）
方案 C: 人口特征 + 10 个字段分别嵌入 → 逐字段 PCA→16d → 拼接 160d
方案 D: 人口特征 + 10 个字段分别嵌入 → 全拼接 3840d → 全局 PCA→128d

数据划分：Train 70% / Val 15% / Test 15%（所有方案用相同索引）
分类模型统一加 class_weight='balanced'

输出：方案对比表、Val/Test 排名、验证曲线、SHAP、学习曲线(2×2)、方案对比图
"""

import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import (
    train_test_split, cross_val_score, learning_curve, validation_curve,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
import shap
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 配置
# ============================================================
QUERY_ID = "20260514_120755"  # 随机抽样
PARQUET_PATH = "data/processed/df_full.parquet"
DB_PATH = "data/results.db"
OUTPUT_DIR = "output/ml_modeling_random"
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15

PCA_DIMS_PER_FIELD = 16    # 方案 C: 每个字段 PCA 降到的维数
PCA_DIMS_GLOBAL = 128      # 方案 D: 全局 PCA 降到的维数

TEXT_FIELDS = [
    "persona", "cultural_background", "professional_persona",
    "sports_persona", "arts_persona", "travel_persona",
    "culinary_persona", "hobbies_and_interests",
    "career_goals_and_ambitions", "skills_and_expertise",
]

FEATURE_COLS = ["age", "sex", "marital_status", "household_type",
                "occupation", "education_level", "departement", "commune"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

import sqlite3
conn = sqlite3.connect(DB_PATH)
df_resp = pd.read_sql_query(f"""
    SELECT persona_id, support_score, stance, similarity,
           age, sex, occupation, education_level, departement
    FROM responses WHERE query_id = '{QUERY_ID}' AND stance IS NOT NULL
""", conn)
conn.close()

df_full = pd.read_parquet(PARQUET_PATH)
merge_cols = ["persona_id", "marital_status", "household_type", "commune", "persona_text"] + TEXT_FIELDS
merge_cols = list(dict.fromkeys(merge_cols))
df_raw = df_resp.merge(df_full[merge_cols], on="persona_id", how="left")

for tf in TEXT_FIELDS:
    if tf in df_raw.columns:
        df_raw[tf] = df_raw[tf].fillna("")

print(f"  响应数据: {len(df_resp)} → 合并后: {len(df_raw)}")
print(f"  缺失值: {df_raw.isnull().sum().sum()}")

# ============================================================
# 1b. 文本嵌入: 10 个字段分别嵌入 + persona_text + PCA
# ============================================================
print("\n" + "=" * 60)
print("1b. 文本嵌入: 10 字段分别编码 + PCA 降维")
print("=" * 60)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "all-MiniLM-L6-v2")
embed_model = SentenceTransformer(MODEL_DIR)

# --- 方案 B: persona_text 整体嵌入 ---
print("  [方案B] 嵌入 persona_text...")
persona_texts = df_raw["persona_text"].fillna("").tolist()
emb_B = embed_model.encode(persona_texts, show_progress_bar=True,
                           batch_size=256, normalize_embeddings=True)
print(f"    persona_text → {emb_B.shape[1]}d")

# --- 方案 C/D: 逐字段嵌入 ---
field_embeddings = {}
print("  [方案C/D] 逐字段嵌入...")
for tf in TEXT_FIELDS:
    if tf not in df_raw.columns:
        print(f"    WARNING: '{tf}' 不在 parquet 中，跳过")
        continue
    texts = df_raw[tf].fillna("").tolist()
    field_embeddings[tf] = embed_model.encode(
        texts, show_progress_bar=False, batch_size=256, normalize_embeddings=True
    )
    print(f"    {tf}: {field_embeddings[tf].shape[1]}d")

valid_fields = [tf for tf in TEXT_FIELDS if tf in field_embeddings]
all_fields_emb = np.hstack([field_embeddings[tf] for tf in valid_fields])
print(f"  全字段拼接: {all_fields_emb.shape[1]}d (={all_fields_emb.shape[1]//len(valid_fields)}d×{len(valid_fields)}字段)")

# --- 方案 C: 逐字段 PCA ---
print(f"  [方案C] 逐字段 PCA → {PCA_DIMS_PER_FIELD}d ...")
field_pcas = {}
emb_C_parts = []
emb_C_cols = []
for tf in valid_fields:
    pca = PCA(n_components=PCA_DIMS_PER_FIELD, random_state=RANDOM_STATE)
    emb_pca = pca.fit_transform(field_embeddings[tf])
    field_pcas[tf] = pca
    emb_C_parts.append(emb_pca)
    emb_C_cols += [f"{tf}_pc{i}" for i in range(PCA_DIMS_PER_FIELD)]
emb_C = np.hstack(emb_C_parts)
print(f"    各字段 PCA → 拼接后 {emb_C.shape[1]}d")

# --- 方案 D: 全局 PCA ---
print(f"  [方案D] 全局 PCA → {PCA_DIMS_GLOBAL}d ...")
pca_global = PCA(n_components=PCA_DIMS_GLOBAL, random_state=RANDOM_STATE)
emb_D = pca_global.fit_transform(all_fields_emb)
emb_D_cols = [f"gpca_{i}" for i in range(PCA_DIMS_GLOBAL)]
print(f"    全字段 PCA → {emb_D.shape[1]}d")

# ============================================================
# 2. 特征工程: 构建 4 种方案的特征矩阵
# ============================================================
print("\n" + "=" * 60)
print("2. 特征工程: 构建 4 种方案特征矩阵")
print("=" * 60)

y_reg = df_raw["support_score"] * 10   # 0-1 → 0-10
y_cls = df_raw["stance"].map({"oppose": 0, "neutral": 1, "support": 2})

# 人口特征编码
X_demo = df_raw[FEATURE_COLS].copy()
encoders = {}
for col in FEATURE_COLS:
    if col != "age":
        le = LabelEncoder()
        X_demo[col] = X_demo[col].astype(str).fillna("unknown")
        X_demo[col] = le.fit_transform(X_demo[col])
        encoders[col] = le
X_demo["age"] = X_demo["age"].fillna(X_demo["age"].median())

# 方案 A: 仅人口特征
X_A = X_demo.copy()
feat_cols_A = list(FEATURE_COLS)

# 方案 B: 人口 + persona_text 384d
emb_cols_B = [f"txt_{i}" for i in range(emb_B.shape[1])]
X_B = pd.concat([X_demo, pd.DataFrame(emb_B, columns=emb_cols_B, index=X_demo.index)], axis=1)
feat_cols_B = FEATURE_COLS + emb_cols_B

# 方案 C: 人口 + 逐字段 PCA 160d
X_C = pd.concat([X_demo, pd.DataFrame(emb_C, columns=emb_C_cols, index=X_demo.index)], axis=1)
feat_cols_C = FEATURE_COLS + emb_C_cols

# 方案 D: 人口 + 全局 PCA 128d
X_D = pd.concat([X_demo, pd.DataFrame(emb_D, columns=emb_D_cols, index=X_demo.index)], axis=1)
feat_cols_D = FEATURE_COLS + emb_D_cols

approach_config = {
    "A": ("仅人口特征(8)", X_A, feat_cols_A),
    "B": ("人口+persona_text(8+384)", X_B, feat_cols_B),
    "C": (f"人口+10字段逐PCA(8+{emb_C.shape[1]})", X_C, feat_cols_C),
    "D": (f"人口+10字段全局PCA(8+{emb_D.shape[1]})", X_D, feat_cols_D),
}

print(f"  方案A 特征: {X_A.shape[1]}d")
print(f"  方案B 特征: {X_B.shape[1]}d")
print(f"  方案C 特征: {X_C.shape[1]}d")
print(f"  方案D 特征: {X_D.shape[1]}d")
print(f"  回归目标: [{y_reg.min():.1f}, {y_reg.max():.1f}]")
print(f"  分类分布: 反对={sum(y_cls==0)}  中立={sum(y_cls==1)}  支持={sum(y_cls==2)}")

# ============================================================
# 3. 数据划分（基于方案 B 的索引，所有方案共用）
# ============================================================
print("\n" + "=" * 60)
print("3. 数据划分: Train 70% / Val 15% / Test 15%（所有方案共用相同索引）")
print("=" * 60)

n = len(y_reg)
idx_temp, idx_test = train_test_split(
    np.arange(n),
    test_size=VAL_SIZE + (1 - TRAIN_SIZE - VAL_SIZE),
    random_state=RANDOM_STATE, stratify=y_cls
)
val_ratio = VAL_SIZE / (VAL_SIZE + (1 - TRAIN_SIZE - VAL_SIZE))
idx_train, idx_val = train_test_split(
    idx_temp,
    test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_cls.iloc[idx_temp]
)

print(f"  Train: {len(idx_train)}  Val: {len(idx_val)}  Test: {len(idx_test)}")

# ============================================================
# 4. 回归模型 — 4 方案对比
# ============================================================
print("\n" + "=" * 60)
print("4. 回归模型 — 4 方案对比（预测 support_score）")
print("=" * 60)

regression_models = {
    "Linear Regression": LinearRegression(),
    "Lasso (α=0.1)": Lasso(alpha=0.1, random_state=RANDOM_STATE),
    "Ridge (α=1.0)": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "ElasticNet (α=0.1)": ElasticNet(alpha=0.1, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12,
                                            random_state=RANDOM_STATE, n_jobs=1),
    "XGBoost": xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                 random_state=RANDOM_STATE, n_jobs=1, verbosity=0),
    "LightGBM": lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.05,
                                   random_state=RANDOM_STATE, verbose=-1),
    "SVR (RBF)": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "MLP (2×64)": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500,
                                random_state=RANDOM_STATE, early_stopping=True),
}

try:
    from catboost import CatBoostRegressor
    regression_models["CatBoost"] = CatBoostRegressor(
        n_estimators=200, depth=6, learning_rate=0.05,
        random_seed=RANDOM_STATE, verbose=0
    )
except ImportError:
    pass

reg_all_results = []   # 所有 (方案, 模型) 结果

for app_label, (app_desc, X_app, feat_cols_app) in approach_config.items():
    X_tr = X_app.iloc[idx_train]
    X_vl = X_app.iloc[idx_val]
    y_tr = y_reg.iloc[idx_train]
    y_vl = y_reg.iloc[idx_val]

    print(f"\n  --- 方案 {app_label}: {app_desc} ---")
    print(f"  {'模型':<25s} {'Val R²':>8s} {'Val MAE':>6s} {'5-Fold CV R²':>16s} {'耗时':>6s}")
    print("  " + "-" * 60)

    for name, model in regression_models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        y_val_pred = model.predict(X_vl)
        val_r2 = r2_score(y_vl, y_val_pred)
        val_mae = mean_absolute_error(y_vl, y_val_pred)
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2", n_jobs=1)
        elapsed = time.time() - t0

        reg_all_results.append({
            "方案": app_label,
            "方案描述": app_desc,
            "模型": name,
            "Val R²": val_r2,
            "Val MAE": val_mae,
            "5-Fold CV R²": f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
            "耗时(s)": f"{elapsed:.1f}",
        })
        print(f"  {name:<25s} {val_r2:>8.4f}  {val_mae:>6.3f}  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}   {elapsed:.0f}s")

df_reg_all = pd.DataFrame(reg_all_results).sort_values(["方案", "Val R²"], ascending=[True, False])

# 每方案最佳
print(f"\n  === 各方案 Val 最佳 ===")
for app_label in ["A", "B", "C", "D"]:
    best_row = df_reg_all[df_reg_all["方案"] == app_label].iloc[0]
    print(f"  方案{app_label}: {best_row['模型']}  Val R²={best_row['Val R²']:.4f}")

# ============================================================
# 5. 分类模型 — 4 方案对比（加 class_weight）
# ============================================================
print("\n" + "=" * 60)
print("5. 分类模型 — 4 方案对比（class_weight='balanced'）")
print("=" * 60)

classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                              class_weight="balanced"),
    "GaussianNB": GaussianNB(),       # sample_weight in fit()
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE,
                                             class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12,
                                             random_state=RANDOM_STATE, n_jobs=1,
                                             class_weight="balanced"),
    "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                  random_state=RANDOM_STATE, n_jobs=1, verbosity=0),
    "LightGBM": lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.05,
                                    random_state=RANDOM_STATE, verbose=-1,
                                    class_weight="balanced"),
}

# 预计算样本权重（用于 GaussianNB 和 XGBoost）
y_cls_train_for_sw = y_cls.iloc[idx_train]
sw_reg = compute_sample_weight(class_weight="balanced", y=y_cls_train_for_sw)

cls_all_results = []

for app_label, (app_desc, X_app, feat_cols_app) in approach_config.items():
    X_tr = X_app.iloc[idx_train]
    X_vl = X_app.iloc[idx_val]
    y_tr = y_cls.iloc[idx_train]
    y_vl = y_cls.iloc[idx_val]

    print(f"\n  --- 方案 {app_label}: {app_desc} ---")
    print(f"  {'模型':<25s} {'Val Acc':>8s} {'Val Macro F1':>12s} {'5-Fold CV Acc':>18s} {'耗时':>6s}")
    print("  " + "-" * 64)

    for name, model in classification_models.items():
        t0 = time.time()
        # GaussianNB 和 XGBoost 通过 sample_weight 处理类别不均衡
        if name in ("GaussianNB", "XGBoost"):
            model.fit(X_tr, y_tr, sample_weight=sw_reg)
        else:
            model.fit(X_tr, y_tr)

        y_val_pred = model.predict(X_vl)
        val_acc = accuracy_score(y_vl, y_val_pred)
        val_f1 = f1_score(y_vl, y_val_pred, average="macro")
        # CV 不加 sample_weight（sklearn 不会按 fold 子集化权重）
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring="accuracy", n_jobs=1)
        elapsed = time.time() - t0

        cls_all_results.append({
            "方案": app_label,
            "方案描述": app_desc,
            "模型": name,
            "Val Accuracy": val_acc,
            "Val Macro F1": val_f1,
            "5-Fold CV Acc": f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
            "耗时(s)": f"{elapsed:.1f}",
        })
        print(f"  {name:<25s} {val_acc:>8.4f}  {val_f1:>12.4f}  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}   {elapsed:.0f}s")

df_cls_all = pd.DataFrame(cls_all_results).sort_values(["方案", "Val Accuracy"], ascending=[True, False])

# 每方案最佳
print(f"\n  === 各方案 Val 最佳 ===")
for app_label in ["A", "B", "C", "D"]:
    subset = df_cls_all[df_cls_all["方案"] == app_label]
    if len(subset) > 0:
        best_row = subset.iloc[0]
        print(f"  方案{app_label}: {best_row['模型']}  Val Acc={best_row['Val Accuracy']:.4f}  Macro F1={best_row['Val Macro F1']:.4f}")

# ============================================================
# 6. Test 最终评估（所有方案最佳模型）
# ============================================================
print("\n" + "=" * 60)
print("6. Test 最终评估（4 方案 × 最佳模型）")
print("=" * 60)

test_summary = []
for app_label, (app_desc, X_app, feat_cols_app) in approach_config.items():
    X_te = X_app.iloc[idx_test]
    y_reg_te = y_reg.iloc[idx_test]
    y_cls_te = y_cls.iloc[idx_test]

    # 回归最佳模型
    subset_reg = df_reg_all[df_reg_all["方案"] == app_label]
    best_reg_name = subset_reg.iloc[0]["模型"]
    best_reg = regression_models[best_reg_name]
    best_reg.fit(X_app.iloc[idx_train], y_reg.iloc[idx_train])
    y_reg_pred = best_reg.predict(X_te)
    reg_r2 = r2_score(y_reg_te, y_reg_pred)
    reg_mae = mean_absolute_error(y_reg_te, y_reg_pred)
    reg_rmse = np.sqrt(mean_squared_error(y_reg_te, y_reg_pred))

    # 分类最佳模型
    subset_cls = df_cls_all[df_cls_all["方案"] == app_label]
    best_cls_name = subset_cls.iloc[0]["模型"]
    best_cls = classification_models[best_cls_name]
    if best_cls_name in ("GaussianNB", "XGBoost"):
        best_cls.fit(X_app.iloc[idx_train], y_cls.iloc[idx_train],
                     sample_weight=sw_reg)
    else:
        best_cls.fit(X_app.iloc[idx_train], y_cls.iloc[idx_train])
    y_cls_pred = best_cls.predict(X_te)
    cls_acc = accuracy_score(y_cls_te, y_cls_pred)
    cls_f1 = f1_score(y_cls_te, y_cls_pred, average="macro")

    test_summary.append({
        "方案": app_label,
        "方案描述": app_desc,
        "回归模型": best_reg_name,
        "Test R²": reg_r2,
        "Test MAE": reg_mae,
        "Test RMSE": reg_rmse,
        "分类模型": best_cls_name,
        "Test Acc": cls_acc,
        "Test Macro F1": cls_f1,
    })

    print(f"\n  方案{app_label} ({app_desc})")
    print(f"    回归 [{best_reg_name}]: R²={reg_r2:.4f}  MAE={reg_mae:.3f}  RMSE={reg_rmse:.3f}")
    print(f"    分类 [{best_cls_name}]: Acc={cls_acc:.4f}  Macro F1={cls_f1:.4f}")
    print(f"    分类报告:")
    print(classification_report(y_cls_te, y_cls_pred,
          target_names=["反对", "中立", "支持"], zero_division=0))

df_test = pd.DataFrame(test_summary)

# ============================================================
# 7. 方案对比图（核心输出）
# ============================================================
print("\n" + "=" * 60)
print("7. 方案对比图")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 回归 Test R² 对比
bar_colors_reg = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
axes[0].bar(df_test["方案"], df_test["Test R²"], color=bar_colors_reg, edgecolor="white", linewidth=0.5)
axes[0].set_ylabel("Test R²")
axes[0].set_title("回归: Test R² 方案对比")
axes[0].set_ylim(0, max(0.4, df_test["Test R²"].max() * 1.2))
for i, v in enumerate(df_test["Test R²"]):
    axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
# 标注方案描述
for i, (app_label, desc) in enumerate(zip(df_test["方案"], df_test["方案描述"])):
    axes[0].text(i, -0.03, desc, ha="center", fontsize=7, color="gray", rotation=0,
                 transform=axes[0].get_xaxis_transform())

# 分类 Test Macro F1 对比
bar_colors_cls = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
axes[1].bar(df_test["方案"], df_test["Test Macro F1"], color=bar_colors_cls, edgecolor="white", linewidth=0.5)
axes[1].set_ylabel("Test Macro F1")
axes[1].set_title("分类: Test Macro F1 方案对比")
axes[1].set_ylim(0, max(0.6, df_test["Test Macro F1"].max() * 1.2))
for i, v in enumerate(df_test["Test Macro F1"]):
    axes[1].text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
for i, (app_label, desc) in enumerate(zip(df_test["方案"], df_test["方案描述"])):
    axes[1].text(i, -0.03, desc, ha="center", fontsize=7, color="gray", rotation=0,
                 transform=axes[1].get_xaxis_transform())

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/approach_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  已保存: approach_comparison.png")

# ============================================================
# 8. 分类详细对比（每方案每模型 Val 排名）
# ============================================================
print("\n" + "=" * 60)
print("8. 分类模型 Val 排名（按方案分组）")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for ax, app_label in zip(axes, ["A", "B", "C", "D"]):
    app_desc = approach_config[app_label][0]
    subset = df_cls_all[df_cls_all["方案"] == app_label].sort_values("Val Macro F1", ascending=False)
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(subset))]
    ax.barh(range(len(subset)), subset["Val Macro F1"].values, color=colors)
    ax.set_yticks(range(len(subset)))
    ax.set_yticklabels(subset["模型"].values, fontsize=8)
    ax.set_xlabel("Val Macro F1")
    ax.set_title(f"方案{app_label}: {app_desc}", fontsize=10)
    ax.invert_yaxis()
    for i, v in enumerate(subset["Val Macro F1"].values):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlim(0, min(1.05, subset["Val Macro F1"].max() * 1.25))

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/classification_by_approach.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  已保存: classification_by_approach.png")

# ============================================================
# 9. 验证曲线 — 最佳方案的 XGBoost
# ============================================================
print("\n" + "=" * 60)
print("9. 验证曲线 — 最佳方案 XGBoost 回归")
print("=" * 60)

# 选择回归 Test R² 最高的方案
best_app_reg = df_test.sort_values("Test R²", ascending=False).iloc[0]
best_app_label = best_app_reg["方案"]
best_app_desc = best_app_reg["方案描述"]
X_best = approach_config[best_app_label][1]
X_tr_best = X_best.iloc[idx_train]
y_tr_best = y_reg.iloc[idx_train]

print(f"  最佳方案: {best_app_label} ({best_app_desc}), Test R²={best_app_reg['Test R²']:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
param_ranges = {
    "max_depth": np.arange(2, 14),
    "n_estimators": np.arange(50, 400, 50),
    "learning_rate": np.logspace(-2, 0, 5),
}

for ax, (param_name, param_range) in zip(axes, param_ranges.items()):
    train_scores, test_scores = validation_curve(
        xgb.XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, verbosity=0),
        X_tr_best, y_tr_best,
        param_name=param_name,
        param_range=param_range,
        cv=5, scoring="r2", n_jobs=1,
    )
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)
    ax.plot(param_range, train_mean, "o-", label="Train R²")
    ax.plot(param_range, test_mean, "s-", label="Val R²")
    ax.fill_between(param_range,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15)
    ax.fill_between(param_range,
                    test_scores.mean(axis=1) - test_scores.std(axis=1),
                    test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.15)
    ax.set_xlabel(param_name)
    ax.set_ylabel("R²")
    ax.set_title(f"验证曲线: {param_name} ({best_app_label})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if param_name == "learning_rate":
        ax.set_xscale("log")

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/validation_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  已保存: validation_curves.png")

# ============================================================
# 10. SHAP 分析 — 方案C（逐字段PCA，特征名含字段来源，可解释）
# ============================================================
print("\n" + "=" * 60)
print("10. SHAP 可解释性分析（方案C: 人口+10字段逐PCA）")
print("=" * 60)

shap_app_label = "C"
_, X_shap, feat_cols_shap = approach_config[shap_app_label]
X_tr_shap = X_shap.iloc[idx_train]
X_te_shap = X_shap.iloc[idx_test]
y_tr_shap = y_reg.iloc[idx_train]

shap_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                               random_state=RANDOM_STATE, verbosity=0)
shap_model.fit(X_tr_shap, y_tr_shap)

print(f"  方案: {shap_app_label}  模型: XGBoost  Test R²={r2_score(y_reg.iloc[idx_test], shap_model.predict(X_te_shap)):.4f}")

explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_te_shap[:500])
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Summary plot
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_te_shap[:500], feature_names=feat_cols_shap,
                  show=False, max_display=20)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# Bar plot
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_te_shap[:500], feature_names=feat_cols_shap,
                  plot_type="bar", show=False, max_display=20)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  已保存: shap_summary.png, shap_bar.png")

# ============================================================
# 11. 学习曲线 2×2（4 种方案各一条）
# ============================================================
print("\n" + "=" * 60)
print("11. 学习曲线 2×2（4 方案对比）")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for ax, app_label in zip(axes, ["A", "B", "C", "D"]):
    app_desc = approach_config[app_label][0]
    X_app = approach_config[app_label][1]
    X_tr_app = X_app.iloc[idx_train]
    y_tr_app = y_reg.iloc[idx_train]

    train_sizes, train_scores, test_scores = learning_curve(
        xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05,
                         random_state=RANDOM_STATE, verbosity=0),
        X_tr_app, y_tr_app, cv=5, scoring="r2",
        train_sizes=np.linspace(0.2, 1.0, 5), random_state=RANDOM_STATE, n_jobs=1
    )
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)
    ax.fill_between(train_sizes, train_mean - train_scores.std(axis=1),
                    train_mean + train_scores.std(axis=1), alpha=0.15)
    ax.fill_between(train_sizes, test_mean - test_scores.std(axis=1),
                    test_mean + test_scores.std(axis=1), alpha=0.15)
    ax.plot(train_sizes, train_mean, "o-", label="Train R²")
    ax.plot(train_sizes, test_mean, "s-", label="Val R²")
    ax.set_xlabel("训练样本数")
    ax.set_ylabel("R²")
    ax.set_title(f"方案{app_label}: {app_desc}", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  已保存: learning_curves.png")

# ============================================================
# 12. 汇总
# ============================================================
print("\n" + "=" * 60)
print("12. 完成汇总")
print("=" * 60)

print(f"\n  === 方案 Test 对比汇总 ===")
print(f"  {'方案':<3s} {'方案描述':<35s} {'回归模型':<20s} {'Test R²':>8s} {'分类模型':<20s} {'Test F1':>8s}")
print(f"  {'-' * 100}")
for _, row in df_test.iterrows():
    print(f"  {row['方案']:<3s} {row['方案描述']:<35s} {row['回归模型']:<20s} {row['Test R²']:>8.4f} {row['分类模型']:<20s} {row['Test Macro F1']:>8.4f}")

print(f"\n  === Val 排名表（回归 Top 3 / 方案）===")
for app_label in ["A", "B", "C", "D"]:
    subset = df_reg_all[df_reg_all["方案"] == app_label].head(3)
    print(f"\n  方案{app_label}:")
    print(subset[["模型", "Val R²", "Val MAE", "5-Fold CV R²"]].to_string(index=False))

print(f"\n  === Val 排名表（分类 Top 3 / 方案）===")
for app_label in ["A", "B", "C", "D"]:
    subset = df_cls_all[df_cls_all["方案"] == app_label].head(3)
    print(f"\n  方案{app_label}:")
    print(subset[["模型", "Val Accuracy", "Val Macro F1", "5-Fold CV Acc"]].to_string(index=False))

print(f"\n  输出文件 ({OUTPUT_DIR}/):")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"    {f}  ({size_kb:.0f} KB)")
