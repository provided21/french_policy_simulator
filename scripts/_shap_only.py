"""Regenerate SHAP figures using approach C (interpretable per-field PCA names)."""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import xgboost as xgb
import shap
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "ml_modeling")
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
PCA_DIMS_PER_FIELD = 16

TEXT_FIELDS = [
    "persona", "cultural_background", "professional_persona",
    "sports_persona", "arts_persona", "travel_persona",
    "culinary_persona", "hobbies_and_interests",
    "career_goals_and_ambitions", "skills_and_expertise",
]
FEATURE_COLS = ["age", "sex", "marital_status", "household_type",
                "occupation", "education_level", "departement", "commune"]

# --- Load data ---
import sqlite3
conn = sqlite3.connect(os.path.join(PROJECT_ROOT, "data", "results.db"))
df_resp = pd.read_sql_query("""
    SELECT persona_id, support_score, stance, age, sex, occupation,
           education_level, departement
    FROM responses
    WHERE query_id = '20260511_022424' AND stance IS NOT NULL
""", conn)
conn.close()

df_full = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "processed", "df_full.parquet"))
merge_cols = ["persona_id", "marital_status", "household_type", "commune", "persona_text"] + TEXT_FIELDS
merge_cols = list(dict.fromkeys(merge_cols))
df_raw = df_resp.merge(df_full[merge_cols], on="persona_id", how="left")
for tf in TEXT_FIELDS:
    if tf in df_raw.columns:
        df_raw[tf] = df_raw[tf].fillna("")

y_reg = df_raw["support_score"] * 10
y_cls = df_raw["stance"].map({"oppose": 0, "neutral": 1, "support": 2})

# --- Encode demographics ---
X_demo = df_raw[FEATURE_COLS].copy()
for col in FEATURE_COLS:
    if col != "age":
        X_demo[col] = X_demo[col].astype(str).fillna("unknown")
        X_demo[col] = LabelEncoder().fit_transform(X_demo[col])
X_demo["age"] = X_demo["age"].fillna(X_demo["age"].median())

# --- Load local model ---
MODEL_DIR = os.path.join(PROJECT_ROOT, "all-MiniLM-L6-v2")
embed_model = SentenceTransformer(MODEL_DIR)

# --- Embed 10 fields ---
field_embeddings = {}
for tf in TEXT_FIELDS:
    if tf not in df_raw.columns:
        continue
    texts = df_raw[tf].fillna("").tolist()
    field_embeddings[tf] = embed_model.encode(texts, show_progress_bar=False,
                                              batch_size=256, normalize_embeddings=True)

valid_fields = [tf for tf in TEXT_FIELDS if tf in field_embeddings]

# --- Approach C: per-field PCA ---
emb_C_parts = []
emb_C_cols = []
for tf in valid_fields:
    pca = PCA(n_components=PCA_DIMS_PER_FIELD, random_state=RANDOM_STATE)
    emb_pca = pca.fit_transform(field_embeddings[tf])
    emb_C_parts.append(emb_pca)
    emb_C_cols += [f"{tf}_pc{i}" for i in range(PCA_DIMS_PER_FIELD)]
emb_C = np.hstack(emb_C_parts)
feat_cols_C = FEATURE_COLS + emb_C_cols
X_C = pd.concat([X_demo, pd.DataFrame(emb_C, columns=emb_C_cols, index=X_demo.index)], axis=1)

# --- Train/test split (same as main script) ---
n = len(y_reg)
idx_temp, idx_test = train_test_split(np.arange(n), test_size=0.3,
                                      random_state=RANDOM_STATE, stratify=y_cls)
idx_train, idx_val = train_test_split(idx_temp, test_size=VAL_SIZE/(TRAIN_SIZE+VAL_SIZE),
                                       random_state=RANDOM_STATE, stratify=y_cls.iloc[idx_temp])

# --- Train XGBoost on approach C ---
X_tr = X_C.iloc[idx_train]
X_te = X_C.iloc[idx_test]
y_tr = y_reg.iloc[idx_train]
y_te = y_reg.iloc[idx_test]

model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                          random_state=RANDOM_STATE, verbosity=0)
model.fit(X_tr, y_tr)
print(f"Approach C XGBoost Test R2 = {r2_score(y_te, model.predict(X_te)):.4f}")

# --- SHAP ---
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_te[:500])
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# SHAP summary
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_te[:500], feature_names=feat_cols_C,
                  show=False, max_display=20)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()

# SHAP bar
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_te[:500], feature_names=feat_cols_C,
                  plot_type="bar", show=False, max_display=20)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

# SHAP dependence for top 2 features
top_feat_idx = np.argsort(np.abs(shap_values).mean(0))[::-1]
for rank, idx in enumerate(top_feat_idx[:2]):
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.dependence_plot(idx, shap_values, X_te[:500], feature_names=feat_cols_C,
                         show=False)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"shap_dependence_{rank+1}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

print("SHAP figures regenerated: shap_summary, shap_bar, shap_dependence_1/2")
