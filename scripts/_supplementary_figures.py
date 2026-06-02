"""Generate supplementary figures for the paper appendix."""
import os, sys, warnings, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "ml_modeling")
PAPER_FIG_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "paper", "figures")
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
PCA_DIMS_PER_FIELD = 16
PCA_DIMS_GLOBAL = 128

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
df_resp = pd.read_sql_query(f"""
    SELECT persona_id, support_score, stance,
           age, sex, occupation, education_level, departement
    FROM responses WHERE query_id = '20260511_022424' AND stance IS NOT NULL
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

# --- Load embedding model ---
MODEL_DIR = os.path.join(PROJECT_ROOT, "all-MiniLM-L6-v2")
embed_model = SentenceTransformer(MODEL_DIR)

# --- Embed ---
persona_texts = df_raw["persona_text"].fillna("").tolist()
emb_B = embed_model.encode(persona_texts, show_progress_bar=True, batch_size=256, normalize_embeddings=True)

field_embeddings = {}
for tf in TEXT_FIELDS:
    if tf not in df_raw.columns:
        continue
    texts = df_raw[tf].fillna("").tolist()
    field_embeddings[tf] = embed_model.encode(texts, show_progress_bar=False, batch_size=256, normalize_embeddings=True)

valid_fields = [tf for tf in TEXT_FIELDS if tf in field_embeddings]
all_fields_emb = np.hstack([field_embeddings[tf] for tf in valid_fields])

# --- Per-field PCA (C) ---
emb_C_parts = []
emb_C_cols = []
for tf in valid_fields:
    pca = PCA(n_components=PCA_DIMS_PER_FIELD, random_state=RANDOM_STATE)
    emb_pca = pca.fit_transform(field_embeddings[tf])
    emb_C_parts.append(emb_pca)
    emb_C_cols += [f"{tf}_pc{i}" for i in range(PCA_DIMS_PER_FIELD)]
emb_C = np.hstack(emb_C_parts)

# --- Global PCA (D) ---
pca_global = PCA(n_components=PCA_DIMS_GLOBAL, random_state=RANDOM_STATE)
emb_D = pca_global.fit_transform(all_fields_emb)
emb_D_cols = [f"gpca_{i}" for i in range(PCA_DIMS_GLOBAL)]

# --- Build X matrices ---
emb_cols_B = [f"txt_{i}" for i in range(emb_B.shape[1])]
X_A = X_demo.copy()
X_B = pd.concat([X_demo, pd.DataFrame(emb_B, columns=emb_cols_B, index=X_demo.index)], axis=1)
X_C = pd.concat([X_demo, pd.DataFrame(emb_C, columns=emb_C_cols, index=X_demo.index)], axis=1)
X_D = pd.concat([X_demo, pd.DataFrame(emb_D, columns=emb_D_cols, index=X_demo.index)], axis=1)

approach_config = {
    "A": ("仅人口特征(8)", X_A, FEATURE_COLS),
    "B": ("+persona_text(8+384)", X_B, FEATURE_COLS + emb_cols_B),
    "C": (f"+逐字段PCA(8+{emb_C.shape[1]})", X_C, FEATURE_COLS + emb_C_cols),
    "D": (f"+全局PCA(8+{emb_D.shape[1]})", X_D, FEATURE_COLS + emb_D_cols),
}

# --- Train/test split (same as main script) ---
n = len(y_reg)
idx_temp, idx_test = train_test_split(np.arange(n), test_size=0.3,
                                       random_state=RANDOM_STATE, stratify=y_cls)
idx_train, idx_val = train_test_split(idx_temp, test_size=VAL_SIZE/(TRAIN_SIZE+VAL_SIZE),
                                       random_state=RANDOM_STATE, stratify=y_cls.iloc[idx_temp])

# --- Models ---
regression_models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(alpha=0.1, random_state=RANDOM_STATE),
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "ElasticNet": ElasticNet(alpha=0.1, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=1),
    "XGBoost": xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=0),
    "LightGBM": lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1),
    "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "MLP": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=RANDOM_STATE, early_stopping=True),
}

try:
    from catboost import CatBoostRegressor
    regression_models["CatBoost"] = CatBoostRegressor(n_estimators=200, depth=6, learning_rate=0.05, random_seed=RANDOM_STATE, verbose=0)
except ImportError:
    pass

classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
    "GaussianNB": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced"),
    "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=0),
    "LightGBM": lgb.LGBMClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1, class_weight="balanced"),
}

y_cls_train = y_cls.iloc[idx_train]
sw_reg = compute_sample_weight(class_weight="balanced", y=y_cls_train)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# Figure S1: Confusion Matrices (4 approaches)
# ================================================================
print("Generating confusion matrices...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
class_names = ["Oppose", "Neutral", "Support"]

for ax, (app_label, (app_desc, X_app, _)) in zip(axes, approach_config.items()):
    X_tr = X_app.iloc[idx_train]
    X_te = X_app.iloc[idx_test]
    y_tr = y_cls.iloc[idx_train]
    y_te = y_cls.iloc[idx_test]

    # Pick best model by Val F1
    best_f1 = -1
    best_name = None
    best_model = None
    for name, model in classification_models.items():
        if name in ("GaussianNB", "XGBoost"):
            model.fit(X_tr, y_tr, sample_weight=sw_reg)
        else:
            model.fit(X_tr, y_tr)
        y_val_pred = model.predict(X_app.iloc[idx_val])
        f1 = f1_score(y_cls.iloc[idx_val], y_val_pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    y_pred = best_model.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Approach {app_label}: {best_name}\n(Macro F1={f1_score(y_te, y_pred, average='macro'):.3f})", fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> confusion_matrices.png")

# ================================================================
# Figure S2: Actual vs Predicted + Residuals (best regression)
# ================================================================
print("Generating actual vs predicted + residuals...")

# Find best regression approach (store name, not fitted model, to avoid CatBoost feature name locking)
best_label = None
best_name = None
best_reg_r2 = -1
for app_label, (app_desc, X_app, _) in approach_config.items():
    X_tr = X_app.iloc[idx_train]
    X_te = X_app.iloc[idx_test]
    y_tr = y_reg.iloc[idx_train]
    y_te = y_reg.iloc[idx_test]

    for name, model in regression_models.items():
        model.fit(X_tr, y_tr)
        r2 = r2_score(y_te, model.predict(X_te))
        if r2 > best_reg_r2:
            best_reg_r2 = r2
            best_label = app_label
            best_name = name

# Re-train best model fresh on its own approach
X_best = approach_config[best_label][1]
X_tr_best = X_best.iloc[idx_train]
X_te_best = X_best.iloc[idx_test]
y_tr_best = y_reg.iloc[idx_train]
y_te_best = y_reg.iloc[idx_test]
best_model = regression_models[best_name]
best_model.fit(X_tr_best, y_tr_best)
y_pred_best = best_model.predict(X_te_best)
residuals = y_te_best - y_pred_best

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Actual vs Predicted
ax = axes[0]
ax.scatter(y_te_best, y_pred_best, alpha=0.3, s=15, c="#3498db", edgecolors="none")
ax.plot([y_te_best.min(), y_te_best.max()], [y_te_best.min(), y_te_best.max()], "r--", linewidth=1.5)
ax.set_xlabel("Actual Support Score")
ax.set_ylabel("Predicted Support Score")
ax.set_title(f"Approach {best_label}: {best_name}\nTest R2={best_reg_r2:.4f}")
ax.grid(True, alpha=0.3)

# Residuals
ax = axes[1]
ax.hist(residuals, bins=40, color="#3498db", edgecolor="white", alpha=0.8)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_xlabel("Residual (Actual - Predicted)")
ax.set_ylabel("Frequency")
ax.set_title(f"Residual Distribution\nMean={residuals.mean():.4f}, Std={residuals.std():.3f}")
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> actual_vs_predicted.png")

# ================================================================
# Figure S3: Feature Importance Grid (all approaches × top models)
# ================================================================
print("Generating feature importance grid...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

for ax, (app_label, (app_desc, X_app, feat_cols)) in zip(axes, approach_config.items()):
    X_tr = X_app.iloc[idx_train]
    y_tr = y_reg.iloc[idx_train]

    # Train XGBoost for feature importance
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                  random_state=RANDOM_STATE, verbosity=0)
    xgb_model.fit(X_tr, y_tr)

    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[-20:]  # top 20

    # Shorten feature names
    short_names = []
    for fn in np.array(feat_cols)[indices]:
        if len(fn) > 35:
            short_names.append(fn[:32] + "...")
        else:
            short_names.append(fn)

    colors = ["#e74c3c" if fn in FEATURE_COLS else "#3498db" for fn in np.array(feat_cols)[indices]]
    ax.barh(range(len(indices)), importances[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Approach {app_label}: {app_desc}", fontsize=10)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#e74c3c", label="Demographic"),
                       Patch(facecolor="#3498db", label="Text-derived")]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "feature_importance_grid.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> feature_importance_grid.png")

# ================================================================
# Figure S4: Model Comparison (all models × regression & classification)
# ================================================================
print("Generating model comparison...")

# Train all regression models on approach C (midpoint) and evaluate on test
app_label = "C"
_, X_app, _ = approach_config[app_label]
X_tr = X_app.iloc[idx_train]
X_te = X_app.iloc[idx_test]
y_tr_reg = y_reg.iloc[idx_train]
y_te_reg = y_reg.iloc[idx_test]
y_tr_cls = y_cls.iloc[idx_train]
y_te_cls = y_cls.iloc[idx_test]

reg_results = []
for name, model in regression_models.items():
    model.fit(X_tr, y_tr_reg)
    y_pred = model.predict(X_te)
    reg_results.append({
        "Model": name,
        "Test R2": r2_score(y_te_reg, y_pred),
        "MAE": mean_absolute_error(y_te_reg, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_te_reg, y_pred)),
    })

cls_results = []
for name, model in classification_models.items():
    if name in ("GaussianNB", "XGBoost"):
        model.fit(X_tr, y_tr_cls, sample_weight=sw_reg)
    else:
        model.fit(X_tr, y_tr_cls)
    y_pred = model.predict(X_te)
    cls_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_te_cls, y_pred),
        "Macro F1": f1_score(y_te_cls, y_pred, average="macro"),
    })

df_reg_res = pd.DataFrame(reg_results).sort_values("Test R2", ascending=False)
df_cls_res = pd.DataFrame(cls_results).sort_values("Macro F1", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Regression
ax = axes[0]
colors_reg = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(df_reg_res))]
ax.barh(range(len(df_reg_res)), df_reg_res["Test R2"].values, color=colors_reg)
ax.set_yticks(range(len(df_reg_res)))
ax.set_yticklabels(df_reg_res["Model"].values, fontsize=9)
ax.set_xlabel("Test R2")
ax.set_title("Regression Models (Approach C)", fontsize=12)
ax.invert_yaxis()
for i, v in enumerate(df_reg_res["Test R2"].values):
    ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=9)
ax.set_xlim(0, min(0.35, df_reg_res["Test R2"].max() * 1.3))

# Classification
ax = axes[1]
colors_cls = ["#2ecc71" if i == 0 else "#e74c3c" for i in range(len(df_cls_res))]
ax.barh(range(len(df_cls_res)), df_cls_res["Macro F1"].values, color=colors_cls)
ax.set_yticks(range(len(df_cls_res)))
ax.set_yticklabels(df_cls_res["Model"].values, fontsize=9)
ax.set_xlabel("Test Macro F1")
ax.set_title("Classification Models (Approach C)", fontsize=12)
ax.invert_yaxis()
for i, v in enumerate(df_cls_res["Macro F1"].values):
    ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=9)
ax.set_xlim(0, min(0.65, df_cls_res["Macro F1"].max() * 1.3))

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> model_comparison.png")

# ================================================================
# Figure S5: Per-class metrics bar chart
# ================================================================
print("Generating per-class metrics...")
from sklearn.metrics import precision_recall_fscore_support

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for ax, (app_label, (app_desc, X_app, _)) in zip(axes, approach_config.items()):
    X_tr = X_app.iloc[idx_train]
    X_te = X_app.iloc[idx_test]
    y_tr = y_cls.iloc[idx_train]
    y_te = y_cls.iloc[idx_test]

    # Best classifier
    best_f1 = -1
    best_model = None
    best_name = ""
    for name, model in classification_models.items():
        if name in ("GaussianNB", "XGBoost"):
            model.fit(X_tr, y_tr, sample_weight=sw_reg)
        else:
            model.fit(X_tr, y_tr)
        y_val_pred = model.predict(X_app.iloc[idx_val])
        f1 = f1_score(y_cls.iloc[idx_val], y_val_pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    y_pred = best_model.predict(X_te)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, labels=[0, 1, 2], zero_division=0)

    x = np.arange(3)
    width = 0.25
    ax.bar(x - width, prec, width, label="Precision", color="#3498db")
    ax.bar(x, rec, width, label="Recall", color="#2ecc71")
    ax.bar(x + width, f1, width, label="F1", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Score")
    ax.set_title(f"Approach {app_label}: {best_name}", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "per_class_metrics.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  -> per_class_metrics.png")

# ================================================================
# Copy to paper figures
# ================================================================
import shutil
generated = [
    "confusion_matrices.png",
    "actual_vs_predicted.png",
    "feature_importance_grid.png",
    "model_comparison.png",
    "per_class_metrics.png",
]
for f in generated:
    src = os.path.join(OUTPUT_DIR, f)
    dst = os.path.join(PAPER_FIG_DIR, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied {f} -> paper/figures/")

print("\nDone. All supplementary figures generated.")
