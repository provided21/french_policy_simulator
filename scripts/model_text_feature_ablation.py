"""
Text feature ablation after structured baselines.

Feature schemes:
T0: A3 structured social baseline
T1: T0 + full persona_text embedding
T2: T0 + career-related text embedding
T3: T0 + lifestyle/culture text embedding
T4: T0 + per-field embeddings reduced by PCA
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier, XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "modeling_samples" / "main_1w_faiss_p2_clean.parquet"
CACHE_PATH = ROOT / "data" / "processed" / "modeling_samples" / "embeddings_main_1w_faiss_p2.npz"
OUT_DIR = ROOT / "output" / "text_feature_ablation"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = ROOT / "all-MiniLM-L6-v2"
RANDOM_STATE = 42

TARGET_MAP = {"oppose": 0, "neutral": 1, "support": 2}
TARGET_NAMES = ["oppose", "neutral", "support"]

BASIC_CAT = ["sex", "occupation", "education_level", "marital_status", "household_type", "departement"]
CAREER_FIELDS = ["professional_persona", "skills_and_expertise", "career_goals_and_ambitions"]
LIFESTYLE_FIELDS = [
    "persona",
    "cultural_background",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "hobbies_and_interests",
]
FIELD_PCA_FIELDS = CAREER_FIELDS + LIFESTYLE_FIELDS


def setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def clean_text_series(df: pd.DataFrame, fields: list[str]) -> list[str]:
    parts = []
    for _, row in df[fields].fillna("").astype(str).iterrows():
        parts.append(" | ".join(v.strip() for v in row.tolist() if v.strip()))
    return parts


def encode_texts(model: SentenceTransformer, texts: list[str], label: str) -> np.ndarray:
    print(f"Encoding {label}: {len(texts):,} texts")
    return model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True).astype(np.float32)


def build_embedding_cache(df: pd.DataFrame) -> dict[str, np.ndarray]:
    if CACHE_PATH.exists():
        data = np.load(CACHE_PATH)
        return {k: data[k] for k in data.files}

    model = SentenceTransformer(str(MODEL_DIR))
    cache: dict[str, np.ndarray] = {}
    cache["persona_text"] = encode_texts(model, df["persona_text"].fillna("").astype(str).tolist(), "persona_text")
    cache["career_text"] = encode_texts(model, clean_text_series(df, CAREER_FIELDS), "career_text")
    cache["lifestyle_text"] = encode_texts(model, clean_text_series(df, LIFESTYLE_FIELDS), "lifestyle_text")

    for field in FIELD_PCA_FIELDS:
        cache[f"field_{field}"] = encode_texts(model, df[field].fillna("").astype(str).tolist(), field)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, **cache)
    print(f"saved embedding cache: {CACHE_PATH}")
    return cache


def add_social_group_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    age = pd.to_numeric(out["age"], errors="coerce")
    out["age"] = age
    out["age_sq"] = age ** 2
    out["near_retirement_55_64"] = ((age >= 55) & (age <= 64)).astype(int)
    out["retired_age_65_plus"] = (age >= 65).astype(int)
    out["young_under_35"] = (age < 35).astype(int)
    bins = [17, 24, 34, 44, 54, 61, 64, 74, 120]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-61", "62-64", "65-74", "75+"]
    out["age_group"] = pd.cut(age, bins=bins, labels=labels).astype(str)

    occupation_map = {
        "Ouvriers": "manual_worker",
        "Employés": "employee",
        "Retraités": "retired",
        "Cadres et professions intellectuelles supérieures": "upper_cadre",
        "Professions intermédiaires": "intermediate",
        "Agriculteurs exploitants": "self_employed",
        "Artisans, commerçants, chefs d'entreprise": "self_employed",
        "Autres sans activité professionnelle": "inactive_other",
    }
    education_map = {
        "Sans diplôme ou CEP": "low",
        "Brevet": "low",
        "CAP ou BEP": "vocational",
        "Baccalauréat": "secondary",
        "Bac+2": "higher_short",
        "Bac+3 ou Bac+4": "higher_mid",
        "Bac+5 ou plus": "higher_high",
    }
    out["occupation_group"] = out["occupation"].fillna("unknown").astype(str).map(occupation_map).fillna("other")
    out["education_group"] = out["education_level"].fillna("unknown").astype(str).map(education_map).fillna("other")
    out["manual_or_lowedu"] = ((out["occupation_group"] == "manual_worker") | (out["education_group"].isin(["low", "vocational"]))).astype(int)
    out["upper_or_highedu"] = ((out["occupation_group"] == "upper_cadre") | (out["education_group"].isin(["higher_mid", "higher_high"]))).astype(int)
    return out


def structured_matrix(df: pd.DataFrame, train_idx: np.ndarray) -> tuple[sparse.csr_matrix, list[str]]:
    work = add_social_group_features(df)
    num_cols = ["age", "age_sq", "near_retirement_55_64", "retired_age_65_plus", "young_under_35", "manual_or_lowedu", "upper_or_highedu"]
    cat_cols = ["age_group", *BASIC_CAT, "occupation_group", "education_group"]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="unknown")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), cat_cols),
    ])
    X = preprocessor.fit_transform(work.iloc[train_idx])
    preprocessor.fit(work.iloc[train_idx])
    X_all = preprocessor.transform(work)
    feature_names = []
    feature_names.extend(num_cols)
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    feature_names.extend(ohe.get_feature_names_out(cat_cols).tolist())
    return sparse.csr_matrix(X_all), feature_names


def pca_field_embeddings(cache: dict[str, np.ndarray], train_idx: np.ndarray, n_components: int = 16) -> np.ndarray:
    parts = []
    for field in FIELD_PCA_FIELDS:
        emb = cache[f"field_{field}"]
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        pca.fit(emb[train_idx])
        parts.append(pca.transform(emb).astype(np.float32))
    return np.hstack(parts)


def make_feature_matrices(df: pd.DataFrame, train_idx: np.ndarray, cache: dict[str, np.ndarray]) -> dict[str, sparse.csr_matrix]:
    X_struct, _ = structured_matrix(df, train_idx)
    matrices = {"T0_structured_A3": X_struct}
    matrices["T1_full_persona_text"] = sparse.hstack([X_struct, sparse.csr_matrix(cache["persona_text"])], format="csr")
    matrices["T2_career_text"] = sparse.hstack([X_struct, sparse.csr_matrix(cache["career_text"])], format="csr")
    matrices["T3_lifestyle_text"] = sparse.hstack([X_struct, sparse.csr_matrix(cache["lifestyle_text"])], format="csr")
    field_pca = pca_field_embeddings(cache, train_idx)
    matrices["T4_field_pca"] = sparse.hstack([X_struct, sparse.csr_matrix(field_pca)], format="csr")
    return matrices


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, pred)
    return {"r2": r2_score(y_true, pred), "mae": mean_absolute_error(y_true, pred), "rmse": float(np.sqrt(mse))}


def classification_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "weighted_f1": f1_score(y_true, pred, average="weighted"),
        "oppose_f1": f1_score(y_true, pred, labels=[0], average="macro", zero_division=0),
        "neutral_f1": f1_score(y_true, pred, labels=[1], average="macro", zero_division=0),
        "support_f1": f1_score(y_true, pred, labels=[2], average="macro", zero_division=0),
    }


def models(task: str) -> dict[str, object]:
    if task == "regression":
        return {
            "Ridge": Ridge(alpha=3.0),
            "XGBoost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        }
    return {
        "LogisticRegression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, eval_metric="mlogloss"),
    }


def save_plot(results: pd.DataFrame) -> None:
    reg = results[results["task"] == "regression"].copy().sort_values("r2", ascending=False)
    cls = results[results["task"] == "classification"].copy().sort_values("macro_f1", ascending=False)

    plt.figure(figsize=(10, 5))
    labels = reg["feature_scheme"] + "\n" + reg["model"]
    plt.barh(labels, reg["r2"], color="#7B5AB6")
    plt.gca().invert_yaxis()
    plt.xlabel("Test R2")
    plt.title("Text Feature Ablation - Regression")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "text_ablation_regression_r2.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    labels = cls["feature_scheme"] + "\n" + cls["model"]
    plt.barh(labels, cls["macro_f1"], color="#7B5AB6")
    plt.gca().invert_yaxis()
    plt.xlabel("Test Macro F1")
    plt.title("Text Feature Ablation - Classification")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "text_ablation_classification_macro_f1.png", dpi=180)
    plt.close()


def run() -> None:
    setup()
    df = pd.read_parquet(DATA_PATH)
    df = df[df["stance"].isin(TARGET_MAP)].reset_index(drop=True)
    df["y_cls"] = df["stance"].map(TARGET_MAP)
    y_reg = df["support_score"].astype(float).to_numpy()
    y_cls = df["y_cls"].to_numpy()

    train_idx, temp_idx = train_test_split(np.arange(len(df)), test_size=0.30, random_state=RANDOM_STATE, stratify=y_cls)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_STATE, stratify=y_cls[temp_idx])
    (OUT_DIR / "split_info.json").write_text(json.dumps({"train": len(train_idx), "val": len(val_idx), "test": len(test_idx), "random_state": RANDOM_STATE}, indent=2), encoding="utf-8")

    cache = build_embedding_cache(df)
    feature_mats = make_feature_matrices(df, train_idx, cache)

    rows = []
    reports = {}
    for scheme, X in feature_mats.items():
        print(f"Feature scheme: {scheme}, shape={X.shape}")
        for task in ["regression", "classification"]:
            for model_name, model in models(task).items():
                y = y_reg if task == "regression" else y_cls
                model.fit(X[train_idx], y[train_idx])
                pred_val = model.predict(X[val_idx])
                pred_test = model.predict(X[test_idx])
                if task == "regression":
                    val_metrics = regression_metrics(y[val_idx], pred_val)
                    test_metrics = regression_metrics(y[test_idx], pred_test)
                else:
                    pred_val = pred_val.astype(int)
                    pred_test = pred_test.astype(int)
                    val_metrics = classification_metrics(y[val_idx], pred_val)
                    test_metrics = classification_metrics(y[test_idx], pred_test)
                    reports[f"{scheme}_{model_name}"] = classification_report(y[test_idx], pred_test, target_names=TARGET_NAMES, output_dict=True, zero_division=0)
                row = {"feature_scheme": scheme, "task": task, "model": model_name, "n_features": X.shape[1]}
                row.update({f"val_{k}": v for k, v in val_metrics.items()})
                row.update(test_metrics)
                rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "text_feature_ablation_results.csv", index=False, encoding="utf-8-sig")
    with (OUT_DIR / "classification_reports.json").open("w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    save_plot(results)

    print("Regression:")
    print(results[results["task"] == "regression"].sort_values("r2", ascending=False).to_string(index=False))
    print("\nClassification:")
    print(results[results["task"] == "classification"].sort_values("macro_f1", ascending=False).to_string(index=False))
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    run()
