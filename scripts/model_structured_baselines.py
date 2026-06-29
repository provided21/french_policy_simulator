"""
Structured-feature baseline modeling for the main 1w clean dataset.

This first modeling round intentionally excludes text embeddings. The goal is
to quantify how much of the LLM-generated attitude structure is explained by
demographics, age engineering, occupation, education, household, and location.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
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
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "modeling_samples" / "main_1w_faiss_p2_clean.parquet"
OUT_DIR = ROOT / "output" / "structured_baselines"
FIG_DIR = OUT_DIR / "figures"
RANDOM_STATE = 42

TARGET_MAP = {"oppose": 0, "neutral": 1, "support": 2}
TARGET_NAMES = ["oppose", "neutral", "support"]

BASIC_CAT = [
    "sex",
    "occupation",
    "education_level",
    "marital_status",
    "household_type",
    "departement",
]


def setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
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
    return out


def add_social_group_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_age_features(df)
    occupation = out["occupation"].fillna("unknown").astype(str)
    education = out["education_level"].fillna("unknown").astype(str)

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
    out["occupation_group"] = occupation.map(occupation_map).fillna("other")
    out["education_group"] = education.map(education_map).fillna("other")
    out["manual_or_lowedu"] = ((out["occupation_group"] == "manual_worker") | (out["education_group"].isin(["low", "vocational"]))).astype(int)
    out["upper_or_highedu"] = ((out["occupation_group"] == "upper_cadre") | (out["education_group"].isin(["higher_mid", "higher_high"]))).astype(int)
    return out


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df = df[df["stance"].isin(TARGET_MAP)].copy()
    df["y_cls"] = df["stance"].map(TARGET_MAP)
    df["y_reg"] = df["support_score"].astype(float)
    return df


def make_feature_frame(df: pd.DataFrame, approach: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    if approach in {"A0_label", "A1_onehot"}:
        work = df[["age", *BASIC_CAT]].copy()
        return work, ["age"], BASIC_CAT
    if approach == "A2_age_engineered":
        work = add_age_features(df)[[
            "age",
            "age_sq",
            "near_retirement_55_64",
            "retired_age_65_plus",
            "young_under_35",
            "age_group",
            *BASIC_CAT,
        ]].copy()
        num = ["age", "age_sq", "near_retirement_55_64", "retired_age_65_plus", "young_under_35"]
        cat = ["age_group", *BASIC_CAT]
        return work, num, cat
    if approach == "A3_social_grouped":
        work = add_social_group_features(df)[[
            "age",
            "age_sq",
            "near_retirement_55_64",
            "retired_age_65_plus",
            "young_under_35",
            "age_group",
            *BASIC_CAT,
            "occupation_group",
            "education_group",
            "manual_or_lowedu",
            "upper_or_highedu",
        ]].copy()
        num = ["age", "age_sq", "near_retirement_55_64", "retired_age_65_plus", "young_under_35", "manual_or_lowedu", "upper_or_highedu"]
        cat = ["age_group", *BASIC_CAT, "occupation_group", "education_group"]
        return work, num, cat
    raise ValueError(f"Unknown approach: {approach}")


def make_preprocessor(approach: str, numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    if approach == "A0_label":
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
    else:
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ])
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", num_transformer, numeric_cols),
        ("cat", cat_transformer, categorical_cols),
    ])


def get_models(task: str) -> dict[str, object]:
    if task == "regression":
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(n_estimators=250, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
            "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.04, max_depth=6, random_state=RANDOM_STATE, verbose=-1),
        }
    return {
        "LogisticRegression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=250, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, eval_metric="mlogloss"),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.04, max_depth=6, random_state=RANDOM_STATE, verbose=-1, class_weight="balanced"),
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mse)),
    }


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "oppose_f1": f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0),
        "neutral_f1": f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0),
        "support_f1": f1_score(y_true, y_pred, labels=[2], average="macro", zero_division=0),
    }


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "named_steps"):
            last = list(transformer.named_steps.values())[-1]
            if hasattr(last, "get_feature_names_out"):
                try:
                    names.extend(last.get_feature_names_out(cols).tolist())
                except TypeError:
                    names.extend(last.get_feature_names_out().tolist())
            else:
                names.extend(cols)
        else:
            names.extend(cols)
    return names


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, approach: str, model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks(range(3), TARGET_NAMES)
    ax.set_yticks(range(3), TARGET_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{approach} / {model_name}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"cm_{approach}_{model_name}.png", dpi=180)
    plt.close()


def plot_model_results(results: pd.DataFrame) -> None:
    reg = results[results["task"] == "regression"].copy()
    cls = results[results["task"] == "classification"].copy()

    fig, ax = plt.subplots(figsize=(11, 5))
    reg["label"] = reg["approach"] + "\n" + reg["model"]
    reg = reg.sort_values("r2", ascending=False).head(20)
    ax.barh(reg["label"], reg["r2"], color="#7B5AB6")
    ax.invert_yaxis()
    ax.set_xlabel("Test R2")
    ax.set_title("Structured Feature Regression Baselines")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "structured_regression_r2.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 5))
    cls["label"] = cls["approach"] + "\n" + cls["model"]
    cls = cls.sort_values("macro_f1", ascending=False).head(20)
    ax.barh(cls["label"], cls["macro_f1"], color="#7B5AB6")
    ax.invert_yaxis()
    ax.set_xlabel("Test Macro F1")
    ax.set_title("Structured Feature Classification Baselines")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "structured_classification_macro_f1.png", dpi=180)
    plt.close()


def run() -> None:
    setup()
    df = load_data()
    train_idx, temp_idx = train_test_split(
        df.index,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["y_cls"],
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=df.loc[temp_idx, "y_cls"],
    )

    split_info = {
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "test": int(len(test_idx)),
        "random_state": RANDOM_STATE,
        "stratified_by": "stance",
    }
    (OUT_DIR / "split_info.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    approaches = ["A0_label", "A1_onehot", "A2_age_engineered", "A3_social_grouped"]
    rows = []
    reports = {}
    best_cls = None

    for approach in approaches:
        X, num_cols, cat_cols = make_feature_frame(df, approach)
        y_reg = df["y_reg"].to_numpy()
        y_cls = df["y_cls"].to_numpy()

        for task in ["regression", "classification"]:
            for model_name, model in get_models(task).items():
                preprocessor = make_preprocessor(approach, num_cols, cat_cols)
                pipeline = Pipeline([
                    ("prep", preprocessor),
                    ("model", model),
                ])
                y = y_reg if task == "regression" else y_cls
                pipeline.fit(X.loc[train_idx], y[train_idx])
                pred_val = pipeline.predict(X.loc[val_idx])
                pred_test = pipeline.predict(X.loc[test_idx])
                if task == "classification":
                    pred_val = np.asarray(pred_val).astype(int)
                    pred_test = np.asarray(pred_test).astype(int)
                    metrics_val = evaluate_classification(y[val_idx], pred_val)
                    metrics_test = evaluate_classification(y[test_idx], pred_test)
                    report = classification_report(y[test_idx], pred_test, target_names=TARGET_NAMES, output_dict=True, zero_division=0)
                    reports[f"{approach}_{model_name}"] = report
                    if best_cls is None or metrics_test["macro_f1"] > best_cls["macro_f1"]:
                        best_cls = {
                            "approach": approach,
                            "model": model_name,
                            "macro_f1": metrics_test["macro_f1"],
                            "y_true": y[test_idx],
                            "y_pred": pred_test,
                        }
                else:
                    metrics_val = evaluate_regression(y[val_idx], pred_val)
                    metrics_test = evaluate_regression(y[test_idx], pred_test)

                row = {
                    "approach": approach,
                    "task": task,
                    "model": model_name,
                    "n_features_after_encoding": pipeline.named_steps["prep"].transform(X.head(1)).shape[1],
                }
                row.update({f"val_{k}": v for k, v in metrics_val.items()})
                row.update({k: v for k, v in metrics_test.items()})
                rows.append(row)

                if task == "classification" and model_name in {"LogisticRegression", "LightGBM"}:
                    save_confusion_matrix(y[test_idx], pred_test, approach, model_name)

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "structured_baseline_results.csv", index=False, encoding="utf-8-sig")
    with (OUT_DIR / "classification_reports.json").open("w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    plot_model_results(results)

    if best_cls:
        save_confusion_matrix(best_cls["y_true"], best_cls["y_pred"], best_cls["approach"], f"BEST_{best_cls['model']}")

    print("Regression top:")
    print(results[results["task"] == "regression"].sort_values("r2", ascending=False).head(10).to_string(index=False))
    print("\nClassification top:")
    print(results[results["task"] == "classification"].sort_values("macro_f1", ascending=False).head(10).to_string(index=False))
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    run()
