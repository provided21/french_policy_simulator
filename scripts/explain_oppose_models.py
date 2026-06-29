"""
Explain the current best oppose-aware models and export representative cases.

This script reproduces:
- direct T4 + XGBoost multiclass baseline;
- best two-stage configuration from oppose optimization.

It writes feature-importance tables, grouped importance summaries, confusion
matrices, and representative cases for PPT discussion.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from model_text_feature_ablation import (
    DATA_PATH,
    FIELD_PCA_FIELDS,
    RANDOM_STATE,
    TARGET_MAP,
    TARGET_NAMES,
    build_embedding_cache,
    pca_field_embeddings,
    structured_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "model_explanations"
FIG_DIR = OUT_DIR / "figures"

CLASS_ORDER = np.array([0, 1, 2])
CLASS_NAMES = ["oppose", "neutral", "support"]

SUPPORT_THRESHOLD = 0.45
OPPOSE_THRESHOLD = 0.16


def setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def split_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_cls = df["y_cls"].to_numpy()
    train_idx, temp_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y_cls,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_cls[temp_idx],
    )
    return train_idx, val_idx, test_idx


def xgb_multiclass() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        eval_metric="mlogloss",
    )


def xgb_binary() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=260,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        eval_metric="logloss",
    )


def build_t4_matrix(df: pd.DataFrame, train_idx: np.ndarray) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    cache = build_embedding_cache(df)
    X_struct, struct_names = structured_matrix(df, train_idx)
    field_pca = pca_field_embeddings(cache, train_idx)
    X_t4 = sparse.hstack([X_struct, sparse.csr_matrix(field_pca)], format="csr")

    pca_names = []
    pca_groups = []
    for field in FIELD_PCA_FIELDS:
        for i in range(16):
            pca_names.append(f"pca_{field}_{i + 1:02d}")
            pca_groups.append(f"text_pca:{field}")

    feature_names = struct_names + pca_names
    feature_groups = [feature_group(name) for name in struct_names] + pca_groups
    return X_t4, feature_names, feature_groups


def feature_group(name: str) -> str:
    if name in {"age", "age_sq", "near_retirement_55_64", "retired_age_65_plus", "young_under_35"}:
        return "age"
    if name in {"manual_or_lowedu", "upper_or_highedu"}:
        return "social_manual_high"
    if name.startswith("occupation") or name.startswith("occupation_group"):
        return "occupation"
    if name.startswith("education") or name.startswith("education_group"):
        return "education"
    if name.startswith("age_group"):
        return "age_group"
    if name.startswith("departement"):
        return "departement"
    if name.startswith("sex"):
        return "sex"
    if name.startswith("marital_status") or name.startswith("household_type"):
        return "family"
    return "other_structured"


def two_stage_predict(stage1, stage2, X: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_support = stage1.predict_proba(X)[:, 1]
    pred = np.full(X.shape[0], 1, dtype=int)
    pred[p_support >= SUPPORT_THRESHOLD] = 2
    non_support_mask = p_support < SUPPORT_THRESHOLD
    p_oppose_full = np.full(X.shape[0], np.nan)
    if non_support_mask.any():
        p_oppose = stage2.predict_proba(X[non_support_mask])[:, 1]
        p_oppose_full[non_support_mask] = p_oppose
        pred[non_support_mask] = np.where(p_oppose >= OPPOSE_THRESHOLD, 0, 1)
    return pred, p_support, p_oppose_full


def importance_table(model, feature_names: list[str], feature_groups: list[str], label: str) -> pd.DataFrame:
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        raise ValueError(f"{label} does not expose feature_importances_.")
    out = pd.DataFrame(
        {
            "feature": feature_names,
            "group": feature_groups,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)
    out.to_csv(OUT_DIR / f"{label}_feature_importance.csv", index=False, encoding="utf-8-sig")

    grouped = out.groupby("group", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    grouped["share"] = grouped["importance"] / grouped["importance"].sum()
    grouped.to_csv(OUT_DIR / f"{label}_grouped_importance.csv", index=False, encoding="utf-8-sig")
    return out


def plot_top_importance(path: Path, imp: pd.DataFrame, title: str, n: int = 18) -> None:
    top = imp.head(n).iloc[::-1]
    labels = top["feature"].str.replace("pca_", "", regex=False)
    plt.figure(figsize=(9, 6), dpi=180)
    plt.barh(labels, top["importance"], color="#7B5AB6")
    plt.xlabel("XGBoost importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_grouped_importance(path: Path, grouped: pd.DataFrame, title: str) -> None:
    top = grouped.sort_values("importance", ascending=True)
    plt.figure(figsize=(9, 6), dpi=180)
    plt.barh(top["group"], top["share"] * 100, color="#4E8F78")
    plt.xlabel("Share of total importance (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_confusion(path: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    plt.figure(figsize=(5.2, 4.6), dpi=180)
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(3), CLASS_NAMES, rotation=30, ha="right")
    plt.yticks(range(3), CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_ORDER, zero_division=0)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(np.mean(f)),
        "oppose_precision": float(p[0]),
        "oppose_recall": float(r[0]),
        "oppose_f1": float(f[0]),
        "neutral_f1": float(f[1]),
        "support_f1": float(f[2]),
        "oppose_support": int(s[0]),
        "neutral_support": int(s[1]),
        "support_support": int(s[2]),
    }


def export_cases(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_support: np.ndarray,
    p_oppose: np.ndarray,
) -> pd.DataFrame:
    cases = df.iloc[test_idx].copy()
    cases["true_label"] = [TARGET_NAMES[v] for v in y_true]
    cases["pred_label"] = [TARGET_NAMES[v] for v in y_pred]
    cases["p_support_stage1"] = p_support
    cases["p_oppose_stage2"] = p_oppose
    cases["case_type"] = "other"
    cases.loc[(y_true == 0) & (y_pred == 0), "case_type"] = "correct_oppose"
    cases.loc[(y_true == 0) & (y_pred == 1), "case_type"] = "oppose_to_neutral"
    cases.loc[(y_true == 0) & (y_pred == 2), "case_type"] = "oppose_to_support"
    cases.loc[(y_true == 1) & (y_pred == 0), "case_type"] = "neutral_to_oppose"

    cols = [
        "case_type",
        "true_label",
        "pred_label",
        "p_support_stage1",
        "p_oppose_stage2",
        "support_score",
        "age",
        "sex",
        "occupation",
        "education_level",
        "marital_status",
        "household_type",
        "departement",
        "reason",
        "persona_text",
    ]
    selected = cases[cases["case_type"] != "other"][cols].copy()
    selected = selected.sort_values(["case_type", "p_oppose_stage2"], ascending=[True, False])
    selected.to_csv(OUT_DIR / "two_stage_representative_cases.csv", index=False, encoding="utf-8-sig")

    sample_rows = []
    for case_type in ["correct_oppose", "oppose_to_neutral", "oppose_to_support", "neutral_to_oppose"]:
        sample = selected[selected["case_type"] == case_type].head(8)
        sample_rows.append(sample)
        sample.to_csv(OUT_DIR / f"cases_{case_type}.csv", index=False, encoding="utf-8-sig")
    return pd.concat(sample_rows, ignore_index=True) if sample_rows else selected


def write_summary(
    direct_metrics: dict[str, float],
    two_metrics: dict[str, float],
    direct_imp: pd.DataFrame,
    stage2_imp: pd.DataFrame,
    cases: pd.DataFrame,
) -> None:
    direct_grouped = pd.read_csv(OUT_DIR / "direct_multiclass_grouped_importance.csv")
    stage2_grouped = pd.read_csv(OUT_DIR / "stage2_neutral_vs_oppose_grouped_importance.csv")

    lines = [
        "# Model Explanation And Cases",
        "",
        "## Compared Models",
        "",
        "- Direct baseline: `T4_field_pca + XGBoost` multiclass.",
        "- Oppose-aware model: two-stage `LogisticRegression + XGBoost` with support_threshold=0.45 and oppose_threshold=0.16.",
        "",
        "## Test Metrics",
        "",
        "| model | accuracy | macro_f1 | oppose_precision | oppose_recall | oppose_f1 | neutral_f1 | support_f1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        metric_row("direct T4 + XGBoost", direct_metrics),
        metric_row("two-stage T4", two_metrics),
        "",
        "## Direct Multiclass Top Features",
        "",
        md_table(direct_imp.head(15)[["feature", "group", "importance"]]),
        "",
        "## Direct Multiclass Group Importance",
        "",
        md_table(direct_grouped.head(12)),
        "",
        "## Stage 2 Neutral-vs-Oppose Top Features",
        "",
        md_table(stage2_imp.head(15)[["feature", "group", "importance"]]),
        "",
        "## Stage 2 Group Importance",
        "",
        md_table(stage2_grouped.head(12)),
        "",
        "## Representative Case Counts",
        "",
        md_table(cases["case_type"].value_counts().rename_axis("case_type").reset_index(name="n")),
        "",
        "## Interpretation",
        "",
        "- Direct multiclass importance is dominated by social structure and text PCA groups, which confirms that the model is learning broad social stratification rather than a clean minority boundary.",
        "- Stage 2 focuses more directly on the neutral/oppose boundary. If high-education, cadre/professional, and status-quo text components appear near the top, this matches the EDA and word-cloud diagnosis.",
        "- Representative cases should be used to show why oppose is difficult: many true oppose reasons read like status quo or fiscal caution, so they are semantically close to neutral.",
        "",
    ]
    (OUT_DIR / "model_explanation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def metric_row(name: str, values: dict[str, float]) -> str:
    return (
        f"| {name} | {values['accuracy']:.4f} | {values['macro_f1']:.4f} | "
        f"{values['oppose_precision']:.4f} | {values['oppose_recall']:.4f} | "
        f"{values['oppose_f1']:.4f} | {values['neutral_f1']:.4f} | {values['support_f1']:.4f} |"
    )


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.5f}")
        else:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else str(x))
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("\n", " ") for col in table.columns) + " |")
    return "\n".join(lines)


def run() -> None:
    setup()
    df = pd.read_parquet(DATA_PATH)
    df = df[df["stance"].isin(TARGET_MAP)].reset_index(drop=True)
    df["y_cls"] = df["stance"].map(TARGET_MAP)
    y = df["y_cls"].to_numpy()
    train_idx, val_idx, test_idx = split_data(df)
    X, feature_names, feature_groups = build_t4_matrix(df, train_idx)

    direct = xgb_multiclass()
    direct.fit(X[train_idx], y[train_idx])
    direct_pred = direct.predict(X[test_idx])
    direct_m = metrics(y[test_idx], direct_pred)
    direct_imp = importance_table(direct, feature_names, feature_groups, "direct_multiclass")
    plot_top_importance(FIG_DIR / "direct_multiclass_top_features.png", direct_imp, "Direct T4 + XGBoost top features")
    plot_grouped_importance(
        FIG_DIR / "direct_multiclass_grouped_importance.png",
        pd.read_csv(OUT_DIR / "direct_multiclass_grouped_importance.csv"),
        "Direct multiclass grouped importance",
    )
    plot_confusion(FIG_DIR / "direct_multiclass_confusion_matrix.png", y[test_idx], direct_pred, "Direct T4 + XGBoost")

    stage1_y = (y == 2).astype(int)
    stage1 = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)
    stage1.fit(X[train_idx], stage1_y[train_idx])

    non_support_train = train_idx[y[train_idx] != 2]
    stage2_y = (y == 0).astype(int)
    stage2 = xgb_binary()
    stage2.fit(X[non_support_train], stage2_y[non_support_train])

    two_pred, p_support, p_oppose = two_stage_predict(stage1, stage2, X[test_idx])
    two_m = metrics(y[test_idx], two_pred)
    stage2_imp = importance_table(stage2, feature_names, feature_groups, "stage2_neutral_vs_oppose")
    plot_top_importance(FIG_DIR / "stage2_neutral_vs_oppose_top_features.png", stage2_imp, "Stage 2 neutral vs oppose top features")
    plot_grouped_importance(
        FIG_DIR / "stage2_neutral_vs_oppose_grouped_importance.png",
        pd.read_csv(OUT_DIR / "stage2_neutral_vs_oppose_grouped_importance.csv"),
        "Stage 2 grouped importance",
    )
    plot_confusion(FIG_DIR / "two_stage_confusion_matrix.png", y[test_idx], two_pred, "Two-stage T4")

    cases = export_cases(df, test_idx, y[test_idx], two_pred, p_support, p_oppose)
    write_summary(direct_m, two_m, direct_imp, stage2_imp, cases)

    metrics_df = pd.DataFrame(
        [
            {"model": "direct_multiclass", **direct_m},
            {"model": "two_stage", **two_m},
        ]
    )
    metrics_df.to_csv(OUT_DIR / "model_explanation_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    run()
