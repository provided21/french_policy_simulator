"""
Validate systemic reasons for limited model performance without extra API calls.

Experiments:
1. Reason upper-bound diagnostic:
   Compare persona-based T4 features with post-hoc reason embeddings. Reason is
   not a valid production feature, but if it predicts much better, it shows the
   original persona lacks direct attitude information.

2. Label-boundary relabel diagnostic:
   Reassign oppose cases whose reasons mainly express status quo / anti-return
   semantics to neutral, then compare classification metrics.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier, XGBRegressor

from model_text_feature_ablation import (
    DATA_PATH,
    MODEL_DIR,
    RANDOM_STATE,
    TARGET_MAP,
    build_embedding_cache,
    make_feature_matrices,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "model_limitation_validation"
FIG_DIR = OUT_DIR / "figures"
REASON_CACHE = ROOT / "data" / "processed" / "modeling_samples" / "reason_embeddings_main_1w_faiss_p2.npz"

TARGET_NAMES = ["oppose", "neutral", "support"]
CLASS_ORDER = np.array([0, 1, 2])

STATUS_QUO_PATTERN = re.compile(r"维持|现状|现行|稳定|安稳|纪律|秩序|保守|谨慎|审慎|财政|可持续|反对.*降低|反对.*恢复|不倾向提前|不支持提前|反对.*提前")
EXPLICIT_RAISE_PATTERN = re.compile(r"进一步提高|提高.*退休年龄|延长.*退休|推迟.*退休|延迟.*退休|65|66|更高退休|继续提高")


def setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def split_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = df["stance"].map(TARGET_MAP).to_numpy()
    train_idx, temp_idx = train_test_split(np.arange(len(df)), test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_STATE, stratify=y[temp_idx])
    return train_idx, val_idx, test_idx


def encode_reasons(df: pd.DataFrame) -> np.ndarray:
    if REASON_CACHE.exists():
        data = np.load(REASON_CACHE)
        return data["reason"]
    model = SentenceTransformer(str(MODEL_DIR))
    emb = model.encode(
        df["reason"].fillna("").astype(str).tolist(),
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    REASON_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REASON_CACHE, reason=emb)
    return emb


def reg_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def cls_model() -> XGBClassifier:
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


def reg_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    return {
        "r2": r2_score(y_true, pred),
        "mae_0_1": mean_absolute_error(y_true, pred),
        "mae_0_10": mean_absolute_error(y_true, pred) * 10,
        "rmse_0_1": rmse,
        "rmse_0_10": rmse * 10,
    }


def cls_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "oppose_f1": f1_score(y_true, pred, labels=[0], average="macro", zero_division=0),
        "neutral_f1": f1_score(y_true, pred, labels=[1], average="macro", zero_division=0),
        "support_f1": f1_score(y_true, pred, labels=[2], average="macro", zero_division=0),
    }


def run_reason_upper_bound(
    df: pd.DataFrame,
    X_t4: sparse.csr_matrix,
    X_reason: sparse.csr_matrix,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_score = df["support_score"].astype(float).to_numpy()
    y_cls = df["stance"].map(TARGET_MAP).to_numpy()
    features = {
        "persona_T4": X_t4,
        "reason_embedding_only": X_reason,
        "persona_T4_plus_reason": sparse.hstack([X_t4, X_reason], format="csr"),
    }

    reg_rows = []
    cls_rows = []
    for name, X in features.items():
        for model_name, model in [
            ("Ridge", Ridge(alpha=3.0)),
            ("XGBoostRegressor", reg_model()),
        ]:
            model.fit(X[train_idx], y_score[train_idx])
            pred = model.predict(X[test_idx])
            reg_rows.append({"feature_set": name, "model": model_name, **reg_metrics(y_score[test_idx], pred)})

        for model_name, model in [
            ("LogisticRegression", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)),
            ("XGBoostClassifier", cls_model()),
        ]:
            model.fit(X[train_idx], y_cls[train_idx])
            pred = model.predict(X[test_idx])
            cls_rows.append({"feature_set": name, "model": model_name, **cls_metrics(y_cls[test_idx], pred)})

    reg = pd.DataFrame(reg_rows).sort_values("r2", ascending=False)
    cls = pd.DataFrame(cls_rows).sort_values("macro_f1", ascending=False)
    reg.to_csv(OUT_DIR / "reason_upper_bound_regression.csv", index=False, encoding="utf-8-sig")
    cls.to_csv(OUT_DIR / "reason_upper_bound_classification.csv", index=False, encoding="utf-8-sig")
    return reg, cls


def relabel_boundary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    reason = out["reason"].fillna("").astype(str)
    is_oppose = out["stance"].eq("oppose")
    status_quo = reason.map(lambda text: bool(STATUS_QUO_PATTERN.search(text)))
    explicit_raise = reason.map(lambda text: bool(EXPLICIT_RAISE_PATTERN.search(text)))
    relabel_mask = is_oppose & status_quo & ~explicit_raise
    out["stance_relabel_boundary"] = out["stance"]
    out.loc[relabel_mask, "stance_relabel_boundary"] = "neutral"
    out["boundary_relabel_reason"] = np.where(relabel_mask, "oppose_status_quo_to_neutral", "unchanged")

    audit = pd.DataFrame(
        [
            {"metric": "rows", "value": len(out)},
            {"metric": "original_oppose", "value": int(is_oppose.sum())},
            {"metric": "relabel_oppose_to_neutral", "value": int(relabel_mask.sum())},
            {"metric": "remaining_oppose", "value": int((out["stance_relabel_boundary"] == "oppose").sum())},
        ]
    )
    audit.to_csv(OUT_DIR / "boundary_relabel_audit.csv", index=False, encoding="utf-8-sig")

    out.loc[relabel_mask, ["uuid", "stance", "stance_relabel_boundary", "support_score", "reason", "persona_text"]].to_csv(
        OUT_DIR / "boundary_relabel_cases.csv", index=False, encoding="utf-8-sig"
    )
    return out


def run_boundary_relabel_experiment(
    df: pd.DataFrame,
    X_t4: sparse.csr_matrix,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    rows = []
    labels = {
        "original": df["stance"].map(TARGET_MAP).to_numpy(),
        "boundary_relabel": df["stance_relabel_boundary"].map(TARGET_MAP).to_numpy(),
    }
    for target_name, y in labels.items():
        for model_name, model in [
            ("LogisticRegression", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)),
            ("XGBoostClassifier", cls_model()),
        ]:
            model.fit(X_t4[train_idx], y[train_idx])
            pred = model.predict(X_t4[test_idx])
            rows.append({"target": target_name, "model": model_name, **cls_metrics(y[test_idx], pred)})
    result = pd.DataFrame(rows).sort_values(["target", "macro_f1"], ascending=[True, False])
    result.to_csv(OUT_DIR / "boundary_relabel_classification.csv", index=False, encoding="utf-8-sig")
    return result


def reason_theme_overlap(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reason = df["reason"].fillna("").astype(str)
    for stance, group in df.groupby("stance"):
        idx = group.index
        status = reason.loc[idx].map(lambda text: bool(STATUS_QUO_PATTERN.search(text)))
        raise_ = reason.loc[idx].map(lambda text: bool(EXPLICIT_RAISE_PATTERN.search(text)))
        rows.append(
            {
                "stance": stance,
                "n": len(group),
                "status_quo_or_fiscal_n": int(status.sum()),
                "status_quo_or_fiscal_pct": float(status.mean() * 100),
                "explicit_raise_n": int(raise_.sum()),
                "explicit_raise_pct": float(raise_.mean() * 100),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "reason_boundary_theme_overlap.csv", index=False, encoding="utf-8-sig")
    return out


def plot_results(reg: pd.DataFrame, boundary: pd.DataFrame) -> None:
    best_reg = reg[reg["model"].eq("XGBoostRegressor")].copy().sort_values("r2")
    plt.figure(figsize=(7.5, 4.5), dpi=180)
    plt.barh(best_reg["feature_set"], best_reg["r2"], color="#7B5AB6")
    plt.xlabel("Test R2")
    plt.title("Reason upper-bound diagnostic")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "reason_upper_bound_r2.png", bbox_inches="tight")
    plt.close()

    pivot = boundary.pivot(index="model", columns="target", values="macro_f1")
    pivot.plot(kind="bar", figsize=(7.5, 4.5), color=["#7B5AB6", "#4E8F78"])
    plt.ylabel("Macro F1")
    plt.title("Boundary relabel diagnostic")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "boundary_relabel_macro_f1.png", bbox_inches="tight")
    plt.close()


def write_summary(reg: pd.DataFrame, cls: pd.DataFrame, boundary: pd.DataFrame, overlap: pd.DataFrame) -> None:
    lines = [
        "# Model Limitation Validation",
        "",
        "## Experiment 1: Reason Upper-bound Diagnostic",
        "",
        "Reason is generated after the label, so it is not a valid production feature. It is used here only as a leakage/upper-bound diagnostic.",
        "",
        "### Regression",
        "",
        md_table(reg),
        "",
        "### Classification",
        "",
        md_table(cls),
        "",
        "## Experiment 2: Boundary Relabel Diagnostic",
        "",
        "Oppose cases with status-quo / anti-return / fiscal-caution wording and no explicit further-raise wording are reassigned to neutral.",
        "",
        "### Theme Overlap",
        "",
        md_table(overlap),
        "",
        "### Classification After Relabel",
        "",
        md_table(boundary),
        "",
        "## Interpretation",
        "",
        "- If reason embeddings outperform persona features, the original persona lacks direct attitude information and the label contains post-hoc reasoning not available at prediction time.",
        "- If boundary relabeling improves macro F1 or changes oppose behavior, neutral/oppose semantics are not cleanly separable under the current label design.",
        "- These diagnostics support the claim that limited model performance is systemic, not just caused by weak algorithms.",
        "",
    ]
    (OUT_DIR / "model_limitation_validation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
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
    train_idx, val_idx, test_idx = split_indices(df)
    _ = val_idx

    cache = build_embedding_cache(df)
    X_t4 = make_feature_matrices(df, train_idx, cache)["T4_field_pca"]
    X_reason = sparse.csr_matrix(encode_reasons(df))

    reg, cls = run_reason_upper_bound(df, X_t4, X_reason, train_idx, test_idx)
    relabeled = relabel_boundary(df)
    overlap = reason_theme_overlap(relabeled)
    boundary = run_boundary_relabel_experiment(relabeled, X_t4, train_idx, test_idx)
    plot_results(reg, boundary)
    write_summary(reg, cls, boundary, overlap)
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    run()
