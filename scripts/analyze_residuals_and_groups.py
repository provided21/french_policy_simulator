"""
Residual and group-level evaluation for the best score model.

Uses T4_field_pca + XGBoostRegressor with the same split as the text ablation
experiment. Outputs residual tables, group metrics, and PPT-ready figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from model_text_feature_ablation import (
    DATA_PATH,
    RANDOM_STATE,
    TARGET_MAP,
    build_embedding_cache,
    make_feature_matrices,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "residual_group_analysis"
FIG_DIR = OUT_DIR / "figures"

GROUP_COLS = [
    "stance",
    "age_bin",
    "occupation",
    "education_level",
    "sex",
    "marital_status",
    "household_type",
]

AGE_BINS = [17, 24, 34, 44, 54, 61, 64, 74, 120]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-61", "62-64", "65-74", "75+"]


def setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def split_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_cls = df["stance"].map(TARGET_MAP).to_numpy()
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


def model() -> XGBRegressor:
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


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": r2_score(y_true, pred) if len(np.unique(y_true)) > 1 and len(y_true) >= 5 else np.nan,
        "mae_0_1": mean_absolute_error(y_true, pred),
        "mae_0_10": mean_absolute_error(y_true, pred) * 10,
        "rmse_0_1": float(np.sqrt(mean_squared_error(y_true, pred))),
        "rmse_0_10": float(np.sqrt(mean_squared_error(y_true, pred))) * 10,
        "bias_0_1": float(np.mean(pred - y_true)),
        "bias_0_10": float(np.mean(pred - y_true) * 10),
    }


def group_metrics(df: pd.DataFrame, group_col: str, min_n: int = 25) -> pd.DataFrame:
    rows = []
    for value, g in df.groupby(group_col, dropna=False):
        if len(g) < min_n:
            continue
        m = metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
        rows.append(
            {
                "group_col": group_col,
                "group_value": str(value),
                "n": len(g),
                "true_mean": g["y_true"].mean(),
                "pred_mean": g["y_pred"].mean(),
                "abs_resid_mean": g["abs_resid"].mean(),
                **m,
            }
        )
    out = pd.DataFrame(rows).sort_values(["mae_0_10", "n"], ascending=[False, False])
    out.to_csv(OUT_DIR / f"group_metrics_by_{group_col}.csv", index=False, encoding="utf-8-sig")
    return out


def score_bin_analysis(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-0.001, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.001]
    labels = ["0-0.25", "0.25-0.35", "0.35-0.45", "0.45-0.55", "0.55-0.65", "0.65-0.75", "0.75-0.85", "0.85-1"]
    work = df.copy()
    work["score_bin"] = pd.cut(work["y_true"], bins=bins, labels=labels)
    out = group_metrics(work, "score_bin", min_n=10)
    out.to_csv(OUT_DIR / "score_bin_metrics.csv", index=False, encoding="utf-8-sig")
    return out


def reason_theme_analysis(df: pd.DataFrame) -> pd.DataFrame:
    patterns = {
        "age_retirement": r"年龄|退休|提前|恢复|早退|年纪",
        "manual_work": r"体力|劳动|工人|辛苦|工作压力|职业",
        "family_life": r"家庭|孩子|生活|节奏|照顾",
        "stability_status_quo": r"稳定|现状|维持|现行|纪律|保守|传统",
        "fiscal_caution": r"财政|经济|养老金|负担|可持续",
        "uncertain": r"未知|信息不足|不明确|无法判断|未明确",
    }
    rows = []
    reason = df["reason"].fillna("").astype(str)
    for name, pattern in patterns.items():
        mask = reason.str.contains(pattern, regex=True)
        subset = df[mask]
        if len(subset) < 10:
            continue
        rows.append(
            {
                "theme": name,
                "n": len(subset),
                "share_pct": len(subset) / len(df) * 100,
                **metrics(subset["y_true"].to_numpy(), subset["y_pred"].to_numpy()),
            }
        )
    out = pd.DataFrame(rows).sort_values("mae_0_10", ascending=False)
    out.to_csv(OUT_DIR / "reason_theme_residual_metrics.csv", index=False, encoding="utf-8-sig")
    return out


def export_extreme_cases(df: pd.DataFrame) -> None:
    cols = [
        "uuid",
        "stance",
        "y_true",
        "y_pred",
        "resid",
        "abs_resid",
        "age",
        "sex",
        "occupation",
        "education_level",
        "marital_status",
        "household_type",
        "reason",
        "persona_text",
    ]
    df.sort_values("resid", ascending=False)[cols].head(80).to_csv(
        OUT_DIR / "largest_over_predictions.csv", index=False, encoding="utf-8-sig"
    )
    df.sort_values("resid", ascending=True)[cols].head(80).to_csv(
        OUT_DIR / "largest_under_predictions.csv", index=False, encoding="utf-8-sig"
    )
    df.sort_values("abs_resid", ascending=False)[cols].head(120).to_csv(
        OUT_DIR / "largest_absolute_residuals.csv", index=False, encoding="utf-8-sig"
    )


def plot_pred_vs_true(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6.2, 5.4), dpi=180)
    colors = {"support": "#4E8F78", "neutral": "#4C78A8", "oppose": "#D4743C"}
    for stance, g in df.groupby("stance"):
        plt.scatter(g["y_true"], g["y_pred"], s=16, alpha=0.55, label=stance, color=colors.get(stance))
    plt.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1)
    plt.xlabel("True support_score")
    plt.ylabel("Predicted support_score")
    plt.title("Predicted vs true support_score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "predicted_vs_true_score.png", bbox_inches="tight")
    plt.close()


def plot_resid_by_stance(df: pd.DataFrame) -> None:
    order = ["support", "neutral", "oppose"]
    data = [df.loc[df["stance"] == s, "abs_resid_0_10"].values for s in order]
    plt.figure(figsize=(6.5, 4.8), dpi=180)
    plt.boxplot(data, tick_labels=order, showfliers=False)
    plt.ylabel("Absolute residual (0-10 scale)")
    plt.title("Prediction error by stance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "absolute_residual_by_stance.png", bbox_inches="tight")
    plt.close()


def plot_group_bar(table: pd.DataFrame, group_col: str, filename: str, top_n: int = 12) -> None:
    top = table.sort_values("mae_0_10", ascending=False).head(top_n).iloc[::-1]
    plt.figure(figsize=(9.5, 5.8), dpi=180)
    plt.barh(top["group_value"], top["mae_0_10"], color="#7B5AB6")
    for i, (_, row) in enumerate(top.iterrows()):
        plt.text(row["mae_0_10"] + 0.02, i, f"n={int(row['n'])}", va="center", fontsize=8)
    plt.xlabel("MAE on 0-10 score scale")
    plt.title(f"Highest-error groups: {group_col}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close()


def write_summary(overall: dict[str, float], group_tables: dict[str, pd.DataFrame], score_bins: pd.DataFrame, themes: pd.DataFrame) -> None:
    lines = [
        "# Residual And Group Analysis",
        "",
        "Model: `T4_field_pca + XGBoostRegressor`.",
        "",
        "## Overall Test Metrics",
        "",
        f"- R2: `{overall['r2']:.4f}`",
        f"- MAE: `{overall['mae_0_1']:.4f}` on 0-1 scale, about `{overall['mae_0_10']:.2f}` points on 0-10 scale",
        f"- RMSE: `{overall['rmse_0_1']:.4f}` on 0-1 scale, about `{overall['rmse_0_10']:.2f}` points on 0-10 scale",
        f"- Bias: `{overall['bias_0_10']:.2f}` points on 0-10 scale",
        "",
        "## Error By Stance",
        "",
        md_table(group_tables["stance"][["group_value", "n", "mae_0_10", "bias_0_10", "r2"]]),
        "",
        "## Error By Score Bin",
        "",
        md_table(score_bins[["group_value", "n", "mae_0_10", "bias_0_10", "r2"]]),
        "",
        "## Highest-error Occupation Groups",
        "",
        md_table(group_tables["occupation"][["group_value", "n", "mae_0_10", "bias_0_10", "r2"]].head(8)),
        "",
        "## Highest-error Education Groups",
        "",
        md_table(group_tables["education_level"][["group_value", "n", "mae_0_10", "bias_0_10", "r2"]].head(8)),
        "",
        "## Reason Theme Residuals",
        "",
        md_table(themes[["theme", "n", "share_pct", "mae_0_10", "bias_0_10"]]),
        "",
        "## Interpretation",
        "",
        "- The low R2 should be read together with MAE. The average error is about one point on a 0-10 score scale, so the model captures broad score levels but not fine-grained individual variation.",
        "- Errors are expected to concentrate near stance boundaries and in heterogeneous groups, especially neutral/status-quo cases.",
        "- If the score-bin and stance tables show larger error around neutral or low-score bins, this supports the claim that label boundaries and minority semantics limit predictability.",
        "",
    ]
    (OUT_DIR / "residual_group_analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


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
    df["age_bin"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)

    train_idx, val_idx, test_idx = split_indices(df)
    cache = build_embedding_cache(df)
    X = make_feature_matrices(df, train_idx, cache)["T4_field_pca"]

    y = df["support_score"].astype(float).to_numpy()
    reg = model()
    reg.fit(X[train_idx], y[train_idx])
    pred = reg.predict(X[test_idx])

    test = df.iloc[test_idx].copy()
    test["y_true"] = y[test_idx]
    test["y_pred"] = pred
    test["resid"] = test["y_pred"] - test["y_true"]
    test["abs_resid"] = test["resid"].abs()
    test["resid_0_10"] = test["resid"] * 10
    test["abs_resid_0_10"] = test["abs_resid"] * 10
    test.to_csv(OUT_DIR / "test_predictions_with_residuals.csv", index=False, encoding="utf-8-sig")

    overall = metrics(test["y_true"].to_numpy(), test["y_pred"].to_numpy())
    pd.DataFrame([overall]).to_csv(OUT_DIR / "overall_regression_metrics.csv", index=False, encoding="utf-8-sig")

    group_tables = {}
    for col in GROUP_COLS:
        group_tables[col] = group_metrics(test, col, min_n=20)

    score_bins = score_bin_analysis(test)
    themes = reason_theme_analysis(test)
    export_extreme_cases(test)

    plot_pred_vs_true(test)
    plot_resid_by_stance(test)
    plot_group_bar(group_tables["occupation"], "occupation", "mae_by_occupation.png")
    plot_group_bar(group_tables["education_level"], "education_level", "mae_by_education.png")
    plot_group_bar(score_bins, "score_bin", "mae_by_score_bin.png")

    write_summary(overall, group_tables, score_bins, themes)
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    run()
