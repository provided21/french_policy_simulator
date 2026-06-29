"""
Explore model-side strategies for recovering the rare oppose class.

This script keeps the main 10k clean data fixed and compares:
1. weighted multiclass classifiers;
2. probability-threshold tuning;
3. two-stage classification;
4. regression-to-threshold classification.

The goal is diagnostic: identify whether oppose can be recovered by model
optimization, and what accuracy/precision tradeoff this creates.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier, XGBRegressor

from model_text_feature_ablation import (
    DATA_PATH,
    RANDOM_STATE,
    TARGET_MAP,
    TARGET_NAMES,
    build_embedding_cache,
    make_feature_matrices,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "oppose_optimization"
FIG_DIR = OUT_DIR / "figures"

FEATURE_SCHEMES = ["T0_structured_A3", "T4_field_pca"]
CLASS_ORDER = np.array([0, 1, 2])
CLASS_NAMES = ["oppose", "neutral", "support"]


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


def class_weight_vector(y: np.ndarray, oppose_multiplier: float = 1.0) -> np.ndarray:
    weights = compute_class_weight(class_weight="balanced", classes=CLASS_ORDER, y=y)
    mapping = {cls: weight for cls, weight in zip(CLASS_ORDER, weights)}
    mapping[0] *= oppose_multiplier
    return np.array([mapping[v] for v in y], dtype=float)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=CLASS_ORDER, average="weighted", zero_division=0),
        "oppose_precision": precision[0],
        "oppose_recall": recall[0],
        "oppose_f1": f1[0],
        "neutral_precision": precision[1],
        "neutral_recall": recall[1],
        "neutral_f1": f1[1],
        "support_precision": precision[2],
        "support_recall": recall[2],
        "support_f1": f1[2],
        "oppose_support": support[0],
        "neutral_support": support[1],
        "support_support": support[2],
    }


def row(method: str, scheme: str, model: str, params: dict, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {
        "method": method,
        "feature_scheme": scheme,
        "model": model,
        "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "split": split,
    }
    out.update(evaluate(y_true, y_pred))
    out["confusion_matrix"] = json.dumps(confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist())
    return out


def multiclass_weighted(
    X: sparse.csr_matrix,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    scheme: str,
) -> list[dict]:
    rows = []

    for model_name, factory in [
        ("LogisticRegression", lambda: LogisticRegression(max_iter=1800, class_weight="balanced", random_state=RANDOM_STATE)),
        ("XGBoost", xgb_multiclass),
    ]:
        for multiplier in [1, 2, 5, 10, 20, 40]:
            model = factory()
            params = {"class_weight": "balanced", "oppose_multiplier": multiplier}
            sample_weight = class_weight_vector(y[train_idx], oppose_multiplier=multiplier)
            if model_name == "LogisticRegression":
                model.fit(X[train_idx], y[train_idx], sample_weight=sample_weight)
            else:
                model.fit(X[train_idx], y[train_idx], sample_weight=sample_weight)
            for split, idx in [("val", val_idx), ("test", test_idx)]:
                pred = model.predict(X[idx])
                rows.append(row("weighted_multiclass", scheme, model_name, params, split, y[idx], pred))

            if hasattr(model, "predict_proba"):
                prob_val = model.predict_proba(X[val_idx])
                prob_test = model.predict_proba(X[test_idx])
                rows.extend(probability_threshold_search(prob_val, prob_test, y, val_idx, test_idx, scheme, model_name, base_params=params))
    return rows


def probability_threshold_search(
    prob_val: np.ndarray,
    prob_test: np.ndarray,
    y: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    scheme: str,
    model_name: str,
    base_params: dict,
) -> list[dict]:
    candidates = []
    for oppose_boost in [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]:
        for oppose_min_prob in [0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18]:
            pred_val = proba_to_prediction(prob_val, oppose_boost, oppose_min_prob)
            metrics = evaluate(y[val_idx], pred_val)
            score = metrics["macro_f1"] + 0.25 * metrics["oppose_recall"] - 0.10 * max(0, 0.50 - metrics["accuracy"])
            candidates.append((score, oppose_boost, oppose_min_prob, metrics))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[:8]
    rows = []
    for _, oppose_boost, oppose_min_prob, _ in candidates:
        params = {**base_params, "oppose_boost": oppose_boost, "oppose_min_prob": oppose_min_prob}
        for split, prob, idx in [("val", prob_val, val_idx), ("test", prob_test, test_idx)]:
            pred = proba_to_prediction(prob, oppose_boost, oppose_min_prob)
            rows.append(row("probability_threshold", scheme, model_name, params, split, y[idx], pred))
    return rows


def proba_to_prediction(prob: np.ndarray, oppose_boost: float, oppose_min_prob: float) -> np.ndarray:
    adjusted = prob.copy()
    adjusted[:, 0] *= oppose_boost
    pred = adjusted.argmax(axis=1)
    pred[(prob[:, 0] >= oppose_min_prob) & (pred != 0)] = 0
    return pred


def two_stage(
    X: sparse.csr_matrix,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    scheme: str,
) -> list[dict]:
    rows = []
    y_stage1 = (y == 2).astype(int)  # 1=support, 0=non-support

    for stage1_model_name, stage1_factory in [
        ("LogisticRegression", lambda: LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)),
        ("XGBoost", xgb_binary),
    ]:
        stage1 = stage1_factory()
        stage1.fit(X[train_idx], y_stage1[train_idx])
        non_support_train = train_idx[y[train_idx] != 2]
        y_stage2 = (y == 0).astype(int)  # 1=oppose, 0=neutral

        for stage2_model_name, stage2_factory in [
            ("LogisticRegression", lambda: LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)),
            ("XGBoost", xgb_binary),
        ]:
            for opp_mult in [1, 2, 5, 10, 20]:
                stage2 = stage2_factory()
                sw = None
                if opp_mult != 1:
                    sw = np.where(y_stage2[non_support_train] == 1, opp_mult, 1.0)
                stage2.fit(X[non_support_train], y_stage2[non_support_train], sample_weight=sw)

                for support_threshold in [0.45, 0.50, 0.55, 0.60]:
                    for oppose_threshold in [0.08, 0.12, 0.16, 0.22, 0.30]:
                        params = {
                            "stage1": stage1_model_name,
                            "stage2": stage2_model_name,
                            "support_threshold": support_threshold,
                            "oppose_threshold": oppose_threshold,
                            "stage2_oppose_multiplier": opp_mult,
                        }
                        for split, idx in [("val", val_idx), ("test", test_idx)]:
                            pred = two_stage_predict(stage1, stage2, X[idx], support_threshold, oppose_threshold)
                            rows.append(row("two_stage", scheme, f"{stage1_model_name}+{stage2_model_name}", params, split, y[idx], pred))
    return rows


def two_stage_predict(
    stage1,
    stage2,
    X_part: sparse.csr_matrix,
    support_threshold: float,
    oppose_threshold: float,
) -> np.ndarray:
    p_support = stage1.predict_proba(X_part)[:, 1]
    pred = np.full(X_part.shape[0], 1, dtype=int)  # neutral by default
    pred[p_support >= support_threshold] = 2
    non_support_mask = p_support < support_threshold
    if non_support_mask.any():
        p_oppose = stage2.predict_proba(X_part[non_support_mask])[:, 1]
        sub_pred = np.where(p_oppose >= oppose_threshold, 0, 1)
        pred[non_support_mask] = sub_pred
    return pred


def regression_thresholds(
    X: sparse.csr_matrix,
    y_cls: np.ndarray,
    y_score: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    scheme: str,
) -> list[dict]:
    rows = []
    for model_name, model in [
        ("Ridge", Ridge(alpha=3.0)),
        (
            "XGBoostRegressor",
            XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            ),
        ),
    ]:
        model.fit(X[train_idx], y_score[train_idx])
        pred_val_score = model.predict(X[val_idx])
        pred_test_score = model.predict(X[test_idx])
        best = []
        for low in np.arange(0.26, 0.48, 0.02):
            for high in np.arange(0.50, 0.76, 0.02):
                if low >= high:
                    continue
                pred_val = score_to_class(pred_val_score, low, high)
                metrics = evaluate(y_cls[val_idx], pred_val)
                score = metrics["macro_f1"] + 0.25 * metrics["oppose_recall"]
                best.append((score, low, high, metrics))
        for _, low, high, _ in sorted(best, key=lambda x: x[0], reverse=True)[:8]:
            params = {"low_threshold": round(float(low), 3), "high_threshold": round(float(high), 3)}
            for split, pred_score, idx in [("val", pred_val_score, val_idx), ("test", pred_test_score, test_idx)]:
                pred = score_to_class(pred_score, low, high)
                rows.append(row("regression_threshold", scheme, model_name, params, split, y_cls[idx], pred))
    return rows


def score_to_class(score: np.ndarray, low: float, high: float) -> np.ndarray:
    pred = np.full(len(score), 1, dtype=int)
    pred[score < low] = 0
    pred[score >= high] = 2
    return pred


def save_confusion_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=180)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_tradeoff_plot(results: pd.DataFrame) -> None:
    test = results[results["split"] == "test"].copy()
    plt.figure(figsize=(8, 5.5), dpi=180)
    for method, group in test.groupby("method"):
        plt.scatter(group["accuracy"], group["oppose_recall"], s=28, alpha=0.7, label=method)
    plt.xlabel("Accuracy")
    plt.ylabel("Oppose recall")
    plt.title("Oppose recovery tradeoff")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.savefig(FIG_DIR / "oppose_recall_accuracy_tradeoff.png", bbox_inches="tight")
    plt.close()


def summarize(results: pd.DataFrame) -> None:
    test = results[results["split"] == "test"].copy()
    view_cols = [
        "method",
        "feature_scheme",
        "model",
        "params",
        "accuracy",
        "macro_f1",
        "oppose_precision",
        "oppose_recall",
        "oppose_f1",
        "neutral_f1",
        "support_f1",
    ]
    top_macro = test.sort_values(["macro_f1", "oppose_recall"], ascending=False).head(12)
    top_oppose = test[test["oppose_recall"] > 0].sort_values(["oppose_f1", "oppose_recall"], ascending=False).head(12)
    recover = test[(test["oppose_recall"] >= 0.25) & (test["accuracy"] >= 0.45)].sort_values(
        ["macro_f1", "accuracy"], ascending=False
    ).head(12)

    top_macro[view_cols].to_csv(OUT_DIR / "top_by_macro_f1.csv", index=False, encoding="utf-8-sig")
    top_oppose[view_cols].to_csv(OUT_DIR / "top_by_oppose_f1.csv", index=False, encoding="utf-8-sig")
    recover[view_cols].to_csv(OUT_DIR / "reasonable_oppose_recovery.csv", index=False, encoding="utf-8-sig")

    best_macro = top_macro.iloc[0]
    best_oppose = top_oppose.iloc[0] if len(top_oppose) else None
    best_recover = recover.iloc[0] if len(recover) else None

    lines = [
        "# Oppose Optimization Exploration",
        "",
        "This experiment keeps the main data fixed and explores model-side ways to recover the rare oppose class.",
        "",
        "## Best Macro F1",
        "",
        format_record(best_macro),
    ]
    if best_oppose is not None:
        lines += ["", "## Best Oppose F1", "", format_record(best_oppose)]
    if best_recover is not None:
        lines += ["", "## Best Reasonable Oppose Recovery", "", format_record(best_recover)]
    lines += [
        "",
        "## Interpretation",
        "",
        "- If a configuration recovers oppose, it usually pays with lower accuracy and more neutral/support false positives.",
        "- This is expected because oppose has only about 2.2% support and its reason semantics often overlap with status-quo neutral.",
        "- Useful next step: decide whether the project wants macro-distribution prediction or minority-position diagnosis. The best model differs by that objective.",
        "",
    ]
    (OUT_DIR / "oppose_optimization_summary.md").write_text("\n".join(lines), encoding="utf-8")


def format_record(row_: pd.Series) -> str:
    return "\n".join(
        [
            f"- method: `{row_['method']}`",
            f"- feature_scheme: `{row_['feature_scheme']}`",
            f"- model: `{row_['model']}`",
            f"- params: `{row_['params']}`",
            f"- accuracy: `{row_['accuracy']:.4f}`",
            f"- macro_f1: `{row_['macro_f1']:.4f}`",
            f"- oppose_precision: `{row_['oppose_precision']:.4f}`",
            f"- oppose_recall: `{row_['oppose_recall']:.4f}`",
            f"- oppose_f1: `{row_['oppose_f1']:.4f}`",
            f"- neutral_f1: `{row_['neutral_f1']:.4f}`",
            f"- support_f1: `{row_['support_f1']:.4f}`",
        ]
    )


def run() -> None:
    setup()
    df = pd.read_parquet(DATA_PATH)
    df = df[df["stance"].isin(TARGET_MAP)].reset_index(drop=True)
    df["y_cls"] = df["stance"].map(TARGET_MAP)
    y_cls = df["y_cls"].to_numpy()
    y_score = df["support_score"].astype(float).to_numpy()
    train_idx, val_idx, test_idx = split_data(df)

    cache = build_embedding_cache(df)
    feature_mats = make_feature_matrices(df, train_idx, cache)

    all_rows = []
    for scheme in FEATURE_SCHEMES:
        X = feature_mats[scheme]
        print(f"Running scheme={scheme}, shape={X.shape}")
        all_rows.extend(multiclass_weighted(X, y_cls, train_idx, val_idx, test_idx, scheme))
        all_rows.extend(two_stage(X, y_cls, train_idx, val_idx, test_idx, scheme))
        all_rows.extend(regression_thresholds(X, y_cls, y_score, train_idx, val_idx, test_idx, scheme))

    results = pd.DataFrame(all_rows)
    results.to_csv(OUT_DIR / "oppose_optimization_results.csv", index=False, encoding="utf-8-sig")
    summarize(results)
    save_tradeoff_plot(results)

    test = results[results["split"] == "test"].copy()
    for name, selector in [
        ("best_macro", test.sort_values(["macro_f1", "oppose_recall"], ascending=False).head(1)),
        ("best_oppose_f1", test[test["oppose_recall"] > 0].sort_values(["oppose_f1", "oppose_recall"], ascending=False).head(1)),
    ]:
        if selector.empty:
            continue
        record = selector.iloc[0]
        cm = np.array(json.loads(record["confusion_matrix"]))
        y_true = np.repeat(CLASS_ORDER, cm.sum(axis=1))
        y_pred_parts = []
        for true_i, row_counts in enumerate(cm):
            for pred_i, count in enumerate(row_counts):
                y_pred_parts.extend([pred_i] * int(count))
        y_pred = np.array(y_pred_parts)
        save_confusion_plot(y_true, y_pred, f"{name}: {record['method']}", FIG_DIR / f"{name}_confusion_matrix.png")

    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    run()
