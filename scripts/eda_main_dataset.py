"""
EDA for the selected 10k clean main dataset.

Outputs tables and figures for discussion before feature engineering/modeling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "modeling_samples" / "main_1w_faiss_p2_clean.parquet"
OUT_DIR = ROOT / "output" / "main_eda"
FIG_DIR = OUT_DIR / "figures"

IFOP = {"support": 61.0, "neutral": 34.0, "oppose": 5.0}
STANCE_ORDER = ["support", "neutral", "oppose"]

REASON_PATTERNS = {
    "age_retirement": r"年龄|退休|接近退休|提前退休|早日退休|年纪|还早|年轻|老",
    "job_work": r"工作|职业|员工|工人|体力|劳动|护理|教师|司机|压力|辛苦",
    "economy": r"收入|经济|负担|财务|生活|稳定|养老金|福利",
    "family": r"家庭|孩子|子女|家人|单亲|父母",
    "uncertain": r"未知|不确定|信息不足|中立|难以判断|无法判断",
}


def setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def save_fig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def stance_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["stance"].value_counts().reindex(STANCE_ORDER).fillna(0)
    pct = counts / len(df) * 100
    out = pd.DataFrame({
        "stance": STANCE_ORDER,
        "n": counts.astype(int).values,
        "pct": pct.round(2).values,
        "ifop_pct": [IFOP[s] for s in STANCE_ORDER],
    })
    out["gap_pp"] = (out["pct"] - out["ifop_pct"]).round(2)
    out.to_csv(OUT_DIR / "stance_distribution.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(7.5, 4.5))
    x = np.arange(len(out))
    width = 0.36
    plt.bar(x - width / 2, out["pct"], width, label="LLM", color="#7B5AB6")
    plt.bar(x + width / 2, out["ifop_pct"], width, label="IFOP", color="#B8B8B8")
    for i, value in enumerate(out["pct"]):
        plt.text(i - width / 2, value + 0.8, f"{value:.1f}", ha="center", fontsize=9)
    for i, value in enumerate(out["ifop_pct"]):
        plt.text(i + width / 2, value + 0.8, f"{value:.1f}", ha="center", fontsize=9)
    plt.title("主数据 stance 分布 vs IFOP 基准")
    plt.xticks(x, out["stance"])
    plt.ylabel("%")
    plt.legend()
    save_fig("stance_vs_ifop.png")
    return out


def score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    score_counts = df["support_score_0_10"].value_counts().sort_index()
    out = pd.DataFrame({
        "score_0_10": score_counts.index.astype(int),
        "n": score_counts.values,
        "pct": (score_counts.values / len(df) * 100).round(2),
    })
    out.to_csv(OUT_DIR / "score_distribution.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 4.5))
    plt.bar(out["score_0_10"].astype(str), out["pct"], color="#7B5AB6")
    plt.title("support_score 0-10 分布")
    plt.xlabel("support_score")
    plt.ylabel("%")
    save_fig("score_distribution.png")
    return out


def demographic_tables(df: pd.DataFrame) -> None:
    for col in ["sex", "occupation", "education_level", "marital_status", "household_type", "departement"]:
        group = (
            df.groupby(col, dropna=False)
            .agg(
                n=("persona_id", "count"),
                support_score_mean=("support_score", "mean"),
                support_pct=("stance", lambda s: (s == "support").mean() * 100),
                neutral_pct=("stance", lambda s: (s == "neutral").mean() * 100),
                oppose_pct=("stance", lambda s: (s == "oppose").mean() * 100),
                similarity_mean=("similarity", "mean"),
            )
            .reset_index()
        )
        group["pct_of_sample"] = group["n"] / len(df) * 100
        for c in ["support_score_mean", "support_pct", "neutral_pct", "oppose_pct", "similarity_mean", "pct_of_sample"]:
            group[c] = group[c].round(3)
        group.sort_values("n", ascending=False).to_csv(OUT_DIR / f"by_{col}.csv", index=False, encoding="utf-8-sig")


def age_analysis(df: pd.DataFrame) -> None:
    bins = [17, 24, 34, 44, 54, 61, 64, 74, 120]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-61", "62-64", "65-74", "75+"]
    work = df.copy()
    work["age_group"] = pd.cut(work["age"], bins=bins, labels=labels)
    age_group = (
        work.groupby("age_group", observed=False)
        .agg(
            n=("persona_id", "count"),
            support_score_mean=("support_score", "mean"),
            support_pct=("stance", lambda s: (s == "support").mean() * 100),
            neutral_pct=("stance", lambda s: (s == "neutral").mean() * 100),
            oppose_pct=("stance", lambda s: (s == "oppose").mean() * 100),
        )
        .reset_index()
    )
    for c in ["support_score_mean", "support_pct", "neutral_pct", "oppose_pct"]:
        age_group[c] = age_group[c].round(2)
    age_group.to_csv(OUT_DIR / "by_age_group.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 4.5))
    plt.plot(age_group["age_group"].astype(str), age_group["support_pct"], marker="o", color="#7B5AB6")
    plt.axhline(IFOP["support"], color="#777777", linestyle="--", linewidth=1, label="IFOP support 61%")
    plt.title("年龄段与支持率")
    plt.xlabel("")
    plt.ylabel("support %")
    plt.legend()
    save_fig("age_group_support.png")

    plt.figure(figsize=(8, 4.5))
    age_bins = np.linspace(df["age"].min(), df["age"].max(), 41)
    bottom = np.zeros(len(age_bins) - 1)
    colors = {"support": "#7B5AB6", "neutral": "#E0B64C", "oppose": "#D65F5F"}
    for stance in STANCE_ORDER:
        counts, _ = np.histogram(df.loc[df["stance"] == stance, "age"], bins=age_bins)
        plt.bar(age_bins[:-1], counts, width=np.diff(age_bins), bottom=bottom, align="edge", color=colors[stance], label=stance, alpha=0.85)
        bottom += counts
    plt.title("年龄分布按 stance 堆叠")
    plt.xlabel("age")
    plt.ylabel("n")
    plt.legend()
    save_fig("age_stance_hist.png")


def reason_analysis(df: pd.DataFrame) -> pd.DataFrame:
    reason = df["reason"].fillna("").astype(str)
    rows = []
    for name, pattern in REASON_PATTERNS.items():
        mask = reason.str.contains(pattern, regex=True)
        rows.append({
            "keyword_group": name,
            "n": int(mask.sum()),
            "pct": round(float(mask.mean() * 100), 2),
            "support_score_mean": round(float(df.loc[mask, "support_score"].mean()), 4) if mask.any() else np.nan,
        })
        df[f"reason_{name}"] = mask
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "reason_keyword_summary.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 4.5))
    plt.bar(out["keyword_group"], out["pct"], color="#7B5AB6")
    plt.title("reason 关键词组覆盖率")
    plt.xlabel("")
    plt.ylabel("%")
    plt.xticks(rotation=20)
    save_fig("reason_keyword_coverage.png")

    examples = []
    for stance in STANCE_ORDER:
        subset = df[df["stance"] == stance].head(20)
        examples.append(subset[["persona_id", "age", "occupation", "support_score_0_10", "stance", "reason"]])
    pd.concat(examples).to_csv(OUT_DIR / "reason_examples.csv", index=False, encoding="utf-8-sig")
    return out


def similarity_analysis(df: pd.DataFrame) -> None:
    sim = df["similarity"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_frame("similarity")
    sim.to_csv(OUT_DIR / "similarity_describe.csv", encoding="utf-8-sig")

    plt.figure(figsize=(8, 4.5))
    for stance in STANCE_ORDER:
        plt.hist(df.loc[df["stance"] == stance, "similarity"], bins=40, histtype="step", density=True, linewidth=1.8, label=stance)
    plt.title("FAISS similarity 分布按 stance")
    plt.xlabel("similarity")
    plt.ylabel("density")
    plt.legend()
    save_fig("similarity_by_stance.png")

    corr = df[["age", "similarity", "support_score"]].corr(numeric_only=True)
    corr.to_csv(OUT_DIR / "numeric_corr.csv", encoding="utf-8-sig")


def write_overview(df: pd.DataFrame, stance: pd.DataFrame, reason: pd.DataFrame) -> None:
    lines = [
        "# Main 1w Clean Dataset EDA",
        "",
        f"Rows: {len(df):,}",
        f"Unique personas: {df['persona_id'].nunique():,}",
        f"Duplicate persona rows: {df['persona_id'].duplicated().sum():,}",
        f"Age range: {df['age'].min()} - {df['age'].max()}",
        f"Mean support_score: {df['support_score'].mean():.4f}",
        "",
        "## Stance Distribution",
        "",
        stance.to_csv(index=False),
        "",
        "## Reason Keyword Summary",
        "",
        reason.to_csv(index=False),
    ]
    (OUT_DIR / "eda_overview.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    df = pd.read_parquet(DATA_PATH)
    stance = stance_distribution(df)
    score_distribution(df)
    demographic_tables(df)
    age_analysis(df)
    reason = reason_analysis(df)
    similarity_analysis(df)
    write_overview(df, stance, reason)
    print(f"rows={len(df):,}, unique_personas={df['persona_id'].nunique():,}")
    print(stance.to_string(index=False))
    print(reason.to_string(index=False))
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
