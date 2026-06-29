"""
Profile the minority oppose class in the clean 10k main dataset.

The goal is diagnostic rather than predictive: identify whether oppose has a
stable social profile, whether its score boundary overlaps with neutral, and
what reason themes appear in the generated labels.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "modeling_samples" / "main_1w_faiss_p2_clean.parquet"
OUT_DIR = ROOT / "output" / "oppose_profile"
FIG_DIR = OUT_DIR / "figures"

STANCE_ORDER = ["support", "neutral", "oppose"]

AGE_BINS = [17, 24, 34, 44, 54, 61, 64, 74, 120]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-61", "62-64", "65-74", "75+"]

REASON_THEMES = {
    "stability_status_quo": r"稳定|安稳|现状|现行|维持|纪律|保守|传统",
    "change_aversion": r"变化|改变|变动|调整|突变|倒退|担忧|谨慎|审慎",
    "fiscal_sustainability": r"财政|可持续|养老金|经济|负担|福利|国家",
    "work_career": r"职业|工作|公务员|员工|企业|劳动",
    "reform_support": r"改革|制度|政策",
    "unknown": r"未知|信息不足|无法判断",
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


def add_age_bin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_bin"] = pd.cut(out["age"], bins=AGE_BINS, labels=AGE_LABELS)
    return out


def stance_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["stance"].value_counts().reindex(STANCE_ORDER).fillna(0).astype(int)
    out = pd.DataFrame(
        {
            "stance": counts.index,
            "n": counts.values,
            "pct": (counts.values / len(df) * 100).round(2),
            "score_mean": [df.loc[df["stance"] == s, "support_score"].mean() for s in counts.index],
            "score_median": [df.loc[df["stance"] == s, "support_score"].median() for s in counts.index],
        }
    )
    out[["score_mean", "score_median"]] = out[["score_mean", "score_median"]].round(4)
    out.to_csv(OUT_DIR / "stance_score_summary.csv", index=False, encoding="utf-8-sig")
    return out


def profile_table(df: pd.DataFrame, col: str, min_n: int = 30) -> pd.DataFrame:
    overall_oppose = (df["stance"] == "oppose").mean()
    grouped = (
        df.groupby(col, dropna=False)
        .agg(
            n=("stance", "size"),
            oppose_n=("stance", lambda s: int((s == "oppose").sum())),
            neutral_n=("stance", lambda s: int((s == "neutral").sum())),
            support_n=("stance", lambda s: int((s == "support").sum())),
            score_mean=("support_score", "mean"),
        )
        .reset_index()
    )
    grouped["oppose_rate"] = grouped["oppose_n"] / grouped["n"]
    grouped["neutral_rate"] = grouped["neutral_n"] / grouped["n"]
    grouped["support_rate"] = grouped["support_n"] / grouped["n"]
    grouped["oppose_lift"] = grouped["oppose_rate"] / overall_oppose
    grouped = grouped[grouped["n"] >= min_n].copy()
    grouped[["oppose_rate", "neutral_rate", "support_rate", "oppose_lift", "score_mean"]] = grouped[
        ["oppose_rate", "neutral_rate", "support_rate", "oppose_lift", "score_mean"]
    ].round(4)
    grouped = grouped.sort_values(["oppose_lift", "oppose_n"], ascending=False)
    grouped.to_csv(OUT_DIR / f"profile_by_{col}.csv", index=False, encoding="utf-8-sig")
    return grouped


def reason_theme_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stance in STANCE_ORDER:
        subset = df[df["stance"] == stance]
        reasons = subset["reason"].fillna("").astype(str)
        for theme, pattern in REASON_THEMES.items():
            matched = reasons.str.contains(pattern, regex=True)
            rows.append(
                {
                    "stance": stance,
                    "theme": theme,
                    "n": int(matched.sum()),
                    "pct": round(float(matched.mean() * 100), 2) if len(subset) else 0.0,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "reason_theme_by_stance.csv", index=False, encoding="utf-8-sig")
    return out


def extract_top_ngrams(texts: pd.Series, ngram_range: tuple[int, int] = (2, 4), top_k: int = 40) -> pd.DataFrame:
    counter: dict[str, int] = {}
    stop = set("，。；：、,.!?！？ 的了和与及是对在以为于")
    for text in texts.fillna("").astype(str):
        clean = re.sub(r"\s+", "", text)
        clean = "".join(ch for ch in clean if ch not in stop)
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(max(0, len(clean) - n + 1)):
                gram = clean[i : i + n]
                if any("\u4e00" <= ch <= "\u9fff" for ch in gram):
                    counter[gram] = counter.get(gram, 0) + 1
    out = pd.DataFrame(sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_k], columns=["ngram", "count"])
    out.to_csv(OUT_DIR / "oppose_reason_top_ngrams.csv", index=False, encoding="utf-8-sig")
    return out


def plot_age(df: pd.DataFrame) -> None:
    age = profile_table(df, "age_bin", min_n=20).sort_values("age_bin")
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(age["age_bin"].astype(str), age["oppose_rate"] * 100, color="#B54545")
    plt.axhline((df["stance"] == "oppose").mean() * 100, color="#444444", linestyle="--", linewidth=1)
    plt.ylabel("Oppose rate (%)")
    plt.title("Oppose rate by age group")
    save_fig("oppose_rate_by_age.png")


def plot_lift(table: pd.DataFrame, col: str, name: str, top_n: int = 10) -> None:
    top = table.sort_values(["oppose_lift", "oppose_n"], ascending=False).head(top_n).iloc[::-1]
    plt.figure(figsize=(9.5, 5.2))
    labels = top[col].astype(str)
    plt.barh(labels, top["oppose_lift"], color="#7B5AB6")
    plt.axvline(1.0, color="#444444", linestyle="--", linewidth=1)
    for i, (_, row) in enumerate(top.iterrows()):
        plt.text(row["oppose_lift"] + 0.03, i, f"n={int(row['n'])}, opp={int(row['oppose_n'])}", va="center", fontsize=8)
    plt.xlabel("Oppose lift vs overall")
    plt.title(f"Top {top_n} {name} groups enriched for oppose")
    save_fig(f"oppose_lift_by_{col}.png")


def plot_scores(df: pd.DataFrame) -> None:
    data = [df.loc[df["stance"] == s, "support_score"].dropna().values for s in STANCE_ORDER]
    plt.figure(figsize=(7.5, 4.8))
    plt.boxplot(data, tick_labels=STANCE_ORDER, showfliers=False)
    plt.ylabel("support_score")
    plt.title("Score distribution by stance")
    save_fig("score_distribution_by_stance.png")


def plot_reason_themes(theme_df: pd.DataFrame) -> None:
    pivot = theme_df.pivot(index="theme", columns="stance", values="pct").reindex(REASON_THEMES.keys())
    ax = pivot[STANCE_ORDER].plot(kind="bar", figsize=(10, 5.2), color=["#7B5AB6", "#B8B8B8", "#B54545"])
    ax.set_ylabel("Share of reasons (%)")
    ax.set_title("Reason theme share by stance")
    plt.xticks(rotation=35, ha="right")
    save_fig("reason_theme_by_stance.png")


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("\n", " ") for col in headers) + " |")
    return "\n".join(lines)


def write_markdown(
    summary: pd.DataFrame,
    age: pd.DataFrame,
    occupation: pd.DataFrame,
    education: pd.DataFrame,
    themes: pd.DataFrame,
    ngrams: pd.DataFrame,
) -> None:
    oppose_n = int(summary.loc[summary["stance"] == "oppose", "n"].iloc[0])
    oppose_pct = float(summary.loc[summary["stance"] == "oppose", "pct"].iloc[0])
    top_occ = occupation.head(5)[["occupation", "n", "oppose_n", "oppose_rate", "oppose_lift"]]
    top_edu = education.head(5)[["education_level", "n", "oppose_n", "oppose_rate", "oppose_lift"]]
    oppose_theme = themes[themes["stance"] == "oppose"].sort_values("pct", ascending=False)

    md = [
        "# Oppose Minority Profile",
        "",
        "## Key Numbers",
        "",
        f"- Clean main dataset rows: {int(summary['n'].sum())}",
        f"- Oppose rows: {oppose_n} ({oppose_pct:.2f}%)",
        "- This is a diagnostic analysis only; it does not rebalance or retrain models.",
        "",
        "## Interpretation",
        "",
        "The oppose class is very small and semantically close to neutral in many generated reasons. Many oppose reasons emphasize stability, status quo, fiscal caution, or opposition to lowering the retirement age, rather than an explicit and enthusiastic preference for further raising the retirement age. This helps explain why direct three-class models fail to learn a stable oppose boundary.",
        "",
        "## Stance And Score Summary",
        "",
        md_table(summary),
        "",
        "## Top Occupation Groups By Oppose Lift",
        "",
        md_table(top_occ),
        "",
        "## Top Education Groups By Oppose Lift",
        "",
        md_table(top_edu),
        "",
        "## Oppose Reason Themes",
        "",
        md_table(oppose_theme[["theme", "n", "pct"]]),
        "",
        "## Frequent Oppose Reason N-grams",
        "",
        md_table(ngrams.head(20)),
        "",
        "## PPT-ready Finding",
        "",
        "> Oppose is not simply a hidden class that can be recovered by a stronger classifier. In the current LLM-generated main data, it is rare, concentrated in some socially advantaged or status-quo-oriented groups, and often phrased as fiscal caution or resistance to lowering retirement age. The next optimization should therefore focus on boundary design and diagnostic sampling, not only model complexity.",
        "",
    ]
    (OUT_DIR / "oppose_profile_summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    setup()
    df = pd.read_parquet(DATA_PATH)
    df = add_age_bin(df)

    summary = stance_summary(df)
    age = profile_table(df, "age_bin", min_n=20)
    occupation = profile_table(df, "occupation", min_n=30)
    education = profile_table(df, "education_level", min_n=30)
    sex = profile_table(df, "sex", min_n=30)
    marital = profile_table(df, "marital_status", min_n=30)
    household = profile_table(df, "household_type", min_n=30)
    departement = profile_table(df, "departement", min_n=30)
    themes = reason_theme_table(df)
    ngrams = extract_top_ngrams(df.loc[df["stance"] == "oppose", "reason"])

    df[df["stance"] == "oppose"].sort_values("support_score")[
        [
            "uuid",
            "age",
            "sex",
            "education_level",
            "occupation",
            "departement",
            "support_score",
            "reason",
            "persona_text",
        ]
    ].to_csv(OUT_DIR / "oppose_cases.csv", index=False, encoding="utf-8-sig")

    plot_age(df)
    plot_lift(occupation, "occupation", "occupation")
    plot_lift(education, "education_level", "education")
    plot_scores(df)
    plot_reason_themes(themes)

    write_markdown(summary, age, occupation, education, themes, ngrams)

    # Touch these variables to make the produced CSVs explicit in logs/readers.
    _ = (sex, marital, household, departement)
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
