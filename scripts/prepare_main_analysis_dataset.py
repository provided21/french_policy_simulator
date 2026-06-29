"""
Prepare the selected 10k main experiment for downstream data science work.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "output" / "sampling_prompt_experiment"
OUT_DIR = ROOT / "data" / "processed" / "modeling_samples"


def find_main_path() -> Path:
    summary_path = EXP_DIR / "summary_1w_main.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    if summary.empty:
        raise ValueError("summary_1w_main.csv is empty")
    return Path(summary.iloc[-1]["path"])


def audit(df: pd.DataFrame, label: str) -> dict[str, object]:
    stance = df["stance"].value_counts(normalize=True).mul(100)
    return {
        "dataset": label,
        "rows": len(df),
        "unique_personas": df["persona_id"].nunique(),
        "duplicate_persona_rows": int(df["persona_id"].duplicated().sum()),
        "api_success_rate": round(float(df["api_success"].mean()), 4) if "api_success" in df else None,
        "parse_success_rate": round(float(df["parse_success"].mean()), 4) if "parse_success" in df else None,
        "stance_consistency_rate": round(float(df["stance_consistent"].mean()), 4) if "stance_consistent" in df else None,
        "support_pct": round(float(stance.get("support", 0)), 2),
        "neutral_pct": round(float(stance.get("neutral", 0)), 2),
        "oppose_pct": round(float(stance.get("oppose", 0)), 2),
        "support_score_mean": round(float(df["support_score"].mean()), 4),
        "support_score_std": round(float(df["support_score"].std()), 4),
    }


def main() -> None:
    source_path = find_main_path()
    df = pd.read_parquet(source_path)

    clean = df[
        (df["api_success"])
        & (df["stance_consistent"])
        & df["support_score"].notna()
        & df["stance"].isin(["support", "neutral", "oppose"])
    ].copy()

    strict = clean[clean["parse_success"]].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full_path = OUT_DIR / "main_1w_faiss_p2_full.parquet"
    clean_path = OUT_DIR / "main_1w_faiss_p2_clean.parquet"
    strict_path = OUT_DIR / "main_1w_faiss_p2_strict_json.parquet"
    audit_path = OUT_DIR / "main_1w_faiss_p2_audit.csv"

    df.to_parquet(full_path, index=False)
    clean.to_parquet(clean_path, index=False)
    strict.to_parquet(strict_path, index=False)

    audit_df = pd.DataFrame([
        audit(df, "full"),
        audit(clean, "clean_api_consistent"),
        audit(strict, "strict_json"),
    ])
    audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")
    print(audit_df.to_string(index=False))
    print(f"source: {source_path}")
    print(f"full: {full_path}")
    print(f"clean: {clean_path}")
    print(f"strict: {strict_path}")
    print(f"audit: {audit_path}")


if __name__ == "__main__":
    main()
