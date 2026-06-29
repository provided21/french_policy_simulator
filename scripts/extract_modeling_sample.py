"""
Extract a lightweight modeling dataset for one LLM query_id.

The responses table stores LLM outputs and a few demographic fields. The full
persona parquet stores the richer persona fields used for prompt construction.
This script joins them once and saves a small per-query dataset so downstream
feature engineering does not repeatedly load the full parquet.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
# Main analysis sample: FAISS similarity retrieval, closest aggregate fit to IFOP.
# Random robustness sample: 20260514_120755.
DEFAULT_QUERY_ID = "20260511_022424"

DB_PATH = ROOT / "data" / "results.db"
FULL_PARQUET = ROOT / "data" / "processed" / "df_full.parquet"
OUT_DIR = ROOT / "data" / "processed" / "modeling_samples"

FULL_FIELDS = [
    "persona_id",
    "uuid",
    "sex",
    "age",
    "marital_status",
    "household_type",
    "education_level",
    "occupation",
    "commune",
    "departement",
    "country",
    "persona",
    "cultural_background",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "skills_and_expertise",
    "skills_and_expertise_list",
    "hobbies_and_interests",
    "hobbies_and_interests_list",
    "career_goals_and_ambitions",
    "persona_text",
]

RESPONSE_FIELDS = [
    "query_id",
    "persona_id",
    "llm_response",
    "support_score",
    "stance",
    "similarity",
    "age",
    "sex",
    "occupation",
    "education_level",
    "departement",
]


def load_responses(query_id: str, valid_only: bool) -> pd.DataFrame:
    where = "query_id = ?"
    if valid_only:
        where += " AND stance IS NOT NULL"
    sql = f"""
        SELECT {", ".join(RESPONSE_FIELDS)}
        FROM responses
        WHERE {where}
        ORDER BY id
    """
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=[query_id])


def load_full_subset(persona_ids: set[str]) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(FULL_PARQUET)
    available_fields = [c for c in FULL_FIELDS if c in parquet_file.schema_arrow.names]
    id_col = "persona_id" if "persona_id" in available_fields else "uuid"
    if id_col not in available_fields:
        raise ValueError("Full parquet has neither persona_id nor uuid")

    wanted_ids = pa.array(list(persona_ids), type=pa.string())
    matched_tables = []
    scanned_rows = 0

    for batch in parquet_file.iter_batches(columns=available_fields, batch_size=200_000):
        table = pa.Table.from_batches([batch])
        id_values = pc.cast(table[id_col], pa.string())
        mask = pc.is_in(id_values, value_set=wanted_ids)
        filtered = table.filter(mask)
        scanned_rows += table.num_rows
        if filtered.num_rows:
            matched_tables.append(filtered)
            print(f"matched {sum(t.num_rows for t in matched_tables):,}/{len(persona_ids):,} after scanning {scanned_rows:,}")
        if sum(t.num_rows for t in matched_tables) >= len(persona_ids):
            break

    if not matched_tables:
        return pd.DataFrame(columns=available_fields)

    df_full = pa.concat_tables(matched_tables).to_pandas()
    if "persona_id" not in df_full.columns and "uuid" in df_full.columns:
        df_full["persona_id"] = df_full["uuid"].astype(str)
    df_full["persona_id"] = df_full["persona_id"].astype(str)
    return df_full.drop_duplicates("persona_id")


def build_dataset(query_id: str, valid_only: bool) -> pd.DataFrame:
    df_resp = load_responses(query_id, valid_only=valid_only)
    if df_resp.empty:
        raise ValueError(f"No responses found for query_id={query_id!r}")

    df_resp["persona_id"] = df_resp["persona_id"].astype(str)
    persona_ids = set(df_resp["persona_id"])
    df_full = load_full_subset(persona_ids)

    overlap_cols = {
        c for c in ["age", "sex", "occupation", "education_level", "departement"]
        if c in df_full.columns
    }
    df_full = df_full.rename(columns={c: f"{c}_full" for c in overlap_cols})

    df = df_resp.merge(df_full, on="persona_id", how="left", validate="many_to_one")

    # Prefer DB values for fields captured at response time; full values remain
    # available as *_full for audits.
    text_cols = [
        "persona", "cultural_background", "professional_persona",
        "sports_persona", "arts_persona", "travel_persona", "culinary_persona",
        "skills_and_expertise", "skills_and_expertise_list",
        "hobbies_and_interests", "hobbies_and_interests_list",
        "career_goals_and_ambitions", "persona_text",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df


def write_outputs(
    df: pd.DataFrame,
    query_id: str,
    valid_only: bool,
    dedup_persona: bool,
) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "valid" if valid_only else "all"
    if dedup_persona:
        suffix += "_dedup_persona"
    parquet_path = OUT_DIR / f"{query_id}_{suffix}_full_features.parquet"
    csv_path = OUT_DIR / f"{query_id}_{suffix}_field_summary.csv"

    df.to_parquet(parquet_path, index=False)

    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
        "non_empty": [
            int((df[c].astype(str).str.len() > 0).sum()) if df[c].dtype == "object" else int(df[c].notna().sum())
            for c in df.columns
        ],
        "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return parquet_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-id", default=DEFAULT_QUERY_ID)
    parser.add_argument("--all", action="store_true", help="Keep rows with missing stance too")
    parser.add_argument(
        "--dedup-persona",
        action="store_true",
        help="Keep the first response per persona_id for leakage-safe modeling audits",
    )
    args = parser.parse_args()

    valid_only = not args.all
    df = build_dataset(args.query_id, valid_only=valid_only)
    if args.dedup_persona:
        before = len(df)
        df = df.drop_duplicates("persona_id").copy()
        print(f"dedup persona_id: {before:,} -> {len(df):,}")
    parquet_path, csv_path = write_outputs(
        df,
        args.query_id,
        valid_only=valid_only,
        dedup_persona=args.dedup_persona,
    )

    print(f"query_id: {args.query_id}")
    print(f"rows: {len(df):,}")
    print(f"columns: {len(df.columns)}")
    print(f"stance counts:\n{df['stance'].value_counts(dropna=False).to_string()}")
    print(f"saved parquet: {parquet_path}")
    print(f"saved summary: {csv_path}")


if __name__ == "__main__":
    main()
