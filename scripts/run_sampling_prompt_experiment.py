"""
Run the sampling-strategy x prompt-version screening experiment.

This script is intentionally separate from the original pipeline so the old
results remain reproducible. It fixes two issues needed for the rerun:

1. FAISS results are mapped through the dataframe used to build the index.
2. Invalid FAISS indices (-1) are filtered before dataframe lookup.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sentence_transformers import SentenceTransformer

from src.llm_client import LLMClient
from src.retriever.build_index import load_index
from src.utils import Config


QUESTION = "将法国法定退休年龄从64岁恢复到62岁。"
IFOP_SUPPORT_PCT = 61.0
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TEMPERATURE = 0.2
INDEX_SAMPLE_SIZE = 100_000
INDEX_RANDOM_STATE = 42

OUT_DIR = ROOT / "output" / "sampling_prompt_experiment"
MAPPING_PATH = ROOT / "data" / "processed" / "faiss_index_mapping_100k_seed42.parquet"
PROGRESS_PATH = OUT_DIR / "progress.json"

DEMO_FIELDS = [
    "age",
    "sex",
    "marital_status",
    "household_type",
    "education_level",
    "occupation",
    "departement",
    "commune",
]

CAREER_FIELDS = [
    "professional_persona",
    "skills_and_expertise",
    "career_goals_and_ambitions",
]

LIFESTYLE_FIELDS = [
    "persona",
    "cultural_background",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "hobbies_and_interests",
]

TEXT_FIELDS = [
    "persona",
    "cultural_background",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "hobbies_and_interests",
    "career_goals_and_ambitions",
    "skills_and_expertise",
    "persona_text",
]

LOAD_FIELDS = list(dict.fromkeys(["uuid", "persona_id", *DEMO_FIELDS, *CAREER_FIELDS, *LIFESTYLE_FIELDS, "persona_text"]))


def clean_text(value: Any, max_chars: int | None = None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        text = "未知"
    else:
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        if not text:
            text = "未知"
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def ensure_persona_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "persona_id" not in df.columns and "uuid" in df.columns:
        df["persona_id"] = df["uuid"].astype(str)
    df["persona_id"] = df["persona_id"].astype(str)
    return df


def load_personas(columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or LOAD_FIELDS
    df = pd.read_parquet(Config.DATA_PATH, columns=[c for c in cols if c != "persona_id"])
    return ensure_persona_id(df)


def sample_positions(total_rows: int, n: int, seed: int) -> np.ndarray:
    return pd.Series(np.arange(total_rows)).sample(n=n, random_state=seed).to_numpy()


def read_positions(positions: np.ndarray, columns: list[str]) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(Config.DATA_PATH)
    wanted = set(int(p) for p in positions)
    order = {int(pos): i for i, pos in enumerate(positions)}
    batches = []
    offset = 0
    read_columns = [c for c in columns if c != "persona_id"]
    for batch in parquet_file.iter_batches(columns=read_columns, batch_size=200_000):
        batch_len = batch.num_rows
        global_positions = [pos for pos in wanted if offset <= pos < offset + batch_len]
        if global_positions:
            global_positions.sort()
            local_indices = [pos - offset for pos in global_positions]
            table = pa.Table.from_batches([batch]).take(pa.array(local_indices, type=pa.int64()))
            table = table.append_column("__sample_order", pa.array([order[pos] for pos in global_positions], type=pa.int64()))
            batches.append(table)
        offset += batch_len
        if sum(t.num_rows for t in batches) >= len(positions):
            break
    if not batches:
        return pd.DataFrame(columns=columns)
    df = ensure_persona_id(pa.concat_tables(batches).to_pandas())
    df = df.sort_values("__sample_order").drop(columns=["__sample_order"]).reset_index(drop=True)
    return df


def build_index_mapping() -> pd.DataFrame:
    if MAPPING_PATH.exists():
        return pd.read_parquet(MAPPING_PATH)

    parquet_file = pq.ParquetFile(Config.DATA_PATH)
    total_rows = parquet_file.metadata.num_rows
    if INDEX_SAMPLE_SIZE and INDEX_SAMPLE_SIZE < total_rows:
        positions = sample_positions(total_rows, INDEX_SAMPLE_SIZE, INDEX_RANDOM_STATE)
        mapping = read_positions(positions, LOAD_FIELDS)
    else:
        mapping = load_personas(LOAD_FIELDS).reset_index(drop=True)
    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(MAPPING_PATH, index=False)
    print(f"saved FAISS mapping: {MAPPING_PATH}")
    return mapping


def sample_random(n: int, seed: int) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(Config.DATA_PATH)
    positions = sample_positions(parquet_file.metadata.num_rows, n, seed)
    return read_positions(positions, LOAD_FIELDS).reset_index(drop=True)


def sample_faiss(n: int) -> pd.DataFrame:
    import faiss  # noqa: F401 - ensures the conda env has FAISS available

    mapping = build_index_mapping()
    index = load_index(Config.INDEX_PATH)
    model_dir = ROOT / "all-MiniLM-L6-v2"
    model = SentenceTransformer(str(model_dir) if model_dir.exists() else "all-MiniLM-L6-v2")
    query_vec = model.encode([QUESTION], normalize_embeddings=True).astype(np.float32)
    if hasattr(index, "nprobe"):
        index.nprobe = min(128, getattr(index, "nlist", 128))

    candidate_k = min(index.ntotal, max(n * 8, n + 2000))
    scores, indices = index.search(query_vec, candidate_k)
    result = pd.DataFrame({"faiss_index": indices[0], "similarity": scores[0]})
    result = result[result["faiss_index"] >= 0].copy()
    result = result.drop_duplicates("faiss_index", keep="first").head(n)
    if len(result) < n:
        raise RuntimeError(f"FAISS returned only {len(result)} valid unique results for n={n}")

    sampled = mapping.iloc[result["faiss_index"].to_numpy()].copy().reset_index(drop=True)
    sampled["similarity"] = result["similarity"].to_numpy()
    if sampled["persona_id"].duplicated().any():
        sampled = sampled.drop_duplicates("persona_id", keep="first").reset_index(drop=True)
    if len(sampled) < n:
        raise RuntimeError(f"FAISS returned only {len(sampled)} unique persona_id rows for n={n}")
    return sampled.head(n).reset_index(drop=True)


def lines_for_fields(persona: dict[str, Any], fields: list[str], max_chars: int | None = None) -> str:
    labels = {
        "age": "年龄",
        "sex": "性别",
        "marital_status": "婚姻状态",
        "household_type": "家庭结构",
        "education_level": "学历",
        "occupation": "职业",
        "departement": "居住省份",
        "commune": "居住城镇",
        "persona": "个人画像",
        "cultural_background": "文化背景",
        "professional_persona": "职业画像",
        "sports_persona": "体育偏好",
        "arts_persona": "艺术偏好",
        "travel_persona": "旅行偏好",
        "culinary_persona": "饮食偏好",
        "hobbies_and_interests": "兴趣爱好",
        "skills_and_expertise": "技能与专长",
        "career_goals_and_ambitions": "职业目标",
        "persona_text": "完整画像",
    }
    return "\n".join(f"- {labels[f]}：{clean_text(persona.get(f), max_chars)}" for f in fields)


def build_experiment_prompt(persona: dict[str, Any], prompt_version: str) -> str:
    if prompt_version == "P0_demo":
        persona_block = lines_for_fields(persona, DEMO_FIELDS)
    elif prompt_version == "P1_career":
        persona_block = lines_for_fields(persona, [*DEMO_FIELDS, *CAREER_FIELDS], max_chars=220)
    elif prompt_version == "P2_full_text":
        persona_block = lines_for_fields(persona, ["persona_text"], max_chars=1800)
    elif prompt_version == "P3_compact_all":
        persona_block = lines_for_fields(persona, [*DEMO_FIELDS, *CAREER_FIELDS, *LIFESTYLE_FIELDS], max_chars=160)
    else:
        raise ValueError(f"Unknown prompt_version: {prompt_version}")

    return f"""你将扮演一位法国居民。请只根据下面的个人画像，判断这位居民对政策主张的支持程度。

政策主张：
“{QUESTION}”

个人画像：
{persona_block}

请输出严格 JSON，不要输出任何额外文字：
{{
  "support_score": 0到10之间的整数,
  "stance": "oppose" | "neutral" | "support",
  "reason": "一句话说明，最多25个中文字符"
}}

评分含义：
0 = 强烈反对恢复到62岁
5 = 中立或不确定
10 = 强烈支持恢复到62岁

分类规则：
0-3 为 oppose，4-6 为 neutral，7-10 为 support"""


def parse_json_response(text: str) -> dict[str, Any]:
    raw = text or ""
    cleaned = raw.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
    except Exception:
        score_match = re.search(r"(\d+(?:\.\d+)?)", raw)
        score = float(score_match.group(1)) if score_match else 5.0
        data = {"support_score": score, "stance": "", "reason": ""}

    score = float(data.get("support_score", 5))
    if score <= 1:
        score *= 10
    score = max(0, min(10, round(score)))
    stance = str(data.get("stance", "")).strip().lower()
    expected = "oppose" if score <= 3 else "neutral" if score <= 6 else "support"
    if stance not in {"oppose", "neutral", "support"}:
        stance = expected
    return {
        "support_score": score / 10.0,
        "support_score_0_10": score,
        "stance": stance,
        "stance_from_score": expected,
        "stance_consistent": stance == expected,
        "reason": clean_text(data.get("reason", ""), max_chars=80),
        "parse_success": bool(match),
    }


def init_experiment_tables(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_runs (
                query_id TEXT PRIMARY KEY,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                sampling TEXT,
                prompt_version TEXT,
                sample_size INTEGER,
                model TEXT,
                base_url TEXT,
                temperature REAL,
                prompt_question TEXT,
                notes TEXT
            )
            """
        )
        conn.commit()


def save_run_metadata(query_id: str, args: argparse.Namespace, sampling: str, prompt_version: str) -> None:
    init_experiment_tables(Path(Config.DB_PATH))
    with sqlite3.connect(Config.DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO experiment_runs
            (query_id, sampling, prompt_version, sample_size, model, base_url, temperature, prompt_question, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_id,
                sampling,
                prompt_version,
                args.n,
                args.model,
                args.base_url,
                args.temperature,
                QUESTION,
                args.run_label,
            ),
        )
        conn.commit()


def write_progress(payload: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = PROGRESS_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(PROGRESS_PATH)


def save_to_existing_tables(query_id: str, question: str, df: pd.DataFrame) -> None:
    with sqlite3.connect(Config.DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO queries (query_id, question, total_responses) VALUES (?, ?, ?)",
            (query_id, question, len(df)),
        )
        rows = [
            (
                query_id,
                row.get("persona_id", ""),
                row.get("llm_response", ""),
                row.get("support_score", None),
                row.get("stance", ""),
                row.get("similarity", None),
                row.get("age", None),
                row.get("sex", ""),
                row.get("occupation", ""),
                row.get("education_level", ""),
                row.get("departement", ""),
            )
            for _, row in df.iterrows()
        ]
        conn.executemany(
            """
            INSERT INTO responses
            (query_id, persona_id, llm_response, support_score, stance, similarity, age, sex, occupation, education_level, departement)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def run_one_group(args: argparse.Namespace, sampling: str, prompt_version: str, source_df: pd.DataFrame) -> dict[str, Any]:
    df = source_df.copy().reset_index(drop=True)
    prompts = [
        {
            "persona_id": f"{row['persona_id']}::{i}",
            "prompt": build_experiment_prompt(row.to_dict(), prompt_version),
        }
        for i, row in df.iterrows()
    ]

    client = LLMClient(
        api_key=args.api_key or Config.LLM_API_KEY,
        model=args.model,
        base_url=args.base_url,
        rpm=args.rpm,
        tpm=args.tpm,
        temperature=args.temperature,
    )

    def progress(done: int, total: int) -> None:
        write_progress({
            "run_label": args.run_label,
            "sampling": sampling,
            "prompt_version": prompt_version,
            "done": done,
            "total": total,
            "pct": round(done / total * 100, 2) if total else 0,
            "temperature": args.temperature,
            "model": args.model,
        })
        print(f"\r  {sampling}/{prompt_version}: {done}/{total}", end="", flush=True)

    responses = client.call_batch_sync(prompts, progress_callback=progress)
    print("")

    parsed_rows = []
    for response in responses:
        text = response.get("response", "") if response and response.get("success") else ""
        parsed = parse_json_response(text)
        parsed_rows.append({
            "llm_response": text,
            "api_success": bool(response and response.get("success")),
            "api_error": "" if response and response.get("success") else str(response.get("error", "")),
            **parsed,
        })

    parsed_df = pd.DataFrame(parsed_rows)
    out = pd.concat([df, parsed_df], axis=1)
    query_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sampling}_{prompt_version}"
    save_to_existing_tables(query_id, f"{QUESTION} [{sampling} {prompt_version}]", out)
    save_run_metadata(query_id, args, sampling, prompt_version)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{query_id}.parquet"
    out.to_parquet(out_path, index=False)
    write_progress({
        "run_label": args.run_label,
        "sampling": sampling,
        "prompt_version": prompt_version,
        "done": len(out),
        "total": len(out),
        "pct": 100.0,
        "temperature": args.temperature,
        "model": args.model,
        "status": "group_complete",
        "query_id": query_id,
        "path": str(out_path),
    })

    stance_pct = out["stance"].value_counts(normalize=True).mul(100)
    summary = {
        "query_id": query_id,
        "sampling": sampling,
        "prompt_version": prompt_version,
        "rows": len(out),
        "api_success_rate": out["api_success"].mean(),
        "parse_success_rate": out["parse_success"].mean(),
        "stance_consistency_rate": out["stance_consistent"].mean(),
        "support_pct": float(stance_pct.get("support", 0)),
        "neutral_pct": float(stance_pct.get("neutral", 0)),
        "oppose_pct": float(stance_pct.get("oppose", 0)),
        "ifop_support_abs_error": abs(float(stance_pct.get("support", 0)) - IFOP_SUPPORT_PCT),
        "support_score_mean": float(out["support_score"].mean()),
        "path": str(out_path),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", nargs="+", default=["random", "faiss"], choices=["random", "faiss"])
    parser.add_argument(
        "--prompt-versions",
        nargs="+",
        default=["P0_demo", "P1_career", "P2_full_text", "P3_compact_all"],
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY", os.getenv("LLM_API_KEY", "")))
    parser.add_argument("--rpm", type=int, default=120)
    parser.add_argument("--tpm", type=int, default=120_000)
    parser.add_argument("--run-label", default="sampling_prompt_screening")
    parser.add_argument("--summary-name", default="summary.csv")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key and not Config.LLM_API_KEY and not args.dry_run:
        raise RuntimeError("Missing LLM_API_KEY. Set it in .env or the current shell environment.")

    samples: dict[str, pd.DataFrame] = {}
    if "random" in args.sampling:
        samples["random"] = sample_random(args.n, args.seed)
    if "faiss" in args.sampling:
        samples["faiss"] = sample_faiss(args.n)

    print("sample audit:")
    for name, df in samples.items():
        print(f"  {name}: rows={len(df):,}, unique_persona={df['persona_id'].nunique():,}, duplicates={df['persona_id'].duplicated().sum():,}")

    if args.dry_run:
        for sampling, df in samples.items():
            row = df.iloc[0].to_dict()
            for prompt_version in args.prompt_versions:
                print(f"\n--- {sampling} {prompt_version} ---")
                print(build_experiment_prompt(row, prompt_version)[:1500])
        return

    summaries = []
    for sampling, df in samples.items():
        for prompt_version in args.prompt_versions:
            print(f"\nRunning {sampling} / {prompt_version} / n={len(df)}")
            summaries.append(run_one_group(args, sampling, prompt_version, df))
            summary_df = pd.DataFrame(summaries)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(OUT_DIR / args.summary_name, index=False, encoding="utf-8-sig")
            print(summary_df.tail(1).to_string(index=False))

    print(f"\nsaved summary: {OUT_DIR / args.summary_name}")


if __name__ == "__main__":
    main()
