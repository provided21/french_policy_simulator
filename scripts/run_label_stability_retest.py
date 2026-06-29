"""
Small-sample LLM label stability retest.

This experiment checks whether LLM-generated support_score / stance labels are
stable for the same persona under repeated calls. It uses a stratified sample
from the clean 10k main dataset and calls DeepSeek directly through the OpenAI
compatible API, intentionally bypassing local prompt cache.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "modeling_samples" / "main_1w_faiss_p2_clean.parquet"
OUT_DIR = ROOT / "output" / "label_stability_retest"
PROGRESS_PATH = OUT_DIR / "progress.json"
CHECKPOINT_PATH = OUT_DIR / "label_stability_retest_raw.jsonl"

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TEMPERATURE = 0.2

STANCE_ORDER = ["support", "neutral", "oppose"]
STANCE_TO_ID = {"oppose": 0, "neutral": 1, "support": 2}

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / ".env")
    load_dotenv()


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


def build_prompt(row: pd.Series) -> str:
    persona_text = clean_text(row.get("persona_text"), max_chars=1800)
    return f"""你将扮演一位法国居民。请只根据下面的个人画像，判断这位居民对政策主张的支持程度。

政策主张：“将法国法定退休年龄从64岁恢复到62岁。”

个人画像：
- 完整画像：{persona_text}

请输出严格 JSON，不要输出任何额外文字：
{{
  "support_score": 0到10之间的整数,
  "stance": "oppose" | "neutral" | "support",
  "reason": "一句话说明，最多35个中文字符"
}}

评分含义：
0 = 强烈反对恢复到62岁
5 = 中立或不确定
10 = 强烈支持恢复到62岁

分类规则：
0-3 为 oppose；4-6 为 neutral；7-10 为 support"""


def parse_response(text: str) -> dict[str, Any]:
    raw = text or ""
    match = re.search(r"\{.*\}", raw.strip(), flags=re.S)
    parse_success = False
    data: dict[str, Any] = {}
    if match:
        try:
            data = json.loads(match.group(0))
            parse_success = True
        except json.JSONDecodeError:
            data = {}

    score_raw = data.get("support_score", None)
    try:
        score_0_10 = int(round(float(score_raw)))
    except Exception:
        score_0_10 = None

    if score_0_10 is not None:
        score_0_10 = max(0, min(10, score_0_10))
        support_score = score_0_10 / 10.0
        stance_from_score = "oppose" if score_0_10 <= 3 else "neutral" if score_0_10 <= 6 else "support"
    else:
        support_score = np.nan
        stance_from_score = ""

    stance = str(data.get("stance", "")).strip().lower()
    if stance not in STANCE_TO_ID:
        stance = stance_from_score

    return {
        "parse_success": parse_success,
        "support_score_0_10": score_0_10,
        "support_score": support_score,
        "stance": stance,
        "stance_from_score": stance_from_score,
        "stance_consistent": bool(stance and stance == stance_from_score),
        "reason": str(data.get("reason", "")).strip(),
    }


def sample_stratified(df: pd.DataFrame, n_per_stance: int, seed: int) -> pd.DataFrame:
    parts = []
    for stance in STANCE_ORDER:
        group = df[df["stance"] == stance]
        n = min(n_per_stance, len(group))
        parts.append(group.sample(n=n, random_state=seed))
    sample = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    sample["retest_sample_id"] = np.arange(len(sample))
    return sample


async def call_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    prompt: str,
    model: str,
    temperature: float,
    max_retries: int = 3,
) -> dict[str, Any]:
    last_error = ""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=temperature,
                )
                text = response.choices[0].message.content.strip()
                return {"api_success": True, "llm_response": text, "api_error": ""}
            except Exception as exc:
                last_error = str(exc)
                await asyncio.sleep(min(2 ** attempt, 8))
    return {"api_success": False, "llm_response": "", "api_error": last_error}


async def run_calls(tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set DEEPSEEK_API_KEY or LLM_API_KEY in .env.")

    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url, timeout=40.0)
    semaphore = asyncio.Semaphore(args.concurrency)
    checkpoint_lock = asyncio.Lock()
    completed_existing = load_checkpoint()
    results = list(completed_existing.values())
    done = 0

    async def wrapped(i: int, payload: dict[str, Any]) -> None:
        nonlocal done
        result = await call_one(client, semaphore, payload["prompt"], args.model, args.temperature)
        clean_payload = {k: v for k, v in payload.items() if k != "prompt"}
        parsed = parse_response(result.get("llm_response", ""))
        record = {**clean_payload, **result, **parsed}
        async with checkpoint_lock:
            append_checkpoint(record)
            results.append(record)
        done += 1
        total_done = len(completed_existing) + done
        total = len(completed_existing) + len(tasks)
        if done % 10 == 0 or done == len(tasks):
            write_progress(total_done, total, args)
            print(f"\rcompleted {total_done}/{total}", end="", flush=True)

    await asyncio.gather(*(wrapped(i, payload) for i, payload in enumerate(tasks)))
    print("")
    return results


def task_key(payload: dict[str, Any]) -> str:
    return f"{payload['retest_sample_id']}::{payload['repeat']}"


def load_checkpoint() -> dict[str, dict[str, Any]]:
    if not CHECKPOINT_PATH.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with CHECKPOINT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("api_success") and "retest_sample_id" in record and "repeat" in record:
                records[task_key(record)] = record
    return records


def append_checkpoint(record: dict[str, Any]) -> None:
    with CHECKPOINT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_progress(done: int, total: int, args: argparse.Namespace) -> None:
    PROGRESS_PATH.write_text(
        json.dumps(
            {
                "done": done,
                "total": total,
                "pct": round(done / total * 100, 2),
                "model": args.model,
                "temperature": args.temperature,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def summarize(out: pd.DataFrame, sample: pd.DataFrame, args: argparse.Namespace) -> None:
    parsed = out[out["api_success"] & out["parse_success"] & out["stance"].isin(STANCE_TO_ID)].copy()
    pivot_score = parsed.pivot_table(index="retest_sample_id", columns="repeat", values="support_score", aggfunc="first")
    pivot_stance = parsed.pivot_table(index="retest_sample_id", columns="repeat", values="stance", aggfunc="first")

    pair = pivot_score.dropna()
    if {1, 2}.issubset(pair.columns):
        score_corr = pair[1].corr(pair[2])
        score_abs_diff = (pair[1] - pair[2]).abs()
        mean_abs_diff = float(score_abs_diff.mean())
        median_abs_diff = float(score_abs_diff.median())
    else:
        score_corr = np.nan
        mean_abs_diff = np.nan
        median_abs_diff = np.nan

    stance_pair = pivot_stance.dropna()
    if {1, 2}.issubset(stance_pair.columns):
        agreement = float((stance_pair[1] == stance_pair[2]).mean())
        kappa = float(cohen_kappa_score(stance_pair[1], stance_pair[2], labels=["oppose", "neutral", "support"]))
    else:
        agreement = np.nan
        kappa = np.nan

    original = sample[["retest_sample_id", "stance", "support_score"]].rename(
        columns={"stance": "original_stance", "support_score": "original_support_score"}
    )
    if {"original_stance", "original_support_score"}.issubset(parsed.columns):
        merged = parsed.copy()
    else:
        merged = parsed.merge(original, on="retest_sample_id", how="left")
    original_agreement = float((merged["stance"] == merged["original_stance"]).mean()) if len(merged) else np.nan
    original_score_mae = float((merged["support_score"] - merged["original_support_score"]).abs().mean()) if len(merged) else np.nan

    rows = [
        {"metric": "sample_personas", "value": sample["retest_sample_id"].nunique()},
        {"metric": "calls", "value": len(out)},
        {"metric": "api_success_rate", "value": out["api_success"].mean()},
        {"metric": "parse_success_rate_all", "value": out["parse_success"].mean()},
        {"metric": "valid_parsed_calls", "value": len(parsed)},
        {"metric": "repeat_score_corr", "value": score_corr},
        {"metric": "repeat_score_mean_abs_diff_0_1", "value": mean_abs_diff},
        {"metric": "repeat_score_mean_abs_diff_0_10", "value": mean_abs_diff * 10 if not np.isnan(mean_abs_diff) else np.nan},
        {"metric": "repeat_score_median_abs_diff_0_10", "value": median_abs_diff * 10 if not np.isnan(median_abs_diff) else np.nan},
        {"metric": "repeat_stance_agreement", "value": agreement},
        {"metric": "repeat_stance_cohen_kappa", "value": kappa},
        {"metric": "agreement_with_original_main_stance", "value": original_agreement},
        {"metric": "score_mae_vs_original_main_0_10", "value": original_score_mae * 10 if not np.isnan(original_score_mae) else np.nan},
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "label_stability_summary.csv", index=False, encoding="utf-8-sig")

    by_original = (
        merged.groupby("original_stance")
        .agg(
            n=("stance", "size"),
            stance_agreement=("stance", lambda s: float((s == merged.loc[s.index, "original_stance"]).mean())),
            score_mae_0_10=("support_score", lambda s: float((s - merged.loc[s.index, "original_support_score"]).abs().mean() * 10)),
        )
        .reset_index()
    )
    by_original.to_csv(OUT_DIR / "agreement_by_original_stance.csv", index=False, encoding="utf-8-sig")

    pivot_compare = original.merge(pivot_stance.reset_index(), on="retest_sample_id", how="left")
    pivot_compare = pivot_compare.merge(pivot_score.reset_index(), on="retest_sample_id", how="left", suffixes=("_stance", "_score"))
    pivot_compare.to_csv(OUT_DIR / "persona_level_repeat_comparison.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Label Stability Retest",
        "",
        f"- n_per_stance: `{args.n_per_stance}`",
        f"- repeats: `{args.repeats}`",
        f"- model: `{args.model}`",
        f"- temperature: `{args.temperature}`",
        "",
        "## Summary",
        "",
        md_table(summary),
        "",
        "## Agreement By Original Stance",
        "",
        md_table(by_original),
        "",
        "## Interpretation",
        "",
        "- Repeat score correlation and stance agreement estimate the stability ceiling of the LLM label itself.",
        "- Agreement with original main labels measures whether the newly generated labels reproduce the existing main dataset.",
        "- If stability is imperfect, downstream supervised models cannot be expected to perfectly predict these labels from persona features.",
        "",
    ]
    (OUT_DIR / "label_stability_summary.md").write_text("\n".join(lines), encoding="utf-8")


def md_table(df: pd.DataFrame) -> str:
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else str(x))
    lines = ["| " + " | ".join(table.columns) + " |", "| " + " | ".join("---" for _ in table.columns) + " |"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("\n", " ") for col in table.columns) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-stance", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=float(os.getenv("DEEPSEEK_TEMPERATURE", DEFAULT_TEMPERATURE)))
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    setup()
    args = parse_args()
    df = pd.read_parquet(DATA_PATH)
    sample = sample_stratified(df, args.n_per_stance, args.seed)
    sample.to_csv(OUT_DIR / "retest_sample.csv", index=False, encoding="utf-8-sig")

    tasks = []
    for _, row in sample.iterrows():
        prompt = build_prompt(row)
        for repeat in range(1, args.repeats + 1):
            tasks.append(
                {
                    "retest_sample_id": int(row["retest_sample_id"]),
                    "persona_id": str(row.get("persona_id", row.get("uuid", ""))),
                    "original_stance": row["stance"],
                    "original_support_score": float(row["support_score"]),
                    "repeat": repeat,
                    "prompt": prompt,
                }
            )

    completed = load_checkpoint()
    if completed:
        tasks = [task for task in tasks if task_key(task) not in completed]
        print(f"resume checkpoint: completed={len(completed)}, remaining={len(tasks)}")

    if args.dry_run:
        if tasks:
            print(tasks[0]["prompt"][:1200])
        print(f"remaining_tasks={len(tasks)}")
        print(f"checkpoint_completed={len(completed)}")
        return

    if tasks:
        raw_results = asyncio.run(run_calls(tasks, args))
    else:
        raw_results = list(load_checkpoint().values())
    out = pd.DataFrame(raw_results)
    out = out.drop_duplicates(["retest_sample_id", "repeat"], keep="last").sort_values(["retest_sample_id", "repeat"])
    out.to_csv(OUT_DIR / "label_stability_retest_raw.csv", index=False, encoding="utf-8-sig")
    out.to_parquet(OUT_DIR / "label_stability_retest_raw.parquet", index=False)
    summarize(out, sample, args)
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
