"""
跨模型对比实验
================================================================
验证 Prompt 敏感度、年龄独大等规律是否跨模型一致。

设计:
  - 随机抽取 1,000 个画像
  - 分别用 4 个模型回答同一问题（Run 2 Prompt）
  - 对比立场分布、平均分、与 IFOP 差距

模型配置 (通过环境变量或此处直接修改):
  MODEL_CONFIGS = [
      {"name": "DeepSeek-V4-Flash", "model": "...", "base_url": "...", "api_key": "..."},
      {"name": "Qwen2.5-72B",       "model": "...", "base_url": "...", "api_key": "..."},
      {"name": "GPT-4o",            "model": "...", "base_url": "...", "api_key": "..."},
      {"name": "Claude-3.5-Sonnet", "model": "...", "base_url": "...", "api_key": "..."},
  ]

输出:
  - data/cross_model_results.parquet
  - data/cross_model_summary.csv

运行: python scripts/06_cross_model.py
"""

import sys
import os
import argparse
import json
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.utils import Config
from src.llm_client import LLMClient, build_batch_prompts, parse_response

# ============================================================
# 实验配置
# ============================================================

K = 1000  # 每个模型的样本量
RANDOM_SEED = 42  # 随机种子，保证可复现

QUESTION = (
    "关于法定退休年龄，您认为应该怎么做？"
    "回到62岁、维持在64岁，还是进一步提高？"
)

# 跨模型配置 —— 每个模型可以指向不同的 API provider
# 环境变量: CROSS_<NAME>_API_KEY / CROSS_<NAME>_BASE_URL / CROSS_<NAME>_MODEL
MODEL_CONFIGS = [
    {
        "name": "DeepSeek-V4-Flash",
        "model": os.getenv("CROSS_DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V4-Flash"),
        "base_url": os.getenv("CROSS_DEEPSEEK_BASE_URL", Config.LLM_BASE_URL),
        "api_key": os.getenv("CROSS_DEEPSEEK_API_KEY", Config.LLM_API_KEY),
    },
    {
        "name": "Qwen2.5-72B",
        "model": os.getenv("CROSS_QWEN_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
        "base_url": os.getenv("CROSS_QWEN_BASE_URL", Config.LLM_BASE_URL),
        "api_key": os.getenv("CROSS_QWEN_API_KEY", Config.LLM_API_KEY),
    },
    {
        "name": "GPT-4o",
        "model": os.getenv("CROSS_GPT_MODEL", "gpt-4o"),
        "base_url": os.getenv("CROSS_GPT_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("CROSS_GPT_API_KEY", ""),
    },
    {
        "name": "Claude-3.5-Sonnet",
        "model": os.getenv("CROSS_CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
        "base_url": os.getenv("CROSS_CLAUDE_BASE_URL", "https://api.anthropic.com/v1"),
        "api_key": os.getenv("CROSS_CLAUDE_API_KEY", ""),
    },
]

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("跨模型对比实验")
    print("=" * 60)

    # 加载数据
    print("\n[1/5] 加载画像数据...")
    df = pd.read_parquet(Config.DATA_PATH)
    print(f"      总画像数: {len(df):,}")

    # 随机采样（固定种子，可复现）
    print(f"\n[2/5] 随机抽样 {K} 个画像 (seed={RANDOM_SEED})...")
    sample_df = df.sample(n=K, random_state=RANDOM_SEED).copy().reset_index(drop=True)
    personas = sample_df.to_dict('records')
    print(f"      抽样完成")

    # 构建 prompts（所有模型共用）
    print(f"\n[3/5] 构建 Prompts...")
    prompts = build_batch_prompts(personas, QUESTION)
    print(f"      生成 {len(prompts)} 条 Prompt")

    # 逐个模型运行
    all_results = {}

    for idx, mcfg in enumerate(MODEL_CONFIGS):
        model_name = mcfg["name"]

        if not mcfg["api_key"]:
            print(f"\n[4.{idx+1}/5] 跳过 {model_name} (无 API Key)")
            continue

        print(f"\n[4.{idx+1}/5] 运行 {model_name}...")
        print(f"      Model: {mcfg['model']}")
        print(f"      Base URL: {mcfg['base_url']}")

        client = LLMClient(
            api_key=mcfg["api_key"],
            model=mcfg["model"],
            base_url=mcfg["base_url"],
            rpm=1500 if "gpt" in model_name.lower() else 3000,
            tpm=200000 if "gpt" in model_name.lower() else 500000,
        )

        def on_progress(done, total):
            pct = done / total * 100
            print(f"\r      进度: {done}/{total} ({pct:.1f}%)", end="", flush=True)

        responses = client.call_batch_sync(prompts, progress_callback=on_progress)
        print("")

        # 解析
        success = sum(1 for r in responses if r and r.get("success"))
        fail = len(responses) - success
        print(f"      成功: {success}, 失败: {fail}")

        # 合并到 DataFrame
        result_df = sample_df.copy()
        resp_dict = {r['persona_id']: r['response'] for r in responses
                     if r and r.get("success") and r.get("response")}
        result_df['llm_response'] = result_df['persona_id'].map(resp_dict)

        mask = result_df['llm_response'].notna() & (result_df['llm_response'] != '')
        parsed = result_df.loc[mask, 'llm_response'].apply(parse_response)
        result_df["support_score"] = None
        result_df["stance"] = None
        result_df.loc[mask, 'support_score'] = parsed.apply(lambda x: x['support_score'])
        result_df.loc[mask, 'stance'] = parsed.apply(lambda x: x['stance'])
        result_df["model"] = model_name

        all_results[model_name] = result_df

    # 汇总统计
    print("\n" + "=" * 60)
    print("[5/5] 跨模型对比统计")
    print("=" * 60)

    summary_rows = []
    for model_name, rdf in all_results.items():
        valid = rdf[rdf["stance"].notna()]
        if len(valid) == 0:
            continue
        stance_pct = valid["stance"].value_counts(normalize=True) * 100
        row = {
            "model": model_name,
            "n_valid": len(valid),
            "mean_score": valid["support_score"].mean(),
            "support_pct": stance_pct.get("support", 0),
            "neutral_pct": stance_pct.get("neutral", 0),
            "oppose_pct": stance_pct.get("oppose", 0),
            "gap_to_ifop": stance_pct.get("support", 0) - 61,  # IFOP baseline = 61%
        }
        summary_rows.append(row)
        print(f"\n  {model_name} (n={row['n_valid']}):")
        print(f"    支持: {row['support_pct']:.1f}%  "
              f"中立: {row['neutral_pct']:.1f}%  "
              f"反对: {row['oppose_pct']:.1f}%")
        print(f"    平均分: {row['mean_score']:.3f}  "
              f"IFOP差距: {row['gap_to_ifop']:+.1f}pp")

    summary_df = pd.DataFrame(summary_rows)

    # 保存
    combined = pd.concat(all_results.values(), ignore_index=True)
    parquet_path = os.path.join(OUTPUT_DIR, "cross_model_results.parquet")
    combined.to_parquet(parquet_path, index=False)
    print(f"\n  详细结果已保存: {parquet_path}")

    csv_path = os.path.join(OUTPUT_DIR, "cross_model_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  汇总统计已保存: {csv_path}")

    print("\n" + "=" * 60)
    print("跨模型实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
