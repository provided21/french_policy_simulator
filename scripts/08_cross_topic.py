"""
跨议题验证实验
================================================================
验证"年龄独大"是否退休议题专属，还是跨议题普适规律。

设计:
  - 随机抽取 1,000 个画像
  - 用 Run 2 Prompt 模板询问移民政策议题
  - 与退休议题结果进行对比

议题:
  "关于移民政策，您认为应该怎么做？
   放宽限制、维持现状、还是进一步收紧？"

输出:
  - data/cross_topic_results.parquet
  - data/cross_topic_summary.csv

运行: python scripts/08_cross_topic.py
"""

import sys
import os
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

K = 1000
RANDOM_SEED = 42

# 跨议题: 移民政策
# 对标法国政治光谱中的另一核心争议
CROSS_TOPIC_QUESTION = (
    "关于移民政策，您认为应该怎么做？"
    "放宽限制、维持现状、还是进一步收紧？"
)

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("跨议题验证实验 —— 移民政策")
    print("=" * 60)
    print(f"\n  议题: {CROSS_TOPIC_QUESTION}")
    print(f"  样本量: K={K}")

    # 加载数据
    print("\n[1/4] 加载画像数据...")
    df = pd.read_parquet(Config.DATA_PATH)
    print(f"      总画像数: {len(df):,}")

    # 随机抽样
    print(f"\n[2/4] 随机抽样 {K} 个画像 (seed={RANDOM_SEED})...")
    sample_df = df.sample(n=K, random_state=RANDOM_SEED).copy().reset_index(drop=True)
    personas = sample_df.to_dict('records')
    print(f"      抽样完成")

    # 构建 prompts
    print(f"\n[3/4] 构建 Prompts...")
    prompts = build_batch_prompts(personas, CROSS_TOPIC_QUESTION)
    print(f"      生成 {len(prompts)} 条 Prompt")

    # 调用 LLM
    if not Config.LLM_API_KEY:
        print("\n[错误] 未找到 LLM_API_KEY")
        return

    client = LLMClient(
        api_key=Config.LLM_API_KEY,
        model=Config.LLM_MODEL,
        base_url=Config.LLM_BASE_URL,
    )

    def on_progress(done, total):
        pct = done / total * 100
        print(f"\r      进度: {done}/{total} ({pct:.1f}%)", end="", flush=True)

    responses = client.call_batch_sync(prompts, progress_callback=on_progress)
    print("")

    success = sum(1 for r in responses if r and r.get("success"))
    fail = len(responses) - success
    print(f"      成功: {success}, 失败: {fail}")

    # 解析
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

    # 统计
    print("\n[4/4] 统计汇总")
    print("=" * 60)
    valid = result_df[result_df["stance"].notna()]
    stance_pct = valid["stance"].value_counts(normalize=True) * 100

    print(f"\n  有效回答数: {len(valid)}/{len(result_df)}")
    print(f"  平均支持度: {valid['support_score'].mean():.3f}")
    print(f"\n  立场分布:")
    stance_labels = {"support": "支持(放宽)", "neutral": "中立(维持)", "oppose": "反对(收紧)"}
    for stance in ["support", "neutral", "oppose"]:
        pct = stance_pct.get(stance, 0)
        label = stance_labels.get(stance, stance)
        print(f"    {label}: {pct:.1f}%")

    # 年龄分组分析
    if "age" in valid.columns:
        print(f"\n  按年龄分组:")
        age_bins = [0, 30, 45, 60, 120]
        age_labels = ["<30", "30-45", "45-60", "60+"]
        valid["age_group"] = pd.cut(valid["age"], bins=age_bins, labels=age_labels)
        age_stats = valid.groupby("age_group")["support_score"].agg(["mean", "count"])
        for ag, row in age_stats.iterrows():
            print(f"    {ag}: {row['mean']:.3f} (n={int(row['count'])})")

    # 保存
    result_df["topic"] = "immigration"
    parquet_path = os.path.join(OUTPUT_DIR, "cross_topic_results.parquet")
    result_df.to_parquet(parquet_path, index=False)
    print(f"\n  详细结果已保存: {parquet_path}")

    # 汇总
    summary = {
        "topic": "immigration",
        "k": K,
        "n_valid": len(valid),
        "mean_score": valid["support_score"].mean(),
        "support_pct": stance_pct.get("support", 0),
        "neutral_pct": stance_pct.get("neutral", 0),
        "oppose_pct": stance_pct.get("oppose", 0),
    }
    csv_path = os.path.join(OUTPUT_DIR, "cross_topic_summary.csv")
    pd.DataFrame([summary]).to_csv(csv_path, index=False)
    print(f"  汇总已保存: {csv_path}")

    print("\n" + "=" * 60)
    print("跨议题实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
