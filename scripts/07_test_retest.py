"""
重测信度实验
================================================================
直接测量 LLM 回答的组内方差，将 R² 天花板论证从理论推导变为实证测量。

设计:
  - 随机抽取 100 个画像
  - 每个画像重复询问 5 次（打乱顺序，间隔足够远避免缓存效应）
  - 计算组内标准差 σ_within → 得到 LLM 采样噪声的无偏估计

核心指标:
  - σ_within: 同一画像重复回答的组内标准差
  - ICC (组内相关系数): 组间方差 / (组间方差 + 组内方差)
  - 对比 σ_within² 与 Var(f(x)) = Var(y) - σ_within²

输出:
  - data/test_retest_results.parquet
  - data/test_retest_stats.csv

运行: python scripts/07_test_retest.py
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
from src.llm_client import LLMClient, build_prompt, parse_response

# ============================================================
# 实验配置
# ============================================================

N_PERSONAS = 100  # 重复测量的画像数
N_REPEATS = 5     # 每个画像重复次数
RANDOM_SEED = 42

QUESTION = (
    "关于法定退休年龄，您认为应该怎么做？"
    "回到62岁、维持在64岁，还是进一步提高？"
)

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("重测信度实验")
    print("=" * 60)
    print(f"\n  N = {N_PERSONAS} 画像 × {N_REPEATS} 次 = {N_PERSONAS * N_REPEATS} 次调用")

    # 加载数据
    print("\n[1/4] 加载画像数据...")
    df = pd.read_parquet(Config.DATA_PATH)
    print(f"      总画像数: {len(df):,}")

    # 随机抽样
    print(f"\n[2/4] 随机抽样 {N_PERSONAS} 个画像 (seed={RANDOM_SEED})...")
    sample_df = df.sample(n=N_PERSONAS, random_state=RANDOM_SEED).copy().reset_index(drop=True)
    print(f"      抽样完成")

    # 构建重复 prompts：每个画像 N_REPEATS 次，打乱顺序
    print(f"\n[3/4] 构建重复 Prompts 并打乱顺序...")
    all_prompts = []
    for _, row in sample_df.iterrows():
        persona = row.to_dict()
        persona_id = persona.get("persona_id", persona.get("uuid", ""))
        for rep in range(N_REPEATS):
            prompt = build_prompt(persona, QUESTION)
            all_prompts.append({
                "persona_id": persona_id,
                "repetition": rep,
                "prompt": prompt,
            })
    # 打乱顺序，使同一画像的重复请求不连续
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(all_prompts)
    print(f"      生成 {len(all_prompts)} 条 Prompt，顺序已打乱")

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

    flat_prompts = [{"persona_id": p["persona_id"], "prompt": p["prompt"]}
                    for p in all_prompts]
    responses = client.call_batch_sync(flat_prompts, progress_callback=on_progress)
    print("")

    success = sum(1 for r in responses if r and r.get("success"))
    fail = len(responses) - success
    print(f"      成功: {success}, 失败: {fail}")

    # 合并结果
    result_df = sample_df.copy()
    for i, r in enumerate(responses):
        rep_info = all_prompts[i]
        result_df.at[rep_info["persona_id"] if "persona_id" in result_df.columns else 0,
                     f"response_{rep_info['repetition']}"] = (
            r.get("response") if r and r.get("success") else None
        )

    # 解析每次回答的分数
    scores = []  # list of (persona_id, repetition, score, stance)
    response_texts = []
    for i, r in enumerate(responses):
        rep_info = all_prompts[i]
        if r and r.get("success") and r.get("response"):
            parsed = parse_response(r["response"])
            scores.append({
                "persona_id": rep_info["persona_id"],
                "repetition": rep_info["repetition"],
                "support_score": parsed["support_score"],
                "stance": parsed["stance"],
            })
            response_texts.append({
                "persona_id": rep_info["persona_id"],
                "repetition": rep_info["repetition"],
                "response": r["response"],
            })

    scores_df = pd.DataFrame(scores)
    texts_df = pd.DataFrame(response_texts)

    # ============================================================
    # 计算重测信度指标
    # ============================================================
    print("\n[4/4] 计算重测信度...")

    # 组内标准差：每个 persona 5 次回答的标准差，再取平均
    within_stds = scores_df.groupby("persona_id")["support_score"].std()
    sigma_within = within_stds.mean()
    sigma_within_se = within_stds.std() / np.sqrt(len(within_stds))

    # 总方差
    total_var = scores_df["support_score"].var()

    # 组间方差（每个 persona 均值的方差）
    persona_means = scores_df.groupby("persona_id")["support_score"].mean()
    between_var = persona_means.var()

    # ICC(1): 组间方差 / 总方差
    icc = between_var / total_var if total_var > 0 else 0

    # 噪声-信号比
    noise_signal_ratio = (sigma_within ** 2) / between_var if between_var > 0 else float('inf')

    # 立场一致性: 同一个人的5次回答中，立场一致的比例
    stance_agreement = 0
    total_pairs = 0
    for pid, group in scores_df.groupby("persona_id"):
        stances = group["stance"].values
        for i in range(len(stances)):
            for j in range(i + 1, len(stances)):
                if stances[i] == stances[j]:
                    stance_agreement += 1
                total_pairs += 1
    stance_agreement_rate = stance_agreement / total_pairs if total_pairs > 0 else 0

    print(f"\n  {'='*50}")
    print(f"  重测信度指标")
    print(f"  {'='*50}")
    print(f"  有效画像数:         {len(within_stds)}")
    print(f"  组内标准差 σ_within:  {sigma_within:.4f} ± {sigma_within_se:.4f}")
    print(f"  总方差 Var(y):        {total_var:.4f}")
    print(f"  组间方差 Var(mean):   {between_var:.4f}")
    print(f"  ICC(1):              {icc:.4f}")
    print(f"  噪声/信号比:          {noise_signal_ratio:.2f}")
    print(f"  立场一致率:           {stance_agreement_rate:.1%}")
    print(f"  {'='*50}")

    # 对比推导值
    # 如果 Var(ε) ≈ 2.6 × Var(f(x)) 成立，则:
    #   noise_signal_ratio ≈ 2.6
    #   σ_within² / between_var ≈ 2.6
    derived_nsr = 2.6  # from R² ceiling analysis
    print(f"\n  R²天花板推导的噪声/信号比: {derived_nsr}")
    if abs(noise_signal_ratio - derived_nsr) < 1.0:
        print(f"  ✓ 实测值与推导值一致（差异 < 1.0）")
    else:
        print(f"  → 实测值({noise_signal_ratio:.2f})与推导值({derived_nsr})存在差异")

    # 保存
    stats = {
        "n_personas": len(within_stds),
        "n_repeats": N_REPEATS,
        "sigma_within": sigma_within,
        "sigma_within_se": sigma_within_se,
        "total_var": total_var,
        "between_var": between_var,
        "icc": icc,
        "noise_signal_ratio": noise_signal_ratio,
        "stance_agreement_rate": stance_agreement_rate,
        "derived_nsr": derived_nsr,
    }

    stats_path = os.path.join(OUTPUT_DIR, "test_retest_stats.csv")
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    print(f"\n  统计已保存: {stats_path}")

    scores_path = os.path.join(OUTPUT_DIR, "test_retest_scores.parquet")
    scores_df.to_parquet(scores_path, index=False)
    print(f"  详细分数已保存: {scores_path}")

    texts_path = os.path.join(OUTPUT_DIR, "test_retest_responses.parquet")
    texts_df.to_parquet(texts_path, index=False)
    print(f"  回答文本已保存: {texts_path}")

    print("\n" + "=" * 60)
    print("重测信度实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
