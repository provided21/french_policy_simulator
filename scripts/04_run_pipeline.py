"""
完整流程脚本

功能：串联所有模块，执行完整的查询流程
运行：
  python scripts/04_run_pipeline.py                        # 默认：相似度检索
  python scripts/04_run_pipeline.py --mode random           # 随机抽样
  python scripts/04_run_pipeline.py --mode random --seed 42 # 随机抽样 + 固定种子
"""

import sys
import os
import argparse
from datetime import datetime

# 强制 UTF-8 输出，避免 GBK 编码崩溃
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.utils import Config
from src.retriever import create_database_adapter
from src.llm_client import LLMClient, build_batch_prompts, parse_response


# ============================================
# 用户配置区域 - 在这里修改问题
# ============================================

QUESTION = "您是否支持将法定退休年龄从64岁恢复到62岁？请给出0-10的支持度分数（0=完全反对，认为应进一步提高退休年龄；5=维持现状64岁；10=完全支持回到62岁）。"
K = 10000  # 采样数量

# ============================================


def main(mode="similarity", seed=42):
    """
    主函数：执行完整流程

    参数:
        mode: "similarity" (FAISS语义检索) 或 "random" (随机抽样)
        seed: 随机种子 (仅 random 模式)
    """
    print("=" * 60)
    print("法国政策模拟器 - 完整流程")
    print("=" * 60)

    print(f"\n问题: {QUESTION}")
    print(f"采样方式: {'FAISS语义相似度检索' if mode == 'similarity' else '随机抽样 (seed=' + str(seed) + ')'}")
    print(f"采样数量: {K}")

    # 1. 检查API Key
    if not Config.LLM_API_KEY:
        print("\n[错误] 未找到LLM_API_KEY")
        print("请在 .env 文件中设置 LLM_API_KEY=your_key")
        return

    # 2. 加载数据
    print("\n[1/5] 加载数据...")
    df = pd.read_parquet(Config.DATA_PATH)
    print(f"      加载完成：{len(df):,} 行")

    # 3. 采样
    if mode == "random":
        print(f"\n[2/5] 随机抽样 (seed={seed})...")
        results_df = df.sample(n=min(K, len(df)), random_state=seed).copy().reset_index(drop=True)
        print(f"      抽样完成：{len(results_df)} 条")
    else:
        # 加载索引和模型 (仅在 similarity 模式导入)
        print("\n[2/5] 加载检索索引...")
        from src.retriever.build_index import load_index
        from src.retriever.search import search_similar
        from sentence_transformers import SentenceTransformer
        index = load_index(Config.INDEX_PATH)
        MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "all-MiniLM-L6-v2")
        model = SentenceTransformer(MODEL_DIR)
        print(f"      加载完成")

        # 检索相似人物
        print("\n[3/5] 检索相似人物...")
        results_df = search_similar(QUESTION, model, index, df, k=K)
        print(f"      检索完成：{len(results_df)} 条")

    step = 4 if mode == "similarity" else 3

    # 4/5. 调用LLM
    print(f"\n[{step}/5] 调用LLM获取回答...")
    personas = results_df.to_dict('records')
    prompts = build_batch_prompts(personas, QUESTION)

    client = LLMClient(
        api_key=Config.LLM_API_KEY,
        model=Config.LLM_MODEL,
        base_url=Config.LLM_BASE_URL
    )
    def on_progress(done, total):
        pct = done / total * 100
        print(f"\r      进度: {done}/{total} ({pct:.1f}%)", end="", flush=True)

    llm_responses = client.call_batch_sync(prompts, progress_callback=on_progress)
    print("")  # 换行

    # 统计成功/失败
    success = sum(1 for r in llm_responses if r and r.get("success"))
    fail = len(llm_responses) - success
    print(f"      成功: {success}, 失败: {fail}")

    # 将回答合并到DataFrame
    response_dict = {r['persona_id']: r['response'] for r in llm_responses if r and r.get("success") and r.get("response")}
    results_df['llm_response'] = results_df['persona_id'].map(response_dict)

    # 仅解析成功回复，失败请求不参与统计
    mask = results_df['llm_response'].notna() & (results_df['llm_response'] != '')
    parsed = results_df.loc[mask, 'llm_response'].apply(parse_response)
    results_df["support_score"] = None
    results_df["stance"] = None
    results_df.loc[mask, 'support_score'] = parsed.apply(lambda x: x['support_score'])
    results_df.loc[mask, 'stance'] = parsed.apply(lambda x: x['stance'])
    print(f"      调用完成")

    # 5/6. 保存到数据库
    print(f"\n[{step+1}/5] 保存结果到数据库...")
    db = create_database_adapter()
    query_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.save(query_id, QUESTION, results_df)
    db.close()
    print(f"      保存完成，Query ID: {query_id}")

    # 7. 输出统计（仅成功请求）
    valid_df = results_df[results_df["stance"].notna()]
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"总请求数: {len(results_df)}")
    print(f"成功回答数: {len(valid_df)}")
    print(f"失败数: {len(results_df) - len(valid_df)}")
    if len(valid_df) > 0:
        print(f"平均支持度: {valid_df['support_score'].mean():.2f}")

        # 立场分布
        stance_counts = valid_df["stance"].value_counts()
        print("\n立场分布:")
        stance_labels = {"oppose": "反对", "neutral": "中立", "support": "支持"}
        for stance, count in stance_counts.items():
            label = stance_labels.get(stance, stance)
            print(f"  {label}: {count} ({count/len(valid_df)*100:.1f}%)")

        print("\n按职业分组:")
        if "occupation" in valid_df.columns:
            prof_stats = valid_df.groupby("occupation")["support_score"].mean().sort_values(ascending=False)
            for prof, score in prof_stats.head(10).items():
                print(f"  {prof}: {score:.2f}")

        print("\n按省份分组(Top 10):")
        if "departement" in valid_df.columns:
            dept_stats = valid_df.groupby("departement")["support_score"].mean().sort_values(ascending=False)
            for dept, score in dept_stats.head(10).items():
                print(f"  {dept}: {score:.2f}")
    else:
        print("无有效回答，所有请求均失败")

    print("\n" + "=" * 60)
    print("流程完成！")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="法国政策模拟器 - 完整流程")
    parser.add_argument("--mode", choices=["similarity", "random"], default="similarity",
                        help="采样方式: similarity=FAISS语义检索(默认), random=随机抽样")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (仅 random 模式, 默认42)")
    args = parser.parse_args()
    main(mode=args.mode, seed=args.seed)