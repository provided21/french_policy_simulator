"""
构建索引脚本

功能：读取本地 parquet -> 生成 Embedding -> 构建 FAISS 索引 -> 保存
运行：python scripts/02_build_index.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever.build_index import build_embeddings, build_faiss_index, build_faiss_index_ivf, save_index
from src.utils import Config


# ============================================
# 配置区域
# ============================================

USE_IVF = True
BATCH_SIZE = 512
SAMPLE_SIZE = 100000  # 采样数量，None=全部
PARQUET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed", "df_full.parquet"
)

# ============================================


def main():
    print("=" * 60)
    print("法国人物画像 - 构建检索索引")
    print("=" * 60)

    # 1. 加载 parquet 数据
    import pandas as pd
    print("\n[1/4] 加载数据...")
    print(f"      数据源: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"      加载完成：{len(df):,} 行")

    # 2. 先采样（减少内存占用）
    if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
        print(f"\n[1.5/4] 采样 {SAMPLE_SIZE:,} 条...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"      采样后：{len(df):,} 行")

    # 3. 构造 persona_id 和 persona_text
    print("\n[2/4] 构造检索字段...")
    df["persona_id"] = df["uuid"].astype(str)
    # 分块构造 persona_text 避免内存溢出
    print("      正在构造 persona_text...")
    chunk_size = 20000
    texts = []
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start+chunk_size]
        arr = chunk.fillna("").astype(str).values
        texts.extend([" | ".join(row) for row in arr])
        del arr
    df["persona_text"] = texts
    print(f"      构造完成")

    # 4. 生成 Embedding
    print("\n[3/4] 生成 Embedding 向量...")
    texts = df["persona_text"].tolist()
    del df["persona_text"]
    embeddings, model = build_embeddings(texts, batch_size=BATCH_SIZE)
    del texts, model
    print(f"      生成完成：向量形状 {embeddings.shape}")

    # 5. 构建 FAISS 索引
    print("\n[4/4] 构建 FAISS 索引...")
    if USE_IVF and len(embeddings) > 10000:
        index = build_faiss_index_ivf(embeddings)
        print("      使用 IVF 索引")
    else:
        index = build_faiss_index(embeddings)
        print("      使用 Flat 索引")

    save_index(index, Config.INDEX_PATH)
    print(f"      索引保存完成: {Config.INDEX_PATH}")

    print("\n" + "=" * 60)
    print("索引构建完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
