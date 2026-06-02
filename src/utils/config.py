"""
配置文件
"""

import os
from dotenv import load_dotenv

# 尝试加载项目目录和外层的 .env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
load_dotenv(env_path)
load_dotenv()  # 再尝试默认路径


class Config:
    """全局配置"""

    # 数据路径 - 可在 .env 中修改本地路径
    # 示例: DATA_PATH=D:/your_local_path/df_full.parquet
    DATA_PATH = os.getenv("DATA_PATH", "data/processed/df_full.parquet")
    INDEX_PATH = os.getenv("INDEX_PATH", "data/index.faiss")

    # LLM 配置
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V4-Flash")

    # 数据集配置
    DATASET_NAME = os.getenv("DATASET_NAME", "your_dataset_name_here")

    # 检索配置
    DEFAULT_K = 10000

    # 数据库配置 - sqlite 或 sqlserver
    DB_TYPE = os.getenv("DB_TYPE", "sqlite")
    DB_PATH = os.getenv("DB_PATH", "data/results.db")

    # SQL Server 配置（仅当 DB_TYPE=sqlserver 时使用）
    SQL_SERVER = os.getenv("SQL_SERVER", "localhost")
    SQL_DATABASE = os.getenv("SQL_DATABASE", "policy_simulator")
    SQL_USER = os.getenv("SQL_USER", "")
    SQL_PASSWORD = os.getenv("SQL_PASSWORD", "")
    SQL_DRIVER = os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server")