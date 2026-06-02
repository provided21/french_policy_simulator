"""
检索模块
"""

from .db_adapter import DatabaseAdapter
from .sqlite_adapter import SQLiteAdapter
from .sqlserver_adapter import SQLServerAdapter
from .db_factory import create_database_adapter
from .save_results import ResultDatabase

# build_index and search require faiss/sentence-transformers — lazy import
# Use: from src.retriever import load_index, search_similar

