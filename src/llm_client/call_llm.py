"""
LLM 调用模块 - 使用 OpenAI 兼容接口（SiliconFlow）

优化：
1. 滑动窗口限流（RPM/TPM 双维度控制，避免 429 限速错误）
2. 响应缓存（相同 prompt 直接命中，避免重复调用）
3. 纯 asyncio 异步 I/O（替代 ThreadPoolExecutor，避免 GIL 开销）
4. 限速错误不重试（直接返回，不浪费配额）
5. SQLite 单连接复用（避免 asyncio 下反复打开连接的 I/O 瓶颈）
"""

import asyncio
import hashlib
import os
import sqlite3
import threading
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import List, Dict, Any
from openai import AsyncOpenAI, OpenAI, APIError

CACHE_VERSION = 2  # 版本号，变更后旧缓存自动失效


def is_rate_limit_error(error) -> bool:
    """检测是否为限速错误（HTTP 429 或限速关键词）"""
    rate_limit_keywords = [
        "rate limit", "too many requests", "quota exceeded",
        "tpm已达上限", "rpm已达上限", "overloaded", "429",
        "RateLimit", "rate_limit", "insufficient_quota",
    ]
    if hasattr(error, "status_code") and getattr(error, "status_code", None) == 429:
        return True
    error_msg = str(error).lower()
    return any(kw.lower() in error_msg for kw in rate_limit_keywords)


class RateLimiter:
    """滑动窗口限流器，同时控制 RPM 和 TPM

    每 10 秒为一个窗口，检查过去 60 秒内的请求数和 token 数。
    如果任一维度超限，则等待窗口刷新。
    """

    def __init__(self, rpm: int, tpm: int):
        self.rpm = rpm
        self.tpm = tpm
        # 记录最近 60 秒的请求：[(timestamp, tokens_used), ...]
        self._requests: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()
        self._wait_interval = 0.5  # 等待间隔（秒）

    async def wait_for_slot(self, estimated_tokens: int = 700):
        """阻塞等待，直到有可用配额"""
        while True:
            async with self._lock:
                now = time.time()
                cutoff = now - 60.0
                # 清理 60 秒前的记录
                self._requests = [(t, tok) for t, tok in self._requests if t > cutoff]
                # 计算当前窗口内的使用量
                current_rpm = len(self._requests)
                current_tpm = sum(tok for _, tok in self._requests)
                # 检查是否超限
                if current_rpm < self.rpm and (current_tpm + estimated_tokens) <= self.tpm:
                    return  # 有配额
            await asyncio.sleep(self._wait_interval)

    async def record(self, tokens_used: int):
        """记录一次请求的实际用量"""
        async with self._lock:
            self._requests.append((time.time(), tokens_used))


class LLMCache:
    """基于 SQLite 的 LLM 响应缓存（单连接复用 + 线程锁）"""

    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, "llm_cache.db")
        self._conn = None
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  prompt_hash TEXT PRIMARY KEY,"
            "  model TEXT NOT NULL,"
            "  version INTEGER NOT NULL DEFAULT 0,"
            "  response TEXT NOT NULL"
            ")"
        )
        try:
            conn.execute("ALTER TABLE cache ADD COLUMN version INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass
        conn.commit()
        conn.close()

    def get(self, prompt: str, model: str) -> str | None:
        h = hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()
        with self._lock:
            row = self._get_conn().execute(
                "SELECT response FROM cache WHERE prompt_hash = ? AND model = ? AND version = ?",
                (h, model, CACHE_VERSION),
            ).fetchone()
        if row:
            resp = row[0]
            try:
                resp.encode('utf-8')
                return resp
            except (UnicodeEncodeError, UnicodeDecodeError):
                return None
        return None

    def put(self, prompt: str, model: str, response: str):
        h = hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()
        with self._lock:
            self._get_conn().execute(
                "INSERT OR REPLACE INTO cache (prompt_hash, model, version, response) VALUES (?, ?, ?, ?)",
                (h, model, CACHE_VERSION, response),
            )
            self._get_conn().commit()

    def stats(self) -> dict:
        with self._lock:
            count = self._get_conn().execute(
                "SELECT COUNT(*) FROM cache WHERE version = ?", (CACHE_VERSION,)
            ).fetchone()[0]
        size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        return {"count": count, "size_bytes": size}


class LLMClient:
    """LLM 客户端"""

    def __init__(self, api_key, model="Qwen/Qwen2.5-7B-Instruct",
                 base_url="https://api.siliconflow.cn/v1", rpm=3000, tpm=500000, temperature=0.7):
        """
        初始化 LLM 客户端

        参数:
            api_key: API 密钥
            model: 模型名称
            base_url: API 基础地址
            rpm: 每分钟最大请求数
            tpm: 每分钟最大 Token 数
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

        # 根据 RPM 推导并发数：平均 LLM 延迟 ~3 秒，需要 rpm * 3/60 个并发 worker
        estimated_tokens_per_req = 700
        tpm_limited_rpm = tpm // estimated_tokens_per_req
        effective_rpm = min(rpm, tpm_limited_rpm)
        self.max_concurrent = max(20, min(300, effective_rpm * 3 // 60))

        self.cache = LLMCache()
        self.rate_limiter = RateLimiter(rpm, tpm)

        # 同步和异步客户端
        timeout_config = {"timeout": 30.0}
        self.client = OpenAI(api_key=api_key, base_url=base_url, **timeout_config)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, **timeout_config)

    async def _call_single_llm(self, persona_id: str, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """调用单个 LLM（带重试 + 缓存，限速错误不重试）"""
        # 先查缓存（单连接复用，asyncio 下不再被 SQLite I/O 阻塞）
        cached = self.cache.get(prompt, self.model)
        if cached is not None:
            return {
                "persona_id": persona_id,
                "success": True,
                "response": cached,
                "cached": True,
                "tokens_used": 0,
            }
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=self.temperature,
                )
                result = response.choices[0].message.content.strip()
                tokens_used = 0
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = getattr(response.usage, 'total_tokens', 0)
                self.cache.put(prompt, self.model, result)
                return {
                    "persona_id": persona_id,
                    "success": True,
                    "response": result,
                    "cached": False,
                    "tokens_used": tokens_used,
                }
            except Exception as e:
                last_error = e
                if is_rate_limit_error(e):
                    return {
                        "persona_id": persona_id,
                        "success": False,
                        "error": str(e),
                        "error_type": "rate_limit",
                        "cached": False,
                        "tokens_used": 0,
                    }
                if attempt < max_retries - 1:
                    wait = min(2 ** attempt, 10)
                    await asyncio.sleep(wait)
        return {
            "persona_id": persona_id,
            "success": False,
            "error": str(last_error),
            "cached": False,
            "tokens_used": 0,
        }

    async def _call_with_counter(self, prompts, counter):
        """异步批量调用，滑动窗口限流 + 并发控制"""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def limited_call(idx, prompt_data):
            async with semaphore:
                # 等待限流窗口有可用配额
                await self.rate_limiter.wait_for_slot()
                result = await self._call_single_llm(
                    prompt_data["persona_id"], prompt_data["prompt"])
                # 记录实际 token 用量
                if result.get("success") and result.get("tokens_used", 0) > 0:
                    await self.rate_limiter.record(result["tokens_used"])
                return idx, result

        coros = [limited_call(i, p) for i, p in enumerate(prompts)]
        total = len(prompts)
        for completed in asyncio.as_completed(coros):
            idx, result = await completed
            counter["results"][idx] = result
            with counter["lock"]:
                counter["done"] += 1

    def call_batch_sync(self, prompts, progress_callback=None):
        """同步包装的批量调用（专用线程 + asyncio 事件循环）

        主线程轮询进度计数器，确保回调在 Streamlit 主线程上下文中执行。
        """
        results = [None] * len(prompts)
        error_holder = [None]
        counter = {
            "done": 0,
            "total": len(prompts),
            "results": results,
            "lock": threading.Lock(),
        }

        def _run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self._call_with_counter(prompts, counter))
            except Exception as e:
                error_holder[0] = e
            finally:
                loop.close()

        thread = threading.Thread(target=_run_async, daemon=True)
        thread.start()

        # 主线程轮询进度，回调在正确上下文中执行
        while thread.is_alive():
            with counter["lock"]:
                done = counter["done"]
            if progress_callback:
                progress_callback(done, counter["total"])
            time.sleep(0.3)

        # 确保最后一次进度更新
        with counter["lock"]:
            done = counter["done"]
        if progress_callback:
            progress_callback(done, counter["total"])

        thread.join()

        # 如果有线程异常，把 None 填充为错误
        if error_holder[0]:
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = {
                        "persona_id": prompts[i]["persona_id"],
                        "success": False,
                        "error": str(error_holder[0]),
                        "cached": False,
                    }

        # 处理 None 结果（未完成的任务）
        for i in range(len(results)):
            if results[i] is None:
                results[i] = {
                    "persona_id": prompts[i]["persona_id"],
                    "success": False,
                    "error": "请求未完成",
                    "cached": False,
                }

        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def call_single_with_id(self, persona_id: str, prompt: str) -> Dict[str, Any]:
        """带 persona_id 的单次调用（带重试）"""
        cached = self.cache.get(prompt, self.model)
        if cached is not None:
            return {
                "persona_id": persona_id,
                "success": True,
                "response": cached,
            }
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=self.temperature,
            )
            result = response.choices[0].message.content.strip()
            self.cache.put(prompt, self.model, result)
            return {
                "persona_id": persona_id,
                "success": True,
                "response": result,
            }
        except Exception as e:
            return {
                "persona_id": persona_id,
                "success": False,
                "error": str(e),
            }

    def call_single(self, prompt: str) -> Dict[str, Any]:
        """单次同步调用（简单场景）"""
        cached = self.cache.get(prompt, self.model)
        if cached is not None:
            return {"success": True, "response": cached}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=self.temperature,
            )
            result = response.choices[0].message.content.strip()
            self.cache.put(prompt, self.model, result)
            return {
                "success": True,
                "response": result,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
