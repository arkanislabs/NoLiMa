"""
OpenRouter embedding client with an SQLite cache, used by the RAG retriever (``rag.py``).

Vectors are cached per (model, endpoint, provider pin, text), so a re-run or another ``top_k`` pays nothing.
Rows are committed after each API request, so an interrupted run keeps what it paid for.
"""

import hashlib
import json
import os
import re
import sqlite3
from typing import Optional

import numpy as np
from openai import AsyncOpenAI

from env_utils import expand_env_vars, load_env, redact_secrets

DEFAULT_EMBEDDING_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "embedding_cache")
EMBEDDING_BATCH_SIZE = 64


def embedding_key(model: str, api_url: str, openrouter_provider: Optional[dict], text: str) -> str:
    """SHA-256 of everything that determines the vector the API returns."""
    payload = json.dumps(
        {"model": model, "api_url": api_url.rstrip("/"), "openrouter_provider": openrouter_provider, "text": text},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """SQLite store of float32 vectors by key. Each call opens its own connection."""

    def __init__(self, path: str):
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=60)
        connection.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        return connection

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found = {}
        connection = self._connect()
        try:
            for start in range(0, len(keys), 500):   # stays under SQLite's variable limit
                chunk = keys[start:start + 500]
                query = f"SELECT key, vector FROM embeddings WHERE key IN ({','.join('?' * len(chunk))})"
                for key, blob in connection.execute(query, chunk):
                    found[key] = np.frombuffer(blob, dtype=np.float32)
        finally:
            connection.close()
        return found

    def put_many(self, vectors: dict[str, np.ndarray]) -> None:
        rows = [(key, np.asarray(vector, dtype=np.float32).tobytes()) for key, vector in vectors.items()]
        connection = self._connect()
        try:
            with connection:
                connection.executemany("INSERT OR IGNORE INTO embeddings (key, vector) VALUES (?, ?)", rows)
        finally:
            connection.close()


class Embedder:
    def __init__(self, model: str, api_key: str, api_url: str, cache: Optional[EmbeddingCache] = None,
                 openrouter_provider: Optional[dict] = None, timeout: float = 60, max_retries: int = 3):
        self.model = model
        self.api_url = api_url
        self.cache = cache
        self.openrouter_provider = openrouter_provider
        self.client = AsyncOpenAI(api_key=api_key, base_url=api_url, timeout=timeout, max_retries=max_retries)
        self.logged_config: dict = {}

    async def _send(self, texts: list[str]) -> tuple[np.ndarray, dict]:
        """One API request. Rows follow the input order; returns the vectors and {"tokens", "cost"}."""
        kwargs = {"model": self.model, "input": texts, "encoding_format": "float"}
        if self.openrouter_provider:
            kwargs["extra_body"] = {"provider": self.openrouter_provider}
        response = await self.client.embeddings.create(**kwargs)
        rows = sorted(response.data, key=lambda item: item.index)
        vectors = np.array([row.embedding for row in rows], dtype=np.float32)
        usage = response.usage
        return vectors, {"tokens": usage.prompt_tokens, "cost": getattr(usage, "cost", None)}

    async def embed(self, texts: list[str]) -> tuple[np.ndarray, dict]:
        """
        Embeds texts (cached ones are not sent again). Returns the rows in input order and
        {"texts_sent", "cache_hits", "tokens", "cost"}, counted over distinct texts. `cost` is 0.0 when
        nothing was sent and None if any request sent did not report one.
        """
        unique = list(dict.fromkeys(texts))
        keys = {text: embedding_key(self.model, self.api_url, self.openrouter_provider, text) for text in unique}
        vectors = self.cache.get_many(list(keys.values())) if self.cache else {}
        by_text = {text: vectors[key] for text, key in keys.items() if key in vectors}
        missing = [text for text in unique if text not in by_text]

        tokens, cost = 0, 0.0
        for start in range(0, len(missing), EMBEDDING_BATCH_SIZE):
            batch = missing[start:start + EMBEDDING_BATCH_SIZE]
            batch_vectors, usage = await self._send(batch)
            for text, vector in zip(batch, batch_vectors):
                by_text[text] = vector
            if self.cache:
                self.cache.put_many({keys[text]: vector for text, vector in zip(batch, batch_vectors)})
            tokens += usage["tokens"]
            cost = None if cost is None or usage["cost"] is None else cost + usage["cost"]

        usage = {"texts_sent": len(missing), "cache_hits": len(unique) - len(missing), "tokens": tokens, "cost": cost}
        if not texts:
            return np.zeros((0, 0), dtype=np.float32), usage
        return np.stack([by_text[text] for text in texts]).astype(np.float32, copy=False), usage


def load_embedder(config_name: str, model_configs_dir: str, cache_dir: str = DEFAULT_EMBEDDING_CACHE_DIR) -> Embedder:
    """Builds an Embedder from `<model_configs_dir>/<config_name>.json`, caching in `<cache_dir>/<model slug>.sqlite`."""
    path = os.path.join(model_configs_dir, config_name + ".json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding configuration {config_name} not found in {model_configs_dir}")
    load_env()
    with open(path, "r") as file:
        raw_config = json.load(file)
    config = expand_env_vars(raw_config, source=path)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", config["model"])
    embedder = Embedder(cache=EmbeddingCache(os.path.join(cache_dir, slug + ".sqlite")), **config)
    embedder.logged_config = redact_secrets(raw_config)
    return embedder
