"""
On-disk cache of LLM requests, so no identical request is sent to an API twice.

A request is described by a *request record*: everything that decides the model's
answer (provider, endpoint, model and the exact arguments of the SDK call, including
the full prompt), but never API keys, timeouts, retry or tokenizer settings. Its
cache key is the SHA-256 of the record's canonical JSON.

Each request is stored as one JSON file, independent of the run results:

    <cache_dir>/<model slug>/<key[:2]>/<key>.json
    {"key": ..., "created_at": ..., "request": <record, long texts digested>, "response": {...}}

Strings longer than INLINE_TEXT_LIMIT characters (the haystack prompt) are stored as
{"sha256", "chars", "head", "tail"}; the key is still computed over the full text.
"""

import asyncio
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

EVALUATION_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REQUEST_CACHE_DIR = os.path.join(EVALUATION_DIR, "results", "request_cache")

CACHE_MARKER = "REQUEST_CACHE.md"   # written at the cache root; gather_results.py skips directories containing it
INLINE_TEXT_LIMIT = 2000            # longer strings are stored as a digest, not verbatim
PREVIEW_CHARS = 200                 # head and tail kept in a digest so an entry stays recognizable

_MARKER_TEXT = """# LLM request cache

One JSON file per LLM request: `<model>/<key[:2]>/<key>.json`, where `key` is the SHA-256
of the request record (provider, endpoint, model, exact call arguments including the prompt).
Each file holds the request record (texts over 2,000 characters stored as a SHA-256 digest
with a head and tail preview) and the response, including the cost reported when it was sent.

Requests found here are not sent again. Deleting an entry, or a model folder, makes those
requests run (and be paid for) again. `gather_results.py` skips this directory.
"""


def request_key(request: dict) -> str:
    """SHA-256 hex of the request's canonical JSON (key order and whitespace don't matter)."""
    canonical = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compact_request(value: Any) -> Any:
    """Copy of a request record with strings over INLINE_TEXT_LIMIT characters replaced by a digest."""
    if isinstance(value, dict):
        return {k: compact_request(v) for k, v in value.items()}
    if isinstance(value, list):
        return [compact_request(v) for v in value]
    if isinstance(value, str) and len(value) > INLINE_TEXT_LIMIT:
        return {
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            "chars": len(value),
            "head": value[:PREVIEW_CHARS],
            "tail": value[-PREVIEW_CHARS:],
        }
    return value


def should_cache(output: dict) -> bool:
    """Responses without a finish reason, or with finish_reason "error", are not cached so a re-run retries them."""
    finish_reason = output.get("finish_reason")
    return finish_reason is not None and str(finish_reason).lower() != "error"


def is_request_cache_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, CACHE_MARKER))


def write_json_atomic(path: str, data: Any, **dump_kwargs) -> None:
    """
    Writes JSON to a temp file next to `path`, then swaps it in with os.replace, so a crash or a
    serialization error never leaves a half-written file and keeps any previous version intact.
    """
    tmp_path = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, **dump_kwargs)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class RequestCache:
    def __init__(self, cache_dir: str = DEFAULT_REQUEST_CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        marker = os.path.join(cache_dir, CACHE_MARKER)
        if not os.path.exists(marker):
            with open(marker, "w") as f:
                f.write(_MARKER_TEXT)
        self._inflight: Dict[str, asyncio.Future] = {}

    def path_for(self, request: dict) -> str:
        key = request_key(request)
        model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(request.get("model", "unknown")))
        return os.path.join(self.cache_dir, model_slug, key[:2], f"{key}.json")

    def get(self, request: dict) -> Optional[dict]:
        """The stored response, or None when there is no entry or it can't be read."""
        path = self.path_for(request)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)["response"]
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"Warning: ignoring unreadable request cache entry {path}: {e!r}")
            return None

    def put(self, request: dict, response: dict) -> str:
        """Writes the entry atomically (temp file + os.replace) and returns its path."""
        path = self.path_for(request)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        entry = {
            "key": request_key(request),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "request": compact_request(request),
            "response": response,
        }
        write_json_atomic(path, entry, indent=1, ensure_ascii=False)
        return path

    async def fetch(self, request: dict, send: Callable[[dict], Awaitable[dict]]) -> Tuple[dict, bool]:
        """
        Returns (response, cache_hit). A stored response, or one already being fetched for an
        identical request by this cache, counts as a hit; otherwise `send` is called and its
        response stored (when should_cache allows).
        """
        cached = self.get(request)
        if cached is not None:
            return cached, True

        key = request_key(request)
        task = self._inflight.get(key)
        if task is not None:
            return await task, True

        async def send_and_store() -> dict:
            response = await send(request)
            if should_cache(response):
                self.put(request, response)
            return response

        task = asyncio.ensure_future(send_and_store())
        self._inflight[key] = task
        task.add_done_callback(lambda _: self._inflight.pop(key, None))
        return await task, False


# ---------------------------------------------------------------- cost accounting

def current_cost(result: dict) -> Optional[float]:
    """
    Cost of a per-request result in the run that produced it: 0 when served from the cache, else what
    the API reported. None when the provider reports no cost, so an unknown spend never shows as $0.
    """
    cost = result.get("cost")
    if result.get("request_cache_hit") and cost is not None:
        return 0.0
    return cost


def cost_summary(results: list) -> str:
    """One line: the cost of the requests sent now (current) and of all requests including cached ones (total)."""
    def total_of(values: list) -> str:
        present = [v for v in values if v is not None]
        return f"${sum(present):.4f}" if present else "n/a"

    n_cached = sum(bool(result.get("request_cache_hit")) for result in results)
    return (f"Cost: current {total_of([current_cost(r) for r in results])}, "
            f"total {total_of([r.get('cost') for r in results])} "
            f"(requests: {len(results) - n_cached} sent, {n_cached} from the request cache)")
