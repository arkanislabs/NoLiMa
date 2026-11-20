# LLM Request Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache every LLM request on disk in its own directory, separate from the run results, so no identical request is sent to an API twice. Report the current cost (requests actually sent) and the total cost (including requests answered from the cache).

**Architecture:** A new module `evaluation/request_cache.py` stores one JSON file per request, keyed by the SHA-256 of a canonical *request record*. The record holds everything that decides the model's answer: provider, endpoint, model and the exact SDK call arguments, including the full prompt. `APIConnector.generate_response` builds this record, asks the cache, and only calls the provider on a miss. Each per-request result gains `request_hash` and `request_cache_hit`. `gather_results.py` uses `request_cache_hit` to split the cost into current and total, and skips the cache directory.

**Tech Stack:** Python 3.12, uv, stdlib only (`hashlib`, `json`, `asyncio`, `os`), pytest (new `dev` dependency group).

**Spec:** No separate spec file. The request, verbatim (2026-11-20):

> plan adding a real cache of llm requests locally so no identical request is run twice. the cost should report current cost (without cache) and total cost (cached reqs included). Current implementation of "caching" requests is strongly coupled with the run results outputs which are used to generate reports. it should store cache in a sepearete indepdnent subdir of the results. please store in cache all params relevant to the call, including the prompt. if any textual prompt gets too long, consider hashing it.

**Prerequisite:** `evaluation/gather_results.py` has an uncommitted "Cost by model" change (`summarize_models`). Commit it before Task 4, which builds on it.

## Design decisions

- **What exists today:** `NoLiMa_Tester.evaluate` hashes the test config (`test_hash`) and skips a whole test when a result file with the same hash is in its `results_dir` (`prevent_duplicate`). That works only per result file, only while the result files exist, and ignores the model config. **It stays unchanged:** its only job now is to keep duplicate result files out of the reports. Saving money is the request cache's job.
- **Where the cache hooks in:** inside `APIConnector`. Only the connector knows exactly what is sent: the system prompt it resolves, `seed: 43`, `max_completion_tokens` for thinking models, Gemini's `system_instruction`, and AWS sampling taken from the model config.
- **Storage:** one JSON file per request. Files are easy to read, writes are atomic (`os.replace`), there is no new dependency, and one model's cache can be deleted as a folder. A marker file at the cache root lets `gather_results.py` skip the cache wherever it lives.
- **Long prompts:** the key is computed over the *full* request. In the stored record, strings longer than 2,000 characters (the haystack prompt) are replaced by `{"sha256", "chars", "head", "tail"}`. The tail keeps the question, so an entry is still recognizable. At 128K context a full prompt would be about 500 KB per request.
- **Cost:** a cached response keeps the `cost` that was reported when it was first fetched. **Current cost** = Σ `cost` over requests with `request_cache_hit == false`. **Total cost** = Σ `cost` over all requests. Result files from before this change have no `request_cache_hit` and count as sent.
- **Failures:** an exception is never cached. A response whose `finish_reason` is `None` or `"error"` is returned but not cached, so a re-run retries it. `"length"` (budget exhausted) is cached: it is a real answer to that exact request.

## Global Constraints

- Default cache dir: `<repo>/evaluation/results/request_cache/`, computed from `request_cache.py`'s own location (not the cwd). It can be overridden with `request_cache_dir`. `evaluation/results/` is already gitignored.
- Entry path: `<cache_dir>/<model slug>/<key[:2]>/<key>.json`, where model slug = `re.sub(r"[^A-Za-z0-9._-]+", "_", request["model"])`.
- Key: SHA-256 hex of `json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")`, computed over the full (uncompacted) request.
- Entry JSON: `{"key": str, "created_at": UTC ISO-8601 str, "request": compact_request(request), "response": <output dict>}`.
- The request record never contains `api_key`, timeouts, retry settings or tokenizer settings. `api_url` is stored with any trailing `/` stripped. `temperature`/`top_p` are cast to `float` and token limits to `int`, so `0` and `0.0` give the same key.
- Inline text limit: `INLINE_TEXT_LIMIT = 2000` characters. Preview: `PREVIEW_CHARS = 200` for head and tail.
- Marker file at the cache root: `REQUEST_CACHE.md`, a short explanation of the layout. A directory containing it is a cache directory.
- New per-request result fields: `request_hash` (str) and `request_cache_hit` (bool).
- Do not add any field to `outputs` before `get_hash` in `async_evaluate.py`. `test_hash` must stay the same so existing result files are still recognized as duplicates.
- No new runtime dependencies. `pytest` goes only in the `dev` dependency group.
- Commit messages follow the repo style (a plain imperative sentence, no `feat:` prefix) and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- Two identical requests in flight at once (e.g. `static_depth` set and the same `{CHAR}` drawn twice in one batch) → one API call; the second caller gets the same response with `request_cache_hit: true`. Test: Task 1 `test_concurrent_identical_requests_send_once`.
- One request in a batch fails after its retries → every other response in the batch is already on disk, and a re-run pays only for the failed one. Tests: Task 1 `test_failed_request_is_not_cached_and_does_not_block_others`, Task 3 `test_failed_request_keeps_successful_responses_in_cache`.
- Provider answers 200 with `finish_reason: "error"` (or none) → not cached, so a re-run retries it. Test: Task 1 `test_error_finish_reason_is_not_cached`.
- `make gather-results` over `evaluation/results/`, which now contains the cache → cache entries are not loaded as result files. Test: Task 4 `test_load_result_files_skips_request_cache`.
- Result files written before this change → still deduplicated (`test_hash` unchanged) and counted as current cost. Tests: Task 3 `test_second_run_is_served_from_cache` (top-level keys unchanged), Task 4 `test_summarized_result_splits_current_and_total_cost`.

---

### Task 1: Request cache module and test setup

**Files:**
- Create: `evaluation/request_cache.py`
- Create: `tests/test_request_cache.py`
- Modify: `pyproject.toml` (add `dev` group, pytest config), `Makefile` (add `make test`), `uv.lock` (regenerated)

**Interfaces:**
- Produces (in `request_cache.py`):
  - `DEFAULT_REQUEST_CACHE_DIR: str`, `CACHE_MARKER = "REQUEST_CACHE.md"`, `INLINE_TEXT_LIMIT = 2000`, `PREVIEW_CHARS = 200`
  - `request_key(request: dict) -> str`
  - `compact_request(value: Any) -> Any`
  - `should_cache(output: dict) -> bool`
  - `is_request_cache_dir(path: str) -> bool`
  - `class RequestCache`:
    - `__init__(self, cache_dir: str = DEFAULT_REQUEST_CACHE_DIR) -> None`: creates the dir and writes the marker if missing
    - `path_for(self, request: dict) -> str`
    - `get(self, request: dict) -> Optional[dict]`: the stored response, or `None` on a miss or an unreadable entry (prints a warning for unreadable)
    - `put(self, request: dict, response: dict) -> str`: atomic write, returns the path
    - `async fetch(self, request: dict, send: Callable[[dict], Awaitable[dict]]) -> Tuple[dict, bool]`: `(response, cache_hit)`

- [ ] **Step 1: Add pytest**

`pyproject.toml`: add `dev = ["pytest>=8.3"]` under `[dependency-groups]` (uv syncs `dev` by default), and:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["evaluation", "."]
```

`Makefile`: add `test` to `.PHONY`, a help line `make test            Run the unit tests`, and a target `test: check-uv` running `uv run pytest`.

Run: `uv lock && uv sync && uv run pytest --version`
Expected: `pytest 8.x`.

- [ ] **Step 2: Write the failing tests** in `tests/test_request_cache.py`

```python
import asyncio, hashlib, json, os
from request_cache import (RequestCache, request_key, compact_request, should_cache, is_request_cache_dir,
                           CACHE_MARKER, INLINE_TEXT_LIMIT, PREVIEW_CHARS)

REQ = {"api_provider": "vllm", "api_url": "https://openrouter.ai/api/v1", "model": "deepseek/deepseek-v4.1-flash-1017",
       "messages": [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Q?"}],
       "seed": 43, "max_tokens": 4096, "temperature": 0.0, "top_p": 1.0}
OUT = {"response": "Veronica", "finish_reason": "stop", "cost": 6.3676e-05, "prompt_tokens": 361}

def counting_send(output=OUT):
    calls = []
    async def send(request):
        calls.append(request)
        await asyncio.sleep(0)
        return dict(output)
    return send, calls

def test_request_key_is_order_independent_and_param_sensitive():
    assert request_key(REQ) == request_key(dict(reversed(list(REQ.items()))))
    assert request_key(REQ) != request_key({**REQ, "max_tokens": 4095})
    assert len(request_key(REQ)) == 64

def test_compact_request_keeps_short_text_and_digests_long_text():
    long = "a" * PREVIEW_CHARS + "x" * INLINE_TEXT_LIMIT + "z" * PREVIEW_CHARS
    compact = compact_request({"short": "Q?", "messages": [{"content": long}], "n": 1})
    assert compact["short"] == "Q?" and compact["n"] == 1
    assert compact["messages"][0]["content"] == {"sha256": hashlib.sha256(long.encode()).hexdigest(), "chars": len(long),
                                                 "head": "a" * PREVIEW_CHARS, "tail": "z" * PREVIEW_CHARS}

def test_miss_then_hit_writes_one_entry(tmp_path):
    cache = RequestCache(str(tmp_path / "cache"))
    send, calls = counting_send()
    assert asyncio.run(cache.fetch(REQ, send)) == (OUT, False)
    assert asyncio.run(cache.fetch(REQ, send)) == (OUT, True)
    assert len(calls) == 1
    key = request_key(REQ)
    path = tmp_path / "cache" / "deepseek_deepseek-v4.1-flash-1017" / key[:2] / f"{key}.json"
    assert cache.path_for(REQ) == str(path)
    assert os.listdir(path.parent) == [path.name]          # atomic write leaves no temp files
    entry = json.loads(path.read_text())
    assert entry["key"] == key and entry["request"] == compact_request(REQ) and entry["response"] == OUT
    assert "created_at" in entry

def test_entries_survive_a_new_instance(tmp_path):
    RequestCache(str(tmp_path)).put(REQ, OUT)
    send, calls = counting_send()
    assert asyncio.run(RequestCache(str(tmp_path)).fetch(REQ, send)) == (OUT, True)
    assert calls == []

def test_concurrent_identical_requests_send_once(tmp_path):
    cache = RequestCache(str(tmp_path))
    send, calls = counting_send()
    async def both():
        return await asyncio.gather(cache.fetch(REQ, send), cache.fetch(REQ, send))
    results = asyncio.run(both())
    assert len(calls) == 1
    assert [r for r, _ in results] == [OUT, OUT]
    assert sorted(hit for _, hit in results) == [False, True]

def test_failed_request_is_not_cached_and_does_not_block_others(tmp_path):
    cache = RequestCache(str(tmp_path))
    other = {**REQ, "seed": 44}
    async def send(request):
        if request["seed"] == 44:
            raise RuntimeError("boom")
        return dict(OUT)
    async def both():
        return await asyncio.gather(cache.fetch(REQ, send), cache.fetch(other, send), return_exceptions=True)
    ok, err = asyncio.run(both())
    assert ok == (OUT, False) and isinstance(err, RuntimeError)
    assert cache.get(REQ) == OUT and cache.get(other) is None

def test_error_finish_reason_is_not_cached(tmp_path):
    cache = RequestCache(str(tmp_path))
    send, calls = counting_send({**OUT, "finish_reason": "error"})
    asyncio.run(cache.fetch(REQ, send))
    asyncio.run(cache.fetch(REQ, send))
    assert len(calls) == 2 and cache.get(REQ) is None

def test_should_cache():
    assert should_cache({"finish_reason": "stop"}) and should_cache({"finish_reason": "length"})
    assert not should_cache({"finish_reason": "error"}) and not should_cache({"finish_reason": None}) and not should_cache({})

def test_unreadable_entry_is_a_miss_and_gets_overwritten(tmp_path):
    cache = RequestCache(str(tmp_path))
    cache.put(REQ, OUT)
    with open(cache.path_for(REQ), "w") as f:
        f.write("{not json")
    send, calls = counting_send()
    assert asyncio.run(cache.fetch(REQ, send)) == (OUT, False)
    assert cache.get(REQ) == OUT

def test_marker_identifies_the_cache_dir(tmp_path):
    RequestCache(str(tmp_path / "c"))
    assert (tmp_path / "c" / CACHE_MARKER).is_file()
    assert is_request_cache_dir(str(tmp_path / "c")) and not is_request_cache_dir(str(tmp_path))
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_request_cache.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'request_cache'`.

- [ ] **Step 4: Implement `evaluation/request_cache.py`**

Use the signatures above and the Global Constraints values. Notes on the parts the tests leave open:
- `put` writes to `<path>.tmp-<pid>-<uuid4 hex>` in the same directory, then calls `os.replace`.
- `fetch` keeps `self._inflight: Dict[str, asyncio.Future]`. On a disk hit it returns `(stored, True)`. If the key is already in `_inflight`, it awaits that task and returns `(response, True)`. Otherwise it wraps "send, then `put` if `should_cache`" in `asyncio.ensure_future`, registers it, removes it with `add_done_callback`, awaits it and returns `(response, False)`. Because every caller awaits the same task, an exception reaches all of them and asyncio logs no "never retrieved" warning.
- `CACHE_MARKER` content: 5–10 lines covering what the directory is, the entry layout, that deleting an entry or a model folder makes those requests run (and be paid for) again, and that `gather_results.py` skips this directory.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `make test`
Expected: 10 passed.

- [ ] **Step 6: Commit**

```bash
git add evaluation/request_cache.py tests/test_request_cache.py pyproject.toml uv.lock Makefile
git commit -m "Add an on-disk LLM request cache keyed by the full request"
```

---

### Task 2: Route `APIConnector` requests through the cache

**Files:**
- Modify: `evaluation/async_api_connector.py` (`APIConnector.__init__`, `generate_response`; new `build_messages`, `build_request`, `APIConnector._send`)
- Create: `tests/conftest.py`, `tests/test_api_connector.py`

**Interfaces:**
- Consumes: `RequestCache`, `request_key` (Task 1).
- Produces (in `async_api_connector.py`):
  - `build_messages(system_prompt: str, user_prompt: Union[str, List[str]], default_system_prompt: str, add_default_system_prompt: bool = True) -> list`: the current message-building logic, moved unchanged
  - `build_request(api_provider: str, api_url: str, model: str, model_config: dict, default_system_prompt: str, messages: list, max_tokens: int, temperature: float, top_p: float) -> dict`
  - `APIConnector.__init__(self, api_key, api_url, api_provider, model, cache: Optional[RequestCache] = None, **kwargs)`: stores `self.api_url`, `self.cache`
  - `async APIConnector._send(self, request: dict) -> dict`: the provider call. Returns the same output dict as today
  - `generate_response(...)` keeps its signature. The returned dict gains `"request_hash": request_key(request)` and `"request_cache_hit": bool`
- Produces (in `tests/conftest.py`): fixtures `fake_tiktoken` (patches `async_api_connector.tiktoken.get_encoding` to return a char-level tokenizer: `encode = [ord(c) ...]`, `decode = "".join(chr ...)`) and `make_connector` (a factory `make(cache=None) -> APIConnector` for `api_provider="vllm"`, `api_key="sk-test-secret"`, `api_url="http://localhost:1/v1"`, `model="test/model"`, `tokenizer_type="tiktoken"`, `tokenizer_model="char"`, `max_retries=0`, `timeout=1`).

`build_request` output, by provider (`api_url` has any trailing `/` stripped in all cases):

| Provider | Keys |
|---|---|
| `openai`, `vllm`, `azure-openai` | `api_provider`, `api_url`, `model`, `messages`, `seed: 43`, then either `max_completion_tokens` (when `model_config["openai_thinking_model"]`) or `max_tokens`, `temperature`, `top_p` |
| `gemini` | `api_provider`, `api_url`, `model`, `system_instruction` (= `default_system_prompt`), `contents` (= `messages[-1]["content"]`), `max_output_tokens`, `temperature`, `top_p`, `thinking_budget` (`model_config.get`, may be `None`), `seed: 43` |
| `aws` | `api_provider`, `api_url`, `model`, `messages`, and `max_tokens`, `temperature`, `top_p`, `region` taken from `model_config` (Bedrock is configured at init, so the call arguments are ignored, as today) |
| other | `ValueError(f"Invalid API provider: {api_provider}")` |

- [ ] **Step 1: Write the failing tests** in `tests/test_api_connector.py` (and the fixtures in `tests/conftest.py`)

```python
import asyncio, json
from types import SimpleNamespace
from async_api_connector import build_messages, build_request
from request_cache import RequestCache, request_key

SYS = "You are a helpful assistant"
MSGS = [{"role": "system", "content": SYS}, {"role": "user", "content": "Q?"}]

def test_build_messages_keeps_current_behavior():
    assert build_messages("ignored", "Q?", SYS) == MSGS
    assert build_messages("Custom", "Q?", "") == [{"role": "system", "content": "Custom"}, {"role": "user", "content": "Q?"}]
    assert build_messages("", ["a", "b"], "") == [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]

def test_build_request_openai_compatible():
    r = build_request("vllm", "https://openrouter.ai/api/v1/", "m", {}, SYS, MSGS, 4096, 0, 1)
    assert r == {"api_provider": "vllm", "api_url": "https://openrouter.ai/api/v1", "model": "m", "messages": MSGS,
                 "seed": 43, "max_tokens": 4096, "temperature": 0.0, "top_p": 1.0}
    assert request_key(r) == request_key(build_request("vllm", "https://openrouter.ai/api/v1", "m", {}, SYS, MSGS, 4096, 0.0, 1.0))
    thinking = build_request("openai", "u", "o3", {"openai_thinking_model": True}, SYS, MSGS, 2048, 0.0, 1.0)
    assert thinking["max_completion_tokens"] == 2048 and not {"max_tokens", "temperature", "top_p"} & set(thinking)

def test_build_request_gemini():
    r = build_request("gemini", "EMPTY", "gemini-2.5-flash", {"thinking_budget": 1536}, SYS, MSGS, 1536, 0.0, 1.0)
    assert r == {"api_provider": "gemini", "api_url": "EMPTY", "model": "gemini-2.5-flash", "system_instruction": SYS,
                 "contents": "Q?", "max_output_tokens": 1536, "temperature": 0.0, "top_p": 1.0, "thinking_budget": 1536, "seed": 43}

def test_build_request_aws_uses_model_config_sampling():
    config = {"temperature": 0.0, "max_tokens": 192, "top_p": 1.0, "region": "us-east-1"}
    r = build_request("aws", "EMPTY", "anthropic.claude", config, "", MSGS, 999, 0.5, 0.5)
    assert r == {"api_provider": "aws", "api_url": "EMPTY", "model": "anthropic.claude", "messages": MSGS,
                 "max_tokens": 192, "temperature": 0.0, "top_p": 1.0, "region": "us-east-1"}

def test_build_request_leaves_out_secrets_and_transport_settings():
    config = {"api_key": "sk-secret", "timeout": 200, "max_retries": 3, "tokenizer_model": "t"}
    r = build_request("vllm", "u", "m", config, SYS, MSGS, 10, 0.0, 1.0)
    assert set(r) == {"api_provider", "api_url", "model", "messages", "seed", "max_tokens", "temperature", "top_p"}

def test_send_passes_the_request_to_the_openai_client(make_connector):
    connector = make_connector()
    captured = {}
    async def create(**params):
        captured.update(params)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12, prompt_tokens_details=None,
                                completion_tokens_details=None, cost=0.001)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Veronica"), finish_reason="stop")], usage=usage)
    connector.api = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    request = build_request("vllm", "http://localhost:1/v1", "test/model", {}, SYS, MSGS, 10, 0.0, 1.0)
    out = asyncio.run(connector._send(request))
    assert captured == {k: v for k, v in request.items() if k not in ("api_provider", "api_url")}
    assert out == {"response": "Veronica", "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
                   "finish_reason": "stop", "cached_tokens": None, "reasoning_tokens": None, "cost": 0.001}

def test_generate_response_uses_the_cache(tmp_path, make_connector):
    cache = RequestCache(str(tmp_path))
    connector = make_connector(cache)
    calls = []
    async def fake_send(request):
        calls.append(request)
        return {"response": "Veronica", "finish_reason": "stop", "cost": 0.001}
    connector._send = fake_send
    first = asyncio.run(connector.generate_response("", "Q?", max_tokens=10))
    second = asyncio.run(connector.generate_response("", "Q?", max_tokens=10))
    third = asyncio.run(connector.generate_response("", "Q2?", max_tokens=10))
    assert len(calls) == 2
    assert (first["request_cache_hit"], second["request_cache_hit"], third["request_cache_hit"]) == (False, True, False)
    assert first["request_hash"] == second["request_hash"] == request_key(calls[0]) != third["request_hash"]
    assert second["response"] == "Veronica" and second["cost"] == 0.001
    assert "sk-test-secret" not in open(cache.path_for(calls[0])).read()

def test_generate_response_without_cache_always_sends(make_connector):
    connector = make_connector(None)
    calls = []
    async def fake_send(request):
        calls.append(request)
        return {"response": "x", "finish_reason": "stop"}
    connector._send = fake_send
    outs = [asyncio.run(connector.generate_response("", "Q?", max_tokens=10)) for _ in range(2)]
    assert len(calls) == 2 and [o["request_cache_hit"] for o in outs] == [False, False]
    assert outs[0]["request_hash"] == request_key(calls[0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_api_connector.py -v`
Expected: `ImportError: cannot import name 'build_messages'`.

- [ ] **Step 3: Implement the refactor in `evaluation/async_api_connector.py`**

`generate_response` becomes: `build_messages` → `build_request` → `self.cache.fetch(request, self._send)` when `self.cache` is set, otherwise `await self._send(request)` with `hit = False` → return `{**output, "request_hash": request_key(request), "request_cache_hit": hit}`. `_send` holds the three existing provider branches with their `@retry` decorators. **Move the Gemini and AWS branches without rewriting them.** The diff should show only their parameters now coming from `request`, because no test reaches those two paths. The OpenAI-compatible branch calls `create(**{k: v for k, v in request.items() if k not in ("api_provider", "api_url")})`.

- [ ] **Step 4: Run all tests to verify they pass**

Run: `make test`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add evaluation/async_api_connector.py tests/conftest.py tests/test_api_connector.py
git commit -m "Build an explicit request record per LLM call and serve repeats from the request cache"
```

---

### Task 3: Turn on the cache for runs and print current/total cost

**Files:**
- Modify: `evaluation/request_cache.py` (add the cost helpers)
- Modify: `evaluation/async_evaluate.py` (`NoLiMa_Tester.__init__`, `evaluate`, `__main__` args)
- Modify: `evaluation/run_tests.py` (args, run total)
- Modify: `README.md` (new "Request cache" subsection under the evaluation steps)
- Create: `tests/test_async_evaluate.py`; add the cost-helper tests to `tests/test_request_cache.py`

**Interfaces:**
- Consumes: `RequestCache`, `DEFAULT_REQUEST_CACHE_DIR` (Task 1); `APIConnector(cache=...)`, the `request_cache_hit` / `request_hash` output fields (Task 2); the `fake_tiktoken` fixture (Task 2).
- Produces:
  - `request_cache.current_cost(result: dict) -> Optional[float]`: `0.0` when `result.get("request_cache_hit")`, else `result.get("cost")`
  - `request_cache.cost_summary(results: list) -> str`: format `"Cost: current $0.0031, total $0.0071 (requests: 3 sent, 1 from the request cache)"`. Sums skip `None`; a sum with nothing to add is `n/a`; amounts use `${:.4f}`
  - `NoLiMa_Tester.__init__(..., request_cache_dir: str = DEFAULT_REQUEST_CACHE_DIR, use_request_cache: bool = True)`
  - `NoLiMa_Tester.evaluate() -> list`: the per-request result dicts it wrote, or `[]` when it skipped
  - CLI/YAML options in `run_tests.py` and `async_evaluate.py`: `request_cache_dir` (str, default `DEFAULT_REQUEST_CACHE_DIR`), `use_request_cache` (bool, default `True`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_request_cache.py`:

```python
from request_cache import current_cost, cost_summary

def test_cost_summary_splits_current_and_total():
    results = [{"cost": 0.001, "request_cache_hit": False}, {"cost": 0.002, "request_cache_hit": False},
               {"cost": 0.004, "request_cache_hit": True}, {"cost": 0.0001}]          # last: written before the cache existed
    assert current_cost(results[2]) == 0.0 and current_cost(results[3]) == 0.0001
    assert cost_summary(results) == "Cost: current $0.0031, total $0.0071 (requests: 3 sent, 1 from the request cache)"
    assert cost_summary([{"cost": None, "request_cache_hit": False}]) == "Cost: current n/a, total n/a (requests: 1 sent, 0 from the request cache)"
```

Create `tests/test_async_evaluate.py`. Its fixture `make_tester(results_dir, **overrides)` uses `fake_tiktoken`, sets a fresh event loop (`asyncio.set_event_loop(asyncio.new_event_loop())`, because `evaluate` calls `asyncio.get_event_loop()` and earlier tests ran `asyncio.run`), and writes:
- `tmp_path/model_configs/fake.json`: `{"model": "test/model", "api_key": "sk-test-secret", "api_url": "http://localhost:1/v1", "api_provider": "vllm", "max_tokens": 10, "temperature": 0.0, "top_p": 1.0, "timeout": 1, "max_retries": 0, "tokenizer_type": "tiktoken", "tokenizer_model": "char"}`
- `tmp_path/book.txt`: `"".join(f"Line {i} of the book.\n" for i in range(300))`

It then builds `NoLiMa_Tester(model_name="fake", needle="Actually, {CHAR} lives next to the Kiasma museum.", retrieval_question="Which character has been to Helsinki?", character_set=json.dumps(["Yuki", "Stuart"]), system_prompt="", use_default_system_prompt=True, task_template="{haystack}\n\nQuestion: {question}", context_length=200, document_depth_percent_intervals=3, metric="contains", test_name="t", request_cache_dir=str(tmp_path / "cache"), ...)`. A `sent` fixture patches `APIConnector._send` with `async def fake_send(self, request)`, which records the request and returns `{"response": "Yuki", "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "finish_reason": "stop", "cost": 0.001}`.

```python
RESULT_KEYS = {"eval_name", "test_name", "model_name", "model_config_path", "retrieval_question", "needle", "gold_answers",
               "system_prompt", "use_default_system_prompt", "task_template", "haystack_path", "haystack_hash", "context_length",
               "character_set", "document_depth_percent_min", "document_depth_percent_max", "document_depth_percent_intervals",
               "shift", "static_depth", "metric", "result_dir", "log_placements_dir", "seed", "results", "test_hash", "model_config"}

def test_second_run_is_served_from_cache(make_tester, sent, tmp_path):
    first = make_tester(str(tmp_path / "run1")).evaluate()
    second = make_tester(str(tmp_path / "run2")).evaluate()
    assert len(sent) == 3
    assert [r["request_cache_hit"] for r in first] == [False] * 3
    assert [r["request_cache_hit"] for r in second] == [True] * 3
    assert [r["response"] for r in second] == [r["response"] for r in first]
    assert [r["request_hash"] for r in second] == [r["request_hash"] for r in first]
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 3
    (result_file,) = (tmp_path / "run2").iterdir()
    assert set(json.loads(result_file.read_text())) == RESULT_KEYS        # nothing new feeds test_hash

def test_disabled_cache_sends_everything_and_writes_nothing(make_tester, sent, tmp_path):
    make_tester(str(tmp_path / "run1"), use_request_cache=False).evaluate()
    make_tester(str(tmp_path / "run2"), use_request_cache=False).evaluate()
    assert len(sent) == 6 and not (tmp_path / "cache").exists()

def test_failed_request_keeps_successful_responses_in_cache(make_tester, sent, fail_first_call, tmp_path):
    with pytest.raises(RuntimeError):
        make_tester(str(tmp_path / "run1")).evaluate()
    assert list((tmp_path / "run1").glob("*.json")) == []
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 2
    make_tester(str(tmp_path / "run2")).evaluate()
    assert len(sent) == 4                                                  # 3 first attempt + 1 retry of the failed one
```

`fail_first_call` makes `fake_send` record the request and then raise `RuntimeError`, on its first call only.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_request_cache.py tests/test_async_evaluate.py -v`
Expected: `ImportError: cannot import name 'current_cost'`, and a `TypeError` for the unexpected keyword `request_cache_dir`.

- [ ] **Step 3: Implement**

- `request_cache.py`: add `current_cost` and `cost_summary`.
- `NoLiMa_Tester.__init__`: pass `cache=RequestCache(request_cache_dir) if use_request_cache else None` to `APIConnector`. Leave `outputs` and `get_hash` untouched.
- `evaluate`: change to `asyncio.gather(*async_tasks, return_exceptions=True)`. When every request has finished, raise the first exception if there is one, before scoring, so no result file is written. Every successful response has already been cached by then. After "Results saved at …", print `cost_summary(outputs["results"])`. Return `outputs["results"]`, or `[]` on both skip paths.
- `run_tests.py` and the `async_evaluate.py` `__main__`: add the two arguments and pass them through. `run_tests.py` collects `all_results += tester.evaluate()` and finishes with `print(f"All tests: {cost_summary(all_results)}")`.
- `README.md` "Request cache" subsection: where the cache lives and how to set `request_cache_dir`; the entry layout and what an entry holds (the request record with long texts digested, and the response); that it is shared by all runs; `use_request_cache: false` to bypass it; to redo requests, delete the model folder or one entry, found via the result's `request_hash`; `finish_reason: "error"` responses are not cached; `prevent_duplicate_tests` still skips tests whose result file exists.

- [ ] **Step 4: Run all tests and check the CLI**

Run: `make test && cd evaluation && PYTHONPATH=.. uv run python run_tests.py --help | grep -E -- '--(request_cache_dir|use_request_cache)'`
Expected: 22 passed. Both options are listed.

- [ ] **Step 5: Commit**

```bash
git add evaluation/request_cache.py evaluation/async_evaluate.py evaluation/run_tests.py README.md tests/test_request_cache.py tests/test_async_evaluate.py
git commit -m "Use the request cache in evaluation runs and print current and total cost"
```

---

### Task 4: Current and total cost in `gather_results.py`, skipping the cache

**Files:**
- Modify: `evaluation/gather_results.py` (`load_result_files`, `get_summarized_result`, `summarize_runs`, `summarize_models`, `build_report`, module docstring)
- Modify: `README.md` (the "Gathering the results" bullets)
- Create: `tests/test_gather_results.py`

**Interfaces:**
- Consumes: `is_request_cache_dir`, `current_cost` (Tasks 1 and 3); `summarize_models` (the prerequisite commit).
- Produces:
  - CSV columns `n_request_cache_hits` and `current_cost`, placed before the existing `total_cost`, which keeps its meaning: all requests
  - Report: the header sentence becomes `… {N:,} requests ({H:,} from the request cache), current cost {X}, total cost {Y}.` The Cost by model table header becomes `| Model | Runs | Requests | From cache | Current cost (USD) | Total cost (USD) | Note |`. Each run gets the bullets `- **Requests:** N (H from the request cache)` and `- **Cost:** current {X}, total {Y}`. The per-length table's `Cost` column is replaced by `Current cost | Total cost`.
  - Footnote under Cost by model, exact text: ``**Current cost**: requests sent to the API when the results were produced. **Total cost**: all requests, including those answered from the local request cache, at the price recorded when they were first sent. Costs are what the provider reported per request (e.g. OpenRouter's `usage.cost`).``

- [ ] **Step 1: Write the failing tests** in `tests/test_gather_results.py`

```python
import json, os
import pytest
from gather_results import load_result_files, get_summarized_result, summarize_runs, summarize_models, build_report
from request_cache import RequestCache

def res(cost, hit=None):
    r = {"metric": 1, "prompt_tokens": 10, "completion_tokens": 2, "finish_reason": "stop", "cost": cost}
    if hit is not None:
        r["request_cache_hit"] = hit
    return r

def result_data():
    return {"model_name": "m", "test_name": "t", "context_length": 1000, "haystack_path": "h.txt",
            "result_dir": "./results/run/CL1K/t", "seed": 1, "test_hash": "x",
            "results": [res(0.001, False), res(0.002, True), res(0.004)]}

def test_load_result_files_skips_request_cache(tmp_path):
    (tmp_path / "run" / "CL1K" / "t").mkdir(parents=True)
    (tmp_path / "run" / "CL1K" / "t" / "x.json").write_text(json.dumps(result_data()))
    RequestCache(str(tmp_path / "request_cache")).put({"model": "m", "messages": []}, {"response": "r", "finish_reason": "stop"})
    assert [os.path.basename(p) for p, _ in load_result_files(str(tmp_path))] == ["x.json"]

def test_summarized_result_splits_current_and_total_cost():
    s = get_summarized_result(result_data())
    assert s["n_request_cache_hits"] == 1
    assert s["current_cost"] == pytest.approx(0.005) and s["total_cost"] == pytest.approx(0.007)

def test_report_shows_current_and_total_cost():
    loaded = [("x.json", result_data())]
    report = build_report(summarize_runs(loaded), summarize_models(loaded), ["results"])
    assert "3 requests (1 from the request cache), current cost $0.0050, total cost $0.0070." in report
    assert "| Model | Runs | Requests | From cache | Current cost (USD) | Total cost (USD) | Note |" in report
    assert "| m | `run` | 3 | 1 | $0.0050 | $0.0070 |  |" in report
    assert "- **Requests:** 3 (1 from the request cache)" in report
    assert "- **Cost:** current $0.0050, total $0.0070" in report
    assert "| Cut off | Avg reasoning tokens | Current cost | Total cost |" in report
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gather_results.py -v`
Expected: 3 failed. The cache entry is loaded as a second result file, `KeyError: 'n_request_cache_hits'`, and the report still has the old single cost column.

- [ ] **Step 3: Implement**

In `load_result_files`, prune the walk in place with `dirnames[:] = [d for d in dirnames if not is_request_cache_dir(os.path.join(dirpath, d))]`, and skip the root itself if it is a cache dir. Everywhere a `cost` list is built, build a parallel `current_cost` list with `current_cost(res)` and count `res.get("request_cache_hit", False)`. `sum_or_none` and `format_cost` stay as they are. Update the module docstring to list the two new CSV columns and what current vs. total means. In the README "Gathering the results" bullet, mention current vs. total cost.

- [ ] **Step 4: Run all tests, then the script on the real results**

Run: `make test && make gather-results && grep -E 'current cost|From cache' evaluation/results/report.md`
Expected: 25 passed. The script reports the same number of result files as before (the cache, if one exists by now, is not counted). For the existing smoke results the report shows `0 from the request cache` and current cost = total cost.

- [ ] **Step 5: Commit**

```bash
git add evaluation/gather_results.py README.md tests/test_gather_results.py
git commit -m "Report current and total cost and skip the request cache when gathering results"
```

---

## Final verification (live, paid; ask the user before running)

Each step makes about 6 OpenRouter requests at 250 tokens, well under $0.001.

- [ ] Run: `cd evaluation && PYTHONPATH=.. uv run python -u run_tests.py --config run_config/smoke_test_deepseek_v4.1_flash_1017_250.yaml --parent_results_dir ./results/cache_check_1/CL250/`
  Expected: the last line is `All tests: Cost: current $0.0…, total $0.0… (requests: 6 sent, 0 from the request cache)`, and `evaluation/results/request_cache/deepseek_deepseek-v4.1-flash-1017/` holds 6 entries.
- [ ] Run the same command with `--parent_results_dir ./results/cache_check_2/CL250/`.
  Expected: `All tests: Cost: current $0.0000, total <same as above> (requests: 0 sent, 6 from the request cache)`.
- [ ] Run: `make gather-results`. In the Cost by model row, `From cache` is 6 and current cost is lower than total cost. Delete `results/cache_check_*` afterwards if they aren't wanted in the report.

## Out of scope

- Storing raw provider responses (e.g. OpenRouter reasoning text). This would be easy to add later as a `raw` field in the entry.
- Deduplicating in-flight requests *across processes*: two `run_tests.py` processes running in parallel can both pay for the same uncached request.
- Seeding the cache from existing result files, which don't store the prompt. Existing results are still skipped by `prevent_duplicate_tests`, so they are not re-paid.
- Cache size limits or eviction.
