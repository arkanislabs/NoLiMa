import asyncio
import hashlib
import json
import os

from request_cache import (
    CACHE_MARKER,
    INLINE_TEXT_LIMIT,
    PREVIEW_CHARS,
    RequestCache,
    compact_request,
    is_request_cache_dir,
    request_key,
    should_cache,
)

REQ = {
    "api_provider": "vllm",
    "api_url": "https://openrouter.ai/api/v1",
    "model": "deepseek/deepseek-v4.1-flash-1017",
    "messages": [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Q?"}],
    "seed": 43,
    "max_tokens": 4096,
    "temperature": 0.0,
    "top_p": 1.0,
}
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
    assert compact["messages"][0]["content"] == {
        "sha256": hashlib.sha256(long.encode()).hexdigest(),
        "chars": len(long),
        "head": "a" * PREVIEW_CHARS,
        "tail": "z" * PREVIEW_CHARS,
    }


def test_miss_then_hit_writes_one_entry(tmp_path):
    cache = RequestCache(str(tmp_path / "cache"))
    send, calls = counting_send()
    assert asyncio.run(cache.fetch(REQ, send)) == (OUT, False)
    assert asyncio.run(cache.fetch(REQ, send)) == (OUT, True)
    assert len(calls) == 1
    key = request_key(REQ)
    path = tmp_path / "cache" / "deepseek_deepseek-v4.1-flash-1017" / key[:2] / f"{key}.json"
    assert cache.path_for(REQ) == str(path)
    assert os.listdir(path.parent) == [path.name]  # atomic write leaves no temp files
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
    assert not should_cache({"finish_reason": "error"})
    assert not should_cache({"finish_reason": None})
    assert not should_cache({})


def test_unreadable_entry_is_a_miss_and_gets_overwritten(tmp_path):
    cache = RequestCache(str(tmp_path))
    cache.put(REQ, OUT)
    with open(cache.path_for(REQ), "w") as f:
        f.write("{not json")
    send, calls = counting_send()
    assert asyncio.run(cache.fetch(REQ, send)) == (OUT, False)
    assert len(calls) == 1
    assert cache.get(REQ) == OUT


def test_marker_identifies_the_cache_dir(tmp_path):
    RequestCache(str(tmp_path / "c"))
    assert (tmp_path / "c" / CACHE_MARKER).is_file()
    assert is_request_cache_dir(str(tmp_path / "c")) and not is_request_cache_dir(str(tmp_path))


def test_cost_summary_splits_current_and_total():
    from request_cache import cost_summary, current_cost

    results = [{"cost": 0.001, "request_cache_hit": False}, {"cost": 0.002, "request_cache_hit": False},
               {"cost": 0.004, "request_cache_hit": True}, {"cost": 0.0001}]  # last: written before the cache existed
    assert current_cost(results[2]) == 0.0 and current_cost(results[3]) == 0.0001
    assert cost_summary(results) == "Cost: current $0.0031, total $0.0071 (requests: 3 sent, 1 from the request cache)"
    assert cost_summary([{"cost": None, "request_cache_hit": False}]) == \
        "Cost: current n/a, total n/a (requests: 1 sent, 0 from the request cache)"
    assert cost_summary([{"cost": 0.004, "request_cache_hit": True}]) == \
        "Cost: current $0.0000, total $0.0040 (requests: 0 sent, 1 from the request cache)"


def test_unknown_cost_is_not_reported_as_zero():
    # providers that report no cost (Gemini, AWS, OpenAI direct, Azure, local vLLM)
    from request_cache import cost_summary, current_cost

    assert current_cost({"cost": None, "request_cache_hit": True}) is None
    results = [{"cost": None, "request_cache_hit": False}, {"cost": None, "request_cache_hit": True}]
    assert cost_summary(results) == "Cost: current n/a, total n/a (requests: 1 sent, 1 from the request cache)"
