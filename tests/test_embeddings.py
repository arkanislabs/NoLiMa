import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

import embeddings
from embeddings import EMBEDDING_BATCH_SIZE, Embedder, EmbeddingCache, embedding_key, load_embedder

MODEL, URL, PIN = "openai/text-embedding-3-small", "https://openrouter.ai/api/v1", {"only": ["openai"], "allow_fallbacks": False}


def vec(text):
    return np.array([len(text), text.count("a"), 1.0], dtype=np.float32)


@pytest.fixture
def make_embedder(tmp_path):
    def make(cache=True):
        return Embedder(model=MODEL, api_key="sk-test", api_url=URL, openrouter_provider=PIN,
                        cache=EmbeddingCache(str(tmp_path / "cache" / "e.sqlite")) if cache else None)
    return make


@pytest.fixture
def batches(monkeypatch):
    sent = []

    async def fake_send(self, texts):
        sent.append(list(texts))
        if "fail" in texts:
            raise RuntimeError("provider failed")
        return np.stack([vec(t) for t in texts]), {"tokens": 10 * len(texts), "cost": 0.001 * len(texts)}

    monkeypatch.setattr(Embedder, "_send", fake_send)
    return sent


def test_cache_round_trip_and_key(tmp_path):
    cache = EmbeddingCache(str(tmp_path / "c" / "e.sqlite"))
    key = embedding_key(MODEL, URL, PIN, "text")
    cache.put_many({key: vec("text")})
    found = cache.get_many([key, "missing"])
    assert list(found) == [key] and found[key].dtype == np.float32 and found[key].tolist() == vec("text").tolist()
    assert embedding_key(MODEL, URL + "/", PIN, "text") == key
    assert len({key, embedding_key("other/model", URL, PIN, "text"), embedding_key(MODEL, URL, None, "text"),
                embedding_key(MODEL, URL, PIN, "text!")}) == 4


def test_second_call_is_served_from_cache(make_embedder, batches):
    vectors, usage = asyncio.run(make_embedder().embed(["a", "bb"]))
    assert usage == {"texts_sent": 2, "cache_hits": 0, "tokens": 20, "cost": pytest.approx(0.002)}
    vectors, usage = asyncio.run(make_embedder().embed(["bb", "a", "ccc"]))   # a new embedder, the same cache file
    assert batches == [["a", "bb"], ["ccc"]]
    assert vectors.tolist() == [vec("bb").tolist(), vec("a").tolist(), vec("ccc").tolist()]
    assert usage == {"texts_sent": 1, "cache_hits": 2, "tokens": 10, "cost": pytest.approx(0.001)}


def test_duplicate_texts_are_sent_once(make_embedder, batches):
    vectors, usage = asyncio.run(make_embedder().embed(["a", "a"]))
    assert batches == [["a"]] and vectors.tolist() == [vec("a").tolist()] * 2 and usage["texts_sent"] == 1


def test_large_calls_are_split_into_batches(make_embedder, batches):
    asyncio.run(make_embedder().embed([f"t{i}" for i in range(EMBEDDING_BATCH_SIZE + 1)]))
    assert [len(b) for b in batches] == [EMBEDDING_BATCH_SIZE, 1]


def test_failed_request_is_not_cached(make_embedder, batches):
    with pytest.raises(RuntimeError):
        asyncio.run(make_embedder().embed(["fail"]))
    with pytest.raises(RuntimeError):
        asyncio.run(make_embedder().embed(["fail"]))
    assert batches == [["fail"], ["fail"]]


def test_without_cache_everything_is_sent(make_embedder, batches):
    asyncio.run(make_embedder(cache=False).embed(["a"]))
    asyncio.run(make_embedder(cache=False).embed(["a"]))
    assert batches == [["a"], ["a"]]


def test_cost_is_unknown_when_the_provider_does_not_report_it(make_embedder, monkeypatch):
    async def send_without_cost(self, texts):
        return np.stack([vec(t) for t in texts]), {"tokens": 5, "cost": None}

    monkeypatch.setattr(Embedder, "_send", send_without_cost)
    assert asyncio.run(make_embedder().embed(["a"]))[1]["cost"] is None
    assert asyncio.run(make_embedder().embed(["a"]))[1]["cost"] == 0.0   # all cached: nothing sent, nothing paid


def test_send_pins_the_provider_requests_floats_and_orders_rows(make_embedder):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        data = [SimpleNamespace(index=1, embedding=[0.0, 2.0]), SimpleNamespace(index=0, embedding=[1.0, 0.0])]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=7, total_tokens=7, cost=0.00014))

    embedder = make_embedder()
    embedder.client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    vectors, usage = asyncio.run(embedder._send(["first", "second"]))
    assert calls == [{"model": MODEL, "input": ["first", "second"], "encoding_format": "float", "extra_body": {"provider": PIN}}]
    assert vectors.dtype == np.float32 and vectors.tolist() == [[1.0, 0.0], [0.0, 2.0]]   # rows follow `index`
    assert usage == {"tokens": 7, "cost": 0.00014}


def test_response_with_missing_rows_raises_and_caches_nothing(make_embedder):
    async def create(**kwargs):
        data = [SimpleNamespace(index=0, embedding=[1.0]), SimpleNamespace(index=2, embedding=[3.0])]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=3, total_tokens=3, cost=None))

    embedder = make_embedder()
    embedder.client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    with pytest.raises(RuntimeError, match="expected indices"):
        asyncio.run(embedder._send(["a", "b", "c"]))
    with pytest.raises(RuntimeError, match="expected indices"):
        asyncio.run(embedder.embed(["a", "b", "c"]))
    assert embedder.cache.get_many([embedding_key(MODEL, URL, PIN, t) for t in "abc"]) == {}


def test_load_embedder_reads_the_config(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_EMBEDDING_KEY", "sk-from-env")
    config = {"model": MODEL, "api_key": "${TEST_EMBEDDING_KEY}", "api_url": URL, "openrouter_provider": PIN,
              "timeout": 60, "max_retries": 3}
    (tmp_path / "emb.json").write_text(json.dumps(config))
    embedder = load_embedder("emb", str(tmp_path), cache_dir=str(tmp_path / "cache"))
    assert embedder.client.api_key == "sk-from-env" and embedder.logged_config["api_key"] == "${TEST_EMBEDDING_KEY}"
    assert embedder.cache.path == str(tmp_path / "cache" / "openai_text-embedding-3-small.sqlite")
