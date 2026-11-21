import asyncio
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


def test_build_request_sends_openrouter_provider_routing_and_keys_the_cache_on_it():
    routing = {"only": ["openai"], "allow_fallbacks": False, "require_parameters": True}
    pinned = build_request("vllm", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini", {"openrouter_provider": routing},
                           SYS, MSGS, 192, 0.0, 1.0)
    unpinned = build_request("vllm", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini", {}, SYS, MSGS, 192, 0.0, 1.0)
    assert pinned["extra_body"] == {"provider": {"only": ["openai"], "allow_fallbacks": False, "require_parameters": True}}
    assert "extra_body" not in unpinned
    assert request_key(pinned) != request_key(unpinned)


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
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Veronica"), finish_reason="stop")],
                               usage=usage, system_fingerprint=None)

    connector.api = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    request = build_request("vllm", "http://localhost:1/v1", "test/model", {}, SYS, MSGS, 10, 0.0, 1.0)
    out = asyncio.run(connector._send(request))
    assert captured == {k: v for k, v in request.items() if k not in ("api_provider", "api_url")}
    assert out == {"response": "Veronica", "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
                   "finish_reason": "stop", "cached_tokens": None, "reasoning_tokens": None, "cost": 0.001,
                   "provider": None, "system_fingerprint": None}


def test_send_forwards_provider_routing_and_records_who_served_it(make_connector):
    connector = make_connector()
    captured = {}

    async def create(**params):
        captured.update(params)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12, prompt_tokens_details=None,
                                completion_tokens_details=None, cost=0.001)
        # OpenRouter adds the serving provider to the completion, next to OpenAI's system_fingerprint
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Veronica"), finish_reason="stop")],
                               usage=usage, system_fingerprint="fp_2bf6f3634d", provider="OpenAI")

    connector.api = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    routing = {"only": ["openai"], "allow_fallbacks": False}
    request = build_request("vllm", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini", {"openrouter_provider": routing},
                            SYS, MSGS, 10, 0.0, 1.0)
    out = asyncio.run(connector._send(request))
    assert captured["extra_body"] == {"provider": {"only": ["openai"], "allow_fallbacks": False}}
    assert (out["provider"], out["system_fingerprint"]) == ("OpenAI", "fp_2bf6f3634d")


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
    with open(cache.path_for(calls[0])) as f:
        assert "sk-test-secret" not in f.read()


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
