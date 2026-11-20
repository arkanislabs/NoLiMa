import pytest

import async_api_connector
from async_api_connector import APIConnector


class CharTokenizer:
    """One token per character, so tests need no tokenizer download."""

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


@pytest.fixture
def fake_tiktoken(monkeypatch):
    monkeypatch.setattr(async_api_connector.tiktoken, "get_encoding", lambda name: CharTokenizer())


@pytest.fixture
def make_connector(fake_tiktoken):
    def make(cache=None):
        return APIConnector(
            api_key="sk-test-secret",
            api_url="http://localhost:1/v1",
            api_provider="vllm",
            model="test/model",
            cache=cache,
            tokenizer_type="tiktoken",
            tokenizer_model="char",
            max_retries=0,
            timeout=1,
        )

    return make
