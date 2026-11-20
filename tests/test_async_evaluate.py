import asyncio
import json

import pytest

from async_api_connector import APIConnector
from async_evaluate import NoLiMa_Tester

MODEL_CONFIG = {
    "model": "test/model",
    "api_key": "sk-test-secret",
    "api_url": "http://localhost:1/v1",
    "api_provider": "vllm",
    "max_tokens": 10,
    "temperature": 0.0,
    "top_p": 1.0,
    "timeout": 1,
    "max_retries": 0,
    "tokenizer_type": "tiktoken",
    "tokenizer_model": "char",
}

# Top-level keys of a result file. All but model_config feed test_hash, so this set must not grow.
RESULT_KEYS = {"eval_name", "test_name", "model_name", "model_config_path", "retrieval_question", "needle", "gold_answers",
               "system_prompt", "use_default_system_prompt", "task_template", "haystack_path", "haystack_hash", "context_length",
               "character_set", "document_depth_percent_min", "document_depth_percent_max", "document_depth_percent_intervals",
               "shift", "static_depth", "metric", "result_dir", "log_placements_dir", "seed", "results", "test_hash", "model_config"}


@pytest.fixture
def make_tester(tmp_path, fake_tiktoken):
    configs_dir = tmp_path / "model_configs"
    configs_dir.mkdir()
    (configs_dir / "fake.json").write_text(json.dumps(MODEL_CONFIG))
    book = tmp_path / "book.txt"
    book.write_text("".join(f"Line {i} of the book.\n" for i in range(300)))
    # evaluate() uses asyncio.get_event_loop(); earlier tests' asyncio.run() leaves no current loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def make(results_dir, **overrides):
        kwargs = dict(
            model_name="fake",
            model_configs_dir=str(configs_dir),
            needle="Actually, {CHAR} lives next to the Kiasma museum.",
            haystack_path=str(book),
            results_dir=results_dir,
            retrieval_question="Which character has been to Helsinki?",
            character_set=json.dumps(["Yuki", "Stuart"]),
            system_prompt="",
            use_default_system_prompt=True,
            task_template="{haystack}\n\nQuestion: {question}",
            context_length=200,
            document_depth_percent_intervals=3,
            metric="contains",
            test_name="t",
            request_cache_dir=str(tmp_path / "cache"),
        )
        kwargs.update(overrides)
        return NoLiMa_Tester(**kwargs)

    yield make
    loop.close()
    asyncio.set_event_loop(None)


@pytest.fixture
def send_state():
    return {"sent": [], "fail_first": False}


@pytest.fixture
def sent(monkeypatch, send_state):
    async def fake_send(self, request):
        send_state["sent"].append(request)
        if send_state["fail_first"] and len(send_state["sent"]) == 1:
            raise RuntimeError("provider failed")
        return {"response": "Yuki", "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                "finish_reason": "stop", "cost": 0.001}

    monkeypatch.setattr(APIConnector, "_send", fake_send)
    return send_state["sent"]


@pytest.fixture
def fail_first_call(send_state):
    send_state["fail_first"] = True


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
    assert set(json.loads(result_file.read_text())) == RESULT_KEYS  # nothing new feeds test_hash


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
    assert len(sent) == 4  # 3 first attempt + 1 retry of the failed one
