import json
import os

import pytest

from gather_results import build_report, get_summarized_result, load_result_files, summarize_models, summarize_runs
from request_cache import RequestCache


def res(cost, hit=None):
    r = {"metric": 1, "prompt_tokens": 10, "completion_tokens": 2, "finish_reason": "stop", "cost": cost}
    if hit is not None:
        r["request_cache_hit"] = hit
    return r


def result_data():
    return {"model_name": "m", "test_name": "t", "context_length": 1000, "haystack_path": "h.txt",
            "result_dir": "./results/run/CL1K/t", "seed": 1, "test_hash": "x",
            "results": [res(0.001, False), res(0.002, True), res(0.004)]}  # last: written before the cache existed


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
