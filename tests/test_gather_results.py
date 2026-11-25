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


def test_summarized_result_drops_seed_without_test_hash():
    data = result_data()
    del data["test_hash"]
    s = get_summarized_result(data)
    assert "seed" not in s and "test_hash" not in s


def test_report_warns_about_repeated_result_files():
    loaded = [("old.json", result_data()), ("new.json", result_data())]
    report = build_report(summarize_runs(loaded), summarize_models(loaded), ["results"])
    assert "1 result file(s) repeat a model, test, haystack and context length already in this run" in report
    single = [("new.json", result_data())]
    assert "repeat a model, test, haystack" not in build_report(summarize_runs(single), summarize_models(single), ["results"])


def rag_data():
    data = result_data()
    data["rag"] = {"chunk_size": 1000, "chunk_overlap": 200, "top_k": 3, "embedding_model": "e",
                   "retriever": {"lexical": "bm25", "bm25_k1": 1.2, "bm25_b": 0.75, "rrf_k": 60},
                   "embedding_config": {"model": "openai/text-embedding-3-small"}}
    ranks = [(1, 2, 1), (4, 1, 6), (None, None, None)]   # (hybrid, bm25, dense); the last needle was in no chunk
    for r, (hybrid, bm25, dense) in zip(data["results"], ranks):
        r["rag"] = {"n_chunks": 10, "context_tokens": 2400, "retrieved": [0, 1, 2], "needle_chunks": [],
                    "needle_retrieved": hybrid is not None and hybrid <= 3,
                    "needle_rank": {"hybrid": hybrid, "bm25": bm25, "dense": dense},
                    "embedding": {"texts_sent": 11, "cache_hits": 0, "tokens": 11000, "cost": 0.00022}}
    data["results"][1]["metric"] = 0
    return data


def test_summarized_result_flattens_rag_settings():
    s = get_summarized_result(rag_data())
    assert s["rag.top_k"] == 3 and s["rag.embedding_model"] == "e" and s["n_needle_retrieved"] == 1
    assert s["rag.retriever"]["lexical"] == "bm25"
    assert s["embedding_tokens"] == 33000 and s["embedding_cost"] == pytest.approx(0.00066)
    assert "n_needle_retrieved" not in get_summarized_result(result_data())


def test_report_shows_retrieval_table_and_warnings():
    loaded = [("x.json", rag_data())]
    report = build_report(summarize_runs(loaded), summarize_models(loaded), ["results"])
    # a file from before `method` was logged counts as hybrid
    assert ("- **RAG:** `method` = `hybrid`, `chunk_size` = 1000, `chunk_overlap` = 200, `top_k` = 3, `embedding_model` = `e`, "
            "`retriever` = `lexical=bm25, bm25_k1=1.2, bm25_b=0.75, rrf_k=60`") in report
    assert ("| Context length | Chunks | Context tokens | Needle retrieved | Hybrid | BM25 alone | Dense alone "
            "| Score if retrieved | Score if missed | Embedding tokens sent | Embedding cost |") in report
    assert "| 1K | 10.0 | 2400 | 33.3 | 33.3 | 66.7 | 33.3 | 100.0 | 50.0 | 33,000 | $0.0007 |" in report
    assert "1 request(s): the needle was not found whole in any chunk, so retrieval counts it as missed." in report
    plain = [("x.json", result_data())]
    assert "### Retrieval" not in build_report(summarize_runs(plain), summarize_models(plain), ["results"])


def bm25_data():
    data = rag_data()
    data["rag"] = {"chunk_size": 1000, "chunk_overlap": 200, "top_k": 3, "embedding_model": None, "method": "bm25",
                   "retriever": {"lexical": "bm25", "bm25_k1": 1.2, "bm25_b": 0.75}}
    for r, bm25 in zip(data["results"], (2, 4, None)):
        r["rag"].update(needle_retrieved=bm25 is not None and bm25 <= 3, needle_rank={"bm25": bm25},
                        embedding={"texts_sent": 0, "cache_hits": 0, "tokens": 0, "cost": 0.0})
    return data


def test_report_of_a_bm25_run_shows_only_the_bm25_rank():
    loaded = [("x.json", bm25_data())]
    report = build_report(summarize_runs(loaded), summarize_models(loaded), ["results"])
    assert "- **RAG:** `method` = `bm25`, `chunk_size` = 1000, `chunk_overlap` = 200, `top_k` = 3, `embedding_model` = none, " in report
    assert "| 1K | 10.0 | 2400 | 33.3 | -- | 33.3 | -- | 100.0 | 50.0 | 0 | $0.0000 |" in report
    assert "1 request(s): the needle was not found whole in any chunk" in report


def test_old_and_new_hybrid_files_count_as_one_rag_setting():
    new = rag_data()
    new["rag"]["method"] = "hybrid"
    loaded = [("a.json", rag_data()), ("b.json", new)]
    assert next(iter(summarize_runs(loaded).values()))["n_rag_settings"] == 1


def test_report_warns_when_a_run_mixes_rag_settings():
    loaded = [("a.json", result_data()), ("b.json", rag_data())]
    report = build_report(summarize_runs(loaded), summarize_models(loaded), ["results"])
    assert "Results mix 2 different RAG settings (no RAG counts as one); their scores are averaged together." in report
