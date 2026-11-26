# RAG Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A retrieval baseline for NoLiMa. The haystack of each request is split into overlapping token chunks, the chunks are ranked by a hybrid of BM25 and `openai/text-embedding-3-small` embeddings, and the model under test answers from the top `top_k` chunks instead of the whole haystack.

**Architecture:** `evaluation/rag.py` holds the retrieval method: chunking, BM25, dense scoring, Reciprocal Rank Fusion and context assembly. `evaluation/embeddings.py` holds the embedding client (OpenRouter, pinned to OpenAI) and an on-disk SQLite embedding cache. `NoLiMa_Tester` gets an optional `rag: RAGConfig`. When it is set, each needle placement runs retrieval and then generation in one coroutine, so a test's placements embed and generate concurrently. Each request logs where the needle ranked and what the embeddings cost. Everything else stays the same: needle placement, prompt template, request cache, result files. `gather_results.py` adds a retrieval table to the report.

**Tech Stack:** Python 3.12, uv, scikit-learn (`CountVectorizer`, for terms and the English stop-word list), the existing `openai` SDK (1.53, `AsyncOpenAI.embeddings`) against OpenRouter, stdlib `sqlite3`, pytest.

**Spec:** No separate spec file. The requests, verbatim (2026-11-23):

> i want to create a rag baseline for nolima. the core idea is that we divide the passage in overlapping chunks of length n (let's say for now 1k, it should be configurable). we do tifdf and embedding of every chunk. embedding should use some cheap, accessible backbone, because the core idea of this baseline is too keep it simple. than we do hybrid retrieval (keep it as simple as possible, the fewer params the better) and gebnerate the answer with the core model. we can retrieve up to m passages. m  2-4?

> word2vec is to archaic. i am okay with paid request via openrourter just choose something simple and rel cheap. implementatino should include caching the embeddings.

> i choose openai/3small update the plan

On 2026-11-24, after a probe showed TF-IDF and BM25 both at chance recall on NoLiMa (see "Expect weak retrieval"):

> update the plan: switch to BM25 the vanilla version

**Prerequisite (outside the code):** the OpenRouter workspace guardrail must allow `openai/text-embedding-3-small`. On 2026-11-23 every embedding model returned 404 "Model blocked by guardrail". The OpenAI provider is already allowed (GPT-4o mini runs on it), so only the model needs adding. Task 2 ends with a live check; stop there if it still returns 404.

## Design decisions

- **Unit of a chunk: generator tokens.** Chunks are cut from `APIConnector.encode(text)` and decoded with `APIConnector.decode`, the same tokenizer that `context_length` is measured in, so `chunk_size: 1000` means the same "1K" as `CL1K`. The text that gets chunked is exactly the text the full-context run puts in the prompt (`placement_output["text"]`, needle and distractor included). The retriever never learns where the needle is.
- **Chunking:** fixed windows of `chunk_size` tokens every `chunk_size - chunk_overlap` tokens. The last window ends at the end of the text and may be shorter. Default overlap is 200 tokens. Every NoLiMa needle (under ~50 tokens) therefore lies whole in at least one chunk.
- **Lexical: vanilla Okapi BM25** (replaced TF-IDF on 2026-11-24), fitted per request on that request's chunks. A chunk's score is the sum, over the question's distinct terms, of `idf · tf · (k1 + 1) / (tf + k1 · (1 − b + b · dl / avgdl))`, with the usual untuned `k1 = 1.2` and `b = 0.75`. `tf` is the term's count in the chunk, `dl` the chunk's term count, `avgdl` the mean `dl` over the request's chunks, `N` the number of chunks and `df` the number of chunks containing the term. `idf = ln(1 + (N − df + 0.5) / (df + 0.5))`, the IDF of Lucene, Elasticsearch and Pyserini. The textbook `ln((N − df + 0.5) / (df + 0.5))` is negative for a word in more than half the chunks and 0 for a word in one of two chunks. With 2 to 40 chunks per request, that would rank chunks containing a question word below chunks containing none. Terms come from `CountVectorizer(stop_words="english")`: lowercased, sklearn's default token pattern, no stemming. Stop words are removed because a word in every chunk still has a small positive IDF. Without that, "which / has / been / to" would give every chunk a slightly different nonzero score, and the tie handling under Hybrid would no longer see "no lexical signal". The switch does not change recall (see "Expect weak retrieval"). BM25 is used because it is the lexical baseline readers expect.
- **Embeddings: `openai/text-embedding-3-small` through OpenRouter,** pinned with `{"only": ["openai"], "allow_fallbacks": false}`. It costs $0.02/M tokens and accepts 8,192 input tokens, so a 1K-token chunk is embedded whole (beyond the limit the API returns an error, never a silent truncation). Vectors have 1,536 dimensions. It is called through the repo's `openai` SDK (`AsyncOpenAI(base_url=OpenRouter)`) with `encoding_format="float"`: without it the SDK asks for base64. Connection settings live in a model config JSON (`model_configs/text_embedding_3_small_openrouter.json`, with `${OPENROUTER_API_KEY}`), like the generator configs.
- **Embedding cache: SQLite, one file per embedding model,** at `evaluation/results/embedding_cache/<model slug>.sqlite`. The key is the SHA-256 of the model, endpoint, provider pin and text; the value is the float32 vector. SQLite rather than the request cache's one-JSON-file-per-entry layout: about 330K vectors × 6 KB ≈ 2 GB for all lengths, against about 10 GB as JSON in 330K files. Rows are committed after each API request, so an interrupted run keeps what it paid for. Failed requests are never cached.
- **Volume and cost.** A chunk that ends before the needle is identical across the 58 tests at the same haystack and depth, so the cache roughly halves the volume. That is about 160M tokens (≈ $3) at 32K and about 330M tokens (≈ $7) for all lengths from 250 to 32K. Re-runs (another `top_k`, the same generator) pay nothing. Chunk texts depend on the generator's tokenizer, so a generator with a different tokenizer embeds again.
- **Hybrid = Reciprocal Rank Fusion with the standard constant k = 60** (Cormack et al., 2009): `score(chunk) = 1/(60 + rank_bm25) + 1/(60 + rank_dense)`. There are no weights and no score normalization, and k (like BM25's k1 and b) is fixed, so nothing is tunable. **Ties share a rank before fusion** (min rank). On NoLiMa the question often shares no word with any chunk, so all BM25 scores are 0: in the 2026-11-24 probe, 86% of requests at about 4K and 39% at about 32K. The chunks that do score (about 5% of them) match topic words such as "drink", "eat" or "character", almost never the needle. If those ties were broken by position, fusion would favour the first chunks. In the prototype that pushed a dense rank-1 needle chunk down to hybrid rank 3. With shared ranks, a retriever that cannot tell chunks apart adds the same amount to every chunk. Exact ties in the fused score are broken by position, which keeps the result deterministic.
- **`top_k` (the request's m): default 3**, configurable. That is about 3K retrieved tokens, between the 2K and 4K full-context points. Every request logs the hybrid, BM25 and dense rank of the needle chunk, so recall@2 and recall@4 come out of the same run. A `top_k` sweep only needs new runs to measure how answer accuracy changes with more context, and its embeddings all come from the cache.
- **Context assembly:** the retrieved chunks' token spans are merged where they overlap or touch, put in document order, decoded, and joined with `"\n\n[...]\n\n"`. If the merged spans cover the whole text, the original text is used unchanged. **Consequence:** while the haystack fits in `top_k` chunks (up to 1000 + 2 × 800 = 2600 tokens with the defaults, i.e. 250 to 2K), the RAG prompt is identical to the full-context prompt. Those lengths are then answered from the request cache (only their embeddings are paid for), and the report's base score is the full-context one. From 4K up, RAG differs from full context.
- **Concurrency:** retrieval runs inside each placement's coroutine, before its generation request, so the existing `asyncio.gather` over a test's 26 placements runs them concurrently. Each placement embeds its question and chunks in one API request (at most 64 inputs per request; a 32K haystack has 40 chunks). BM25 takes about 0.04 s for 40 chunks and runs in the coroutine.
- **Cost tracking:** each request logs `rag.embedding = {texts_sent, cache_hits, tokens, cost}` for its own embedding call. `cost` is what OpenRouter reports, or `None` if it doesn't. The report's retrieval table and the end of a run show embedding tokens and cost, kept apart from the generator's current/total cost.
- **Prompt:** unchanged `DEFAULT_TASK_TEMPLATE`, with `{haystack}` = the retrieved context.
- **Expect weak retrieval; that is the finding.** A probe (rand_book_1, 16K window, 20 chunks of 1000/200, the 10 one-hop needle templates × 3 depths, so 30 cases; random recall@3 = 0.15) gave recall@3 of 0.13 for TF-IDF and 0.23 to 0.27 for two small local embedders. A second probe (2026-11-24: all 58 needle/question pairs × the 5 books × 5 depths, chunks of 750 words ≈ 1000 tokens with 150 overlap, recall@3 with random tie-breaking) put BM25 and per-request TF-IDF both at chance: 0.600 / 0.315 / 0.074 at about 4K / 8K / 32K, against 0.60 / 0.30 / 0.07 for random retrieval. The needle chunk scored above 0 in only 4% to 6% of requests. With the distractor needle set (`needle_set_w_Distractor.json`, not used by these configs), BM25 ranks the distractor's chunk first in 75% to 97% of requests. NoLiMa removes literal overlap, and one needle sentence is about 2% of a 1K-token chunk. text-embedding-3-small is stronger but still sees the needle diluted. The per-request retrieval log separates retrieval misses from reading errors.
- **Results:** RAG runs get their own `parent_results_dir` (`<model>_rag_c<size>_k<top_k>`), so `gather_results` reports them as a separate run next to the full-context run. The result file name is unchanged. The `rag` settings, the retrieval method and the redacted embedding config are a top-level key of the result file (only when RAG is on), so the existing overwrite warning catches RAG and full-context results written to the same folder. **The method is code, not config,** so the result file records it: `rag.retriever` is `RETRIEVER` from `rag.py`, `{"lexical": "bm25", "bm25_k1": 1.2, "bm25_b": 0.75, "rrf_k": 60}`. **Update 2026-11-24:** `rag.method` (`hybrid`, the default; `dense`; or `bm25`) now picks which ranking selects the chunks, for dense-only and lexical-only ablations (`*_rag_dense_*` and `*_rag_bm25_*` configs). `rag.retriever` is `RETRIEVERS[method]`. A `bm25` run makes no embedding requests (its `embedding_model` must be null), so it logs only the BM25 needle rank. Result files from before this change have no `method` and count as `hybrid`. Nothing becomes tunable, but every result says how it was retrieved, and a changed method (another lexical scorer, other constants, a dense-only ablation) also triggers the overwrite warning and the report's mixed-settings warning.

## Global Constraints

- RAG is off unless the run config has a `rag:` block. Without it, requests, result files (top-level keys = `RESULT_KEYS` in `tests/test_async_evaluate.py`) and reports stay exactly as they are now.
- Defaults: `chunk_size = 1000`, `chunk_overlap = 200`, `top_k = 3`, `embedding_model = "text_embedding_3_small_openrouter"` (a config name in `model_configs_dir`). Constants, not configurable: `RRF_K = 60`, `BM25_K1 = 1.2`, `BM25_B = 0.75`, `RETRIEVER` (the method and its constants, written to every RAG result file) and `CHUNK_SEPARATOR = "\n\n[...]\n\n"` in `rag.py`, `EMBEDDING_BATCH_SIZE = 64` in `embeddings.py`.
- Chunk sizes are in generator tokens (`APIConnector.encode`/`decode`).
- The embedding cache defaults to `<repo>/evaluation/results/embedding_cache/` (`DEFAULT_EMBEDDING_CACHE_DIR`, computed from `embeddings.py`'s own location), which is already git-ignored. Cache files are `.sqlite`, so `gather_results` (which loads `.json` only) ignores them.
- New runtime dependency: only `scikit-learn`. Embeddings use the existing `openai` SDK; there is no torch, onnxruntime or model2vec.
- Tests never touch the network: they replace `Embedder._send`, the embedder's `client`, or `async_evaluate.load_embedder`.
- Do not edit `evaluation/run_tests.sh`. A DeepSeek full-context run may be using it right now (bash reads a script while it runs), and full RAG runs cost money, so the user starts them.
- Commit messages follow the repo style (a plain imperative sentence, no `feat:` prefix) and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- **The question shares no word with any chunk** (NoLiMa's design), so every BM25 score is 0. Fusion must then follow the dense ranking, not chunk position. Test: Task 1 `test_retrieve_context_finds_the_needle_when_bm25_has_no_signal`. The prototype confirmed it fails when ties are broken by position.
- **BM25 is the vanilla formula, and its IDF never goes negative.** `k1 = 1.2`, `b = 0.75`, the `(k1 + 1)` factor, length normalization by the mean chunk length, and a word in every chunk still counts a little. Test: Task 1 `test_bm25_scores`, which checks one score against the formula written out by hand.
- **Vectors must map to the right texts.** The API may return rows out of order, the SDK defaults to base64, and one call can contain the same text twice. Any of these would silently scramble retrieval. Tests: Task 2 `test_send_pins_the_provider_requests_floats_and_orders_rows`, `test_duplicate_texts_are_sent_once`.
- **The embedding cache answers only identical requests and keeps paid-for work.** A different model, endpoint or provider pin must miss. A failed request must not be cached, and the batches that did succeed must stay cached. Tests: Task 2 `test_cache_round_trip_and_key`, `test_second_call_is_served_from_cache`, `test_failed_request_is_not_cached`, and the Task 5 re-run of the 1K smoke test (0 tokens sent).
- **The haystack fits in `top_k` chunks** (250 to 2K with the defaults). The prompt must then be the full-context prompt byte for byte, even with a lossy `decode`, so scores match and the request cache answers. Tests: Task 1 `test_context_is_the_original_text_when_every_chunk_is_retrieved`, Task 3 `test_rag_that_retrieves_everything_reuses_full_context_requests`, Task 5 smoke run at 1K (all 52 generations from the request cache).
- **A result file records the retrieval method.** The method lives in code, so `rag.retriever` is its only record, and a result produced by another method must trigger the overwrite warning. Tests: Task 3 `test_rag_prompts_hold_only_the_retrieved_chunks` and `test_a_changed_retrieval_method_warns_on_overwrite`, Task 4 `test_report_shows_retrieval_table_and_warnings`.
- **The needle crosses a chunk boundary.** It must lie whole in the next chunk, and recall must not count it as missed. If no chunk holds it whole, the report must say so rather than silently counting a miss. Tests: Task 1 `test_needle_across_a_chunk_boundary_is_whole_in_the_next_chunk` and `test_needle_missing_from_every_chunk_has_no_rank`, Task 4 `test_report_shows_retrieval_table_and_warnings`.

---

### Task 1: Hybrid retriever module

**Files:**
- Create: `evaluation/rag.py`
- Create: `tests/test_rag.py`
- Modify: `pyproject.toml` (add `scikit-learn` to `dependencies`), `uv.lock` (regenerated by `uv add`)

**Interfaces:**
- Consumes: nothing from other tasks. `encode`/`decode` have the signatures of `APIConnector.encode(text) -> list` and `APIConnector.decode(tokens) -> str`. `embed` has the signature of `Embedder.embed` from Task 2: `async (texts: list[str]) -> tuple[np.ndarray, dict]`.
- Produces (in `rag.py`):
  - `RRF_K = 60`, `BM25_K1 = 1.2`, `BM25_B = 0.75`, `CHUNK_SEPARATOR = "\n\n[...]\n\n"`
  - `RETRIEVER = {"lexical": "bm25", "bm25_k1": BM25_K1, "bm25_b": BM25_B, "rrf_k": RRF_K}`: the retrieval method as written to result files; change it together with the method
  - `@dataclass(frozen=True) class RAGConfig: chunk_size: int = 1000; chunk_overlap: int = 200; top_k: int = 3; embedding_model: str = "text_embedding_3_small_openrouter"`. `__post_init__` raises `ValueError` unless `chunk_size >= 1`, `0 <= chunk_overlap < chunk_size` and `top_k >= 1`.
  - `chunk_spans(n_tokens: int, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]`
  - `merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]`: sorted, overlapping or touching spans merged
  - `ordinal_ranks(scores: np.ndarray) -> np.ndarray`: 1-based, higher score first, ties by position (what a stand-alone top-k retriever returns)
  - `tied_ranks(scores: np.ndarray) -> np.ndarray`: 1-based, equal scores share the best rank (min / "competition" ranking, numpy only)
  - `rrf_scores(*ranks: np.ndarray) -> np.ndarray`: `Σ 1 / (RRF_K + rank)`
  - `bm25_scores(question: str, chunks: list[str]) -> np.ndarray`: each chunk's BM25 score (Design decisions); all zeros when no chunk has a term or no question term is in the chunks' vocabulary
  - `async retrieve_context(text: str, question: str, needle: str, config: RAGConfig, encode: Callable, decode: Callable, embed: Callable) -> tuple[str, dict]`. It returns the context and an info dict with exactly these keys:
    - `n_chunks: int`
    - `retrieved: list[int]`: chunk indices in hybrid rank order
    - `needle_chunks: list[int]`: indices of chunks containing `needle` whole
    - `needle_rank: {"hybrid", "bm25", "dense"}`: the best 1-based ordinal rank of a needle chunk under each ranking, `None` if `needle_chunks` is empty
    - `needle_retrieved: bool`: hybrid rank ≤ `top_k`
    - `context_tokens: int`: total length of the merged spans
    - `embedding: dict`: the usage dict returned by `embed`, passed through unchanged

- [ ] **Step 1: Write the failing tests** in `tests/test_rag.py`

```python
import asyncio
import math

import numpy as np
import pytest

from rag import (CHUNK_SEPARATOR, RAGConfig, bm25_scores, chunk_spans, merge_spans, ordinal_ranks, retrieve_context,
                 rrf_scores, tied_ranks)

encode = lambda text: [ord(c) for c in text]   # one token per character
decode = lambda tokens: "".join(map(chr, tokens))
USAGE = {"texts_sent": 0, "cache_hits": 0, "tokens": 0, "cost": None}


async def keyword_embed(texts):
    """Dense stand-in that 'knows' the Kiasma museum is in Helsinki."""
    return np.array([[float("Kiasma" in t or "Helsinki" in t), 0.1] for t in texts]), USAGE


def retrieve(text, config=None, decode=decode):
    return asyncio.run(retrieve_context(text, QUESTION, NEEDLE, config or CONFIG, encode, decode, keyword_embed))


FILLER = "".join(f"Line {i} of the old book.\n" for i in range(40))
NEEDLE = "Actually, Yuki lives next to the Kiasma museum."
QUESTION = "Which character has been to Helsinki?"
CONFIG = RAGConfig(chunk_size=200, chunk_overlap=60, top_k=2)


def test_chunk_spans():
    assert chunk_spans(2500, 1000, 200) == [(0, 1000), (800, 1800), (1600, 2500)]
    assert chunk_spans(1000, 1000, 200) == [(0, 1000)]
    assert chunk_spans(300, 1000, 200) == [(0, 300)]
    assert chunk_spans(0, 1000, 200) == []


@pytest.mark.parametrize("kwargs", [{"chunk_size": 0}, {"chunk_overlap": -1}, {"chunk_overlap": 1000}, {"top_k": 0}])
def test_rag_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        RAGConfig(**kwargs)


def test_merge_spans():
    assert merge_spans([(1600, 2500), (0, 1000), (800, 1800)]) == [(0, 2500)]
    assert merge_spans([(20, 30), (0, 10)]) == [(0, 10), (20, 30)]
    assert merge_spans([(0, 10), (10, 20)]) == [(0, 20)]


def test_ranks_and_fusion():
    scores = np.array([0.0, 0.5, 0.0, 0.5])
    assert ordinal_ranks(scores).tolist() == [3, 1, 4, 2]
    assert tied_ranks(scores).tolist() == [3, 1, 3, 1]
    assert rrf_scores(np.array([1, 2]), np.array([2, 1])).tolist() == pytest.approx([1 / 61 + 1 / 62] * 2)


def test_bm25_scores():
    chunks = ["dog dog cat", "cat bird", "fish"]   # 3, 2 and 1 terms: avgdl = 2
    idf_dog = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    assert bm25_scores("Which dog was it?", chunks).tolist() == pytest.approx(
        [idf_dog * 2 * (1.2 + 1) / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / 2)), 0.0, 0.0])
    assert bm25_scores("Which one was it there?", chunks).tolist() == [0.0, 0.0, 0.0]   # stop words only
    in_every_chunk = bm25_scores("Which cat?", ["cat dog", "cat bird"])
    assert in_every_chunk[0] == in_every_chunk[1] > 0   # IDF stays positive, unlike ln((N - df + 0.5) / (df + 0.5))
    assert bm25_scores("Which dog?", ["the", "it is"]).tolist() == [0.0, 0.0]   # no chunk has a term


def test_retrieve_context_finds_the_needle_when_bm25_has_no_signal():
    text = FILLER + NEEDLE + "\n" + FILLER   # no question word in any chunk: all BM25 scores are 0
    context, info = retrieve(text)
    assert NEEDLE in context
    assert info["needle_retrieved"] is True
    assert info["needle_rank"]["dense"] == 1 and info["needle_rank"]["hybrid"] == 1
    assert info["n_chunks"] == len(chunk_spans(len(text), 200, 60)) and len(info["retrieved"]) == 2
    assert info["context_tokens"] <= 2 * 200 and info["embedding"] == USAGE


def test_context_is_the_original_text_when_every_chunk_is_retrieved():
    text = "Line one.\n" + NEEDLE + "\nLine two.\n"
    context, info = retrieve(text, decode=lambda tokens: decode(tokens).replace("\n", " "))   # lossy decode
    assert context == text and info["n_chunks"] == 1 and info["context_tokens"] == len(text)


def test_non_adjacent_chunks_are_joined_in_document_order():
    text = NEEDLE + "\n" + FILLER + "Helsinki is far.\n"   # dense hits in the first and the last chunk
    context, info = retrieve(text)
    assert info["retrieved"] == [info["n_chunks"] - 1, 0]   # last chunk also matches "helsinki" in BM25
    assert context.startswith(NEEDLE) and context.endswith("Helsinki is far.\n")
    assert context.count(CHUNK_SEPARATOR) == 1


def test_needle_across_a_chunk_boundary_is_whole_in_the_next_chunk():
    text = "x" * 180 + NEEDLE + "y" * 400   # chunk [0, 200) cuts the needle; chunk [140, 340) holds it whole
    _, info = retrieve(text)
    assert info["needle_chunks"] == [1]


def test_needle_missing_from_every_chunk_has_no_rank():
    _, info = retrieve(FILLER)
    assert info["needle_chunks"] == [] and info["needle_retrieved"] is False
    assert info["needle_rank"] == {"hybrid": None, "bm25": None, "dense": None}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_rag.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'rag'`

- [ ] **Step 3: Add the dependency**

Run: `uv add scikit-learn`
Expected: `pyproject.toml` lists it under `dependencies`, `uv.lock` is updated, and `uv run python -c "import sklearn"` exits 0.

- [ ] **Step 4: Implement `evaluation/rag.py`** with the interfaces above. `chunk_spans` starts a window every `chunk_size - chunk_overlap` tokens and stops after the first window that reaches `n_tokens`. `bm25_scores` fits `CountVectorizer(stop_words="english")` on the chunks, takes the question's distinct terms through the same analyzer (`build_analyzer()`) and keeps those in the vocabulary. `fit_transform` raises `ValueError` when no chunk has a term; that case returns zeros. `retrieve_context` works in this order:

```text
tokens = encode(text); spans = chunk_spans(len(tokens), ...); chunks = [decode(tokens[a:b]) for a, b in spans]
vectors, usage = await embed([question] + chunks)
bm25   = bm25_scores(question, chunks)
dense  = cosine of vectors[0] against each of vectors[1:] (a zero vector scores 0)
hybrid = ordinal_ranks(rrf_scores(tied_ranks(bm25), tied_ranks(dense)))
retrieved = chunk indices with hybrid rank 1..top_k, in rank order
merged = merge_spans(spans of retrieved)
context = text if merged == [(0, len(tokens))] else CHUNK_SEPARATOR.join(decode(tokens[a:b]) for a, b in merged)
needle_rank[name] = min ordinal rank of a needle chunk under hybrid / ordinal_ranks(bm25) / ordinal_ranks(dense)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -v`
Expected: all tests pass, including the existing ones.

- [ ] **Step 6: Commit**

```bash
git add evaluation/rag.py tests/test_rag.py pyproject.toml uv.lock
git commit -m "Add a hybrid BM25 and embedding retriever for a RAG baseline"
```

---

### Task 2: Embedding client with an SQLite cache

**Files:**
- Create: `evaluation/embeddings.py`
- Create: `evaluation/model_configs/text_embedding_3_small_openrouter.json`
- Create: `tests/test_embeddings.py`

**Interfaces:**
- Consumes: `load_env`, `expand_env_vars`, `redact_secrets` from `env_utils.py`.
- Produces (in `embeddings.py`):
  - `DEFAULT_EMBEDDING_CACHE_DIR: str` (`<evaluation>/results/embedding_cache`), `EMBEDDING_BATCH_SIZE = 64`
  - `embedding_key(model: str, api_url: str, openrouter_provider: Optional[dict], text: str) -> str`: SHA-256 hex of `json.dumps({"model", "api_url" (trailing "/" stripped), "openrouter_provider", "text"}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))`
  - `class EmbeddingCache(path: str)` with attribute `path`, `get_many(keys: list[str]) -> dict[str, np.ndarray]` and `put_many(vectors: dict[str, np.ndarray]) -> None`. Table `embeddings(key TEXT PRIMARY KEY, vector BLOB NOT NULL)` holds float32 bytes. `put_many` uses `INSERT OR IGNORE` and commits. Each call opens its own connection, and the parent folder and table are created on first use.
  - `class Embedder(model: str, api_key: str, api_url: str, cache: Optional[EmbeddingCache] = None, openrouter_provider: Optional[dict] = None, timeout: float = 60, max_retries: int = 3)`, with attributes `cache`, `client` (`AsyncOpenAI(api_key=..., base_url=api_url, timeout=..., max_retries=...)`) and `logged_config` (set by `load_embedder`, `{}` otherwise)
    - `async _send(texts: list[str]) -> tuple[np.ndarray, dict]`: one API request; float32 rows in input order and `{"tokens": usage.prompt_tokens, "cost": getattr(usage, "cost", None)}`
    - `async embed(texts: list[str]) -> tuple[np.ndarray, dict]`: rows in input order and usage `{"texts_sent", "cache_hits", "tokens", "cost"}`. Counts are over distinct texts. `cost` is `0.0` when nothing was sent, and `None` if any request sent did not report one.
  - `load_embedder(config_name: str, model_configs_dir: str, cache_dir: str = DEFAULT_EMBEDDING_CACHE_DIR) -> Embedder`: reads `<model_configs_dir>/<config_name>.json`, expands `${VAR}`, and caches in `<cache_dir>/<model slug>.sqlite` (slug = `re.sub(r"[^A-Za-z0-9._-]+", "_", model)`)

- [ ] **Step 1: Write the failing tests** in `tests/test_embeddings.py`

```python
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


def test_load_embedder_reads_the_config(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_EMBEDDING_KEY", "sk-from-env")
    config = {"model": MODEL, "api_key": "${TEST_EMBEDDING_KEY}", "api_url": URL, "openrouter_provider": PIN,
              "timeout": 60, "max_retries": 3}
    (tmp_path / "emb.json").write_text(json.dumps(config))
    embedder = load_embedder("emb", str(tmp_path), cache_dir=str(tmp_path / "cache"))
    assert embedder.client.api_key == "sk-from-env" and embedder.logged_config["api_key"] == "${TEST_EMBEDDING_KEY}"
    assert embedder.cache.path == str(tmp_path / "cache" / "openai_text-embedding-3-small.sqlite")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'embeddings'`

- [ ] **Step 3: Implement `evaluation/embeddings.py`** with the interfaces above. `embed`:
  1. dedupes the texts in order;
  2. looks the keys up in the cache;
  3. sends the missing texts in batches of `EMBEDDING_BATCH_SIZE`, one batch after another, calling `cache.put_many` after each batch;
  4. returns rows in input order.

  `_send` calls `self.client.embeddings.create(model=..., input=texts, encoding_format="float", extra_body={"provider": openrouter_provider})`, leaving out `extra_body` when there is no pin. It orders `response.data` by `index`. Retries come from the SDK's `max_retries`.

- [ ] **Step 4: Write `evaluation/model_configs/text_embedding_3_small_openrouter.json`**

```json
{
    "model": "openai/text-embedding-3-small",
    "api_key": "${OPENROUTER_API_KEY}",
    "api_url": "https://openrouter.ai/api/v1",
    "openrouter_provider": {
        "only": ["openai"],
        "allow_fallbacks": false
    },
    "timeout": 60,
    "max_retries": 3
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 6: Live check: the guardrail allows the model, and the cache works** (two texts, a fraction of a cent)

Run (from `evaluation/`): `for i in 1 2; do uv run python -c "import asyncio; from embeddings import load_embedder; e = load_embedder('text_embedding_3_small_openrouter', 'model_configs'); v, u = asyncio.run(e.embed(['Which character has been to Helsinki?', 'Actually, Yuki lives next to the Kiasma museum.'])); print(v.shape, u)"; done`
Expected: first line `(2, 1536) {'texts_sent': 2, 'cache_hits': 0, 'tokens': <about 20>, 'cost': <number or None>}`, second line `(2, 1536) {'texts_sent': 0, 'cache_hits': 2, 'tokens': 0, 'cost': 0.0}`. A 404 "Model blocked by guardrail" means the prerequisite is not met: stop and tell the user. Note whether `cost` is reported, for the README.

- [ ] **Step 7: Commit**

```bash
git add evaluation/embeddings.py evaluation/model_configs/text_embedding_3_small_openrouter.json tests/test_embeddings.py
git commit -m "Add an OpenRouter embedding client with an SQLite embedding cache"
```

---

### Task 3: Wire RAG into the tester and the run config

**Files:**
- Modify: `evaluation/async_evaluate.py` (`NoLiMa_Tester.__init__`, `NoLiMa_Tester.evaluate`)
- Modify: `evaluation/run_tests.py` (new `--rag` and `--embedding_cache_dir` arguments, pass them to the tester, print embedding usage)
- Test: `tests/test_async_evaluate.py`

**Interfaces:**
- Consumes:
  - from Task 1: `RAGConfig`, `RETRIEVER` and `retrieve_context`;
  - from Task 2: `load_embedder`, `DEFAULT_EMBEDDING_CACHE_DIR`, and `Embedder.embed` / `Embedder.logged_config`.

  All are imported by name into `async_evaluate`, so tests can patch `async_evaluate.load_embedder` and `async_evaluate.RETRIEVER`.
- Produces:
  - `NoLiMa_Tester(..., rag: Optional[RAGConfig] = None, embedding_cache_dir: str = DEFAULT_EMBEDDING_CACHE_DIR)`. When `rag` is set, `__init__` creates `self.embedder = load_embedder(rag.embedding_model, model_configs_dir, embedding_cache_dir)`.
  - Result file: top-level `"rag": {**dataclasses.asdict(rag), "retriever": dict(RETRIEVER), "embedding_config": self.embedder.logged_config}`, only when RAG is on. Each per-request result gets `"rag"`: the info dict from `retrieve_context`.
  - Run config: an optional `rag:` block with the `RAGConfig` fields, where omitted fields take their defaults, and `embedding_cache_dir`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_async_evaluate.py`; add `import numpy as np`, `import async_evaluate` and `from rag import RETRIEVER, RAGConfig` to its imports)

```python
class KiasmaEmbedder:
    logged_config = {"model": "fake/embedding"}

    async def embed(self, texts):
        vectors = np.array([[float("Kiasma" in t or "Helsinki" in t), 0.1] for t in texts])
        return vectors, {"texts_sent": len(texts), "cache_hits": 0, "tokens": 10 * len(texts), "cost": None}


@pytest.fixture
def fake_embedder(monkeypatch):
    monkeypatch.setattr(async_evaluate, "load_embedder", lambda *args, **kwargs: KiasmaEmbedder())


SMALL_RAG = RAGConfig(chunk_size=100, chunk_overlap=60, top_k=1)


def test_rag_prompts_hold_only_the_retrieved_chunks(make_tester, sent, fake_embedder, tmp_path):
    results = make_tester(str(tmp_path / "run"), rag=SMALL_RAG).evaluate()
    prompts = [request["messages"][-1]["content"] for request in sent]
    assert len(prompts) == 3 and all("Kiasma" in p and len(p) < 200 for p in prompts)   # full haystack alone is 200+
    assert all(r["rag"]["n_chunks"] > 1 and len(r["rag"]["retrieved"]) == 1 and r["rag"]["context_tokens"] <= 100
               and r["rag"]["embedding"]["texts_sent"] == r["rag"]["n_chunks"] + 1 for r in results)
    (result_file,) = (tmp_path / "run").iterdir()
    saved = json.loads(result_file.read_text())
    assert set(saved) == RESULT_KEYS | {"rag"}
    assert saved["rag"] == {"chunk_size": 100, "chunk_overlap": 60, "top_k": 1,
                            "embedding_model": "text_embedding_3_small_openrouter",
                            "retriever": {"lexical": "bm25", "bm25_k1": 1.2, "bm25_b": 0.75, "rrf_k": 60},
                            "embedding_config": {"model": "fake/embedding"}}


def test_rag_that_retrieves_everything_reuses_full_context_requests(make_tester, sent, fake_embedder, tmp_path):
    make_tester(str(tmp_path / "full")).evaluate()
    results = make_tester(str(tmp_path / "rag"), rag=RAGConfig()).evaluate()   # 1000-token chunks > 200-token haystack
    assert len(sent) == 3 and all(r["request_cache_hit"] for r in results)


def test_changing_rag_settings_warns_on_overwrite(make_tester, sent, fake_embedder, tmp_path, capsys):
    make_tester(str(tmp_path / "run")).evaluate()
    make_tester(str(tmp_path / "run"), rag=SMALL_RAG).evaluate()
    assert "produced with different settings: rag." in capsys.readouterr().out


def test_a_changed_retrieval_method_warns_on_overwrite(make_tester, sent, fake_embedder, tmp_path, capsys, monkeypatch):
    make_tester(str(tmp_path / "run"), rag=SMALL_RAG).evaluate()
    monkeypatch.setattr(async_evaluate, "RETRIEVER", {**RETRIEVER, "lexical": "tfidf"})
    make_tester(str(tmp_path / "run"), rag=SMALL_RAG).evaluate()   # same settings, another method
    assert "produced with different settings: rag." in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_async_evaluate.py -v`
Expected: the four new tests fail, first with `ImportError: cannot import name 'RAGConfig'`, or `TypeError: ... unexpected keyword argument 'rag'` once the import works. The old tests still pass.

- [ ] **Step 3: Implement.** In `evaluate()`, when `self.rag` is set, the task appended to `async_tasks` for a placement is a coroutine that does three things:
  1. awaits `retrieve_context(placement_output["text"], retrieval_question, needle, self.rag, self.api_connector.encode, self.api_connector.decode, self.embedder.embed)`;
  2. awaits `generate_response` on the template filled with the returned context;
  3. returns `{**response, "rag": info}`.

  The existing merge loop then copies `rag` into the result. `retrieval_question` and `needle` are the locals that already have `{CHAR}` substituted; pass them as arguments, not closures over loop variables. Without RAG the task stays exactly as it is now. Add the top-level `outputs["rag"]` (as in Produces, reading the module-level `RETRIEVER` when the file is written) next to the `distractor` key, only when set.

  In `run_tests.py`:
  - add `parser.add_argument("--rag", type=Optional[RAGConfig], default=None, help="RAG baseline: answer from the top_k retrieved chunks of the haystack instead of the whole haystack")` and `parser.add_argument("--embedding_cache_dir", type=str, default=DEFAULT_EMBEDDING_CACHE_DIR, help="Directory of the embedding cache, shared by all runs")`;
  - parse with `args = parser.instantiate_classes(parser.parse_args())`; without it, `args.rag` is a `Namespace`, not a `RAGConfig`;
  - pass both arguments to `NoLiMa_Tester`;
  - when RAG is on, append `f" (RAG: top {k} of {size}-token chunks, overlap {overlap})"` to the start line, and print a line after the cost summary: `f"{label} embeddings: {tokens:,} tokens sent, cost {cost}, {hits:,} texts from the embedding cache"`. The values are sums of `r["rag"]["embedding"]` over all results. `cost` is formatted `$x.xxxx`, or `n/a` when no request reported one.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 5: Check that a partial `rag` setting parses with the defaults filled in**

Run (from `evaluation/`): `PYTHONPATH=../ uv run python run_tests.py --config run_config/smoke_test_deepseek_v4.1_flash_1017_4K.yaml --rag.top_k 4 --print_config | grep -A4 '^rag:'`
Expected: `rag:` followed by `chunk_size: 1000`, `chunk_overlap: 200`, `top_k: 4`, `embedding_model: text_embedding_3_small_openrouter`. The YAML form is exercised by the Task 5 smoke runs.

- [ ] **Step 6: Commit**

```bash
git add evaluation/async_evaluate.py evaluation/run_tests.py tests/test_async_evaluate.py
git commit -m "Answer from retrieved chunks when a run config has a rag block"
```

---

### Task 4: Retrieval diagnostics in the CSV and the report

**Files:**
- Modify: `evaluation/gather_results.py` (`get_summarized_result`, `summarize_runs`, `build_report`, module docstring)
- Test: `tests/test_gather_results.py`

**Interfaces:**
- Consumes: the result-file fields from Task 3: top-level `rag` (with `top_k` and `retriever`) and per-request `rag` (`n_chunks`, `context_tokens`, `needle_retrieved`, `needle_rank`, `embedding`).
- Produces:
  - CSV: `rag.<field>` columns, flattened like `model_config.<field>`, plus `n_needle_retrieved`, `embedding_tokens` and `embedding_cost`, all only for RAG files.
  - Report, per run with RAG results:
    - a `- **RAG:** ` settings line (the four `RAGConfig` fields, then `retriever` as `key=value` pairs in file order) under the model config line;
    - a `### Retrieval` table between `### Per context length` and `### Per test`, with these columns:
      - context length;
      - mean chunks and mean context tokens;
      - share (%) of requests whose needle chunk is in the top `top_k` by hybrid rank ("Needle retrieved"), by BM25 rank alone and by dense rank alone;
      - accuracy (%) on the retrieved requests and on the missed ones (`--` when there are none);
      - embedding tokens sent and embedding cost (`format_cost`; `n/a` when no request reported a cost);
    - a warning counting requests whose needle was in no chunk.
  - Report, per run: a warning when its files have more than one distinct `rag` value (no RAG counts as one value).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_gather_results.py`)

```python
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
    assert ("- **RAG:** `chunk_size` = 1000, `chunk_overlap` = 200, `top_k` = 3, `embedding_model` = `e`, "
            "`retriever` = `lexical=bm25, bm25_k1=1.2, bm25_b=0.75, rrf_k=60`") in report
    assert ("| Context length | Chunks | Context tokens | Needle retrieved | BM25 alone | Dense alone "
            "| Score if retrieved | Score if missed | Embedding tokens sent | Embedding cost |") in report
    assert "| 1K | 10.0 | 2400 | 33.3 | 66.7 | 33.3 | 100.0 | 50.0 | 33,000 | $0.0007 |" in report
    assert "1 request(s): the needle was not found whole in any chunk, so retrieval counts it as missed." in report
    plain = [("x.json", result_data())]
    assert "### Retrieval" not in build_report(summarize_runs(plain), summarize_models(plain), ["results"])


def test_report_warns_when_a_run_mixes_rag_settings():
    loaded = [("a.json", result_data()), ("b.json", rag_data())]
    report = build_report(summarize_runs(loaded), summarize_models(loaded), ["results"])
    assert "Results mix 2 different RAG settings (no RAG counts as one); their scores are averaged together." in report
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_gather_results.py -v`
Expected: the three new tests fail (`KeyError: 'rag.top_k'` and missing report lines). The old tests pass.

- [ ] **Step 3: Implement** the CSV columns, per-length retrieval counters in `summarize_runs`, the settings line, the `### Retrieval` table and the two warnings, with the exact strings from the tests. Use the existing `pct` / `format_length` / `format_cost` / `md_table` helpers. "Chunks" uses one decimal; "Context tokens" is a rounded integer mean; "Embedding tokens sent" uses thousands separators. Add the new CSV columns to the module docstring.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add evaluation/gather_results.py tests/test_gather_results.py
git commit -m "Report needle retrieval rates and embedding cost for RAG runs"
```

---

### Task 5: Run configs, README, and smoke runs

**Files:**
- Create: `evaluation/run_config/nolima_deepseek_v4.1_flash_1017_nothink_openrouter_rag_c1000_k3_{250,500,1K,2K,4K,8K,16K,32K}.yaml` (8 files)
- Create: `evaluation/run_config/smoke_test_rag_deepseek_v4.1_flash_1017_nothink_openrouter_{1K,32K}.yaml`
- Modify: `README.md` (a "RAG baseline" subsection under "Additional Notes")

**Interfaces:**
- Consumes: the `rag:` run-config block from Task 3, the embedding config from Task 2 and the report from Task 4.
- Produces: configs the user can add to `run_tests.sh` later. The generator is DeepSeek V4.1 Flash, thinking off, because it is the model with full-context runs configured from 250 to 32K.

- [ ] **Step 1: Write the 8 full-run configs.** Each is a copy of the matching `nolima_deepseek_v4.1_flash_1017_nothink_openrouter_<len>.yaml` with two changes. `parent_results_dir` becomes `./results/deepseek_v4.1_flash_1017_nothink_openrouter_rag_c1000_k3/CL<len>/`. This block is appended:

```yaml
# RAG baseline: answer from the top_k chunks of the haystack (hybrid BM25 + text-embedding-3-small, RRF k=60)
rag:
  chunk_size: 1000
  chunk_overlap: 200
  top_k: 3
  embedding_model: text_embedding_3_small_openrouter
```

  Update the header comment: the run command with the new file name, and a note that 250 to 2K equal full context, so their generations come from the request cache.

- [ ] **Step 2: Write the 2 smoke configs.** Each is a copy of the matching full-run RAG config from Step 1, with `haystack_files: [rand_book_1.txt]`, `test_names: [0401_T17_C02_onehop, 0401_T17_C02_twohop]` and `parent_results_dir: ./results/smoke_rag_deepseek_v4.1_flash_1017_nothink_openrouter/CL<len>/`:
  - `_1K`: `context_length: 1000`, `document_depth_percent_intervals: 26` (52 requests). Same placements as the full-context run, so every generation should come from the request cache. About 120K embedding tokens, under $0.01.
  - `_32K`: `context_length: 32000`, `document_depth_percent_intervals: 3` (6 generations, about 250K embedding tokens, a few cents in total).

- [ ] **Step 3: Smoke run at 1K, twice: RAG must equal full context, and embeddings must come from the cache the second time**

Run (from `evaluation/`): `for i in 1 2; do PYTHONPATH=../ TOKENIZERS_PARALLELISM=false uv run python -u run_tests.py --config run_config/smoke_test_rag_deepseek_v4.1_flash_1017_nothink_openrouter_1K.yaml | tail -2; done`
Expected: both runs end their cost line with `(requests: 0 sent, 52 from the request cache)`. The first run's embeddings line shows tokens sent; the second shows `0 tokens sent`. If any generation is sent, the RAG prompt differs from the full-context prompt at 1K: stop and investigate.

- [ ] **Step 4: Smoke run at 32K, then gather**

Run (from `evaluation/`): `PYTHONPATH=../ TOKENIZERS_PARALLELISM=false uv run python -u run_tests.py --config run_config/smoke_test_rag_deepseek_v4.1_flash_1017_nothink_openrouter_32K.yaml && uv run python gather_results.py results/smoke_rag_deepseek_v4.1_flash_1017_nothink_openrouter -o results/smoke_rag_deepseek_v4.1_flash_1017_nothink_openrouter/gathered.csv -r results/smoke_rag_deepseek_v4.1_flash_1017_nothink_openrouter/report.md && grep -A5 '### Retrieval' results/smoke_rag_deepseek_v4.1_flash_1017_nothink_openrouter/report.md`
Expected: 6 generations sent, and a Retrieval table with two rows. 1K: 2.0 chunks (1000 tokens plus the needle), 100.0 needle retrieved. 32K: 40.0 chunks, 2,200 to 3,000 context tokens, embedding tokens in the hundreds of thousands.

- [ ] **Step 5: README.** Add a short "RAG baseline" subsection under "Additional Notes". It covers:
  - what the `rag:` block does, and its four fields with their defaults;
  - that chunk sizes are in the model's tokens;
  - that hybrid means RRF (k = 60) of BM25 (vanilla Okapi, `k1 = 1.2`, `b = 0.75`, the non-negative Lucene IDF) and `text-embedding-3-small` via OpenRouter, pinned to OpenAI;
  - the embedding cache: location, key, deleting a model's `.sqlite` to re-embed, `embedding_cache_dir`;
  - whether OpenRouter reports embedding cost (from the Task 2 live check);
  - that lengths up to `chunk_size + (top_k − 1) × (chunk_size − chunk_overlap)` equal full context and come from the request cache;
  - that result files record the retrieval method under `rag.retriever`, and the report shows it on the RAG line;
  - the report's Retrieval table.

- [ ] **Step 6: Commit**

```bash
git add evaluation/run_config/nolima_deepseek_v4.1_flash_1017_nothink_openrouter_rag_c1000_k3_*.yaml evaluation/run_config/smoke_test_rag_deepseek_v4.1_flash_1017_nothink_openrouter_*.yaml README.md
git commit -m "Add RAG baseline run configs for DeepSeek V4.1 Flash and document the baseline"
```

---

## Not in this plan

- **Full RAG runs** (250 to 32K). Embeddings cost about $7 for all lengths. Generations for 250 to 2K come from the request cache. From 4K up, each length sends 7,540 prompts of about 3K tokens, roughly the cost of the full-context 2K to 4K runs. Add the configs to `run_tests.sh` once the current DeepSeek run is done.
- **Sweeps.** `top_k` 2 and 4: recall comes from any run's logs and embeddings from the cache, so only re-run to measure answer accuracy. `chunk_size`: smaller chunks raise the needle's share of a chunk and are probably the biggest lever for dense recall; each new size embeds again.
- An experiment-log entry, once results exist.
