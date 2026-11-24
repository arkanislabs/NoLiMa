"""
Hybrid retriever for the RAG baseline: BM25 and dense embeddings fused with Reciprocal Rank Fusion.

A haystack is cut into overlapping windows of generator tokens. The windows are ranked against
the question by BM25 and by embedding cosine similarity, and the fused top-k windows become
the context the generator sees. The retriever never learns where the needle is; the needle is
only used afterwards, to log where it ranked.
"""
import asyncio
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

RRF_K = 60
BM25_K1 = 1.2
BM25_B = 0.75
CHUNK_SEPARATOR = "\n\n[...]\n\n"
# The retrieval method as written to result files; change it together with the method.
RETRIEVER = {"lexical": "bm25", "bm25_k1": BM25_K1, "bm25_b": BM25_B, "rrf_k": RRF_K}


@dataclass(frozen=True)
class RAGConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 3
    embedding_model: str = "text_embedding_3_small_openrouter"

    def __post_init__(self):
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(f"chunk_overlap must be in [0, chunk_size), got {self.chunk_overlap}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")


def chunk_spans(n_tokens: int, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """Token windows of `chunk_size`, one every `chunk_size - chunk_overlap`, up to the first that reaches the end."""
    spans = []
    start = 0
    while start < n_tokens:
        end = min(start + chunk_size, n_tokens)
        spans.append((start, end))
        if end >= n_tokens:
            break
        start += chunk_size - chunk_overlap
    return spans


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort spans and merge those that overlap or touch."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def ordinal_ranks(scores: np.ndarray) -> np.ndarray:
    """1-based ranks, higher score first, ties broken by position."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def tied_ranks(scores: np.ndarray) -> np.ndarray:
    """1-based ranks, higher score first; equal scores share the best rank."""
    scores = np.asarray(scores, dtype=float)
    return np.array([1 + int(np.sum(scores > s)) for s in scores], dtype=int)


def rrf_scores(*ranks: np.ndarray) -> np.ndarray:
    """Reciprocal Rank Fusion: the sum over rankings of 1 / (RRF_K + rank)."""
    return sum(1.0 / (RRF_K + np.asarray(r, dtype=float)) for r in ranks)


def bm25_scores(question: str, chunks: list[str]) -> np.ndarray:
    """Okapi BM25 of each chunk for the question's distinct terms, with Lucene's non-negative IDF."""
    vectorizer = CountVectorizer(stop_words="english")
    try:
        counts = vectorizer.fit_transform(chunks).tocsc()
    except ValueError:  # no chunk has a term
        return np.zeros(len(chunks))
    vocabulary = vectorizer.vocabulary_
    terms = {t for t in vectorizer.build_analyzer()(question) if t in vocabulary}
    scores = np.zeros(len(chunks))
    if not terms:
        return scores
    doc_len = np.asarray(counts.sum(axis=1)).ravel()
    norm = BM25_K1 * (1 - BM25_B + BM25_B * doc_len / doc_len.mean())
    for term in terms:
        tf = counts[:, vocabulary[term]].toarray().ravel()
        df = np.count_nonzero(tf)
        idf = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
        scores += idf * tf * (BM25_K1 + 1) / (tf + norm)
    return scores


def _cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine of `query` against each row of `matrix`; a zero vector scores 0."""
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    dots = matrix @ query
    return np.divide(dots, norms, out=np.zeros(len(matrix)), where=norms > 0)


def _chunk(text: str, config: RAGConfig, encode: Callable, decode: Callable) -> tuple[list, list, list[str]]:
    """The tokens of `text`, its chunk spans and the decoded chunks."""
    tokens = encode(text)
    spans = chunk_spans(len(tokens), config.chunk_size, config.chunk_overlap)
    return tokens, spans, [decode(tokens[a:b]) for a, b in spans]


async def retrieve_context(text: str, question: str, needle: str, config: RAGConfig,
                           encode: Callable, decode: Callable, embed: Callable,
                           encode_in_thread: bool = False) -> tuple[str, dict]:
    """
    Return the top-k hybrid-ranked chunks of `text` as context, and a log of the retrieval. With `encode_in_thread`,
    tokenizing and chunking run in a worker thread, so that concurrent requests tokenize in parallel; only pass it
    for a thread-safe tokenizer that releases the GIL (`APIConnector.encode_in_thread`).
    """
    if encode_in_thread:
        tokens, spans, chunks = await asyncio.to_thread(_chunk, text, config, encode, decode)
    else:
        tokens, spans, chunks = _chunk(text, config, encode, decode)
    ranked = len(chunks) > config.top_k
    if ranked:
        vectors, usage = await embed([question] + chunks)
        vectors = np.asarray(vectors, dtype=float)
        bm25 = bm25_scores(question, chunks)
        dense = _cosine(vectors[0], vectors[1:])
    else:
        # Every chunk is retrieved whatever the scores: skip the embedding request. Ranks are chunk positions,
        # all within top_k, so the needle counts as retrieved under every ranking, as it would after scoring.
        usage = {"texts_sent": 0, "cache_hits": 0, "tokens": 0, "cost": 0.0}
        bm25 = dense = np.zeros(len(chunks))
    hybrid = ordinal_ranks(rrf_scores(tied_ranks(bm25), tied_ranks(dense)))

    retrieved = [int(i) for i in np.argsort(hybrid)[:config.top_k]]
    merged = merge_spans([spans[i] for i in retrieved])
    if merged == [(0, len(tokens))]:
        context = text
    else:
        context = CHUNK_SEPARATOR.join(decode(tokens[a:b]) for a, b in merged)

    needle_chunks = [i for i, chunk in enumerate(chunks) if needle in chunk]
    rankings = {"hybrid": hybrid, "bm25": ordinal_ranks(bm25), "dense": ordinal_ranks(dense)}
    needle_rank = {name: int(min(r[needle_chunks])) if needle_chunks else None for name, r in rankings.items()}
    info = {
        "n_chunks": len(chunks),
        "ranked": ranked,
        "retrieved": retrieved,
        "needle_chunks": needle_chunks,
        "needle_rank": needle_rank,
        "needle_retrieved": needle_rank["hybrid"] is not None and needle_rank["hybrid"] <= config.top_k,
        "context_tokens": sum(b - a for a, b in merged),
        "embedding": usage,
    }
    return context, info
