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
