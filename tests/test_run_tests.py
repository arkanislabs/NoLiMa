from run_tests import EmbeddingTally


def log(tokens, hits, cost, sent=1, ranked=True):
    return {"ranked": ranked, "embedding": {"texts_sent": sent, "cache_hits": hits, "tokens": tokens, "cost": cost}}


def test_embedding_tally_is_silent_until_something_is_embedded():
    tally = EmbeddingTally()
    tally.add(log(0, 0, 0.0, sent=0, ranked=False))
    assert tally.postfix() == ""
    assert tally.summary() == "none needed (every chunk retrieved in all 1 requests)"


def test_embedding_tally_sums_tokens_cost_and_cache_hits():
    tally = EmbeddingTally()
    tally.add(log(1000, 2, 0.00002))
    tally.add(log(5000, 0, 0.0001))
    tally.add(log(0, 0, 0.0, sent=0, ranked=False))
    assert tally.postfix() == " | embeddings $0.0001 (6,000 tokens)"
    assert tally.summary() == ("6,000 tokens sent, cost $0.0001, 2 texts from the embedding cache, "
                               "1 requests not embedded (every chunk retrieved)")
    tally.add(log(10, 0, None))
    assert "+ unreported" in tally.postfix()
