from run_tests import EmbeddingProgress


def log(tokens, hits, cost, sent=1, ranked=True):
    return {"ranked": ranked, "embedding": {"texts_sent": sent, "cache_hits": hits, "tokens": tokens, "cost": cost}}


def test_embedding_progress_sums_tokens_cost_and_cache_hits():
    progress = EmbeddingProgress(total=3, desc="CL4K embeddings")
    progress.add(log(1000, 2, 0.00002))
    progress.add(log(500, 0, 0.00001))
    progress.add(log(0, 0, 0.0, sent=0, ranked=False))
    assert progress.bar.n == 3
    assert progress.summary() == ("1,500 tokens sent, cost $0.0000, 2 texts from the embedding cache, "
                                  "1 requests not embedded (every chunk retrieved)")
    progress.add(log(10, 0, None))
    assert "+ unreported" in progress.summary()
    progress.close()
