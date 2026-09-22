from rag_bench.metrics import recall_at_k, mean_reciprocal_rank


def test_recall_at_k_hit():
    retrieved_ids = ["c3", "c1", "c5"]
    assert recall_at_k(retrieved_ids, relevant_id="c1", k=3) == 1.0
    assert recall_at_k(retrieved_ids, relevant_id="c1", k=1) == 0.0


def test_mean_reciprocal_rank():
    # relevant doc is rank 2 (1-indexed) -> RR = 1/2
    runs = [(["c3", "c1", "c5"], "c1"), (["c2", "c4"], "c9")]  # second run: not found -> RR 0
    mrr = mean_reciprocal_rank(runs)
    assert abs(mrr - 0.25) < 1e-9  # (0.5 + 0.0) / 2
