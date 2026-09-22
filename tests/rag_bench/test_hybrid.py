from rag_bench.hybrid import BM25Index, reciprocal_rank_fusion


def test_bm25_index_ranks_exact_keyword_matches_highest():
    index = BM25Index(ids=["a", "b", "c"], documents=[
        "The Eiffel Tower is in Paris.",
        "Mount Everest is the tallest mountain.",
        "Python is a programming language.",
    ])

    results = index.query("Eiffel Tower Paris", top_k=1)

    assert results[0] == "a"


def test_reciprocal_rank_fusion_merges_two_rankings():
    dense_ranked_ids = ["a", "b", "c"]
    sparse_ranked_ids = ["b", "a", "c"]

    fused = reciprocal_rank_fusion([dense_ranked_ids, sparse_ranked_ids], top_k=2)

    # "a" and "b" both rank in the top 2 of both lists -> both should beat "c"
    assert set(fused) == {"a", "b"}


def test_reciprocal_rank_fusion_rewards_consistent_high_rank():
    # "x" is #1 in both lists; "y" is #1 in one and absent from the other
    fused = reciprocal_rank_fusion([["x", "y"], ["x", "z"]], top_k=1)
    assert fused[0] == "x"
