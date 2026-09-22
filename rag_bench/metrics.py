from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], relevant_id: str, k: int) -> float:
    return 1.0 if relevant_id in retrieved_ids[:k] else 0.0


def mean_reciprocal_rank(runs: list[tuple[list[str], str]]) -> float:
    reciprocal_ranks = []
    for retrieved_ids, relevant_id in runs:
        if relevant_id in retrieved_ids:
            rank = retrieved_ids.index(relevant_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
