from __future__ import annotations

import random


def bootstrap_pass_rate_ci(
    results: list[bool], n_resamples: int = 2000, confidence: float = 0.95, seed: int | None = None
) -> tuple[float, float]:
    """95%-style bootstrap confidence interval on a pass rate.

    Resamples `results` with replacement `n_resamples` times, computes the pass rate of
    each resample, and returns the (lower, upper) percentile bounds. This is what makes
    "62% pass rate" honest: with 8 test cases, that number alone hides how little signal
    it actually carries.
    """
    if not results:
        return (0.0, 0.0)

    rng = random.Random(seed)
    n = len(results)
    resampled_rates = []
    for _ in range(n_resamples):
        resample = [results[rng.randrange(n)] for _ in range(n)]
        resampled_rates.append(sum(resample) / n)

    resampled_rates.sort()
    alpha = (1 - confidence) / 2
    low_idx = int(alpha * n_resamples)
    high_idx = int((1 - alpha) * n_resamples) - 1
    return (resampled_rates[low_idx], resampled_rates[max(low_idx, high_idx)])
