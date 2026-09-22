from promptbench.stats import bootstrap_pass_rate_ci


def test_bootstrap_ci_is_tight_for_unanimous_results():
    results = [True] * 20
    low, high = bootstrap_pass_rate_ci(results, n_resamples=1000, seed=42)
    assert low == 1.0
    assert high == 1.0


def test_bootstrap_ci_widens_for_small_mixed_sample():
    results = [True, False, True]
    low, high = bootstrap_pass_rate_ci(results, n_resamples=1000, seed=42)
    assert 0.0 <= low <= high <= 1.0
    assert high - low > 0.2  # small n -> wide interval, not a false-precise point estimate


def test_bootstrap_ci_is_deterministic_given_a_seed():
    results = [True, False, True, True, False]
    first = bootstrap_pass_rate_ci(results, n_resamples=500, seed=7)
    second = bootstrap_pass_rate_ci(results, n_resamples=500, seed=7)
    assert first == second


def test_bootstrap_ci_handles_empty_results():
    assert bootstrap_pass_rate_ci([], n_resamples=100, seed=1) == (0.0, 0.0)
