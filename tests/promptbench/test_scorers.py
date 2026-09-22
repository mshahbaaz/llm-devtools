from unittest.mock import MagicMock

from common.model_client import ModelResponse
from promptbench.scorers import exact_match, llm_judge, regex_match, get_scorer


def test_exact_match():
    assert exact_match(output="positive", expected="positive") == 1.0
    assert exact_match(output="Positive", expected="positive") == 1.0  # case-insensitive
    assert exact_match(output="negative", expected="positive") == 0.0


def test_regex_match():
    assert regex_match(output="The answer is 42.", expected=r"\b42\b") == 1.0
    assert regex_match(output="The answer is 7.", expected=r"\b42\b") == 0.0


def test_llm_judge_parses_score():
    mock_client = MagicMock()
    mock_client.complete.return_value = ModelResponse(
        text="SCORE: 0.8\nREASON: Mostly correct but missing nuance.",
        model="openai/gpt-4o-mini",
        prompt_tokens=50,
        completion_tokens=20,
        latency_seconds=0.5,
    )
    score = llm_judge(mock_client, output="Paris is the capital", expected="The capital of France is Paris", judge_model="openai/gpt-4o-mini")
    assert score == 0.8


def test_get_scorer_dispatches_by_name():
    assert get_scorer("exact_match") is exact_match
    assert get_scorer("regex_match") is regex_match
    try:
        get_scorer("not_a_scorer")
        assert False, "expected ValueError"
    except ValueError:
        pass
