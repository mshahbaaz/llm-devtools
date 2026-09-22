from unittest.mock import MagicMock

from common.model_client import ModelResponse
from promptbench.config import EvalConfig, TestCase
from promptbench.runner import run_eval


def test_run_eval_produces_one_result_per_case_per_model():
    config = EvalConfig(
        models=["openai/gpt-4o-mini"],
        prompt_template="Classify: {input}",
        scorer="exact_match",
        cases=[
            TestCase(id="c1", input="great!", expected="positive"),
            TestCase(id="c2", input="awful", expected="negative"),
        ],
    )
    mock_client = MagicMock()
    mock_client.complete.side_effect = [
        ModelResponse(text="positive", model="openai/gpt-4o-mini", prompt_tokens=5, completion_tokens=1, latency_seconds=0.1),
        ModelResponse(text="neutral", model="openai/gpt-4o-mini", prompt_tokens=5, completion_tokens=1, latency_seconds=0.1),
    ]

    run = run_eval(mock_client, config)

    assert len(run.results) == 2
    assert run.results[0].passed is True
    assert run.results[0].score == 1.0
    assert run.results[1].passed is False
    assert run.results[1].score == 0.0
    assert mock_client.complete.call_count == 2
    first_prompt = mock_client.complete.call_args_list[0].kwargs["prompt"]
    assert "great!" in first_prompt
