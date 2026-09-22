from unittest.mock import MagicMock

from common.model_client import ModelResponse
from rag_bench.faithfulness import score_faithfulness


def test_score_faithfulness_parses_verdict():
    mock_client = MagicMock()
    mock_client.complete.return_value = ModelResponse(
        text="VERDICT: grounded\nREASON: The answer only uses facts present in the retrieved context.",
        model="openai/gpt-4o-mini",
        prompt_tokens=80,
        completion_tokens=20,
        latency_seconds=0.6,
    )
    score = score_faithfulness(
        mock_client,
        answer="The Eiffel Tower is in Paris.",
        context=["The Eiffel Tower is located in Paris, France."],
        judge_model="openai/gpt-4o-mini",
    )
    assert score == 1.0


def test_score_faithfulness_flags_hallucination():
    mock_client = MagicMock()
    mock_client.complete.return_value = ModelResponse(
        text="VERDICT: hallucinated\nREASON: The answer states a fact not present in the context.",
        model="openai/gpt-4o-mini",
        prompt_tokens=80,
        completion_tokens=20,
        latency_seconds=0.6,
    )
    score = score_faithfulness(mock_client, answer="It was built in 1600.", context=["The Eiffel Tower was completed in 1889."], judge_model="openai/gpt-4o-mini")
    assert score == 0.0
