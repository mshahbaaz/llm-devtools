from unittest.mock import MagicMock

from common.model_client import ModelResponse
from git_why.explain import explain_diff


def test_explain_diff_calls_model_with_diff_and_parses_sections():
    mock_client = MagicMock()
    mock_client.complete.return_value = ModelResponse(
        text=(
            "SUMMARY: Adds input validation to the login form.\n"
            "WHY: Prevents empty-password submissions reported in issue #42.\n"
            "RISK: Touches auth flow; verify existing login tests still pass.\n"
            "COMMIT_MESSAGE: fix(auth): reject empty passwords on login"
        ),
        model="openai/gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=40,
        latency_seconds=1.2,
    )

    explanation = explain_diff(mock_client, diff_text="diff --git a/login.py...", model="openai/gpt-4o-mini")

    mock_client.complete.assert_called_once()
    call_kwargs = mock_client.complete.call_args.kwargs
    assert "diff --git a/login.py" in call_kwargs["prompt"]

    assert explanation.summary == "Adds input validation to the login form."
    assert "issue #42" in explanation.why
    assert "auth flow" in explanation.risk
    assert explanation.commit_message == "fix(auth): reject empty passwords on login"


def test_explain_diff_includes_test_references_in_prompt_when_provided():
    mock_client = MagicMock()
    mock_client.complete.return_value = ModelResponse(
        text="SUMMARY: s\nWHY: w\nRISK: r\nCOMMIT_MESSAGE: c",
        model="openai/gpt-4o-mini",
        prompt_tokens=1,
        completion_tokens=1,
        latency_seconds=0.1,
    )

    explain_diff(
        mock_client,
        diff_text="diff --git a/auth.py...",
        model="openai/gpt-4o-mini",
        test_references=["test_auth.py"],
    )

    prompt = mock_client.complete.call_args.kwargs["prompt"]
    assert "test_auth.py" in prompt
