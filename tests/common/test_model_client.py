import httpx
import pytest
import respx

from common.model_client import ModelClient, ModelResponse


@respx.mock
def test_complete_returns_text_and_usage():
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello world"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                "model": "openai/gpt-4o-mini",
            },
        )
    )
    client = ModelClient(api_key="test-key")
    resp = client.complete(model="openai/gpt-4o-mini", prompt="say hello")

    assert isinstance(resp, ModelResponse)
    assert resp.text == "hello world"
    assert resp.prompt_tokens == 10
    assert resp.completion_tokens == 2
    assert resp.model == "openai/gpt-4o-mini"


@respx.mock
def test_complete_raises_on_http_error():
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "invalid key"}})
    )
    client = ModelClient(api_key="bad-key")
    with pytest.raises(RuntimeError, match="invalid key"):
        client.complete(model="openai/gpt-4o-mini", prompt="say hello")
