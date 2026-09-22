from __future__ import annotations

import time

import httpx
from pydantic import BaseModel

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class ModelResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float


class ModelClient:
    """Thin wrapper over the OpenRouter chat completions API.

    One client, any model string OpenRouter supports (e.g. "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku", "meta-llama/llama-3.1-8b-instruct:free").
    """

    def __init__(self, api_key: str, timeout: float = 60.0, referer: str | None = None, tracer=None, tool_name: str = "unknown"):
        self._api_key = api_key
        self._timeout = timeout
        self._referer = referer or "https://github.com/mshahbaaz/llm-devtools"
        self._tracer = tracer
        self._tool_name = tool_name

    def complete(self, model: str, prompt: str, system: str | None = None, temperature: float = 0.0) -> ModelResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": self._referer,
            "X-Title": "llm-devtools",
        }
        payload = {"model": model, "messages": messages, "temperature": temperature}

        start = time.monotonic()
        response = httpx.post(OPENROUTER_URL, json=payload, headers=headers, timeout=self._timeout)
        latency = time.monotonic() - start

        if response.status_code != 200:
            detail = response.json().get("error", {}).get("message", response.text)
            raise RuntimeError(f"OpenRouter request failed ({response.status_code}): {detail}")

        data = response.json()
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        result = ModelResponse(
            text=choice,
            model=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_seconds=latency,
        )

        if self._tracer is not None:
            self._tracer.log(
                tool=self._tool_name,
                model=result.model,
                prompt=prompt,
                response_text=result.text,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                latency_seconds=result.latency_seconds,
                cost_usd=0.0,  # cost is estimated by callers (see promptbench.runner._estimate_cost)
            )
        return result
