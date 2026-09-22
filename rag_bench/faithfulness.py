from __future__ import annotations

from common.model_client import ModelClient

JUDGE_SYSTEM_PROMPT = (
    "You check whether an answer is fully supported by the given context (no invented facts). "
    "Respond in EXACTLY this format:\n"
    "VERDICT: <grounded or hallucinated>\n"
    "REASON: <one sentence>"
)


def score_faithfulness(client: ModelClient, answer: str, context: list[str], judge_model: str) -> float:
    context_block = "\n---\n".join(context)
    prompt = f"Context:\n{context_block}\n\nAnswer to check:\n{answer}"
    response = client.complete(model=judge_model, prompt=prompt, system=JUDGE_SYSTEM_PROMPT, temperature=0.0)
    for line in response.text.splitlines():
        if line.strip().startswith("VERDICT:"):
            verdict = line.strip()[len("VERDICT:"):].strip().lower()
            return 1.0 if verdict == "grounded" else 0.0
    return 0.0
