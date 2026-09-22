from __future__ import annotations

from pydantic import BaseModel

from common.model_client import ModelClient

SYSTEM_PROMPT = (
    "You are a senior engineer explaining a code change to a teammate. "
    "Be precise and concrete; refer to actual function/file names from the diff. "
    "Respond in EXACTLY this format, one line per field:\n"
    "SUMMARY: <one sentence, what changed>\n"
    "WHY: <one sentence, the most likely intent, inferred from the diff itself>\n"
    "RISK: <one sentence on what could break, or 'Low risk' if the change is trivial>\n"
    "COMMIT_MESSAGE: <a single conventional-commit-style line>"
)

MAX_DIFF_CHARS = 12000


class DiffExplanation(BaseModel):
    summary: str
    why: str
    risk: str
    commit_message: str
    raw: str


def _truncate(diff_text: str) -> str:
    if len(diff_text) <= MAX_DIFF_CHARS:
        return diff_text
    return diff_text[:MAX_DIFF_CHARS] + "\n... [diff truncated for length]"


def _parse(raw: str) -> DiffExplanation:
    fields = {"SUMMARY": "", "WHY": "", "RISK": "", "COMMIT_MESSAGE": ""}
    for line in raw.splitlines():
        for key in fields:
            prefix = f"{key}:"
            if line.strip().startswith(prefix):
                fields[key] = line.strip()[len(prefix):].strip()
    return DiffExplanation(
        summary=fields["SUMMARY"] or "(no summary returned)",
        why=fields["WHY"] or "(no rationale returned)",
        risk=fields["RISK"] or "(no risk assessment returned)",
        commit_message=fields["COMMIT_MESSAGE"] or "(no commit message returned)",
        raw=raw,
    )


def explain_diff(client: ModelClient, diff_text: str, model: str, test_references: list[str] | None = None) -> DiffExplanation:
    if not diff_text.strip():
        raise ValueError("No diff content to explain (empty diff).")

    prompt = f"Explain this diff:\n\n{_truncate(diff_text)}"
    if test_references:
        prompt += f"\n\nFiles that reference the changed function(s), for risk assessment: {', '.join(test_references)}"
    else:
        prompt += "\n\nNo files were found referencing the changed function(s). Factor this into RISK."

    response = client.complete(model=model, prompt=prompt, system=SYSTEM_PROMPT, temperature=0.0)
    return _parse(response.text)
