from __future__ import annotations

import os

import gradio as gr

from common.model_client import ModelClient
from git_why.explain import explain_diff

DEMO_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
RATE_LIMIT_NOTE = (
    "Public demo, runs on a free rate-limited model via OpenRouter. "
    "For real use, `pip install llm-devtools` and run `git-why` locally with your own key."
)


def run_demo(diff_text: str) -> str:
    if not diff_text.strip():
        return "Paste a `git diff` / `git show` output above first."
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "Demo is misconfigured (missing server-side API key). Try the CLI instead."

    client = ModelClient(api_key=api_key)
    try:
        explanation = explain_diff(client, diff_text, model=DEMO_MODEL)
    except Exception as exc:
        return f"Error: {exc}"

    return (
        f"### Summary\n{explanation.summary}\n\n"
        f"### Why\n{explanation.why}\n\n"
        f"### Risk\n{explanation.risk}\n\n"
        f"### Suggested commit message\n`{explanation.commit_message}`"
    )


demo = gr.Interface(
    fn=run_demo,
    inputs=gr.Textbox(lines=15, label="Paste diff output (git diff / git show)"),
    outputs=gr.Markdown(label="Explanation"),
    title="git-why",
    description=f"Turn a diff into a plain-English explanation. {RATE_LIMIT_NOTE}",
)

if __name__ == "__main__":
    demo.queue(max_size=10).launch()
