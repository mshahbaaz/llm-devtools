from __future__ import annotations

import os

import gradio as gr

from common.model_client import ModelClient
from common.report import render_html
from promptbench.config import EvalConfig, TestCase
from promptbench.runner import run_eval

DEMO_MODELS = ["meta-llama/llama-3.1-8b-instruct:free", "openai/gpt-4o-mini"]


def run_demo(prompt_template: str, cases_csv: str, scorer: str) -> str:
    if "{input}" not in prompt_template:
        return "<p>prompt_template must contain an <code>{input}</code> placeholder.</p>"

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "<p>Demo is misconfigured (missing server-side API key).</p>"

    cases = []
    for i, line in enumerate(cases_csv.strip().splitlines()):
        if "," not in line:
            continue
        input_text, expected = line.split(",", 1)
        cases.append(TestCase(id=f"c{i}", input=input_text.strip(), expected=expected.strip()))

    if not cases:
        return "<p>Add at least one line as <code>input,expected</code>.</p>"

    config = EvalConfig(models=DEMO_MODELS, prompt_template=prompt_template, scorer=scorer, cases=cases)
    client = ModelClient(api_key=api_key)
    result = run_eval(client, config)
    return render_html(result)


demo = gr.Interface(
    fn=run_demo,
    inputs=[
        gr.Textbox(label="Prompt template (must include {input})", value="Classify sentiment as positive/negative: {input}"),
        gr.Textbox(lines=6, label="Test cases, one per line: input,expected", value="I love this,positive\nThis is terrible,negative"),
        gr.Dropdown(choices=["exact_match", "regex_match"], value="exact_match", label="Scorer"),
    ],
    outputs=gr.HTML(label="Results"),
    title="promptbench",
    description="Evaluate a prompt against a couple of free/cheap models via OpenRouter.",
)

if __name__ == "__main__":
    demo.queue(max_size=10).launch()
