from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr

from common.model_client import ModelClient
from rag_bench.config import QaCase, RagConfig
from rag_bench.pipeline import run_pipeline

DEMO_MODEL = "meta-llama/llama-3.1-8b-instruct:free"


def run_demo(corpus_text: str, question: str, expected_answer: str) -> str:
    if not corpus_text.strip() or not question.strip():
        return "Paste some corpus text and a question first."

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "Demo is misconfigured (missing server-side API key)."

    with tempfile.TemporaryDirectory() as tmp_dir:
        corpus_path = Path(tmp_dir) / "corpus.txt"
        corpus_path.write_text(corpus_text)

        config = RagConfig(
            corpus_paths=[corpus_path],
            chunk_size=50,
            overlap=10,
            top_k=2,
            generation_model=DEMO_MODEL,
            judge_model=DEMO_MODEL,
            cases=[QaCase(id="q1", question=question, relevant_doc_id="", expected_answer=expected_answer or question)],
        )
        client = ModelClient(api_key=api_key)
        try:
            results = run_pipeline(client, config)
        except Exception as exc:
            return f"Error: {exc}"

    r = results[0]
    return (
        f"**Answer:** {r.answer}\n\n"
        f"**Retrieved chunks:** {', '.join(r.retrieved_chunk_ids)}\n\n"
        f"**Faithfulness (grounded in context):** {r.faithfulness_score:.1f}\n\n"
        f"**Answer correctness vs expected:** {r.answer_score:.1f}"
    )


demo = gr.Interface(
    fn=run_demo,
    inputs=[
        gr.Textbox(lines=10, label="Corpus text (paste a paragraph or two)"),
        gr.Textbox(label="Question"),
        gr.Textbox(label="Expected answer (optional, for scoring)"),
    ],
    outputs=gr.Markdown(label="Result"),
    title="rag-bench",
    description="Runs retrieval + generation + faithfulness scoring on your own pasted text.",
)

if __name__ == "__main__":
    demo.queue(max_size=10).launch()
