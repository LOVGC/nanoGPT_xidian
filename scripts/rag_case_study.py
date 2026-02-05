from __future__ import annotations

from pathlib import Path
import sys

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import (
    DEFAULT_DOCS_DIR,
    DEFAULT_EMBED_MODEL,
    SYSTEM_PROMPT,
    build_user_message,
    build_vectorstore,
    call_ollama_generate,
    retrieve_context,
)


QUESTIONS = [
    "What is my student ID?",
    "Where do I study?",
    "What is my favorite topic?",
]

KB_TEXT = (
    "My name is Zhang San.\n"
    "I study at Xidian University.\n"
    "My student ID is 123.\n"
    "My favorite topic is RAG.\n"
)


def ensure_kb_file(docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    kb_path = docs_dir / "profile.txt"
    kb_path.write_text(KB_TEXT, encoding="utf-8")


def main() -> None:
    docs_dir = DEFAULT_DOCS_DIR
    index_dir = ROOT / "rag_index_case_study"

    ensure_kb_file(docs_dir)

    vectordb = build_vectorstore(
        docs_dir=docs_dir,
        index_dir=index_dir,
        rebuild=True,
        embed_model=DEFAULT_EMBED_MODEL,
    )

    enc = tiktoken.get_encoding("gpt2")
    max_input_tokens = 512

    rows = []
    for question in QUESTIONS:
        context_on = retrieve_context(vectordb, question, top_k=3)
        user_on = build_user_message(question, context_on, enc, max_input_tokens)
        prompt_on = f"System: {SYSTEM_PROMPT}\n\n{user_on}"
        answer_on = call_ollama_generate(
            base_url="http://localhost:11434",
            model_name="qwen3:4b",
            prompt=prompt_on,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
        )

        user_off = build_user_message(question, "", enc, max_input_tokens)
        prompt_off = f"System: {SYSTEM_PROMPT}\n\n{user_off}"
        answer_off = call_ollama_generate(
            base_url="http://localhost:11434",
            model_name="qwen3:4b",
            prompt=prompt_off,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
        )

        rows.append((question, answer_off, answer_on))

    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "rag_case_study.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# RAG Case Study (Ollama qwen3:4b)\n\n")
        f.write("| Question | RAG OFF | RAG ON |\n")
        f.write("|---|---|---|\n")
        for question, off, on in rows:
            off = off.replace("\n", "<br>")
            on = on.replace("\n", "<br>")
            f.write(f"| {question} | {off} | {on} |\n")

    print(f"Saved case study: {report_path}")


if __name__ == "__main__":
    main()
