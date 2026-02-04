from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset


TARGET_SIZE = 800
RNG = random.Random(42)


def contains_cjk(text: str) -> bool:
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
            return True
    return False


def is_english(text: str) -> bool:
    return text and not contains_cjk(text)


def load_streaming_dataset(name: str, split: str = "train"):
    try:
        return load_dataset(name, split=split, streaming=True)
    except Exception as exc:
        print(f"warning: failed to load {name}: {exc}")
        return None


def take_samples(dataset, max_items: int, seed: int) -> list[dict]:
    if dataset is None:
        return []
    try:
        dataset = dataset.shuffle(buffer_size=10000, seed=seed)
    except Exception:
        pass
    samples = []
    for item in dataset:
        samples.append(item)
        if len(samples) >= max_items:
            break
    return samples


def sample_alpaca(max_items: int) -> list[dict[str, str]]:
    raw = load_streaming_dataset("tatsu-lab/alpaca")
    items = []
    for record in take_samples(raw, max_items * 2, seed=1):
        instruction = record.get("instruction", "").strip()
        input_text = record.get("input", "").strip()
        output_text = record.get("output", "").strip()
        combined = instruction + " " + input_text + " " + output_text
        if not is_english(combined):
            continue
        if not instruction or not output_text:
            continue
        items.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
            }
        )
        if len(items) >= max_items:
            break
    return items


def sample_dolly(max_items: int) -> list[dict[str, str]]:
    raw = load_streaming_dataset("databricks/databricks-dolly-15k")
    items = []
    for record in take_samples(raw, max_items * 2, seed=2):
        instruction = record.get("instruction", "").strip()
        input_text = record.get("context", "").strip()
        output_text = record.get("response", "").strip()
        combined = instruction + " " + input_text + " " + output_text
        if not is_english(combined):
            continue
        if not instruction or not output_text:
            continue
        items.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
            }
        )
        if len(items) >= max_items:
            break
    return items


def sample_squad(max_items: int) -> list[dict[str, str]]:
    raw = load_streaming_dataset("squad")
    items = []
    instruction = (
        "Answer the question using the context. "
        'If the answer is not in the context, say "I don\'t know."'
    )
    for record in take_samples(raw, max_items * 2, seed=3):
        context = record.get("context", "").strip()
        question = record.get("question", "").strip()
        answers = record.get("answers", {}).get("text", [])
        if not answers:
            continue
        answer = str(answers[0]).strip()
        combined = context + " " + question + " " + answer
        if not is_english(combined):
            continue
        input_text = f"Context: {context}\nQuestion: {question}"
        items.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": answer,
            }
        )
        if len(items) >= max_items:
            break
    return items


def sample_squad_unanswerable(max_items: int) -> list[dict[str, str]]:
    raw = load_streaming_dataset("squad_v2")
    items = []
    instruction = (
        "Answer the question using the context. "
        'If the answer is not in the context, say "I don\'t know."'
    )
    for record in take_samples(raw, max_items * 3, seed=4):
        if not record.get("is_impossible", False):
            continue
        context = record.get("context", "").strip()
        question = record.get("question", "").strip()
        combined = context + " " + question
        if not is_english(combined):
            continue
        input_text = f"Context: {context}\nQuestion: {question}"
        items.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": "I don't know.",
            }
        )
        if len(items) >= max_items:
            break
    return items


def fallback_examples() -> list[dict[str, str]]:
    return [
        {
            "instruction": "Summarize the text in one sentence.",
            "input": "A storm knocked out power across the town. Residents used candles and power returned by morning.",
            "output": "A storm cut power overnight and electricity returned by morning.",
        },
        {
            "instruction": "Rewrite to be more formal.",
            "input": "Can you fix this ASAP?",
            "output": "Could you please resolve this as soon as possible?",
        },
        {
            "instruction": "Extract the action items.",
            "input": "We should review the draft, send feedback by Friday, and schedule a follow-up call.",
            "output": "Review the draft; send feedback by Friday; schedule a follow-up call.",
        },
        {
            "instruction": 'Answer the question using the context. If the answer is not in the context, say "I don\'t know."',
            "input": "Context: The library opens at 9 a.m. on weekdays.\nQuestion: What time does the library open on weekdays?",
            "output": "It opens at 9 a.m. on weekdays.",
        },
        {
            "instruction": 'Answer the question using the context. If the answer is not in the context, say "I don\'t know."',
            "input": "Context: The meeting is on Tuesday in Room 204.\nQuestion: What time is the meeting?",
            "output": "I don't know.",
        },
        {
            "instruction": "Translate to English.",
            "input": "Bonjour",
            "output": "Hello.",
        },
        {
            "instruction": "Provide a short, helpful response.",
            "input": "I missed the deadline. What should I do?",
            "output": "Let your team know promptly, explain the delay briefly, and propose a new timeline.",
        },
        {
            "instruction": "Classify the sentiment as positive, negative, or neutral.",
            "input": "The service was slow and the food was cold.",
            "output": "negative",
        },
        {
            "instruction": "Generate three bullet points.",
            "input": "Tips for staying focused while studying.",
            "output": "- Remove distractions\n- Use timed study sessions\n- Take short breaks",
        },
        {
            "instruction": "Answer the question.",
            "input": "What is the capital of Japan?",
            "output": "Tokyo.",
        },
    ]


def build_dataset() -> list[dict[str, str]]:
    dataset: list[dict[str, str]] = []

    dataset.extend(sample_alpaca(320))
    dataset.extend(sample_dolly(200))
    dataset.extend(sample_squad(200))
    dataset.extend(sample_squad_unanswerable(80))

    if len(dataset) < 500:
        dataset.extend(fallback_examples())

    RNG.shuffle(dataset)

    if len(dataset) < 500:
        raise RuntimeError(f"dataset too small: {len(dataset)}")
    if len(dataset) > 1000:
        dataset = dataset[:1000]
    if len(dataset) > TARGET_SIZE:
        dataset = dataset[:TARGET_SIZE]
    return dataset


def main() -> None:
    output_path = Path("data") / "sft_data_en.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset()
    with output_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"wrote {len(dataset)} records to {output_path}")


if __name__ == "__main__":
    main()
