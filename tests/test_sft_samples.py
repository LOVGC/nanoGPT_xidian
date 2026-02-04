from __future__ import annotations

from pathlib import Path
import sys
import json
import random

import torch
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model


def format_sft_prompt(instruction: str, input_text: str) -> str:
    if input_text:
        return (
            "### Instruction:\n"
            + instruction
            + "\n### Input:\n"
            + input_text
            + "\n### Response:\n"
        )
    return "### Instruction:\n" + instruction + "\n### Response:\n"


def generate_sample(
    model_instance: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    top_k: int,
    eot_id: int,
    newline_ids: list[int],
    min_tokens_before_eot: int,
) -> torch.Tensor:
    for step in range(max_new_tokens):
        input_cond = input_ids[:, -block_size:]
        logits = model_instance(input_cond)
        logits = logits[:, -1, :]
        logits = logits / max(temperature, 1e-6)
        if step < min_tokens_before_eot and 0 <= eot_id < logits.shape[-1]:
            logits[:, eot_id] = -float("inf")
        if step < min_tokens_before_eot and newline_ids:
            for token_id in newline_ids:
                if 0 <= token_id < logits.shape[-1]:
                    logits[:, token_id] = -float("inf")
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_values, torch.full_like(logits, -float("inf")), logits
            )
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids


def decode_response(enc: tiktoken.Encoding, response_ids: list[int]) -> str:
    eot_id = enc.eot_token
    if eot_id in response_ids:
        response_ids = response_ids[: response_ids.index(eot_id)]
    return enc.decode(response_ids)


def load_sft_samples(data_path: Path, count: int, seed: int) -> list[dict[str, str]]:
    if not data_path.exists():
        raise FileNotFoundError(f"missing SFT data: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    filtered = []
    for record in records:
        instruction = record.get("instruction", "").strip()
        output_text = record.get("output", "").strip()
        if not instruction or not output_text:
            continue
        filtered.append(record)

    if len(filtered) <= count:
        return filtered

    rng = random.Random(seed)
    rng.shuffle(filtered)
    return filtered[:count]


def main() -> None:
    checkpoint_path = ROOT / "model_trained" / "sft_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = model.ModelArgs(**checkpoint["model_args"])
    model_instance = model.Transformer(model_args).to(device)
    model_instance.load_state_dict(checkpoint["model_state_dict"])
    model_instance.eval()

    enc = tiktoken.get_encoding("gpt2")

    max_new_tokens = 128
    block_size = model_args.max_seq_len
    temperature = 0.7
    top_k = 50
    min_tokens_before_eot = 5
    newline_ids = enc.encode("\n")
    samples = load_sft_samples(ROOT / "data" / "sft_data_en.jsonl", count=3, seed=42)

    for idx, sample in enumerate(samples, start=1):
        instruction = sample.get("instruction", "").strip()
        input_text = sample.get("input", "").strip()
        expected_output = sample.get("output", "").strip()

        prompt = format_sft_prompt(instruction, input_text)
        print(f"=== SAMPLE {idx} PROMPT ===")
        print(prompt)

        prompt_ids = enc.encode(prompt)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(
            0
        )

        with torch.no_grad():
            logits = model_instance(input_ids)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            top_values, top_indices = torch.topk(next_logits, k=5, dim=-1)
            print(f"=== SAMPLE {idx} TOP-5 TOKENS ===")
            for score, token_id in zip(top_values[0].tolist(), top_indices[0].tolist()):
                token_text = enc.decode([token_id])
                print(f"{token_id}: {repr(token_text)} ({score:.4f})")

            output_ids = generate_sample(
                model_instance,
                input_ids,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                temperature=temperature,
                top_k=top_k,
                eot_id=enc.eot_token,
                newline_ids=newline_ids,
                min_tokens_before_eot=min_tokens_before_eot,
            )

        response_ids = output_ids[0, len(prompt_ids) :].tolist()
        response_text = decode_response(enc, response_ids)

        print(f"=== SAMPLE {idx} EXPECTED ===")
        print(expected_output)
        print(f"=== SAMPLE {idx} OUTPUT ===")
        print(response_text)
        print(f"=== SAMPLE {idx} OUTPUT (repr) ===")
        print(repr(response_text))
        print("=== END ===\n")


if __name__ == "__main__":
    main()
