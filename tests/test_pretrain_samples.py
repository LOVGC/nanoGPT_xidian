from __future__ import annotations

from pathlib import Path
import sys

import torch
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model


def generate_sample(
    model_instance: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -block_size:]
        logits = model_instance(input_cond)
        logits = logits[:, -1, :]
        logits = logits / max(temperature, 1e-6)
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
    return enc.decode(response_ids).strip()


def main() -> None:
    checkpoint_path = ROOT / "model_trained" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = model.ModelArgs(**checkpoint["model_args"])
    model_instance = model.Transformer(model_args).to(device)
    model_instance.load_state_dict(checkpoint["model_state_dict"])
    model_instance.eval()

    enc = tiktoken.get_encoding("gpt2")

    max_new_tokens = 200
    block_size = model_args.max_seq_len
    temperature = 0.8
    top_k = 50

    prompts = [
        "Once upon a time",
        "In a small village",
    ]

    for idx, prompt in enumerate(prompts, start=1):
        print(f"=== PROMPT {idx} ===")
        print(prompt)

        prompt_ids = enc.encode(prompt)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(
            0
        )

        with torch.no_grad():
            output_ids = generate_sample(
                model_instance,
                input_ids,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                temperature=temperature,
                top_k=top_k,
            )

        response_ids = output_ids[0, len(prompt_ids) :].tolist()
        response_text = decode_response(enc, response_ids)

        print(f"=== OUTPUT {idx} ===")
        print(response_text)
        print("=== END ===\n")


if __name__ == "__main__":
    main()
