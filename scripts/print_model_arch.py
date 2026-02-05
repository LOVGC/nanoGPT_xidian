from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import torch
from torchinfo import summary

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


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

    total_params, trainable_params = count_parameters(model_instance)

    print("=== ModelArgs ===")
    print(asdict(model_args))
    print("\n=== Module Tree ===")
    print(model_instance)
    print("\n=== Parameter Counts ===")
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")

    print("\n=== Architecture Summary ===")
    input_ids = torch.randint(0, model_args.vocab_size, (2, 16), device=device)
    print(
        summary(
            model_instance,
            input_data=(input_ids,),
            verbose=0,
            row_settings=("ascii_only",),
        )
    )

    print("\n=== MoE Configuration ===")
    print(f"n_expert: {model_args.n_expert}")
    print(f"moe_top_k: {model_args.moe_top_k}")
    print(f"moe_jitter: {model_args.moe_jitter}")


if __name__ == "__main__":
    main()
