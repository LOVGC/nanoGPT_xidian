from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model
from train import SFT_CONFIG, build_sft_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MoE expert load")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model_trained/sft_best.pt",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/sft_data_en.jsonl",
    )
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--plot_dir", type=str, default="plots")
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def compute_entropy(probs: torch.Tensor) -> float:
    eps = 1e-12
    probs = probs.clamp(min=eps)
    return float(-(probs * probs.log()).sum().item())


def plot_heatmap(ratios: torch.Tensor, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(ratios, aspect="auto", cmap="viridis")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Layer")
    ax.set_title("MoE Expert Selection Ratio (Top-2)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overall_bar(ratios: torch.Tensor, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(ratios)), ratios.cpu().tolist())
    ax.set_xlabel("Expert")
    ax.set_ylabel("Selection Ratio")
    ax.set_title("Overall MoE Expert Selection Ratio")
    ax.set_ylim(0, max(0.05, float(ratios.max().item()) * 1.2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"missing data file: {data_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = model.ModelArgs(**checkpoint["model_args"])
    if model_args.n_expert <= 1:
        print("MoE is disabled (n_expert <= 1).")
        return

    model_instance = model.Transformer(model_args).to(device)
    model_instance.load_state_dict(checkpoint["model_state_dict"])
    model_instance.eval()

    hp = replace(
        SFT_CONFIG, block_size=min(SFT_CONFIG.block_size, model_args.max_seq_len)
    )
    _, val_loader = build_sft_dataloaders(data_path, hp, seed=1337)

    num_layers = model_args.n_layer
    num_experts = model_args.n_expert
    top_k = model_args.moe_top_k

    counts = torch.zeros((num_layers, num_experts), dtype=torch.long)
    weight_sums = torch.zeros((num_layers, num_experts), dtype=torch.float32)
    total_tokens = torch.zeros(num_layers, dtype=torch.long)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            input_ids, _, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            _, _, router_logits = model_instance(
                input_ids,
                attention_mask=attention_mask,
                return_router_logits=True,
            )

            valid_mask = attention_mask.reshape(-1) == 1
            for layer_idx, layer_logits in enumerate(router_logits):
                if layer_logits is None:
                    continue
                logits = layer_logits[valid_mask]
                if logits.numel() == 0:
                    continue
                probs = torch.softmax(logits.float(), dim=-1)
                topk = torch.topk(probs, k=top_k, dim=-1)
                topk_indices = topk.indices.reshape(-1).cpu()
                topk_weights = topk.values.reshape(-1).cpu()

                layer_counts = torch.bincount(topk_indices, minlength=num_experts)
                counts[layer_idx] += layer_counts
                weight_sums[layer_idx] += torch.bincount(
                    topk_indices, weights=topk_weights, minlength=num_experts
                )
                total_tokens[layer_idx] += logits.shape[0]

    selection_ratios = counts.float() / (total_tokens.float().unsqueeze(-1) * top_k)
    selection_ratios = torch.nan_to_num(selection_ratios, nan=0.0)

    overall_counts = counts.sum(dim=0)
    overall_ratio = overall_counts.float() / overall_counts.sum().clamp(min=1)

    print("\nMoE expert load summary")
    for layer_idx in range(num_layers):
        if total_tokens[layer_idx] == 0:
            continue
        layer_ratio = selection_ratios[layer_idx]
        min_ratio = float(layer_ratio.min().item())
        max_ratio = float(layer_ratio.max().item())
        entropy = compute_entropy(layer_ratio)
        ratio_span = max_ratio / max(min_ratio, 1e-9)
        print(
            f"Layer {layer_idx}: min={min_ratio:.4f}, max={max_ratio:.4f}, "
            f"entropy={entropy:.4f}, max/min={ratio_span:.2f}"
        )

    collapse_threshold = 0.01
    collapsed = (overall_ratio < collapse_threshold).sum().item()
    print(f"Overall: experts below {collapse_threshold:.2%} = {collapsed}")

    plot_dir = Path(args.plot_dir)
    plot_heatmap(selection_ratios, plot_dir / "moe_expert_load.png")
    plot_overall_bar(overall_ratio, plot_dir / "moe_expert_load_overall.png")
    print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
