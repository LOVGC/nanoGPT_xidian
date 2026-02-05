from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TRAIN_LOG_DIR = ROOT / "training_log"
PLOTS_DIR = ROOT / "plots"


def plot_loss(csv_path: Path, out_path: Path, title: str) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"missing log file: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"empty log file: {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 5))

    for split, label in [("train", "Train"), ("eval", "Eval")]:
        subset = df[df["split"] == split]
        if subset.empty:
            continue
        subset = subset.sort_values("step")
        ax.plot(subset["step"], subset["loss"], label=label)

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pretrain_csv = TRAIN_LOG_DIR / "pretrain_loss.csv"
    sft_csv = TRAIN_LOG_DIR / "sft_loss.csv"

    plot_loss(pretrain_csv, PLOTS_DIR / "pretrain_loss.png", "Pretrain Loss")
    plot_loss(sft_csv, PLOTS_DIR / "sft_loss.png", "SFT Loss")

    print(f"Saved: {PLOTS_DIR / 'pretrain_loss.png'}")
    print(f"Saved: {PLOTS_DIR / 'sft_loss.png'}")


if __name__ == "__main__":
    main()
