from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path
import sys

import torch
import torch.nn as nn
import tiktoken
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model
from train import SFT_CONFIG, SFT_DATA_PATH, build_sft_dataloaders

REPORTS_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "model_trained"

SFT_CKPT = MODEL_DIR / "sft_best.pt"
SFT_INT8 = MODEL_DIR / "sft_best_int8.pt"
SFT_FP32_WEIGHTS = MODEL_DIR / "sft_best_fp32_weights.pt"

EVAL_BATCHES = 20


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def load_fp32_checkpoint(path: Path) -> tuple[model.Transformer, model.ModelArgs]:
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    model_args = model.ModelArgs(**checkpoint["model_args"])
    model_instance = model.Transformer(model_args)
    model_instance.load_state_dict(checkpoint["model_state_dict"])
    model_instance.eval()
    return model_instance, model_args


def quantize_linear(model_instance: torch.nn.Module) -> torch.nn.Module:
    model_instance.eval()
    return torch.quantization.quantize_dynamic(
        model_instance, {nn.Linear}, dtype=torch.qint8
    )


def load_int8_checkpoint(path: Path) -> torch.nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"missing int8 checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    model_args = model.ModelArgs(**checkpoint["model_args"])
    model_instance = model.Transformer(model_args)
    quantized_model = quantize_linear(model_instance)
    quantized_model.load_state_dict(checkpoint["model_state_dict"])
    quantized_model.eval()
    return quantized_model


def save_fp32_weights(
    path: Path, model_args: model.ModelArgs, model_instance: torch.nn.Module
) -> None:
    torch.save(
        {
            "model_args": asdict(model_args),
            "model_state_dict": model_instance.state_dict(),
        },
        path,
    )


def build_sft_val_loader(model_args: model.ModelArgs):
    hp = replace(
        SFT_CONFIG,
        block_size=min(SFT_CONFIG.block_size, model_args.max_seq_len),
    )
    _, val_loader = build_sft_dataloaders(SFT_DATA_PATH, hp, seed=1337)
    return val_loader


def compute_ppl(
    model_instance: torch.nn.Module,
    val_loader,
    eval_batches: int,
    desc: str,
) -> tuple[float, float]:
    model_instance.eval()
    losses = []
    iterator = iter(val_loader)
    for _ in tqdm(range(eval_batches), desc=desc, unit="batch"):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        input_ids, labels, attention_mask = batch
        input_ids = input_ids.to("cpu")
        labels = labels.to("cpu")
        attention_mask = attention_mask.to("cpu")
        _, loss = model_instance(
            input_ids, attention_mask=attention_mask, labels=labels
        )
        losses.append(loss.item())
    if not losses:
        return float("nan"), float("nan")
    loss = sum(losses) / len(losses)
    ppl = math.exp(loss)
    return loss, ppl


@torch.no_grad()
def generate_latency(
    model_instance: torch.nn.Module,
    enc: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    block_size: int,
    warmup: int = 10,
    runs: int = 50,
    desc: str = "",
) -> tuple[float, float]:
    input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0)

    def run_once():
        ids = input_ids.clone()
        for _ in range(max_new_tokens):
            input_cond = ids[:, -block_size:]
            logits = model_instance(input_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_values, torch.full_like(logits, -float("inf")), logits
                )
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_token], dim=1)

    for _ in tqdm(range(warmup), desc=f"{desc} warmup", unit="run", leave=False):
        run_once()

    durations = []
    for _ in tqdm(range(runs), desc=f"{desc} timed", unit="run", leave=False):
        start = time.perf_counter()
        run_once()
        durations.append(time.perf_counter() - start)

    return float(statistics.mean(durations)), float(statistics.median(durations))


def format_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    idx = 0
    size = float(num_bytes)
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.2f} {units[idx]}"


def make_row(
    name: str,
    fp32_ckpt: Path,
    fp32_weights: Path,
    int8_ckpt: Path,
    fp32_loss: float,
    fp32_ppl: float,
    int8_loss: float,
    int8_ppl: float,
    fp32_latency: float,
    int8_latency: float,
) -> list[str]:
    ckpt_size = file_size(fp32_ckpt)
    int8_size = file_size(int8_ckpt)
    weights_size = file_size(fp32_weights)
    ratio_ckpt = ckpt_size / int8_size if int8_size > 0 else float("nan")
    ratio_weights = weights_size / int8_size if int8_size > 0 else float("nan")
    ppl_delta = (
        (int8_ppl - fp32_ppl) / fp32_ppl * 100.0
        if fp32_ppl == fp32_ppl and int8_ppl == int8_ppl and fp32_ppl > 0
        else float("nan")
    )
    speedup = fp32_latency / int8_latency if int8_latency > 0 else float("nan")

    return [
        name,
        format_bytes(ckpt_size),
        format_bytes(int8_size),
        f"{ratio_ckpt:.2f}x" if ratio_ckpt == ratio_ckpt else "n/a",
        f"{ratio_weights:.2f}x" if ratio_weights == ratio_weights else "n/a",
        f"{fp32_ppl:.4f}",
        f"{int8_ppl:.4f}",
        f"{ppl_delta:.2f}%" if ppl_delta == ppl_delta else "n/a",
        f"{fp32_latency:.3f}s",
        f"{int8_latency:.3f}s",
        f"{speedup:.2f}x" if speedup == speedup else "n/a",
    ]


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")

    # SFT
    sft_fp32, sft_args = load_fp32_checkpoint(SFT_CKPT)
    save_fp32_weights(SFT_FP32_WEIGHTS, sft_args, sft_fp32)
    sft_int8 = load_int8_checkpoint(SFT_INT8)
    sft_val = build_sft_val_loader(sft_args)
    sft_fp32_loss, sft_fp32_ppl = compute_ppl(
        sft_fp32, sft_val, EVAL_BATCHES, desc="FP32 eval"
    )
    sft_int8_loss, sft_int8_ppl = compute_ppl(
        sft_int8, sft_val, EVAL_BATCHES, desc="INT8 eval"
    )
    sft_fp32_lat, _ = generate_latency(
        sft_fp32,
        enc,
        prompt=(
            "### Instruction:\nTranslate to English.\n### Input:\nBonjour\n### Response:\n"
        ),
        max_new_tokens=64,
        temperature=0.7,
        top_k=40,
        block_size=sft_args.max_seq_len,
        desc="FP32",
    )
    sft_int8_lat, _ = generate_latency(
        sft_int8,
        enc,
        prompt=(
            "### Instruction:\nTranslate to English.\n### Input:\nBonjour\n### Response:\n"
        ),
        max_new_tokens=64,
        temperature=0.7,
        top_k=40,
        block_size=sft_args.max_seq_len,
        desc="INT8",
    )

    headers = [
        "Model",
        "FP32 size",
        "INT8 size",
        "Ratio (ckpt/int8)",
        "Ratio (weights/int8)",
        "FP32 PPL",
        "INT8 PPL",
        "ΔPPL%",
        "FP32 latency",
        "INT8 latency",
        "Speedup",
    ]

    rows = [
        make_row(
            "sft",
            SFT_CKPT,
            SFT_FP32_WEIGHTS,
            SFT_INT8,
            sft_fp32_loss,
            sft_fp32_ppl,
            sft_int8_loss,
            sft_int8_ppl,
            sft_fp32_lat,
            sft_int8_lat,
        )
    ]

    report_path = REPORTS_DIR / "quantization_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Quantization Report\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")

    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
