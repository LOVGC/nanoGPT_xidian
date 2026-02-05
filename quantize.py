from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn as nn
import tiktoken

import model
from train import (
    PRETRAIN_CONFIG,
    SFT_CONFIG,
    SFT_DATA_PATH,
    build_dataloader,
    build_sft_dataloaders,
    evaluate,
)


PRETRAIN_CHECKPOINT = Path("model_trained") / "best.pt"
SFT_CHECKPOINT = Path("model_trained") / "sft_best.pt"


def load_checkpoint(path: Path) -> tuple[model.Transformer, model.ModelArgs]:
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    model_args = model.ModelArgs(**checkpoint["model_args"])
    model_instance = model.Transformer(model_args)
    model_instance.load_state_dict(checkpoint["model_state_dict"])
    model_instance.eval()
    return model_instance, model_args


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


def quantize_linear(model_instance: torch.nn.Module) -> torch.nn.Module:
    model_instance.eval()
    return torch.quantization.quantize_dynamic(
        model_instance, {nn.Linear}, dtype=torch.qint8
    )


def save_quantized_weights(
    path: Path,
    model_args: model.ModelArgs,
    quantized_model: torch.nn.Module,
) -> None:
    torch.save(
        {
            "model_args": asdict(model_args),
            "model_state_dict": quantized_model.state_dict(),
            "quantized": True,
        },
        path,
    )


def load_quantized_checkpoint(path: Path) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu")
    model_args = model.ModelArgs(**checkpoint["model_args"])
    model_instance = model.Transformer(model_args)
    quantized_model = quantize_linear(model_instance)
    quantized_model.load_state_dict(checkpoint["model_state_dict"])
    quantized_model.eval()
    return quantized_model


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def evaluate_ppl(
    model_instance: torch.nn.Module,
    val_loader_factory,
    eval_steps: int = 0,
) -> tuple[float, float]:
    val_loader = val_loader_factory()
    loss = evaluate(model_instance, val_loader, torch.device("cpu"), eval_steps)
    ppl = math.exp(loss) if loss == loss else float("nan")
    return loss, ppl


def build_pretrain_val_loader(model_args: model.ModelArgs):
    def factory():
        return build_dataloader(
            split="validation",
            block_size=model_args.max_seq_len,
            batch_size=PRETRAIN_CONFIG.batch_size,
            seed=1337,
            shuffle=False,
            buffer_size=PRETRAIN_CONFIG.shuffle_buffer,
            repeat=False,
            drop_last=False,
        )

    return factory


def build_sft_val_loader(model_args: model.ModelArgs):
    hp = replace(
        SFT_CONFIG,
        block_size=min(SFT_CONFIG.block_size, model_args.max_seq_len),
    )

    def factory():
        _, val_loader = build_sft_dataloaders(SFT_DATA_PATH, hp, seed=1337)
        return val_loader

    return factory


def decode_until_eot(enc: tiktoken.Encoding, token_ids: list[int]) -> str:
    eot_id = enc.eot_token
    if eot_id in token_ids:
        token_ids = token_ids[: token_ids.index(eot_id)]
    return enc.decode(token_ids)


@torch.no_grad()
def generate_sample(
    model_instance: torch.nn.Module,
    enc: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    top_k: int,
) -> str:
    input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0)
    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -block_size:]
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
        input_ids = torch.cat([input_ids, next_token], dim=1)
    response_ids = input_ids[0, len(enc.encode(prompt)) :].tolist()
    return decode_until_eot(enc, response_ids).strip()


def quantize_and_report(
    label: str,
    checkpoint_path: Path,
    fp32_out: Path,
    int8_out: Path,
    val_loader_factory_builder,
    sample_prompt: str,
) -> None:
    print(f"\n=== {label.upper()} ===")
    print(f"Checkpoint: {checkpoint_path}")

    model_instance, model_args = load_checkpoint(checkpoint_path)
    val_factory = val_loader_factory_builder(model_args)

    fp32_loss, fp32_ppl = evaluate_ppl(model_instance, val_factory, eval_steps=0)
    print(f"FP32 loss: {fp32_loss:.4f} | PPL: {fp32_ppl:.4f}")

    save_fp32_weights(fp32_out, model_args, model_instance)

    quantized_model = quantize_linear(model_instance)
    save_quantized_weights(int8_out, model_args, quantized_model)

    int8_loss, int8_ppl = evaluate_ppl(quantized_model, val_factory, eval_steps=0)
    print(f"INT8 loss: {int8_loss:.4f} | PPL: {int8_ppl:.4f}")

    if fp32_ppl == fp32_ppl and int8_ppl == int8_ppl and fp32_ppl > 0:
        delta = (int8_ppl - fp32_ppl) / fp32_ppl * 100.0
        print(f"PPL increase: {delta:.2f}%")

    fp32_size = file_size(fp32_out)
    int8_size = file_size(int8_out)
    ckpt_size = file_size(checkpoint_path)
    if int8_size > 0:
        if ckpt_size > 0:
            ratio_ckpt = ckpt_size / int8_size
            print(f"Checkpoint size: {ckpt_size} bytes")
            print(f"Compression ratio (ckpt vs int8): {ratio_ckpt:.2f}x")
        if fp32_size > 0:
            ratio_weights = fp32_size / int8_size
            print(f"FP32 weights size: {fp32_size} bytes")
            print(f"INT8 weights size: {int8_size} bytes")
            print(f"Compression ratio (weights vs int8): {ratio_weights:.2f}x")

    enc = tiktoken.get_encoding("gpt2")
    sample = generate_sample(
        quantized_model,
        enc,
        prompt=sample_prompt,
        max_new_tokens=64,
        block_size=model_args.max_seq_len,
        temperature=0.7,
        top_k=40,
    )
    print("Sample output:")
    print(sample)

    _ = load_quantized_checkpoint(int8_out)
    print(f"Loaded quantized checkpoint: {int8_out}")


def main() -> None:
    quantize_and_report(
        label="pretrain",
        checkpoint_path=PRETRAIN_CHECKPOINT,
        fp32_out=Path("model_trained") / "best_fp32_weights.pt",
        int8_out=Path("model_trained") / "best_int8.pt",
        val_loader_factory_builder=build_pretrain_val_loader,
        sample_prompt="Once upon a time",
    )

    quantize_and_report(
        label="sft",
        checkpoint_path=SFT_CHECKPOINT,
        fp32_out=Path("model_trained") / "sft_best_fp32_weights.pt",
        int8_out=Path("model_trained") / "sft_best_int8.pt",
        val_loader_factory_builder=build_sft_val_loader,
        sample_prompt=(
            "### Instruction:\nTranslate to English.\n### Input:\nBonjour\n### Response:\n"
        ),
    )


if __name__ == "__main__":
    main()
