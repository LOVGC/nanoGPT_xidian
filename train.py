from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from datasets import load_dataset
import tiktoken

import model


class TinyStoriesIterableDataset(IterableDataset):
    def __init__(
        self,
        split: str,
        block_size: int,
        seed: int,
        shuffle: bool,
        buffer_size: int,
        add_eot: bool,
        repeat: bool,
        encoding_name: str = "gpt2",
    ) -> None:
        super().__init__()
        self.split = split
        self.block_size = block_size
        self.seed = seed
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.add_eot = add_eot
        self.repeat = repeat
        self.enc = tiktoken.get_encoding(encoding_name)
        self.eot = self.enc.eot_token

    def _get_dataset(self):
        dataset = load_dataset(
            "roneneldan/TinyStories", split=self.split, streaming=True
        )
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size, seed=self.seed)
        return dataset

    def __iter__(self) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            buffer: list[int] = []
            dataset = self._get_dataset()
            for sample in dataset:
                text = sample.get("text")
                if not text:
                    continue
                tokens = self.enc.encode(text)
                if self.add_eot:
                    tokens.append(self.eot)
                buffer.extend(tokens)
                while len(buffer) >= self.block_size + 1:
                    chunk = buffer[: self.block_size + 1]
                    buffer = buffer[self.block_size + 1 :]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield x, y
            if not self.repeat:
                break


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_dataloader(
    split: str,
    block_size: int,
    batch_size: int,
    seed: int,
    shuffle: bool,
    buffer_size: int,
    repeat: bool,
) -> DataLoader:
    dataset = TinyStoriesIterableDataset(
        split=split,
        block_size=block_size,
        seed=seed,
        shuffle=shuffle,
        buffer_size=buffer_size,
        add_eot=True,
        repeat=repeat,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def get_lr(
    step: int, base_lr: float, min_lr: float, warmup_steps: int, max_steps: int
) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    if step >= max_steps:
        return min_lr
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


@torch.no_grad()
def evaluate(
    model_instance: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    eval_steps: int,
) -> float:
    model_instance.eval()
    losses = []
    data_iter = iter(data_loader)
    for _ in range(eval_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            break
        x = x.to(device)
        y = y.to(device)
        _, loss = model_instance(x, labels=y)
        losses.append(loss.item())
    model_instance.train()
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


@torch.no_grad()
def generate(
    model_instance: torch.nn.Module,
    enc: tiktoken.Encoding,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    block_size: int,
) -> str:
    model_instance.eval()
    token_ids = enc.encode(prompt)
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -block_size:]
        logits = model_instance(input_cond)
        logits = logits[:, -1, :]
        logits = logits / max(temperature, 1e-6)
        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_values, torch.full_like(logits, -float("inf")), logits
            )
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    model_instance.train()
    return enc.decode(input_ids.squeeze(0).tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train Transformer on TinyStories")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--sample_interval", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--shuffle_buffer", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    model_args = model.ModelArgs(
        vocab_size=vocab_size,
        max_seq_len=args.block_size,
        n_layer=6,
        n_head=5,
        n_kv_head=5,
        n_embd=320,
        ffn_hidden_size=832,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        dropout=0.0,
        attention_dropout=0.0,
        bias=False,
        n_expert=4,
        moe_top_k=2,
        moe_jitter=0.0,
        tie_embeddings=True,
    )

    if not (4 <= model_args.n_expert <= 8):
        raise ValueError("n_expert must be between 4 and 8 for this setup")
    if model_args.moe_top_k != 2:
        raise ValueError("moe_top_k must be 2 for Top-2 routing")

    model_instance = model.Transformer(model_args).to(device)
    total_params = count_parameters(model_instance)
    print(f"Model parameters: {total_params}")
    print(f"Model args: {asdict(model_args)}")

    train_loader = build_dataloader(
        split="train",
        block_size=args.block_size,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=True,
        buffer_size=args.shuffle_buffer,
        repeat=True,
    )
    try:
        val_loader = build_dataloader(
            split="validation",
            block_size=args.block_size,
            batch_size=args.batch_size,
            seed=args.seed + 1,
            shuffle=False,
            buffer_size=args.shuffle_buffer,
            repeat=False,
        )
    except Exception:
        val_loader = build_dataloader(
            split="train",
            block_size=args.block_size,
            batch_size=args.batch_size,
            seed=args.seed + 1,
            shuffle=False,
            buffer_size=args.shuffle_buffer,
            repeat=False,
        )

    optimizer = torch.optim.AdamW(
        model_instance.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model_instance.train()
    start_time = time.time()
    train_iter = iter(train_loader)

    for step in range(1, args.max_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)

        lr = get_lr(step - 1, args.lr, args.min_lr, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        _, loss = model_instance(x, labels=y)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model_instance.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            tokens = step * args.batch_size * args.block_size
            tokens_per_sec = tokens / max(elapsed, 1e-6)
            print(
                f"step {step}/{args.max_steps} | loss {loss.item():.4f} | lr {lr:.2e} | tokens/s {tokens_per_sec:.0f}"
            )

        if step % args.eval_interval == 0:
            val_loss = evaluate(model_instance, val_loader, device, args.eval_steps)
            print(f"eval loss at step {step}: {val_loss:.4f}")

        if step % args.sample_interval == 0:
            sample = generate(
                model_instance,
                enc,
                prompt="Once upon a time",
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                block_size=args.block_size,
            )
            print("--- sample ---")
            print(sample)
            print("--- end sample ---")


if __name__ == "__main__":
    main()
