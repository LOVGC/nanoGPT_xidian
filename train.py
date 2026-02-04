from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Iterable, Iterator, Optional

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset

from datasets import load_dataset
import tiktoken

import model


SFT_DATA_PATH = Path("data") / "sft_data.jsonl"
PRETRAIN_LOG_NAME = "pretrain_loss.csv"
SFT_LOG_NAME = "sft_loss.csv"
PRETRAIN_SAMPLE_PROMPT = "Once upon a time"
SFT_SAMPLE_PROMPT = "### Instruction:\n翻译成英文\n### Input:\n你好\n### Response:\n"


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    block_size: int
    max_steps: int
    lr: float
    min_lr: float
    warmup_steps: int
    weight_decay: float
    grad_clip: float
    eval_interval: int
    eval_steps: int
    sample_interval: int
    max_new_tokens: int
    temperature: float
    top_k: int
    log_interval: int
    shuffle_buffer: int


PRETRAIN_CONFIG = TrainingConfig(
    batch_size=32,
    block_size=128,
    max_steps=2000,
    lr=1e-3,
    min_lr=1e-4,
    warmup_steps=200,
    weight_decay=0.1,
    grad_clip=1.0,
    eval_interval=100,
    eval_steps=50,
    sample_interval=100,
    max_new_tokens=120,
    temperature=0.8,
    top_k=50,
    log_interval=20,
    shuffle_buffer=10000,
)

SFT_CONFIG = TrainingConfig(
    batch_size=16,
    block_size=128,
    max_steps=200, # number of weight updates
    lr=5e-5,
    min_lr=1e-5,
    warmup_steps=50,
    weight_decay=0.0,
    grad_clip=1.0,
    eval_interval=100,
    eval_steps=20,
    sample_interval=200,
    max_new_tokens=120,
    temperature=0.7,
    top_k=50,
    log_interval=20,
    shuffle_buffer=1000,
)


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

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
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


class SFTJsonlDataset(Dataset):
    def __init__(
        self, data_path: Path, block_size: int, encoding_name: str = "gpt2"
    ) -> None:
        self.block_size = block_size
        self.enc = tiktoken.get_encoding(encoding_name)
        self.eot = self.enc.eot_token
        self.samples: list[dict[str, list[int]]] = []

        if not data_path.exists():
            raise FileNotFoundError(f"SFT data not found: {data_path}")

        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                instruction = record.get("instruction", "").strip()
                input_text = record.get("input", "").strip()
                output_text = record.get("output", "").strip()
                if not instruction or not output_text:
                    continue
                prompt = format_sft_prompt(instruction, input_text)
                prompt_tokens = self.enc.encode(prompt)
                response_tokens = self.enc.encode(output_text) + [self.eot]
                if len(response_tokens) > self.block_size:
                    continue
                if len(prompt_tokens) + len(response_tokens) > self.block_size:
                    keep = self.block_size - len(response_tokens)
                    if keep > 0:
                        prompt_tokens = prompt_tokens[-keep:]
                    else:
                        prompt_tokens = []
                input_ids = prompt_tokens + response_tokens
                labels = [-100] * len(prompt_tokens) + response_tokens
                self.samples.append({"input_ids": input_ids, "labels": labels})

        if not self.samples:
            raise RuntimeError("No valid SFT samples found")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.samples[idx]


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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def apply_overrides(cfg: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    updates = {}
    for field in fields(cfg):
        if hasattr(args, field.name):
            value = getattr(args, field.name)
            if value is not None:
                updates[field.name] = value
    if updates:
        return replace(cfg, **updates)
    return cfg


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


def build_sft_dataloaders(
    data_path: Path,
    hp: TrainingConfig,
    seed: int,
    val_ratio: float = 0.1,
) -> tuple[DataLoader, DataLoader]:
    dataset = SFTJsonlDataset(data_path, block_size=hp.block_size)
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    def collate_fn(batch: list[dict[str, list[int]]]):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), max_len), dataset.eot, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for idx, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[idx, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
            labels[idx, :length] = torch.tensor(item["labels"], dtype=torch.long)
            attention_mask[idx, :length] = 1
        return input_ids, labels, attention_mask

    train_loader = DataLoader(
        train_set,
        batch_size=hp.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=hp.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


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


def format_eta(seconds: float) -> str:
    if seconds != seconds or seconds < 0:
        return "N/A"
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def append_log(
    log_path: Path,
    step: int,
    split: str,
    loss: float,
    lr: float,
    eta_seconds: float,
    wall_time: float,
) -> None:
    with log_path.open("a", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            [
                step,
                split,
                f"{loss:.6f}",
                f"{lr:.8f}",
                f"{eta_seconds:.2f}",
                f"{wall_time:.2f}",
            ]
        )


def setup_log_file(log_dir: Path, filename: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename
    if not log_path.exists():
        with log_path.open("w", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(["step", "split", "loss", "lr", "eta_seconds", "wall_time"])
    return log_path


def prepare_batch(
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if len(batch) == 2:
        x, y = batch
        attention_mask = None
    else:
        x, y, attention_mask = batch
    x = x.to(device)
    y = y.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return x, y, attention_mask


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
            batch = next(data_iter)
        except StopIteration:
            break
        x, y, attention_mask = prepare_batch(batch, device)
        _, loss = model_instance(x, attention_mask=attention_mask, labels=y)
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
        if top_k is not None and top_k > 0:
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


def load_checkpoint_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[model.Transformer, model.ModelArgs]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = model.ModelArgs(**checkpoint["model_args"])
    model_instance = model.Transformer(model_args).to(device)
    model_instance.load_state_dict(checkpoint["model_state_dict"])
    return model_instance, model_args


def run_training(
    model_instance: torch.nn.Module,
    model_args: model.ModelArgs,
    enc: tiktoken.Encoding,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hp: TrainingConfig,
    device: torch.device,
    log_path: Path,
    output_dir: Path,
    best_name: str,
    final_name: str,
    sample_prompt: str,
) -> None:
    optimizer = torch.optim.AdamW(
        model_instance.parameters(),
        lr=hp.lr,
        weight_decay=hp.weight_decay,
    )

    model_instance.train()
    start_time = time.time()
    train_iter = iter(train_loader)
    best_eval_loss = float("inf")

    for step in range(1, hp.max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x, y, attention_mask = prepare_batch(batch, device)

        lr = get_lr(step - 1, hp.lr, hp.min_lr, hp.warmup_steps, hp.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        _, loss = model_instance(x, attention_mask=attention_mask, labels=y)
        loss.backward()
        if hp.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model_instance.parameters(), hp.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if hp.log_interval > 0 and step % hp.log_interval == 0:
            elapsed = time.time() - start_time
            avg_step_time = elapsed / max(step, 1)
            eta_seconds = (hp.max_steps - step) * avg_step_time
            print(
                f"step {step}/{hp.max_steps} | train_loss {loss.item():.4f} | lr {lr:.2e} | ETA {format_eta(eta_seconds)}"
            )
            append_log(
                log_path,
                step,
                "train",
                loss.item(),
                lr,
                eta_seconds,
                elapsed,
            )

        if hp.eval_interval > 0 and step % hp.eval_interval == 0:
            val_loss = evaluate(model_instance, val_loader, device, hp.eval_steps)
            elapsed = time.time() - start_time
            avg_step_time = elapsed / max(step, 1)
            eta_seconds = (hp.max_steps - step) * avg_step_time
            print(
                f"eval loss at step {step}: {val_loss:.4f} | lr {lr:.2e} | ETA {format_eta(eta_seconds)}"
            )
            append_log(
                log_path,
                step,
                "eval",
                val_loss,
                lr,
                eta_seconds,
                elapsed,
            )
            if not math.isnan(val_loss) and val_loss < best_eval_loss:
                best_eval_loss = val_loss
                best_path = output_dir / best_name
                torch.save(
                    {
                        "model_state_dict": model_instance.state_dict(),
                        "model_args": asdict(model_args),
                        "step": step,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "eval_loss": val_loss,
                    },
                    best_path,
                )
                print(f"saved best checkpoint: {best_path}")

        if hp.sample_interval > 0 and step % hp.sample_interval == 0:
            sample = generate(
                model_instance,
                enc,
                prompt=sample_prompt,
                device=device,
                max_new_tokens=hp.max_new_tokens,
                temperature=hp.temperature,
                top_k=hp.top_k,
                block_size=hp.block_size,
            )
            print("--- sample ---")
            print(sample)
            print("--- end sample ---")

    final_path = output_dir / final_name
    torch.save(
        {
            "model_state_dict": model_instance.state_dict(),
            "model_args": asdict(model_args),
            "step": hp.max_steps,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_path,
    )
    print(f"saved final checkpoint: {final_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train or SFT the model")
    parser.add_argument(
        "--stage",
        choices=["pretrain", "sft"],
        default="sft",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--sample_interval", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--shuffle_buffer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="model_trained")
    parser.add_argument("--log_dir", type=str, default="training_log")
    return parser.parse_args()


def run_pretrain(
    args: argparse.Namespace, hp: TrainingConfig, device: torch.device
) -> None:
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    model_args = model.ModelArgs(
        vocab_size=vocab_size,
        max_seq_len=hp.block_size,
        n_layer=6,
        n_head=5,
        n_kv_head=5,
        n_embd=320,
        ffn_hidden_size=896,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        dropout=0.0,
        attention_dropout=0.0,
        bias=False,
        n_expert=6,
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_log_file(Path(args.log_dir), PRETRAIN_LOG_NAME)

    train_loader = build_dataloader(
        split="train",
        block_size=hp.block_size,
        batch_size=hp.batch_size,
        seed=args.seed,
        shuffle=True,
        buffer_size=hp.shuffle_buffer,
        repeat=True,
    )
    try:
        val_loader = build_dataloader(
            split="validation",
            block_size=hp.block_size,
            batch_size=hp.batch_size,
            seed=args.seed + 1,
            shuffle=False,
            buffer_size=hp.shuffle_buffer,
            repeat=False,
        )
    except Exception:
        print("validation split unavailable, falling back to train split for eval")
        val_loader = build_dataloader(
            split="train",
            block_size=hp.block_size,
            batch_size=hp.batch_size,
            seed=args.seed + 1,
            shuffle=False,
            buffer_size=hp.shuffle_buffer,
            repeat=False,
        )

    run_training(
        model_instance=model_instance,
        model_args=model_args,
        enc=enc,
        train_loader=train_loader,
        val_loader=val_loader,
        hp=hp,
        device=device,
        log_path=log_path,
        output_dir=output_dir,
        best_name="best.pt",
        final_name="ckpt.pt",
        sample_prompt=PRETRAIN_SAMPLE_PROMPT,
    )


def run_sft(args: argparse.Namespace, hp: TrainingConfig, device: torch.device) -> None:
    enc = tiktoken.get_encoding("gpt2")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_log_file(Path(args.log_dir), SFT_LOG_NAME)

    base_checkpoint = output_dir / "best.pt"
    model_instance, model_args = load_checkpoint_model(base_checkpoint, device)

    block_size = min(hp.block_size, model_args.max_seq_len)
    if block_size != hp.block_size:
        print(
            f"SFT block_size {hp.block_size} exceeds model max_seq_len {model_args.max_seq_len}, using {block_size}"
        )
        hp = replace(hp, block_size=block_size)

    train_loader, val_loader = build_sft_dataloaders(SFT_DATA_PATH, hp, seed=args.seed)

    total_params = count_parameters(model_instance)
    print(f"Model parameters: {total_params}")
    print(f"Model args: {asdict(model_args)}")

    run_training(
        model_instance=model_instance,
        model_args=model_args,
        enc=enc,
        train_loader=train_loader,
        val_loader=val_loader,
        hp=hp,
        device=device,
        log_path=log_path,
        output_dir=output_dir,
        best_name="sft_best.pt",
        final_name="sft_ckpt.pt",
        sample_prompt=SFT_SAMPLE_PROMPT,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    if args.stage == "pretrain":
        hp = apply_overrides(PRETRAIN_CONFIG, args)
        run_pretrain(args, hp, device)
    elif args.stage == "sft":
        hp = apply_overrides(SFT_CONFIG, args)
        run_sft(args, hp, device)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
