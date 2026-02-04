# nanoGPT_xidian

## Overview
This repository trains a decoder-only Transformer with modern components (RMSNorm, RoPE, SwiGLU, MoE). It supports two stages:

- Pretraining on TinyStories
- Supervised fine-tuning (SFT) on an English instruction dataset for assistant-style behavior

## Datasets

### Pretrain Dataset
- Source: `roneneldan/TinyStories`
- Mode: non-streaming (downloaded and cached locally)
- Loader: on-the-fly tokenization in `train.py`

### SFT Dataset
- File: `data/sft_data_en.jsonl`
- Generation script: `scripts/build_sft_data_en.py`
- Composition: Alpaca + Dolly + SQuAD + SQuAD-v2 ("I don't know" cases)

To (re)generate the SFT dataset:
```bash
uv run python scripts/build_sft_data_en.py
```

## Pretrain

Run pretraining:
```bash
uv run python train.py --stage pretrain
```

Outputs:
- Logs: `training_log/pretrain_loss.csv`
- Checkpoints: `model_trained/best.pt`, `model_trained/ckpt.pt`

Notes:
- The first run will download TinyStories to the Hugging Face cache.
- `max_steps` controls training length. Epochs are estimated and printed.

## SFT

Run SFT (uses `data/sft_data_en.jsonl` by default):
```bash
uv run python train.py --stage sft
```

Outputs:
- Logs: `training_log/sft_loss.csv`
- Checkpoints: `model_trained/sft_best.pt`, `model_trained/sft_ckpt.pt`

## Quick Samples

SFT samples (assistant responses):
```bash
uv run python tests/test_sft_samples.py
```

Pretrain samples (story continuation):
```bash
uv run python tests/test_pretrain_samples.py
```

## Notes on Hyperparameters

Key controls in `train.py`:
- `max_steps`: total number of training steps
- `eval_interval`: how often to run evaluation
- `sample_interval`: how often to print sample outputs

You can override defaults via command-line flags, for example:
```bash
uv run python train.py --stage pretrain --max_steps 2000 --eval_interval 50
```
