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

## Quantization

Run dynamic quantization (W8A16, `nn.Linear` only):
```bash
uv run python quantize.py
```

Outputs:
- Quantized weights: `model_trained/best_int8.pt`, `model_trained/sft_best_int8.pt`
- FP32 weights-only snapshots: `model_trained/best_fp32_weights.pt`, `model_trained/sft_best_fp32_weights.pt`

Reports:
- FP32 vs INT8 loss/PPL
- Compression ratio (ckpt vs int8) and (weights vs int8)

Notes:
- Quantization runs on CPU and uses full validation by default (`eval_steps=0`).

## RAG Inference

Prepare your knowledge base under `data/knowledge_base/` and build the index:

Local backend (your SFT model):
```bash
uv run --python 3.13 python inference.py --backend local --docs_dir data/knowledge_base --index_dir rag_index --rebuild
```

Ollama backend (local model via Ollama):
```bash
uv run --python 3.13 python inference.py --backend ollama --ollama_model qwen3:4b --ollama_base_url http://localhost:11434 --docs_dir data/knowledge_base --index_dir rag_index --rebuild
```

Notes:
- Ollama must be running and the model pulled (e.g. `ollama pull qwen3:4b`).
- Chroma + onnxruntime require Python 3.13 on Windows; use `--python 3.13` when running RAG.

## Notes on Hyperparameters

Key controls in `train.py`:
- `max_steps`: total number of training steps
- `eval_interval`: how often to run evaluation
- `sample_interval`: how often to print sample outputs

You can override defaults via command-line flags, for example:
```bash
uv run python train.py --stage pretrain --max_steps 2000 --eval_interval 50
```
