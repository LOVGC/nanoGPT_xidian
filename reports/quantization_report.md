# Quantization Report

| Model | FP32 size | INT8 size | Ratio (ckpt/int8) | Ratio (weights/int8) | FP32 PPL | INT8 PPL | ΔPPL% | FP32 latency | INT8 latency | Speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| sft | 560.34 MB | 108.75 MB | 5.15x | 1.74x | 8.7782 | 8.6860 | -1.05% | 1.698s | 1.464s | 1.16x |
