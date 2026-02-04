# 任务说明

- 从 0 到 1 构建一个参数量小于 50M 的微型 LLM，并攻克架构优化（MoE）、指令微调（SFT）、模型压缩（Quantization） 和 知识增强（RAG） 四大核心技术难点。


- 基于 PyTorch 框架，构建一个具备指令跟随能力的 Decoder-only Transformer 模型。该模型需在保持轻量级（<50M 参数）的同时，通过混合专家（MoE）架构提升容量，通过量化适配端侧推理，并通过 RAG 外挂知识库解决“幻觉”问题。

# 任务分解

1. 原生 transformer 实现：
- 禁止直接调用 HuggingFace Transformers 库中的现成模型（如 AutoModel）
- 需手动基于 torch.nn 编写 Decoder-only 架构

2. 混合专家模型 (MoE)：
- 引入稀疏门控机制（Sparse Gating），在不显著增加推理 FLOPs 的前提下增加模型总参数量

3. 指令微调（SFT）
- 实现从 Pre-training（续写）到 Instruction Tuning（对话）的范式转变

4.	训练后动态量化 (PTQ)：
- 将模型权重从 FP32 压缩至 INT8，实现体积缩小 4 倍以上。

5.	检索增强生成 (RAG)：
- 实现向量检索系统，赋予小模型查阅外部知识库的能力。
