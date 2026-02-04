from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    vocab_size: int
    max_seq_len: int = 2048
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: Optional[int] = None
    n_embd: int = 768
    ffn_hidden_size: Optional[int] = None
    ffn_multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    bias: bool = False
    n_expert: int = 0
    moe_top_k: int = 2
    moe_jitter: float = 0.0
    tie_embeddings: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _compute_ffn_hidden_dim(
    dim: int,
    multiple_of: int,
    ffn_dim_multiplier: Optional[float],
    ffn_hidden_size: Optional[int],
) -> int:
    if ffn_hidden_size is not None:
        return ffn_hidden_size
    hidden_dim = int((2.0 / 3.0) * (4 * dim))
    if ffn_dim_multiplier is not None:
        hidden_dim = int(hidden_dim * ffn_dim_multiplier)
    hidden_dim = multiple_of * math.ceil(hidden_dim / multiple_of)
    return hidden_dim


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        if args.n_embd % args.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = args.n_head
        self.n_kv_head = args.n_kv_head or args.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        self.head_dim = args.n_embd // args.n_head
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = args.attention_dropout

        self.q_proj = nn.Linear(
            args.n_embd, args.n_head * self.head_dim, bias=args.bias
        )
        self.k_proj = nn.Linear(
            args.n_embd, self.n_kv_head * self.head_dim, bias=args.bias
        )
        self.v_proj = nn.Linear(
            args.n_embd, self.n_kv_head * self.head_dim, bias=args.bias
        )
        self.o_proj = nn.Linear(
            args.n_head * self.head_dim, args.n_embd, bias=args.bias
        )

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=args.max_seq_len,
            base=args.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(
                0
            )

        hidden_shape = (batch_size, seq_len, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        key_states = repeat_kv(key_states, self.n_head // self.n_kv_head)
        value_states = repeat_kv(value_states, self.n_head // self.n_kv_head)

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        else:
            mask = torch.triu(
                torch.ones(
                    seq_len, seq_len, device=hidden_states.device, dtype=torch.bool
                ),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(
                mask, torch.finfo(attn_weights.dtype).min
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        return self.o_proj(attn_output)


class MoE(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        router_jitter: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter = router_jitter
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(dim, hidden_dim, bias=bias) for _ in range(num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor, return_router_logits: bool = False):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if self.training and self.router_jitter > 0:
            jitter = torch.empty_like(hidden_states).uniform_(
                1.0 - self.router_jitter, 1.0 + self.router_jitter
            )
            hidden_states = hidden_states * jitter
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.router(hidden_states)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current_state = hidden_states[token_idx]
            current_hidden_states = self.experts[expert_idx](current_state)
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
        if return_router_logits:
            return final_hidden_states, router_logits
        return final_hidden_states


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(args)
        ffn_hidden_dim = _compute_ffn_hidden_dim(
            args.n_embd,
            args.ffn_multiple_of,
            args.ffn_dim_multiplier,
            args.ffn_hidden_size,
        )
        self.mlp = (
            MoE(
                args.n_embd,
                ffn_hidden_dim,
                num_experts=args.n_expert,
                top_k=args.moe_top_k,
                router_jitter=args.moe_jitter,
                bias=args.bias,
            )
            if args.n_expert and args.n_expert > 1
            else SwiGLU(args.n_embd, ffn_hidden_dim, bias=args.bias)
        )
        self.input_layernorm = RMSNorm(args.n_embd, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.n_embd, eps=args.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_router_logits: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        router_logits = None
        if isinstance(self.mlp, MoE):
            if return_router_logits:
                hidden_states, router_logits = self.mlp(
                    hidden_states, return_router_logits=True
                )
            else:
                hidden_states = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if return_router_logits:
            return hidden_states, router_logits
        return hidden_states


def _build_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full(
        (1, 1, seq_len, seq_len), min_dtype, device=device, dtype=dtype
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    if attention_mask is None:
        return causal_mask
    expanded = attention_mask[:, None, None, :].to(dtype=dtype)
    padding_mask = (1.0 - expanded) * min_dtype
    return causal_mask + padding_mask


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.n_embd)
        self.layers = nn.ModuleList(
            [TransformerBlock(args) for _ in range(args.n_layer)]
        )
        self.norm = RMSNorm(args.n_embd, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        if args.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight
        self.rotary_emb = RotaryEmbedding(
            args.n_embd // args.n_head,
            max_position_embeddings=args.max_seq_len,
            base=args.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_router_logits: bool = False,
    ):
        batch_size, seq_len = input_ids.shape
        hidden_states = self.tok_embeddings(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = _build_causal_mask(
            seq_len, hidden_states.device, hidden_states.dtype, attention_mask
        )

        router_logits: list[torch.Tensor] = []
        for layer in self.layers:
            if return_router_logits:
                hidden_states, layer_router_logits = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    return_router_logits=True,
                )
                router_logits.append(layer_router_logits)
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        if labels is None and not return_router_logits:
            return logits
        if return_router_logits:
            return logits, loss, tuple(router_logits)
        return logits, loss
