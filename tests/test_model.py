from pathlib import Path
import sys

import torch

from transformers import LlamaConfig, MixtralConfig
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralSparseMoeBlock,
    MixtralRotaryEmbedding,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model


def _make_llama_config(
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
) -> LlamaConfig:
    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=max_seq_len,
        rope_theta=10000.0,
        mlp_bias=False,
    )
    config._attn_implementation = "eager"
    return config


def _make_mixtral_config(
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
    num_experts: int,
    top_k: int,
) -> MixtralConfig:
    config = MixtralConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=max_seq_len,
        rope_theta=10000.0,
        num_local_experts=num_experts,
        num_experts_per_tok=top_k,
        router_jitter_noise=0.0,
        hidden_act="silu",
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "batched_mm"
    return config


def _make_attention_mask(config, hidden_states, position_ids):
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    return create_causal_mask(
        config=config,
        input_embeds=hidden_states,
        attention_mask=None,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )


def test_rmsnorm_matches_llama():
    torch.manual_seed(0)
    hidden_states = torch.randn(2, 4, 64)
    ours = model.RMSNorm(64, eps=1e-6)
    ref = LlamaRMSNorm(64, eps=1e-6)
    ref.weight.data.copy_(ours.weight.data)
    ours.eval()
    ref.eval()

    with torch.no_grad():
        out_ours = ours(hidden_states)
        out_ref = ref(hidden_states)

    assert torch.allclose(out_ours, out_ref, atol=1e-5, rtol=1e-5)


def test_rope_matches_llama():
    torch.manual_seed(0)
    batch = 2
    seq_len = 8
    head_dim = 16
    num_heads = 4
    hidden_size = head_dim * num_heads
    config = _make_llama_config(hidden_size, 64, num_heads, num_heads, seq_len)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    hidden_states = torch.zeros(batch, seq_len, hidden_size)

    ours = model.RotaryEmbedding(
        head_dim, max_position_embeddings=seq_len, base=10000.0
    )
    ref = LlamaRotaryEmbedding(config)

    cos_ours, sin_ours = ours(hidden_states, position_ids)
    cos_ref, sin_ref = ref(hidden_states, position_ids)

    assert torch.allclose(cos_ours, cos_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sin_ours, sin_ref, atol=1e-5, rtol=1e-5)

    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)
    q_ours, k_ours = model.apply_rotary_pos_emb(q, k, cos_ours, sin_ours)
    q_ref, k_ref = llama_apply_rotary_pos_emb(q, k, cos_ref, sin_ref)

    assert torch.allclose(q_ours, q_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(k_ours, k_ref, atol=1e-5, rtol=1e-5)


def test_swiglu_matches_llama_mlp():
    torch.manual_seed(0)
    hidden_size = 32
    intermediate_size = 64
    config = _make_llama_config(hidden_size, intermediate_size, 4, 4, 16)

    ours = model.SwiGLU(hidden_size, intermediate_size, bias=False)
    ref = LlamaMLP(config)

    ours.gate_proj.weight.data.copy_(ref.gate_proj.weight.data)
    ours.up_proj.weight.data.copy_(ref.up_proj.weight.data)
    ours.down_proj.weight.data.copy_(ref.down_proj.weight.data)

    hidden_states = torch.randn(2, 5, hidden_size)
    ours.eval()
    ref.eval()

    with torch.no_grad():
        out_ours = ours(hidden_states)
        out_ref = ref(hidden_states)

    assert torch.allclose(out_ours, out_ref, atol=1e-5, rtol=1e-5)


def test_attention_matches_llama():
    torch.manual_seed(0)
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 4
    seq_len = 8
    config = _make_llama_config(hidden_size, 128, num_heads, num_kv_heads, seq_len)

    args = model.ModelArgs(
        vocab_size=32000,
        max_seq_len=seq_len,
        n_layer=1,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=hidden_size,
        ffn_hidden_size=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        dropout=0.0,
        attention_dropout=0.0,
        bias=False,
    )

    ours = model.CausalSelfAttention(args)
    ref = LlamaAttention(config, layer_idx=0)

    ours.q_proj.weight.data.copy_(ref.q_proj.weight.data)
    ours.k_proj.weight.data.copy_(ref.k_proj.weight.data)
    ours.v_proj.weight.data.copy_(ref.v_proj.weight.data)
    ours.o_proj.weight.data.copy_(ref.o_proj.weight.data)

    hidden_states = torch.randn(2, seq_len, hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    rope = MixtralRotaryEmbedding(config)
    position_embeddings = rope(hidden_states, position_ids)
    attention_mask = _make_attention_mask(config, hidden_states, position_ids)

    ours.eval()
    ref.eval()

    with torch.no_grad():
        out_ours = ours(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        out_ref, _ = ref(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

    assert torch.allclose(out_ours, out_ref, atol=1e-5, rtol=1e-5)


def test_moe_matches_mixtral():
    torch.manual_seed(0)
    hidden_size = 32
    intermediate_size = 64
    num_experts = 4
    top_k = 2
    seq_len = 6

    config = _make_mixtral_config(
        hidden_size,
        intermediate_size,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=16,
        num_experts=num_experts,
        top_k=top_k,
    )

    ours = model.MoE(
        hidden_size,
        intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        router_jitter=0.0,
        bias=False,
    )
    ref = MixtralSparseMoeBlock(config)

    ours.router.weight.data.copy_(ref.gate.weight.data)
    for idx in range(num_experts):
        gate_up = ref.experts.gate_up_proj[idx]
        gate, up = gate_up.chunk(2, dim=0)
        ours.experts[idx].gate_proj.weight.data.copy_(gate)
        ours.experts[idx].up_proj.weight.data.copy_(up)
        ours.experts[idx].down_proj.weight.data.copy_(ref.experts.down_proj[idx])

    hidden_states = torch.randn(2, seq_len, hidden_size)
    ours.eval()
    ref.eval()

    with torch.no_grad():
        out_ours = ours(hidden_states)
        out_ref = ref(hidden_states)

    assert torch.allclose(out_ours, out_ref, atol=1e-5, rtol=1e-5)


def test_block_matches_llama_decoder():
    torch.manual_seed(0)
    hidden_size = 64
    intermediate_size = 128
    num_heads = 4
    num_kv_heads = 4
    seq_len = 8
    config = _make_llama_config(
        hidden_size, intermediate_size, num_heads, num_kv_heads, seq_len
    )

    args = model.ModelArgs(
        vocab_size=32000,
        max_seq_len=seq_len,
        n_layer=1,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=hidden_size,
        ffn_hidden_size=intermediate_size,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        dropout=0.0,
        attention_dropout=0.0,
        bias=False,
        n_expert=0,
    )

    ours = model.TransformerBlock(args)
    ref = LlamaDecoderLayer(config, layer_idx=0)

    ours.attn.q_proj.weight.data.copy_(ref.self_attn.q_proj.weight.data)
    ours.attn.k_proj.weight.data.copy_(ref.self_attn.k_proj.weight.data)
    ours.attn.v_proj.weight.data.copy_(ref.self_attn.v_proj.weight.data)
    ours.attn.o_proj.weight.data.copy_(ref.self_attn.o_proj.weight.data)
    ours.input_layernorm.weight.data.copy_(ref.input_layernorm.weight.data)
    ours.post_attention_layernorm.weight.data.copy_(
        ref.post_attention_layernorm.weight.data
    )
    ours.mlp.gate_proj.weight.data.copy_(ref.mlp.gate_proj.weight.data)
    ours.mlp.up_proj.weight.data.copy_(ref.mlp.up_proj.weight.data)
    ours.mlp.down_proj.weight.data.copy_(ref.mlp.down_proj.weight.data)

    hidden_states = torch.randn(2, seq_len, hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    rope = LlamaRotaryEmbedding(config)
    position_embeddings = rope(hidden_states, position_ids)
    attention_mask = _make_attention_mask(config, hidden_states, position_ids)

    ours.eval()
    ref.eval()

    with torch.no_grad():
        out_ours = ours(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        out_ref = ref(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

    assert torch.allclose(out_ours, out_ref, atol=1e-5, rtol=1e-5)


def test_moe_block_matches_mixtral_decoder():
    torch.manual_seed(0)
    hidden_size = 32
    intermediate_size = 64
    num_heads = 4
    num_kv_heads = 4
    num_experts = 4
    top_k = 2
    seq_len = 8

    config = _make_mixtral_config(
        hidden_size,
        intermediate_size,
        num_heads,
        num_kv_heads,
        max_seq_len=seq_len,
        num_experts=num_experts,
        top_k=top_k,
    )

    args = model.ModelArgs(
        vocab_size=32000,
        max_seq_len=seq_len,
        n_layer=1,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=hidden_size,
        ffn_hidden_size=intermediate_size,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        dropout=0.0,
        attention_dropout=0.0,
        bias=False,
        n_expert=num_experts,
        moe_top_k=top_k,
        moe_jitter=0.0,
    )

    ours = model.TransformerBlock(args)
    ref = MixtralDecoderLayer(config, layer_idx=0)

    ours.attn.q_proj.weight.data.copy_(ref.self_attn.q_proj.weight.data)
    ours.attn.k_proj.weight.data.copy_(ref.self_attn.k_proj.weight.data)
    ours.attn.v_proj.weight.data.copy_(ref.self_attn.v_proj.weight.data)
    ours.attn.o_proj.weight.data.copy_(ref.self_attn.o_proj.weight.data)
    ours.input_layernorm.weight.data.copy_(ref.input_layernorm.weight.data)
    ours.post_attention_layernorm.weight.data.copy_(
        ref.post_attention_layernorm.weight.data
    )
    ours.mlp.router.weight.data.copy_(ref.mlp.gate.weight.data)
    for idx in range(num_experts):
        gate_up = ref.mlp.experts.gate_up_proj[idx]
        gate, up = gate_up.chunk(2, dim=0)
        ours.mlp.experts[idx].gate_proj.weight.data.copy_(gate)
        ours.mlp.experts[idx].up_proj.weight.data.copy_(up)
        ours.mlp.experts[idx].down_proj.weight.data.copy_(
            ref.mlp.experts.down_proj[idx]
        )

    hidden_states = torch.randn(2, seq_len, hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    rope = LlamaRotaryEmbedding(config)
    position_embeddings = rope(hidden_states, position_ids)
    attention_mask = _make_attention_mask(config, hidden_states, position_ids)

    ours.eval()
    ref.eval()

    with torch.no_grad():
        out_ours = ours(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        out_ref = ref(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )

    assert torch.allclose(out_ours, out_ref, atol=1e-5, rtol=1e-5)
