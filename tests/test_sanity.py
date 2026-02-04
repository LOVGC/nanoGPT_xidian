from pathlib import Path
import sys

import torch
from torchinfo import summary

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model


def test_transformer_sanity():
    args = model.ModelArgs(
        vocab_size=32000,
        max_seq_len=128,
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

    model_instance = model.Transformer(args).eval()

    assert 4 <= args.n_expert <= 8
    assert isinstance(model_instance.layers[0].mlp, model.MoE)
    assert model_instance.layers[0].mlp.top_k == 2

    input_ids = torch.randint(0, args.vocab_size, (2, 16))
    with torch.no_grad():
        _ = model_instance(input_ids)

    total_params = sum(p.numel() for p in model_instance.parameters())
    print(f"Total parameters: {total_params}")
    assert total_params < 50_000_000

    print(
        summary(
            model_instance,
            input_data=(input_ids,),
            verbose=0,
            row_settings=("ascii_only",),
        )
    )
