"""Extract standard attention weights from Transformer layers.

The GraniteMoeHybrid model doesn't return attention weights via output_attentions,
so we manually compute them from Q, K projections using hooks.
"""

import math
import torch
import torch.nn.functional as F


def extract_transformer_attention_via_hooks(
    model,
    attention_layer_indices: list[int],
) -> tuple[list, dict]:
    """Register hooks to capture and compute attention weights from self_attn layers.

    Hooks Q and K projections, then computes softmax(QK^T / sqrt(d_k)) in a
    forward hook on the self_attn module.

    Returns (hooks, storage_dict) where storage_dict maps layer_idx -> tensor
    after forward pass.
    """
    storage = {}
    hooks = []
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads

    for layer_idx in attention_layer_indices:
        layer = model.model.layers[layer_idx]
        if not hasattr(layer, "self_attn"):
            continue

        # Per-layer storage for Q and K
        qk_store = {}

        def make_q_hook(store):
            def hook_fn(module, input, output):
                store["Q"] = output.detach()
            return hook_fn

        def make_k_hook(store):
            def hook_fn(module, input, output):
                store["K"] = output.detach()
            return hook_fn

        h_q = layer.self_attn.q_proj.register_forward_hook(make_q_hook(qk_store))
        h_k = layer.self_attn.k_proj.register_forward_hook(make_k_hook(qk_store))
        hooks.extend([h_q, h_k])

        def make_attn_hook(idx, store, n_heads, n_kv_heads, h_dim):
            def hook_fn(module, args, kwargs, output):
                if "Q" not in store or "K" not in store:
                    return
                Q = store["Q"].float()
                K = store["K"].float()
                batch, seq_len, _ = Q.shape

                # Reshape to [batch, heads, seq, head_dim]
                Q = Q.view(batch, seq_len, n_heads, h_dim).transpose(1, 2)
                K = K.view(batch, seq_len, n_kv_heads, h_dim).transpose(1, 2)

                # Repeat KV heads to match Q heads (GQA)
                if n_kv_heads != n_heads:
                    repeats = n_heads // n_kv_heads
                    K = K.repeat_interleave(repeats, dim=1)

                # Compute attention weights
                scale = math.sqrt(h_dim)
                attn_weights = torch.matmul(Q, K.transpose(-1, -2)) / scale

                # Apply causal mask
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=Q.device),
                    diagonal=1,
                )
                attn_weights = attn_weights + causal_mask

                attn_weights = F.softmax(attn_weights, dim=-1)
                storage[idx] = attn_weights.detach()

            return hook_fn

        h_attn = layer.self_attn.register_forward_hook(
            make_attn_hook(layer_idx, qk_store, num_heads, num_kv_heads, head_dim),
            with_kwargs=True,
        )
        hooks.append(h_attn)

    return hooks, storage
