"""Extract MLP neuron activations from Granite hybrid model layers."""

import torch


def register_neuron_hooks(model, arch_map):
    """Register forward hooks on shared_mlp to capture SwiGLU intermediate activations.

    The Granite MoE Hybrid MLP (GraniteMoeSharedMLP) computes:
        projected = input_linear(x)          # [batch, seq, intermediate_size * 2]
        chunks = projected.chunk(2, dim=-1)  # two [batch, seq, intermediate_size]
        intermediate = silu(chunks[0]) * chunks[1]  # <-- we capture this
        output = output_linear(intermediate)

    Returns:
        hooks: list of hook handles to remove after forward pass
        storage: dict mapping layer_idx -> Tensor[seq_len, intermediate_size]
    """
    storage = {}
    hooks = []

    for layer_idx in range(arch_map.num_layers):
        shared_mlp = model.model.layers[layer_idx].shared_mlp

        def hook_fn(module, input, output, _idx=layer_idx):
            hidden_states = input[0]  # [batch, seq, hidden_size]
            with torch.no_grad():
                projected = module.input_linear(hidden_states)
                chunks = projected.chunk(2, dim=-1)
                intermediate = module.activation(chunks[0]) * chunks[1]
                # Squeeze batch dimension (always 1 for inference)
                storage[_idx] = intermediate[0].detach()

        h = shared_mlp.register_forward_hook(hook_fn)
        hooks.append(h)

    return hooks, storage
