"""Visualization for MLP neuron activations."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def create_neuron_activation_heatmap(
    activations,
    tokens,
    layer_idx,
    layer_type,
    top_n=50,
):
    """Create a heatmap of top-N most active neurons across all tokens.

    Args:
        activations: Tensor [seq_len, intermediate_size] — raw SwiGLU intermediate values.
        tokens: list of token strings.
        layer_idx: layer index for title.
        layer_type: "mamba" or "attention" for title.
        top_n: number of top neurons to display (by max abs activation across tokens).

    Returns:
        matplotlib Figure.
    """
    act = activations.float()
    abs_act = act.abs()
    seq_len, n_neurons = abs_act.shape

    # Find top-N neurons by max absolute activation across all tokens
    max_per_neuron, _ = abs_act.max(dim=0)  # [n_neurons]
    top_n = min(top_n, n_neurons)
    _, top_neuron_indices = max_per_neuron.topk(top_n)
    top_neuron_indices, _ = top_neuron_indices.sort()  # sort by index for readability

    # Extract the activation values for these neurons (use abs for magnitude)
    heatmap_data = abs_act[:, top_neuron_indices].numpy()  # [seq_len, top_n]
    neuron_labels = [f"N{idx.item()}" for idx in top_neuron_indices]

    # Truncate token labels for display
    display_tokens = []
    for t in tokens:
        t_clean = t.replace("\u0120", " ").replace("\u010a", "\\n")
        if len(t_clean) > 12:
            t_clean = t_clean[:10] + ".."
        display_tokens.append(t_clean)

    # Figure sizing
    fig_width = max(10, top_n * 0.25)
    fig_height = max(4, seq_len * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(heatmap_data, aspect="auto", cmap="plasma", interpolation="nearest")

    # Labels
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(display_tokens, fontsize=8, fontfamily="monospace")

    if top_n <= 60:
        ax.set_xticks(range(top_n))
        ax.set_xticklabels(neuron_labels, fontsize=6, rotation=90, fontfamily="monospace")
    else:
        # Too many labels — show every 5th
        tick_positions = list(range(0, top_n, 5))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([neuron_labels[i] for i in tick_positions], fontsize=6, rotation=90)

    ax.set_xlabel("Neuron Index", fontsize=10)
    ax.set_ylabel("Token", fontsize=10)
    ax.set_title(
        f"Neuron Activation — Layer {layer_idx} ({layer_type.upper()}) — Top {top_n} neurons",
        fontsize=12,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("|Activation|", fontsize=9)

    fig.tight_layout()
    return fig


def format_neuron_info(activations, tokens, layer_idx, layer_type, top_k=10):
    """Format neuron activation info as a text string.

    Args:
        activations: Tensor [seq_len, intermediate_size].
        tokens: list of token strings.
        layer_idx: layer index.
        layer_type: "mamba" or "attention".
        top_k: number of top neurons to show per token.

    Returns:
        Formatted info string.
    """
    act = activations.float()
    abs_act = act.abs()
    seq_len, n_neurons = abs_act.shape

    mean_act = abs_act.mean().item()
    max_act = abs_act.max().item()
    sparsity = (abs_act < 0.1).float().mean().item() * 100

    lines = [
        f"Layer {layer_idx} ({layer_type.upper()})",
        f"Total neurons: {n_neurons}",
        f"Mean |activation|: {mean_act:.4f}",
        f"Max |activation|: {max_act:.4f}",
        f"Sparsity (|act| < 0.1): {sparsity:.1f}%",
        "",
        f"Per-token top-{top_k} neurons:",
    ]

    for tok_idx, tok in enumerate(tokens):
        tok_clean = tok.replace("\u0120", " ").replace("\u010a", "\\n")
        if len(tok_clean) > 15:
            tok_clean = tok_clean[:13] + ".."

        tok_abs = abs_act[tok_idx]
        top_vals, top_ids = tok_abs.topk(min(top_k, n_neurons))

        neuron_strs = [f"N{top_ids[i].item()}:{top_vals[i].item():.1f}" for i in range(len(top_vals))]
        lines.append(f"  {tok_clean!r} -> {', '.join(neuron_strs)}")

    return "\n".join(lines)
