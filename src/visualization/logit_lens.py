"""Logit Lens visualization — prediction evolution across layers."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def create_logit_lens_heatmap(logit_lens_data: dict, k: int = 10) -> plt.Figure:
    """Create a heatmap showing how top-k predictions evolve through layers.

    Args:
        logit_lens_data: Dict with "position", "position_token", and "layers" list.
            Each layer entry has "layer", "tokens", and "probs".
        k: Number of top tokens to display per layer.

    Returns:
        matplotlib Figure
    """
    layers = logit_lens_data["layers"]
    position_token = logit_lens_data["position_token"]
    num_layers = len(layers)

    if num_layers == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No logit lens data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.axis("off")
        return fig

    # Clamp k to actual available tokens
    k = min(k, len(layers[0]["probs"]))

    # Build probability matrix [num_layers, k]
    prob_matrix = np.zeros((num_layers, k))
    token_labels = [[""]*k for _ in range(num_layers)]

    for row, layer_data in enumerate(layers):
        for col in range(k):
            if col < len(layer_data["probs"]):
                prob_matrix[row, col] = layer_data["probs"][col]
                token_labels[row][col] = layer_data["tokens"][col]

    fig_width = max(10, k * 1.5)
    fig_height = max(8, num_layers * 0.42)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(prob_matrix, cmap="YlOrRd", aspect="auto", vmin=0,
                   vmax=max(prob_matrix.max(), 0.01))

    # Annotate cells with token and probability
    vmax = prob_matrix.max()
    threshold = vmax * 0.65 if vmax > 0 else 1.0

    for row in range(num_layers):
        for col in range(k):
            prob = prob_matrix[row, col]
            token = token_labels[row][col]

            # Truncate long tokens
            if len(token) > 10:
                token = token[:9] + "…"

            color = "white" if prob > threshold else "black"

            # Token name
            ax.text(col, row - 0.12, token, ha="center", va="center",
                    fontsize=7, fontweight="bold", color=color)
            # Probability percentage
            ax.text(col, row + 0.18, f"{prob:.1%}", ha="center", va="center",
                    fontsize=6, color=color)

    # Axis labels
    layer_indices = [ld["layer"] for ld in layers]
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f"L{idx}" for idx in layer_indices], fontsize=8)

    ax.set_xticks(range(k))
    ax.set_xticklabels([f"Top {i+1}" for i in range(k)], fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Clean token for title display
    display_token = position_token.replace("Ġ", " ").replace("▁", " ").strip()
    if not display_token:
        display_token = "□"
    ax.set_title(
        f'Logit Lens — Prediction Evolution\n(predicting next token after "{display_token}")',
        fontsize=13, pad=20,
    )

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label="Probability")
    fig.tight_layout()

    return fig
