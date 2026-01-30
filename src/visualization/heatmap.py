"""Single-layer attention heatmaps."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch


def create_attention_heatmap(
    attention: torch.Tensor,
    tokens: list[str],
    layer_idx: int,
    layer_type: str,
    head_agg: str = "mean",
    title: str | None = None,
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """Create a heatmap for a single layer's attention pattern.

    Args:
        attention: [batch, heads, seq, seq] tensor
        tokens: List of token strings
        layer_idx: Layer index for labeling
        layer_type: "mamba" or "attention"
        head_agg: How to aggregate heads — "mean" or "max"
        title: Optional custom title
        figsize: Optional figure size

    Returns:
        matplotlib Figure
    """
    # Aggregate over batch and heads
    attn = attention[0]  # first batch element, [heads, seq, seq]

    if head_agg == "mean":
        attn = attn.mean(dim=0)  # [seq, seq]
    elif head_agg == "max":
        attn = attn.max(dim=0).values  # [seq, seq]
    else:
        attn = attn.mean(dim=0)

    attn = attn.numpy()
    seq_len = len(tokens)

    # Truncate token labels for display
    display_tokens = [_truncate_token(t, 12) for t in tokens]

    if figsize is None:
        size = max(6, min(16, seq_len * 0.5))
        figsize = (size, size)

    colormap = "magma" if layer_type == "mamba" else "viridis"

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # For mamba hidden attention, use symmetric log scale if values span wide range
    vmin = 0
    vmax = np.percentile(attn[attn > 0], 99) if np.any(attn > 0) else 1.0

    if layer_type == "mamba" and vmax > 0:
        # Use power norm for better visualization of skewed distributions
        im = ax.imshow(
            attn,
            cmap=colormap,
            aspect="auto",
            norm=mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax),
        )
    else:
        im = ax.imshow(attn, cmap=colormap, aspect="auto", vmin=vmin, vmax=vmax)

    # Labels
    if seq_len <= 40:
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(display_tokens, rotation=90, fontsize=max(5, 10 - seq_len // 10))
        ax.set_yticklabels(display_tokens, fontsize=max(5, 10 - seq_len // 10))
    else:
        # Too many tokens for labels, use numeric ticks
        tick_step = max(1, seq_len // 20)
        ticks = list(range(0, seq_len, tick_step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    ax.set_xlabel("Key (source token)")
    ax.set_ylabel("Query (target token)")

    if title is None:
        type_label = "Mamba-2 Hidden" if layer_type == "mamba" else "Transformer"
        title = f"Layer {layer_idx} — {type_label} Attention ({head_agg})"

    ax.set_title(title, fontsize=12)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    return fig


def create_all_layers_overview(
    mamba_attention: dict[int, torch.Tensor],
    transformer_attention: dict[int, torch.Tensor],
    tokens: list[str],
    layer_types: list[str],
    head_agg: str = "mean",
) -> plt.Figure:
    """Create a grid of small heatmaps for all layers."""
    num_layers = len(layer_types)
    cols = 8
    rows = (num_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_layers):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        layer_type = layer_types[i]
        if layer_type == "mamba" and i in mamba_attention:
            attn = mamba_attention[i][0]  # [heads, seq, seq]
        elif layer_type == "attention" and i in transformer_attention:
            attn = transformer_attention[i][0]
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"L{i}", fontsize=8)
            ax.axis("off")
            continue

        if head_agg == "mean":
            attn = attn.mean(dim=0).numpy()
        else:
            attn = attn.max(dim=0).values.numpy()

        cmap = "magma" if layer_type == "mamba" else "viridis"
        vmax = np.percentile(attn[attn > 0], 99) if np.any(attn > 0) else 1.0

        if layer_type == "mamba" and vmax > 0:
            ax.imshow(
                attn, cmap=cmap, aspect="auto",
                norm=mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax),
            )
        else:
            ax.imshow(attn, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)

        type_tag = "M" if layer_type == "mamba" else "A"
        ax.set_title(f"L{i} [{type_tag}]", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for i in range(num_layers, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle("All Layers Overview (magma=Mamba-2, viridis=Transformer)", fontsize=12)
    fig.tight_layout()
    return fig


def _truncate_token(token: str, max_len: int = 12) -> str:
    """Truncate a token string for display."""
    # Clean up common HF tokenizer prefixes
    token = token.replace("Ġ", "▁").replace("▁", " ").strip()
    if not token:
        token = "□"
    if len(token) > max_len:
        token = token[:max_len - 1] + "…"
    return token
