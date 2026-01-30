"""Side-by-side Mamba vs Transformer comparison views."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from .heatmap import _truncate_token


def create_comparison_view(
    mamba_attn: torch.Tensor,
    transformer_attn: torch.Tensor,
    tokens: list[str],
    mamba_layer_idx: int,
    transformer_layer_idx: int,
    head_agg: str = "mean",
) -> plt.Figure:
    """Create side-by-side comparison of Mamba-2 hidden attention vs Transformer attention.

    Args:
        mamba_attn: [batch, heads, seq, seq] from Mamba layer
        transformer_attn: [batch, heads, seq, seq] from Transformer layer
        tokens: Token strings
        mamba_layer_idx: Mamba layer index
        transformer_layer_idx: Transformer layer index
        head_agg: "mean" or "max"

    Returns:
        matplotlib Figure with side-by-side plots
    """
    seq_len = len(tokens)
    display_tokens = [_truncate_token(t, 10) for t in tokens]

    # Aggregate
    m_attn = mamba_attn[0]
    t_attn = transformer_attn[0]

    if head_agg == "mean":
        m_attn = m_attn.mean(dim=0).numpy()
        t_attn = t_attn.mean(dim=0).numpy()
    else:
        m_attn = m_attn.max(dim=0).values.numpy()
        t_attn = t_attn.max(dim=0).values.numpy()

    size = max(6, min(12, seq_len * 0.4))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(size * 2 + 1, size))

    # Mamba heatmap
    m_vmax = np.percentile(m_attn[m_attn > 0], 99) if np.any(m_attn > 0) else 1.0
    if m_vmax > 0:
        im1 = ax1.imshow(
            m_attn, cmap="magma", aspect="auto",
            norm=mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=m_vmax),
        )
    else:
        im1 = ax1.imshow(m_attn, cmap="magma", aspect="auto")

    ax1.set_title(f"Layer {mamba_layer_idx} — Mamba-2 Hidden Attention", fontsize=11)

    # Transformer heatmap
    t_vmax = t_attn.max() if t_attn.max() > 0 else 1.0
    im2 = ax2.imshow(t_attn, cmap="viridis", aspect="auto", vmin=0, vmax=t_vmax)
    ax2.set_title(
        f"Layer {transformer_layer_idx} — Transformer Attention", fontsize=11
    )

    # Labels
    for ax in (ax1, ax2):
        if seq_len <= 40:
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))
            fontsize = max(5, 9 - seq_len // 10)
            ax.set_xticklabels(display_tokens, rotation=90, fontsize=fontsize)
            ax.set_yticklabels(display_tokens, fontsize=fontsize)
        else:
            tick_step = max(1, seq_len // 20)
            ticks = list(range(0, seq_len, tick_step))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        ax.set_xlabel("Key (source)")
        ax.set_ylabel("Query (target)")

    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Mamba-2 vs Transformer Attention Comparison ({head_agg} over heads)",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def create_dt_heatmap(
    dt_values: dict[int, torch.Tensor],
    tokens: list[str],
) -> plt.Figure:
    """Create a heatmap of dt (dwell time) values across layers and tokens.

    Fallback visualization when full hidden attention isn't available.

    Args:
        dt_values: Dict mapping layer_idx -> dt tensor [batch, seq, heads]
        tokens: Token strings
    """
    if not dt_values:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No dt values captured", ha="center", va="center")
        return fig

    layers = sorted(dt_values.keys())
    seq_len = len(tokens)
    display_tokens = [_truncate_token(t, 10) for t in tokens]

    # Average dt over heads for each layer
    # Result: [num_mamba_layers, seq_len]
    dt_matrix = []
    for idx in layers:
        dt = dt_values[idx][0]  # first batch, [seq, heads]
        dt_matrix.append(dt.mean(dim=-1).numpy())

    dt_matrix = np.stack(dt_matrix)

    fig, ax = plt.subplots(figsize=(max(8, seq_len * 0.4), max(4, len(layers) * 0.3)))

    im = ax.imshow(dt_matrix, cmap="inferno", aspect="auto")
    ax.set_ylabel("Mamba Layer Index")
    ax.set_xlabel("Token")

    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(l) for l in layers], fontsize=8)

    if seq_len <= 40:
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(display_tokens, rotation=90, fontsize=max(5, 9 - seq_len // 10))

    ax.set_title("Mamba-2 dt (Dwell Time) — Higher = More Attention", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
