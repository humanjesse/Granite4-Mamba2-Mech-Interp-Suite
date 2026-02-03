"""Visualization for multi-step token generation analysis."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import plotly.graph_objects as go


# ── Color scheme ─────────────────────────────────────────────────────────────
MAMBA_COLOR = "#ab47bc"
TRANSFORMER_COLOR = "#26a69a"


def create_multistep_dashboard(step_data, arch_map, head_agg="mean"):
    """Create a 2x2 dashboard for a single generation step.

    Layout:
        Top-left:     Token timeline with current step highlighted
        Top-right:    Logit lens (last 8 layers)
        Bottom-left:  Mamba attention heatmap (representative layer)
        Bottom-right: Transformer attention heatmap (representative layer)

    Args:
        step_data: Single step dict from run_multistep_generation().steps[i].
        arch_map: ArchitectureMap instance.
        head_agg: "mean" or "max" for head aggregation.

    Returns:
        matplotlib Figure.
    """
    result = step_data["extraction_result"]
    tokens = result["tokens"]
    step_idx = step_data["step_idx"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"Multi-Step Generation — Step {step_idx + 1}",
        fontsize=14,
        fontweight="bold",
    )

    _plot_token_timeline(axes[0, 0], step_data)
    _plot_logit_lens_compact(axes[0, 1], result.get("logit_lens"), n_layers=8)
    _plot_attention_panel(axes[1, 0], result, arch_map, "mamba", head_agg)
    _plot_attention_panel(axes[1, 1], result, arch_map, "attention", head_agg)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _plot_token_timeline(ax, step_data):
    """Show tokens as colored cells with the generated token highlighted."""
    result = step_data["extraction_result"]
    tokens = result["tokens"]
    step_idx = step_data["step_idx"]
    generated_token = step_data["generated_token"]
    top_preds = step_data["top_predictions"]

    n_tokens = len(tokens)
    # Show last 20 tokens if sequence is long
    max_display = 20
    offset = max(0, n_tokens - max_display)
    display_tokens = tokens[offset:]
    n_display = len(display_tokens)

    # Draw token boxes
    for i, tok in enumerate(display_tokens):
        tok_label = _truncate_token(tok, 10)
        # Last token is the one just before generation at this step
        # Color: light gray for context, lighter for earlier tokens
        is_last = (i == n_display - 1)
        if is_last:
            color = "#FFB74D"  # orange — the token the model is predicting from
            fontweight = "bold"
        else:
            color = "#E0E0E0"  # gray
            fontweight = "normal"

        ax.barh(0, 1, left=i, height=0.6, color=color, edgecolor="#999", linewidth=0.5)
        ax.text(i + 0.5, 0, tok_label, ha="center", va="center",
                fontsize=max(5, 8 - n_display // 10), fontweight=fontweight,
                fontfamily="monospace")

    if offset > 0:
        ax.text(-0.5, 0, f"...+{offset}", ha="right", va="center", fontsize=7, color="#888")

    # Show the predicted next token
    gen_label = _truncate_token(generated_token, 15)
    ax.barh(0, 1, left=n_display, height=0.6, color="#66BB6A", edgecolor="#333", linewidth=1.5)
    ax.text(n_display + 0.5, 0, gen_label, ha="center", va="center",
            fontsize=9, fontweight="bold", fontfamily="monospace", color="white")

    # Arrow
    ax.annotate("", xy=(n_display, 0), xytext=(n_display - 0.15, 0),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    # Top predictions below
    pred_text = "Top predictions: " + "  |  ".join(
        f'"{t}" {p:.1%}' for t, p in top_preds[:5]
    )
    ax.text(n_display / 2, -0.7, pred_text, ha="center", va="center",
            fontsize=8, fontfamily="monospace", color="#444")

    ax.set_xlim(-1 if offset > 0 else -0.3, n_display + 1.5)
    ax.set_ylim(-1.2, 0.8)
    ax.set_title(f"Token Timeline — Generated: \"{gen_label}\"", fontsize=11)
    ax.axis("off")


def _plot_logit_lens_compact(ax, logit_lens_data, n_layers=8):
    """Compact logit lens showing last N layers with top-5 predictions."""
    if not logit_lens_data or not logit_lens_data.get("layers"):
        ax.text(0.5, 0.5, "No logit lens data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Logit Lens")
        return

    all_layers = logit_lens_data["layers"]
    # Take the last n_layers
    layers = all_layers[-n_layers:] if len(all_layers) > n_layers else all_layers
    n_show = len(layers)
    k = min(5, len(layers[0]["tokens"]) if layers else 5)

    # Build heatmap data
    prob_matrix = np.zeros((n_show, k))
    annotations = []

    for i, layer_data in enumerate(layers):
        row_labels = []
        for j in range(k):
            if j < len(layer_data["probs"]):
                prob_matrix[i, j] = layer_data["probs"][j]
                tok = layer_data["tokens"][j] if j < len(layer_data["tokens"]) else "?"
                tok = tok[:8] + "…" if len(tok) > 8 else tok
                row_labels.append(f"{tok}\n{layer_data['probs'][j]:.0%}")
            else:
                row_labels.append("")
        annotations.append(row_labels)

    im = ax.imshow(prob_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Annotations
    for i in range(n_show):
        for j in range(k):
            val = prob_matrix[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, annotations[i][j], ha="center", va="center",
                    fontsize=6, fontfamily="monospace", color=text_color)

    ax.set_xticks(range(k))
    ax.set_xticklabels([f"Top {j+1}" for j in range(k)], fontsize=8)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels([f"L{d['layer']}" for d in layers], fontsize=8, fontfamily="monospace")
    ax.set_title(f"Logit Lens — Last {n_show} Layers", fontsize=11)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probability")


def _plot_attention_panel(ax, result, arch_map, layer_type, head_agg):
    """Plot attention heatmap for a representative layer of the given type."""
    if layer_type == "mamba":
        attn_dict = result.get("mamba_attention", {})
        indices = arch_map.mamba_indices
        cmap = "magma"
        label = "Mamba-2"
    else:
        attn_dict = result.get("transformer_attention", {})
        indices = arch_map.attention_indices
        cmap = "viridis"
        label = "Transformer"

    # Pick a representative layer (middle for Mamba, last for Transformer)
    available = [idx for idx in indices if idx in attn_dict]
    if not available:
        ax.text(0.5, 0.5, f"No {label} attention data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(f"{label} Attention")
        return

    if layer_type == "mamba":
        chosen = available[len(available) // 2]
    else:
        chosen = available[-1]

    attn = attn_dict[chosen][0]  # [heads, seq, seq]
    if head_agg == "mean":
        attn = attn.mean(dim=0).numpy()
    else:
        attn = attn.max(dim=0).values.numpy()

    tokens = result["tokens"]
    seq_len = len(tokens)
    display_tokens = [_truncate_token(t, 10) for t in tokens]

    vmax = np.percentile(attn[attn > 0], 99) if np.any(attn > 0) else 1.0

    if layer_type == "mamba" and vmax > 0:
        im = ax.imshow(attn, cmap=cmap, aspect="auto",
                       norm=mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax))
    else:
        im = ax.imshow(attn, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)

    if seq_len <= 30:
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        fontsize = max(5, 8 - seq_len // 8)
        ax.set_xticklabels(display_tokens, rotation=90, fontsize=fontsize)
        ax.set_yticklabels(display_tokens, fontsize=fontsize)
    else:
        tick_step = max(1, seq_len // 15)
        ticks = list(range(0, seq_len, tick_step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    ax.set_xlabel("Key", fontsize=8)
    ax.set_ylabel("Query", fontsize=8)
    ax.set_title(f"{label} Attention — Layer {chosen} ({head_agg})", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _empty_plotly(message: str) -> go.Figure:
    """Return a Plotly figure with centered instructional text."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color="gray"),
    )
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def create_token_timeline_plotly(step_data):
    """Horizontal token boxes with the generated token highlighted."""
    if step_data is None:
        return _empty_plotly("No step data available.")

    result = step_data["extraction_result"]
    tokens = result["tokens"]
    step_idx = step_data["step_idx"]
    generated_token = step_data["generated_token"]
    top_preds = step_data["top_predictions"]

    n_tokens = len(tokens)
    max_display = 20
    offset = max(0, n_tokens - max_display)
    display_tokens = tokens[offset:]
    n_display = len(display_tokens)

    fig = go.Figure()

    # Draw token boxes as bar segments
    for i, tok in enumerate(display_tokens):
        tok_label = _truncate_token(tok, 10)
        is_last = (i == n_display - 1)
        color = "#FFB74D" if is_last else "#E0E0E0"
        border_color = "#333" if is_last else "#999"

        fig.add_shape(
            type="rect", x0=i, x1=i + 1, y0=-0.3, y1=0.3,
            fillcolor=color, line=dict(color=border_color, width=1),
        )
        fig.add_annotation(
            x=i + 0.5, y=0, text=tok_label, showarrow=False,
            font=dict(size=max(8, 11 - n_display // 8), family="monospace",
                      color="black"),
        )

    # Generated token (green box)
    gen_label = _truncate_token(generated_token, 15)
    fig.add_shape(
        type="rect", x0=n_display, x1=n_display + 1, y0=-0.3, y1=0.3,
        fillcolor="#66BB6A", line=dict(color="#333", width=2),
    )
    fig.add_annotation(
        x=n_display + 0.5, y=0, text=f"<b>{gen_label}</b>", showarrow=False,
        font=dict(size=11, family="monospace", color="white"),
    )

    # Arrow annotation
    fig.add_annotation(
        x=n_display, y=0, ax=n_display - 0.3, ay=0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.5, arrowcolor="#333",
    )

    # Offset indicator
    if offset > 0:
        fig.add_annotation(
            x=-0.3, y=0, text=f"...+{offset}", showarrow=False,
            font=dict(size=9, color="#888"),
        )

    # Top predictions as subtitle
    pred_text = "Top: " + " | ".join(
        f'"{t}" {p:.1%}' for t, p in top_preds[:5]
    )
    fig.add_annotation(
        x=(n_display + 1) / 2, y=-0.6, text=pred_text, showarrow=False,
        font=dict(size=10, family="monospace", color="#666"),
    )

    fig.update_layout(
        title=f"Token Timeline — Step {step_idx + 1} → \"{gen_label}\"",
        xaxis=dict(visible=False, range=[-1 if offset > 0 else -0.3, n_display + 1.5]),
        yaxis=dict(visible=False, range=[-1, 0.6]),
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def create_logit_lens_compact_plotly(step_data, n_layers=8):
    """Heatmap of last 8 layers showing top-k predictions and probabilities."""
    if step_data is None:
        return _empty_plotly("No step data available.")

    result = step_data["extraction_result"]
    logit_lens_data = result.get("logit_lens")
    if not logit_lens_data or not logit_lens_data.get("layers"):
        return _empty_plotly("No logit lens data available for this step.")

    all_layers = logit_lens_data["layers"]
    layers = all_layers[-n_layers:] if len(all_layers) > n_layers else all_layers
    n_show = len(layers)
    k = min(5, len(layers[0]["tokens"]) if layers else 5)

    # Build data
    prob_matrix = np.zeros((n_show, k))
    annotations = []
    for i, layer_data in enumerate(layers):
        row = []
        for j in range(k):
            if j < len(layer_data["probs"]):
                prob_matrix[i, j] = layer_data["probs"][j]
                tok = layer_data["tokens"][j] if j < len(layer_data["tokens"]) else "?"
                tok = tok[:8] + "…" if len(tok) > 8 else tok
                row.append(f"{tok}<br>{layer_data['probs'][j]:.0%}")
            else:
                row.append("")
        annotations.append(row)

    # Hover text
    hover = []
    for i, layer_data in enumerate(layers):
        row = []
        for j in range(k):
            if j < len(layer_data["probs"]):
                tok = layer_data["tokens"][j] if j < len(layer_data["tokens"]) else "?"
                row.append(f"Layer {layer_data['layer']}, Top-{j+1}: {tok} (p={layer_data['probs'][j]:.3f})")
            else:
                row.append("")
        hover.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=prob_matrix,
        x=[f"Top {j+1}" for j in range(k)],
        y=[f"L{d['layer']}" for d in layers],
        colorscale="YlOrRd", zmin=0, zmax=1,
        hovertext=hover, hoverinfo="text",
        colorbar=dict(title="Probability"),
        text=annotations, texttemplate="%{text}",
    ))

    fig.update_layout(
        title=f"Logit Lens — Last {n_show} Layers",
        height=400,
    )
    return fig


def _get_attention_plotly(step_data, arch_map, layer_type, head_agg):
    """Build a Plotly heatmap for a representative layer of the given type."""
    if step_data is None:
        return _empty_plotly("No step data available.")

    result = step_data["extraction_result"]
    if layer_type == "mamba":
        attn_dict = result.get("mamba_attention", {})
        indices = arch_map.mamba_indices
        colorscale = "Magma"
        label = "Mamba-2"
    else:
        attn_dict = result.get("transformer_attention", {})
        indices = arch_map.attention_indices
        colorscale = "Viridis"
        label = "Transformer"

    available = [idx for idx in indices if idx in attn_dict]
    if not available:
        return _empty_plotly(f"No {label} attention data available.")

    # Pick representative layer
    chosen = available[len(available) // 2] if layer_type == "mamba" else available[-1]

    attn = attn_dict[chosen][0]  # [heads, seq, seq]
    if head_agg == "mean":
        attn = attn.mean(dim=0).numpy()
    else:
        attn = attn.max(dim=0).values.numpy()

    tokens = result["tokens"]
    display_tokens = [_truncate_token(t, 10) for t in tokens]

    fig = go.Figure(data=go.Heatmap(
        z=attn, x=display_tokens, y=display_tokens,
        colorscale=colorscale,
        hovertemplate="Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>",
        colorbar=dict(title="Weight"),
    ))

    fig.update_layout(
        title=f"{label} Attention — Layer {chosen} ({head_agg})",
        xaxis_title="Key",
        yaxis_title="Query",
        height=500,
    )
    return fig


def create_mamba_attention_plotly(step_data, arch_map, head_agg="mean"):
    """Attention heatmap for a representative Mamba layer."""
    return _get_attention_plotly(step_data, arch_map, "mamba", head_agg)


def create_transformer_attention_plotly(step_data, arch_map, head_agg="mean"):
    """Attention heatmap for a representative Transformer layer."""
    return _get_attention_plotly(step_data, arch_map, "attention", head_agg)


def format_multistep_info(multistep_result, current_step):
    """Format multi-step generation info for the text panel.

    Args:
        multistep_result: Full output from run_multistep_generation().
        current_step: Currently selected step index.

    Returns:
        Formatted info string.
    """
    steps = multistep_result["steps"]
    num_steps = len(steps)
    original = multistep_result["original_prompt"]

    lines = [
        f"Multi-Step Generation — Step {current_step + 1}/{num_steps}",
        f"Original prompt: \"{original}\"",
        "",
    ]

    # Build generation sequence
    gen_tokens = []
    for i, s in enumerate(steps):
        tok = s["generated_token"].strip() or repr(s["generated_token"])
        if i == current_step:
            gen_tokens.append(f"[>{tok}<]")
        else:
            gen_tokens.append(f"[{tok}]")

    lines.append("Generated: " + " ".join(gen_tokens))
    lines.append(f"Full text: \"{multistep_result['final_text']}\"")
    lines.append("")

    # Current step details
    step = steps[current_step]
    gen_tok = step["generated_token"].strip() or repr(step["generated_token"])
    lines.append(f"Step {current_step + 1} generated: \"{gen_tok}\"")
    lines.append("Top 5 predictions:")
    for i, (tok, prob) in enumerate(step["top_predictions"]):
        marker = " <--" if i == 0 else ""
        lines.append(f"  {i+1}. \"{tok}\" — {prob:.1%}{marker}")

    lines.append("")
    lines.append(f"Sequence length at this step: {len(step['extraction_result']['tokens'])} tokens")

    return "\n".join(lines)


def _truncate_token(token, max_len=12):
    """Truncate a token string for display."""
    token = token.replace("Ġ", " ").replace("▁", " ").strip()
    if not token:
        token = "□"
    if len(token) > max_len:
        token = token[:max_len - 1] + "…"
    return token
