"""Visualization for circuit discovery results.

Three main visualizations:
1. Path matrix heatmap — [num_layers x num_layers] showing connection importance
2. Component importance — attention heads within Transformer layers
3. Circuit diagram — graph of discovered nodes and edges

Plus a combined dashboard and info formatting.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np


# ── Color scheme (consistent with intervention_viz) ──────────────────────────
MAMBA_COLOR = "#ab47bc"
TRANSFORMER_COLOR = "#26a69a"
MAMBA_BG = "#2d0040"
TRANSFORMER_BG = "#002d20"


def create_circuit_dashboard(circuit_result, arch_map) -> plt.Figure:
    """Create a multi-panel dashboard for circuit discovery results.

    Layout depends on whether component data is available:
      FAST mode:  2x2 (path matrix, layer importance, circuit diagram, summary)
      DETAILED mode: 3x2 (adds component importance panel)
    """
    if circuit_result is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No circuit discovery results available.",
                ha="center", va="center", fontsize=14, color="gray",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    has_components = len(circuit_result.component_results) > 0

    if has_components:
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 2, 2], hspace=0.35, wspace=0.3)
        ax_path = fig.add_subplot(gs[0, 0])
        ax_layers = fig.add_subplot(gs[0, 1])
        ax_components = fig.add_subplot(gs[1, 0])
        ax_circuit = fig.add_subplot(gs[1, 1])
        ax_summary = fig.add_subplot(gs[2, :])
    else:
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.35, wspace=0.3)
        ax_path = fig.add_subplot(gs[0, 0])
        ax_layers = fig.add_subplot(gs[0, 1])
        ax_circuit = fig.add_subplot(gs[1, 0])
        ax_summary = fig.add_subplot(gs[1, 1])
        ax_components = None

    fig.suptitle(
        f"Circuit Discovery — \"{circuit_result.correct_token}\" vs \"{circuit_result.incorrect_token}\"",
        fontsize=14, fontweight="bold",
    )

    _plot_path_matrix(ax_path, circuit_result, arch_map)
    _plot_layer_importance(ax_layers, circuit_result, arch_map)
    if ax_components is not None:
        _plot_component_importance(ax_components, circuit_result, arch_map)
    _plot_circuit_diagram(ax_circuit, circuit_result, arch_map)
    _plot_circuit_summary(ax_summary, circuit_result, arch_map)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def create_path_matrix_figure(circuit_result, arch_map) -> plt.Figure:
    """Standalone path matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    _plot_path_matrix(ax, circuit_result, arch_map)
    fig.tight_layout()
    return fig


def create_circuit_diagram_figure(circuit_result, arch_map) -> plt.Figure:
    """Standalone circuit diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_circuit_diagram(ax, circuit_result, arch_map)
    fig.tight_layout()
    return fig


# ── Internal plotting functions ──────────────────────────────────────────────

def _plot_path_matrix(ax, circuit_result, arch_map):
    """Plot the [num_layers x num_layers] path patching heatmap.

    Entry [i,j] = recovery when patching the path from layer i to layer j.
    Only upper triangle is meaningful (source < target).
    """
    matrix = circuit_result.path_matrix.numpy()
    num_layers = matrix.shape[0]

    # Mask lower triangle (meaningless: source >= target)
    mask = np.tril(np.ones_like(matrix, dtype=bool), k=0)
    masked = np.ma.array(matrix, mask=mask)

    vmax = max(abs(matrix).max(), 0.1)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(masked, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")

    # Overlay layer type indicators on axes
    tick_step = max(1, num_layers // 16)
    tick_positions = list(range(0, num_layers, tick_step))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)

    xlabels = []
    ylabels = []
    for i in tick_positions:
        lt = arch_map.layer_type(i)
        marker = "M" if lt == "mamba" else "T"
        xlabels.append(f"{i}{marker}")
        ylabels.append(f"{i}{marker}")

    ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(ylabels, fontsize=7)

    ax.set_xlabel("Target Layer (downstream)", fontsize=10)
    ax.set_ylabel("Source Layer (upstream)", fontsize=10)
    ax.set_title("Path Patching Matrix", fontsize=12)

    # Highlight Transformer layer rows/columns
    for idx in arch_map.attention_indices:
        ax.axhline(y=idx, color=TRANSFORMER_COLOR, linewidth=0.5, alpha=0.3)
        ax.axvline(x=idx, color=TRANSFORMER_COLOR, linewidth=0.5, alpha=0.3)

    plt.colorbar(im, ax=ax, label="Fractional Recovery", shrink=0.8)


def _plot_layer_importance(ax, circuit_result, arch_map):
    """Bar chart of layer-level importance from standard activation patching."""
    importance = circuit_result.layer_importance.numpy()
    layers = circuit_result.layer_indices
    threshold = circuit_result.threshold

    colors = [
        MAMBA_COLOR if arch_map.layer_type(idx) == "mamba" else TRANSFORMER_COLOR
        for idx in layers
    ]

    bars = ax.bar(range(len(layers)), importance, color=colors, width=0.8)

    # Highlight bars above threshold
    for i, val in enumerate(importance):
        if abs(val) >= threshold:
            bars[i].set_edgecolor("gold")
            bars[i].set_linewidth(1.5)

    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.axhline(y=threshold, color="gold", linewidth=1, linestyle="--", alpha=0.6, label=f"threshold={threshold}")
    ax.axhline(y=-threshold, color="gold", linewidth=1, linestyle="--", alpha=0.6)

    # Background shading
    for i, idx in enumerate(layers):
        lt = arch_map.layer_type(idx)
        bg = MAMBA_BG if lt == "mamba" else TRANSFORMER_BG
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color=bg)

    tick_step = max(1, len(layers) // 16)
    ax.set_xticks(range(0, len(layers), tick_step))
    ax.set_xticklabels([str(layers[i]) for i in range(0, len(layers), tick_step)], fontsize=8)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Fractional Recovery", fontsize=10)
    ax.set_title("Layer Importance (Activation Patching)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        handles=[
            mpatches.Patch(color=MAMBA_COLOR, label="Mamba"),
            mpatches.Patch(color=TRANSFORMER_COLOR, label="Transformer"),
        ],
        fontsize=9, loc="upper right",
    )


def _plot_component_importance(ax, circuit_result, arch_map):
    """Plot attention head importance for Transformer layers."""
    components = circuit_result.component_results
    if not components:
        ax.text(0.5, 0.5, "No component data\n(use DETAILED mode)",
                ha="center", va="center", fontsize=12, color="gray",
                transform=ax.transAxes)
        ax.set_title("Component Importance", fontsize=12)
        return

    # Group by layer
    layers_with_heads = sorted(set(c.layer_idx for c in components))
    num_heads = max(c.component_idx for c in components) + 1

    # Build matrix: [num_attn_layers x num_heads]
    matrix = np.zeros((len(layers_with_heads), num_heads))
    for c in components:
        row = layers_with_heads.index(c.layer_idx)
        matrix[row, c.component_idx] = c.recovery_score

    vmax = max(abs(matrix).max(), 0.1)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")

    ax.set_yticks(range(len(layers_with_heads)))
    ax.set_yticklabels([f"L{idx}" for idx in layers_with_heads], fontsize=9)
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([f"H{i}" for i in range(num_heads)], fontsize=8)

    ax.set_xlabel("Attention Head", fontsize=10)
    ax.set_ylabel("Transformer Layer", fontsize=10)
    ax.set_title("Attention Head Importance", fontsize=12)

    # Annotate cells with values
    threshold = circuit_result.threshold
    for row in range(len(layers_with_heads)):
        for col in range(num_heads):
            val = matrix[row, col]
            if abs(val) >= threshold:
                ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="white" if abs(val) > vmax * 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Recovery", shrink=0.8)


def _plot_circuit_diagram(ax, circuit_result, arch_map):
    """Plot a simplified circuit diagram showing nodes and edges.

    Nodes are placed vertically by layer index, with Mamba on the left
    and Transformer on the right. Edges show connections with thickness
    proportional to importance.
    """
    nodes = circuit_result.circuit_nodes
    edges = circuit_result.circuit_edges

    if not nodes:
        ax.text(0.5, 0.5, "No circuit found above threshold\n(try lowering the threshold)",
                ha="center", va="center", fontsize=12, color="gray",
                transform=ax.transAxes)
        ax.set_title("Discovered Circuit", fontsize=12)
        return

    num_layers = arch_map.num_layers

    # Position nodes: y = evenly spaced within each type column, x = type
    # Group nodes by type for even spacing
    mamba_nodes = [n for n in nodes if n.layer_type == "mamba"]
    attn_nodes = [n for n in nodes if n.layer_type != "mamba"]
    # Sort by layer index so vertical order matches layer order
    mamba_nodes.sort(key=lambda n: n.layer_idx)
    attn_nodes.sort(key=lambda n: n.layer_idx)

    node_positions = {}
    for group, x_pos in [(mamba_nodes, 0.3), (attn_nodes, 0.7)]:
        n = len(group)
        for i, node in enumerate(group):
            # Space evenly with margin, or center if single node
            y = 1.0 - (i / max(n - 1, 1)) if n > 1 else 0.5
            node_positions[node.layer_idx] = (x_pos, y)

    # Draw edges first (behind nodes)
    max_edge_importance = max((abs(e.importance) for e in edges), default=1.0)
    for edge in edges:
        if edge.source_layer in node_positions and edge.target_layer in node_positions:
            x1, y1 = node_positions[edge.source_layer]
            x2, y2 = node_positions[edge.target_layer]

            # Thickness proportional to importance
            width = 0.5 + 3.0 * abs(edge.importance) / max(max_edge_importance, 1e-6)
            alpha = 0.3 + 0.5 * abs(edge.importance) / max(max_edge_importance, 1e-6)

            color = "#e53935" if edge.importance > 0 else "#1e88e5"
            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.15,head_length=0.1",
                    color=color, lw=width, alpha=min(alpha, 0.9),
                    connectionstyle="arc3,rad=0.1",
                ),
            )

    # Draw nodes
    for node in nodes:
        x, y = node_positions[node.layer_idx]
        color = MAMBA_COLOR if node.layer_type == "mamba" else TRANSFORMER_COLOR
        size = 200 + 400 * abs(node.importance)
        ax.scatter(x, y, s=size, c=color, edgecolors="white", linewidths=1.5, zorder=5)
        ax.text(x, y, f"L{node.layer_idx}", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=6)

        # Label importance next to node
        label_x = x + (0.06 if node.layer_type == "mamba" else -0.06)
        ha = "left" if node.layer_type == "mamba" else "right"
        ax.text(label_x, y, f"{node.importance:+.2f}", ha=ha, va="center",
                fontsize=7, color=color)

    # Layout
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Discovered Circuit", fontsize=12)
    ax.set_xticks([0.3, 0.7])
    ax.set_xticklabels(["Mamba", "Transformer"], fontsize=10)
    ax.set_ylabel("Layer (top=early, bottom=late)", fontsize=9)
    ax.set_yticks([])

    # Legend for edge colors
    ax.legend(
        handles=[
            mpatches.Patch(color="#e53935", label="Positive path"),
            mpatches.Patch(color="#1e88e5", label="Negative path"),
            mpatches.Patch(color=MAMBA_COLOR, label="Mamba node"),
            mpatches.Patch(color=TRANSFORMER_COLOR, label="Transformer node"),
        ],
        fontsize=8, loc="lower right",
    )


def _plot_circuit_summary(ax, circuit_result, arch_map):
    """Text summary of circuit discovery results."""
    ax.axis("off")

    nodes = circuit_result.circuit_nodes
    edges = circuit_result.circuit_edges
    path_matrix = circuit_result.path_matrix.numpy()

    lines = [
        f"Correct: {circuit_result.correct_token_display}",
        f"Incorrect: {circuit_result.incorrect_token_display}",
        f"Clean logit diff:     {circuit_result.clean_logit_diff:+.3f}   P(correct): {circuit_result.clean_prob_correct:.4f}",
        f"Corrupted logit diff: {circuit_result.corrupted_logit_diff:+.3f}   P(correct): {circuit_result.corrupted_prob_correct:.4f}",
        f"Threshold: {circuit_result.threshold}   Granularity: {circuit_result.granularity.value}",
        "",
        f"Circuit: {len(nodes)} nodes, {len(edges)} edges",
    ]

    # Node summary
    if nodes:
        n_mamba = sum(1 for n in nodes if n.layer_type == "mamba")
        n_attn = sum(1 for n in nodes if n.layer_type == "attention")
        lines.append(f"  Mamba nodes: {n_mamba}   Transformer nodes: {n_attn}")
        top_nodes = sorted(nodes, key=lambda n: abs(n.importance), reverse=True)[:5]
        lines.append("  Top nodes:")
        for n in top_nodes:
            lines.append(f"    Layer {n.layer_idx:2d} ({n.layer_type:>5s}): {n.importance:+.3f}")

    # Top paths
    if path_matrix.size > 0:
        lines.append("")
        # Flatten upper triangle
        upper = np.triu_indices_from(path_matrix, k=1)
        values = path_matrix[upper]
        if len(values) > 0:
            top_k = min(5, len(values))
            top_flat_idx = np.argsort(np.abs(values))[::-1][:top_k]
            lines.append("Top paths (source → target):")
            for fi in top_flat_idx:
                s, t = upper[0][fi], upper[1][fi]
                val = values[fi]
                s_type = "M" if arch_map.layer_type(s) == "mamba" else "T"
                t_type = "M" if arch_map.layer_type(t) == "mamba" else "T"
                lines.append(f"  L{s}{s_type} → L{t}{t_type}: {val:+.3f}")

    # Component summary
    if circuit_result.component_results:
        lines.append("")
        top_comps = sorted(
            circuit_result.component_results,
            key=lambda c: abs(c.recovery_score), reverse=True
        )[:5]
        lines.append("Top attention heads:")
        for c in top_comps:
            lines.append(f"  Layer {c.layer_idx} Head {c.component_idx}: {c.recovery_score:+.3f}")

    text = "\n".join(lines)
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes, fontsize=9, fontfamily="monospace",
        verticalalignment="top",
    )


def format_circuit_info(circuit_result) -> str:
    """Format circuit discovery results as text for the Gradio info panel."""
    if circuit_result is None:
        return "No circuit discovery results available."

    nodes = circuit_result.circuit_nodes
    edges = circuit_result.circuit_edges

    lines = [
        f"Circuit Discovery — {circuit_result.granularity.value} mode",
        f"Correct token: {circuit_result.correct_token_display}",
        f"Incorrect token: {circuit_result.incorrect_token_display}",
        f"Clean logit diff: {circuit_result.clean_logit_diff:+.3f}  |  P(correct): {circuit_result.clean_prob_correct:.4f}",
        f"Corrupted logit diff: {circuit_result.corrupted_logit_diff:+.3f}  |  P(correct): {circuit_result.corrupted_prob_correct:.4f}",
        "",
        f"Threshold: {circuit_result.threshold}",
        f"Circuit: {len(nodes)} nodes, {len(edges)} edges",
    ]

    if nodes:
        top_node = max(nodes, key=lambda n: abs(n.importance))
        lines.append(
            f"Most important node: Layer {top_node.layer_idx} "
            f"({top_node.layer_type}) recovery={top_node.importance:+.3f}"
        )

    if edges:
        top_edge = max(edges, key=lambda e: abs(e.importance))
        lines.append(
            f"Strongest path: L{top_edge.source_layer} → L{top_edge.target_layer} "
            f"recovery={top_edge.importance:+.3f}"
        )

    path_matrix = circuit_result.path_matrix.numpy()
    upper = np.triu_indices_from(path_matrix, k=1)
    values = path_matrix[upper]
    if len(values) > 0:
        lines.append(f"Mean path recovery: {np.abs(values).mean():.4f}")
        lines.append(f"Max path recovery: {np.abs(values).max():.4f}")

    return "\n".join(lines)
