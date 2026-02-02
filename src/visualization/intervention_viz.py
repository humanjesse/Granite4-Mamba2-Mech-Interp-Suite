"""Visualization for causal intervention / activation patching results."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Patch

from src.extraction.causal_intervention import MIN_RECOVERY_DENOMINATOR


def create_layer_sweep_dashboard(sweep_result, arch_map) -> plt.Figure:
    """Create a 2-panel dashboard for layer sweep results.

    Top: Bar chart of recovery per layer, colored by layer type.
    Bottom: Baseline comparison and summary statistics.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"Activation Patching — {sweep_result.intervention_type.value}",
        fontsize=14,
        fontweight="bold",
    )

    _plot_layer_recovery_bar(axes[0], sweep_result, arch_map)
    _plot_baselines_summary(axes[1], sweep_result)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def create_full_sweep_dashboard(sweep_result, arch_map) -> plt.Figure:
    """Create a 2x2 dashboard for full position x layer sweep results.

    Top-left: Position x Layer heatmap of recovery scores.
    Top-right: Per-layer aggregated recovery (bar chart).
    Bottom-left: Per-position aggregated recovery (bar chart).
    Bottom-right: Summary statistics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Activation Patching — {sweep_result.intervention_type.value} (Full Sweep)",
        fontsize=14,
        fontweight="bold",
    )

    _plot_position_layer_heatmap(axes[0, 0], sweep_result, arch_map)
    _plot_layer_marginal(axes[0, 1], sweep_result, arch_map)
    _plot_position_marginal(axes[1, 0], sweep_result)
    _plot_baselines_summary(axes[1, 1], sweep_result)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _plot_layer_recovery_bar(ax, sweep_result, arch_map):
    """Bar chart of recovery score per layer, colored by layer type."""
    recovery = sweep_result.recovery_matrix.numpy()
    if recovery.ndim > 1:
        # For full sweep, aggregate over positions
        recovery = recovery.mean(axis=1)

    layers = sweep_result.layer_indices
    colors = [
        "#ab47bc" if arch_map.layer_type(idx) == "mamba" else "#26a69a"
        for idx in layers
    ]

    ax.bar(range(len(layers)), recovery, color=colors, width=0.8)

    # Reference lines
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.axhline(y=1.0, color="#4caf50", linewidth=1, linestyle="--", alpha=0.5)

    # Background shading by layer type
    for i, idx in enumerate(layers):
        lt = arch_map.layer_type(idx)
        bg = "#2d0040" if lt == "mamba" else "#002d20"
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color=bg)

    ax.set_xticks(range(0, len(layers), max(1, len(layers) // 16)))
    ax.set_xticklabels(
        [str(layers[i]) for i in range(0, len(layers), max(1, len(layers) // 16))],
        fontsize=8,
    )
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Fractional Recovery", fontsize=10)
    ax.set_title("Recovery by Layer", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # Legend
    ax.legend(
        handles=[
            Patch(color="#ab47bc", label="Mamba"),
            Patch(color="#26a69a", label="Transformer"),
        ],
        fontsize=9,
        loc="upper right",
    )

    # Mamba cascade annotation
    n_mamba = sum(1 for idx in layers if arch_map.layer_type(idx) == "mamba")
    if n_mamba > 0:
        ax.text(
            0.01, -0.12,
            "Note: Mamba layers propagate state sequentially — patching cascades to later positions, "
            "which may yield higher recovery than Transformer layers.",
            transform=ax.transAxes,
            fontsize=7,
            fontstyle="italic",
            color="#888888",
        )


def _plot_position_layer_heatmap(ax, sweep_result, arch_map):
    """Position x Layer heatmap — the primary full-sweep visualization."""
    matrix = sweep_result.recovery_matrix.numpy()
    tokens = sweep_result.corrupted_tokens

    # Diverging colormap centered at 0
    vmin = min(matrix.min(), -0.1)
    vmax = max(matrix.max(), 0.1)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")

    # Y-axis: layers
    layers = sweep_result.layer_indices
    ax.set_yticks(range(0, len(layers), max(1, len(layers) // 16)))
    ax.set_yticklabels(
        [str(layers[i]) for i in range(0, len(layers), max(1, len(layers) // 16))],
        fontsize=7,
    )
    ax.set_ylabel("Layer", fontsize=10)

    # X-axis: tokens (truncated labels)
    truncated = [_truncate_token(t) for t in tokens]
    if len(truncated) <= 30:
        ax.set_xticks(range(len(truncated)))
        ax.set_xticklabels(truncated, fontsize=6, rotation=60, ha="right")
    else:
        step = max(1, len(truncated) // 20)
        ax.set_xticks(range(0, len(truncated), step))
        ax.set_xticklabels(
            [truncated[i] for i in range(0, len(truncated), step)],
            fontsize=6, rotation=60, ha="right",
        )
    ax.set_xlabel("Token Position", fontsize=10)
    ax.set_title("Recovery Heatmap (Position × Layer)", fontsize=12)

    plt.colorbar(im, ax=ax, label="Fractional Recovery", shrink=0.8)


def _plot_layer_marginal(ax, sweep_result, arch_map):
    """Per-layer mean recovery, marginalized over positions."""
    matrix = sweep_result.recovery_matrix.numpy()
    layer_means = matrix.mean(axis=1)
    layers = sweep_result.layer_indices

    colors = [
        "#ab47bc" if arch_map.layer_type(idx) == "mamba" else "#26a69a"
        for idx in layers
    ]

    ax.barh(range(len(layers)), layer_means, color=colors, height=0.8)
    ax.axvline(x=0, color="gray", linewidth=0.5)

    ax.set_yticks(range(0, len(layers), max(1, len(layers) // 16)))
    ax.set_yticklabels(
        [str(layers[i]) for i in range(0, len(layers), max(1, len(layers) // 16))],
        fontsize=7,
    )
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_xlabel("Mean Recovery", fontsize=10)
    ax.set_title("Per-Layer Mean Recovery", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()


def _plot_position_marginal(ax, sweep_result):
    """Per-position mean recovery, marginalized over layers."""
    matrix = sweep_result.recovery_matrix.numpy()
    pos_means = matrix.mean(axis=0)
    tokens = sweep_result.corrupted_tokens

    ax.bar(range(len(pos_means)), pos_means, color="#64b5f6", width=0.8)
    ax.axhline(y=0, color="gray", linewidth=0.5)

    truncated = [_truncate_token(t) for t in tokens]
    if len(truncated) <= 30:
        ax.set_xticks(range(len(truncated)))
        ax.set_xticklabels(truncated, fontsize=6, rotation=60, ha="right")
    else:
        step = max(1, len(truncated) // 20)
        ax.set_xticks(range(0, len(truncated), step))
        ax.set_xticklabels(
            [truncated[i] for i in range(0, len(truncated), step)],
            fontsize=6, rotation=60, ha="right",
        )
    ax.set_xlabel("Token Position", fontsize=10)
    ax.set_ylabel("Mean Recovery", fontsize=10)
    ax.set_title("Per-Position Mean Recovery", fontsize=12)
    ax.grid(axis="y", alpha=0.3)


def _plot_baselines_summary(ax, sweep_result):
    """Text summary of baselines and top results."""
    ax.axis("off")

    recovery = sweep_result.recovery_matrix.numpy()

    lines = [
        f"Correct: {sweep_result.correct_token_display}",
        f"Incorrect: {sweep_result.incorrect_token_display}",
        "",
        f"Clean logit diff:     {sweep_result.clean_logit_diff:+.3f}   P(correct): {sweep_result.clean_prob_correct:.4f}",
        f"Corrupted logit diff: {sweep_result.corrupted_logit_diff:+.3f}   P(correct): {sweep_result.corrupted_prob_correct:.4f}",
        "",
    ]

    if recovery.ndim == 1:
        # Layer sweep
        top_k = min(5, len(recovery))
        top_idx = np.argsort(recovery)[::-1][:top_k]
        lines.append("Top layers by recovery:")
        for idx in top_idx:
            layer = sweep_result.layer_indices[idx]
            lines.append(f"  Layer {layer:2d}: {recovery[idx]:+.3f}")
    else:
        # Full sweep
        flat = recovery.flatten()
        top_k = min(5, len(flat))
        top_flat_idx = np.argsort(flat)[::-1][:top_k]
        lines.append("Top (layer, position) by recovery:")
        for fi in top_flat_idx:
            li = fi // recovery.shape[1]
            pi = fi % recovery.shape[1]
            layer = sweep_result.layer_indices[li]
            token = sweep_result.corrupted_tokens[pi] if pi < len(sweep_result.corrupted_tokens) else f"pos{pi}"
            lines.append(f"  Layer {layer:2d}, pos {pi} ({_truncate_token(token)}): {flat[fi]:+.3f}")

    text = "\n".join(lines)
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes, fontsize=9, fontfamily="monospace",
        verticalalignment="top",
    )


def format_intervention_info(sweep_result) -> str:
    """Format intervention results as text for the Gradio info panel."""
    lines = [
        f"Intervention: {sweep_result.intervention_type.value}",
        f"Correct token: {sweep_result.correct_token_display}",
        f"Incorrect token: {sweep_result.incorrect_token_display}",
        f"Clean logit diff: {sweep_result.clean_logit_diff:+.3f}  |  P(correct): {sweep_result.clean_prob_correct:.4f}",
        f"Corrupted logit diff: {sweep_result.corrupted_logit_diff:+.3f}  |  P(correct): {sweep_result.corrupted_prob_correct:.4f}",
        "",
    ]

    # Baseline quality warnings
    denominator = sweep_result.clean_logit_diff - sweep_result.corrupted_logit_diff
    if abs(denominator) <= MIN_RECOVERY_DENOMINATOR:
        lines.append(
            "⚠ WARNING: Clean and corrupted logit diffs are nearly identical "
            f"(denominator={denominator:+.2e}). Recovery scores are meaningless — "
            "the model may not distinguish these prompts for the chosen tokens."
        )
        lines.append("")
    elif sweep_result.clean_logit_diff < 0:
        lines.append(
            "⚠ WARNING: Clean logit diff is NEGATIVE — the model already prefers "
            f"the 'incorrect' token on the clean prompt (diff={sweep_result.clean_logit_diff:+.3f}). "
            "Recovery scores may be inverted. Check your token assignments."
        )
        lines.append("")
    elif abs(denominator) < 1.0:
        lines.append(
            f"⚠ CAUTION: Small denominator ({denominator:+.3f}). The model only weakly "
            "distinguishes these prompts, so recovery scores may be noisy."
        )
        lines.append("")

    recovery = sweep_result.recovery_matrix.numpy()
    if recovery.ndim == 1:
        top_idx = int(np.argmax(recovery))
        layer = sweep_result.layer_indices[top_idx]
        lines.append(f"Most important layer: {layer} (recovery: {recovery[top_idx]:+.3f})")
        lines.append(f"Mean recovery across layers: {recovery.mean():.3f}")
    else:
        flat_max = int(np.argmax(recovery.flatten()))
        li = flat_max // recovery.shape[1]
        pi = flat_max % recovery.shape[1]
        layer = sweep_result.layer_indices[li]
        token = sweep_result.corrupted_tokens[pi] if pi < len(sweep_result.corrupted_tokens) else f"pos{pi}"
        lines.append(
            f"Most important: layer {layer}, pos {pi} "
            f"(\"{_truncate_token(token)}\") recovery: {recovery.flatten()[flat_max]:+.3f}"
        )

    return "\n".join(lines)


def _truncate_token(token: str, max_len: int = 12) -> str:
    """Truncate a token string for display."""
    # Remove common HF tokenizer prefixes
    if token.startswith("▁"):
        token = token[1:]
    if token.startswith("Ġ"):
        token = token[1:]
    if len(token) > max_len:
        token = token[:max_len - 1] + "…"
    return token
