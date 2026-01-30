"""Visualization for activation differences between two prompts."""

import matplotlib.pyplot as plt
import numpy as np


def create_activation_diff_summary(diff_result: dict, arch_map) -> plt.Figure:
    """Create a 2x2 dashboard comparing activations between two prompts.

    Subplots:
        Top-left: Residual stream cosine similarity per layer
        Top-right: Attention JSD per layer (colored by layer type)
        Bottom-left: Top changed neurons (horizontal bar chart)
        Bottom-right: Logit lens prediction diff

    Args:
        diff_result: Output of compute_activation_diff().
        arch_map: ArchitectureMap for layer type coloring.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Activation Diff — {diff_result['shared_len']} shared tokens",
        fontsize=14,
        fontweight="bold",
    )

    _plot_residual_similarity(axes[0, 0], diff_result["residual_similarity"], arch_map)
    _plot_attention_divergence(axes[0, 1], diff_result["attention_divergence"])
    _plot_neuron_changes(axes[1, 0], diff_result["top_changed_neurons"])
    _plot_logit_lens_diff(axes[1, 1], diff_result["logit_lens_diff"])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _plot_residual_similarity(ax, residual_sim: dict, arch_map):
    """Line plot of cosine similarity per layer."""
    layers = residual_sim.get("layers", [])
    cosine = residual_sim.get("cosine_sim_mean", [])

    if not layers:
        ax.text(0.5, 0.5, "No residual stream data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Residual Cosine Similarity")
        return

    # Background shading by layer type
    for i, layer_idx in enumerate(layers):
        lt = arch_map.layer_type(layer_idx)
        color = "#2d0040" if lt == "mamba" else "#002d20"
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color=color)

    ax.plot(range(len(layers)), cosine, "o-", color="#4fc3f7", linewidth=1.5, markersize=3)
    ax.set_xticks(range(0, len(layers), max(1, len(layers) // 10)))
    ax.set_xticklabels([str(layers[i]) for i in range(0, len(layers), max(1, len(layers) // 10))], fontsize=7)
    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Cosine Similarity", fontsize=9)
    ax.set_title("Residual Stream Cosine Similarity", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)


def _plot_attention_divergence(ax, attn_div: dict):
    """Bar chart of JSD per layer, colored by layer type."""
    combined = attn_div.get("combined", [])

    if not combined:
        ax.text(0.5, 0.5, "No attention divergence data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Attention JSD")
        return

    indices = range(len(combined))
    jsd_vals = [c[1] for c in combined]
    colors = ["#ab47bc" if c[2] == "mamba" else "#26a69a" for c in combined]
    labels = [str(c[0]) for c in combined]

    ax.bar(indices, jsd_vals, color=colors, width=0.8)
    ax.set_xticks(range(0, len(combined), max(1, len(combined) // 10)))
    ax.set_xticklabels(
        [labels[i] for i in range(0, len(combined), max(1, len(combined) // 10))],
        fontsize=7,
    )
    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel("Jensen-Shannon Divergence", fontsize=9)
    ax.set_title("Attention Pattern Divergence", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color="#ab47bc", label="Mamba"), Patch(color="#26a69a", label="Transformer")],
        fontsize=8,
        loc="upper right",
    )


def _plot_neuron_changes(ax, top_neurons: list):
    """Horizontal bar chart of top changed neurons."""
    if not top_neurons:
        ax.text(0.5, 0.5, "No neuron data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Top Changed Neurons")
        return

    # Show up to 15 for readability
    display = top_neurons[:15]
    display.reverse()  # largest at top

    labels = [f"L{n['layer']}.N{n['neuron_idx']}" for n in display]
    deltas = [n["delta"] for n in display]
    colors = ["#ef5350" if d > 0 else "#42a5f5" for d in deltas]

    ax.barh(range(len(display)), deltas, color=colors, height=0.7)
    ax.set_yticks(range(len(display)))
    ax.set_yticklabels(labels, fontsize=7, fontfamily="monospace")
    ax.set_xlabel("Activation Delta (B - A)", fontsize=9)
    ax.set_title("Top Changed Neurons", fontsize=11)
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)


def _plot_logit_lens_diff(ax, logit_diff: dict):
    """Table-style plot of logit lens prediction differences."""
    layers = logit_diff.get("layers", [])

    if not layers:
        ax.text(0.5, 0.5, "No logit lens data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Logit Lens Diff")
        return

    # Sample layers evenly if too many
    if len(layers) > 20:
        step = len(layers) // 20
        layers = layers[::step]

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(layers) - 0.5)
    ax.set_title("Logit Lens — Top-1 Prediction Diff", fontsize=11)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{d['layer']}" for d in layers], fontsize=7, fontfamily="monospace")
    ax.set_xticks([])
    ax.invert_yaxis()

    for i, d in enumerate(layers):
        bg = "#ffebee" if not d["same_prediction"] else "#e8f5e9"
        ax.axhspan(i - 0.4, i + 0.4, color=bg, alpha=0.5)

        text_a = f"A: {d['top1_a']} ({d['prob_a']:.2f})"
        text_b = f"B: {d['top1_b']} ({d['prob_b']:.2f})"

        ax.text(0.25, i, text_a, ha="center", va="center", fontsize=7, fontfamily="monospace")
        ax.text(0.75, i, text_b, ha="center", va="center", fontsize=7, fontfamily="monospace")

    ax.grid(False)


def format_diff_info(diff_result: dict) -> str:
    """Format activation diff summary as text for the info panel."""
    lines = [
        f"Prompt A tokens: {len(diff_result['tokens_a'])}",
        f"Prompt B tokens: {len(diff_result['tokens_b'])}",
        f"Shared positions compared: {diff_result['shared_len']}",
        "",
    ]

    # Residual similarity summary
    res = diff_result["residual_similarity"]
    if res["cosine_sim_mean"]:
        mean_cos = np.mean(res["cosine_sim_mean"])
        min_cos = np.min(res["cosine_sim_mean"])
        min_layer = res["layers"][np.argmin(res["cosine_sim_mean"])]
        lines.append(f"Residual cosine sim — mean: {mean_cos:.4f}, min: {min_cos:.4f} (layer {min_layer})")

    # Attention divergence summary
    combined = diff_result["attention_divergence"].get("combined", [])
    if combined:
        jsd_vals = [c[1] for c in combined]
        max_jsd = max(jsd_vals)
        max_entry = combined[jsd_vals.index(max_jsd)]
        lines.append(f"Max attention JSD: {max_jsd:.4f} (layer {max_entry[0]}, {max_entry[2]})")

    # Top neurons
    neurons = diff_result["top_changed_neurons"]
    if neurons:
        lines.append("")
        lines.append("Top 5 changed neurons:")
        for n in neurons[:5]:
            lines.append(f"  L{n['layer']}.N{n['neuron_idx']}: delta={n['delta']:+.3f}")

    return "\n".join(lines)
