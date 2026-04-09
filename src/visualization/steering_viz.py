"""Visualization for SAE training, feature browser, and activation steering."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Color scheme ─────────────────────────────────────────────────────────────
MAMBA_COLOR = "#ab47bc"
TRANSFORMER_COLOR = "#26a69a"
BEE_COLOR = "#FFC107"
BEE_COLOR_LIGHT = "#FFF3CD"
BASELINE_COLOR = "#78909C"


def _empty_plotly(message: str) -> go.Figure:
    """Return a Plotly figure with centered instructional text."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color="gray"),
    )
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# ── SAE Training Visualization ──────────────────────────────────────────────


def create_training_loss_plotly(training_result) -> go.Figure:
    """Line chart of SAE training loss over steps."""
    if training_result is None or not training_result.loss_history:
        return _empty_plotly("No training data available.")

    steps = list(range(1, len(training_result.loss_history) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=training_result.loss_history,
        name="Total Loss", line=dict(color=BEE_COLOR, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=training_result.reconstruction_history,
        name="Reconstruction (MSE)", line=dict(color=MAMBA_COLOR, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=training_result.l1_history,
        name="L1 Sparsity", line=dict(color=TRANSFORMER_COLOR, width=2),
    ))
    fig.update_layout(
        title="SAE Training Loss",
        xaxis_title="Training Step",
        yaxis_title="Loss",
        template="plotly_dark",
        height=500,
        legend=dict(x=0.7, y=0.95),
    )
    return fig


def create_feature_density_plotly(training_result) -> go.Figure:
    """Bar chart showing dead vs alive features and mean active count."""
    if training_result is None:
        return _empty_plotly("No training data available.")

    alive = training_result.num_latent_features - training_result.dead_features
    dead = training_result.dead_features

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Alive Features", "Dead Features"],
        y=[alive, dead],
        marker_color=[BEE_COLOR, "#616161"],
        text=[alive, dead],
        textposition="auto",
    ))
    fig.add_annotation(
        text=f"Mean active features per sample: {training_result.mean_active_features:.1f}",
        xref="paper", yref="paper", x=0.5, y=1.05,
        showarrow=False, font=dict(size=13),
    )
    fig.update_layout(
        title="SAE Feature Density",
        yaxis_title="Count",
        template="plotly_dark",
        height=500,
    )
    return fig


def create_training_summary_markdown(training_result) -> str:
    """Markdown summary of SAE training config and results."""
    if training_result is None:
        return "No SAE trained yet."

    c = training_result.config
    r = training_result
    return f"""# SAE Training Summary

**Architecture**: Vanilla SAE (Linear + ReLU encoder, unit-normalized decoder)

| Parameter | Value |
|-----------|-------|
| Target Layer | {c.layer_idx} |
| Input Dim | {r.sae.input_dim} |
| Latent Dim | {r.sae.latent_dim} ({c.expansion_factor}x expansion) |
| L1 Coefficient | {c.l1_coefficient} |
| Learning Rate | {c.learning_rate} |
| Training Samples | {c.num_activations:,} |
| Batch Size | {c.batch_size} |
| Epochs | {c.num_epochs} |

**Results**:
- Final total loss: **{r.final_loss:.6f}**
- Final reconstruction loss: **{r.final_reconstruction_loss:.6f}**
- Final L1 loss: **{r.final_l1_loss:.6f}**
- Mean active features per sample: **{r.mean_active_features:.1f}** / {r.num_latent_features}
- Dead features: **{r.dead_features}** / {r.num_latent_features} ({100*r.dead_features/max(r.num_latent_features,1):.1f}%)
"""


# ── Feature Browser Visualization ───────────────────────────────────────────


def create_feature_search_plotly(search_result) -> go.Figure:
    """Bar chart of top features by selectivity score."""
    if search_result is None or not search_result.top_features:
        return _empty_plotly("No feature search results available.")

    features = search_result.top_features
    labels = [f"Feature {f['feature_idx']}" for f in features]
    concept_acts = [f["concept_activation"] for f in features]
    baseline_acts = [f["baseline_activation"] for f in features]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=concept_acts,
        name=f"{search_result.concept.capitalize()} Texts",
        marker_color=BEE_COLOR,
    ))
    fig.add_trace(go.Bar(
        x=labels, y=baseline_acts,
        name="Baseline Texts",
        marker_color=BASELINE_COLOR,
    ))
    fig.update_layout(
        title=f"Top Features for '{search_result.concept}' (by Selectivity)",
        xaxis_title="SAE Feature",
        yaxis_title="Mean Activation",
        barmode="group",
        template="plotly_dark",
        height=500,
        xaxis=dict(tickangle=-45),
    )
    return fig


def create_feature_activations_plotly(search_result, feature_rank: int = 0) -> go.Figure:
    """Per-text activation heatmap for a selected top feature.

    Args:
        search_result: FeatureSearchResult.
        feature_rank: Index into top_features list (0 = most selective).
    """
    if search_result is None or not search_result.top_features:
        return _empty_plotly("No feature search results available.")

    feature_rank = max(0, min(feature_rank, len(search_result.top_features) - 1))
    feature_info = search_result.top_features[feature_rank]
    feature_idx = feature_info["feature_idx"]

    # Get per-text activations for this feature
    concept_vals = search_result.concept_activations[:, feature_rank].numpy()
    baseline_vals = search_result.baseline_activations[:, feature_rank].numpy()

    # Truncate text labels
    concept_labels = [t[:60] + "..." if len(t) > 60 else t for t in search_result.concept_texts]
    baseline_labels = [t[:60] + "..." if len(t) > 60 else t for t in search_result.baseline_texts]

    all_labels = concept_labels + baseline_labels
    all_vals = list(concept_vals) + list(baseline_vals)
    colors = [BEE_COLOR] * len(concept_vals) + [BASELINE_COLOR] * len(baseline_vals)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=all_labels, x=all_vals, orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in all_vals],
        textposition="auto",
    ))
    fig.update_layout(
        title=f"Feature {feature_idx} Activation Per Text (selectivity: {feature_info['selectivity_score']:.1f}x)",
        xaxis_title="Mean Feature Activation",
        template="plotly_dark",
        height=max(400, 30 * len(all_labels)),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=350),
    )
    return fig


def create_feature_search_summary_markdown(search_result) -> str:
    """Markdown summary of feature search results."""
    if search_result is None:
        return "No feature search performed yet."

    lines = [
        f"# Feature Search: '{search_result.concept}'",
        "",
        f"Searched {len(search_result.concept_texts)} concept texts vs "
        f"{len(search_result.baseline_texts)} baseline texts.",
        "",
        "## Top Features",
        "",
        "| Rank | Feature | Concept Act. | Baseline Act. | Selectivity |",
        "|------|---------|-------------|--------------|-------------|",
    ]
    for i, f in enumerate(search_result.top_features[:10]):
        lines.append(
            f"| {i+1} | {f['feature_idx']} | {f['concept_activation']:.4f} | "
            f"{f['baseline_activation']:.4f} | {f['selectivity_score']:.1f}x |"
        )

    if search_result.top_features:
        best = search_result.top_features[0]
        lines.extend([
            "",
            f"**Best candidate**: Feature {best['feature_idx']} "
            f"with {best['selectivity_score']:.1f}x selectivity",
            "",
            "Use this feature index in the **Activation Steering** tab to steer the model.",
        ])

    return "\n".join(lines)


# ── Steering Visualization ──────────────────────────────────────────────────


def create_steered_comparison_plotly(steering_result) -> go.Figure:
    """Side-by-side comparison of steered vs unsteered generation."""
    if steering_result is None:
        return _empty_plotly("No steering results available.")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Unsteered (Baseline)", "Steered (I-Bee-M)"],
        vertical_spacing=0.15,
    )

    # Unsteered tokens
    unsteered_tokens = steering_result.unsteered_tokens
    fig.add_trace(go.Bar(
        x=list(range(len(unsteered_tokens))),
        y=[1] * len(unsteered_tokens),
        text=[t.strip() or repr(t) for t in unsteered_tokens],
        textposition="inside",
        marker_color=BASELINE_COLOR,
        hoverinfo="text",
        showlegend=False,
    ), row=1, col=1)

    # Steered tokens
    steered_tokens = steering_result.steered_tokens
    fig.add_trace(go.Bar(
        x=list(range(len(steered_tokens))),
        y=[1] * len(steered_tokens),
        text=[t.strip() or repr(t) for t in steered_tokens],
        textposition="inside",
        marker_color=BEE_COLOR,
        hoverinfo="text",
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        title=f"Generation Comparison (coefficient={steering_result.config.steering_coefficient})",
        template="plotly_dark",
        height=400,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Token Position", row=2, col=1)
    fig.update_yaxes(visible=False)

    return fig


def create_steering_top5_plotly(steering_result) -> go.Figure:
    """Per-token top-5 predictions comparing steered vs unsteered."""
    if steering_result is None:
        return _empty_plotly("No steering results available.")

    steered_t5 = steering_result.steered_top5
    unsteered_t5 = steering_result.unsteered_top5
    n_steps = max(len(steered_t5), len(unsteered_t5))

    if n_steps == 0:
        return _empty_plotly("No generation steps to display.")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Unsteered Top-5", "Steered Top-5"],
        horizontal_spacing=0.12,
    )

    # Build heatmap data: steps x rank (top-5)
    for col_idx, (top5_list, title) in enumerate(
        [(unsteered_t5, "Unsteered"), (steered_t5, "Steered")], start=1
    ):
        n = len(top5_list)
        z_vals = np.zeros((n, 5))
        text_vals = [[""]*5 for _ in range(n)]

        for step, preds in enumerate(top5_list):
            for rank, (token, prob) in enumerate(preds[:5]):
                z_vals[step, rank] = prob
                text_vals[step][rank] = f"{token} ({prob:.3f})"

        colorscale = [[0, "#1a1a2e"], [1, BEE_COLOR if col_idx == 2 else BASELINE_COLOR]]

        fig.add_trace(go.Heatmap(
            z=z_vals,
            text=text_vals,
            texttemplate="%{text}",
            colorscale=colorscale,
            showscale=False,
            xgap=2, ygap=2,
        ), row=1, col=col_idx)

    fig.update_layout(
        title="Top-5 Predictions Per Generation Step",
        template="plotly_dark",
        height=max(350, 35 * n_steps),
    )
    fig.update_xaxes(title_text="Rank", tickvals=list(range(5)), ticktext=["1st","2nd","3rd","4th","5th"])
    fig.update_yaxes(title_text="Step", row=1, col=1)

    return fig


def create_steering_summary_markdown(steering_result) -> str:
    """Markdown summary of steering configuration and results."""
    if steering_result is None:
        return "No steering results yet."

    c = steering_result.config
    return f"""# Activation Steering Summary

**Prompt**: "{steering_result.prompt}"

| Parameter | Value |
|-----------|-------|
| Layer | {c.layer_idx} |
| Feature | {c.feature_idx} |
| Coefficient | {c.steering_coefficient} |
| Max Tokens | {c.max_tokens} |
| Direction Norm | {steering_result.feature_direction_norm:.4f} |

## Unsteered Output
> {steering_result.unsteered_text}

## Steered Output (I-Bee-M)
> {steering_result.steered_text}

---
*Tokens generated — Steered: {len(steering_result.steered_tokens)}, Unsteered: {len(steering_result.unsteered_tokens)}*
"""


# ── Info formatters ─────────────────────────────────────────────────────────


def format_training_info(training_result) -> str:
    """Text summary for the Gradio info panel."""
    if training_result is None:
        return "No SAE trained yet."
    r = training_result
    return (
        f"SAE trained: {r.sae.input_dim}→{r.sae.latent_dim} | "
        f"Loss: {r.final_loss:.6f} (recon: {r.final_reconstruction_loss:.6f}, "
        f"L1: {r.final_l1_loss:.6f}) | "
        f"Active: {r.mean_active_features:.0f}/{r.num_latent_features} | "
        f"Dead: {r.dead_features}"
    )


def format_feature_search_info(search_result) -> str:
    """Text summary for the Gradio info panel."""
    if search_result is None:
        return "No feature search performed."
    if not search_result.top_features:
        return "Feature search found no selective features."
    best = search_result.top_features[0]
    return (
        f"Best '{search_result.concept}' feature: #{best['feature_idx']} "
        f"(selectivity: {best['selectivity_score']:.1f}x, "
        f"concept act: {best['concept_activation']:.4f}, "
        f"baseline act: {best['baseline_activation']:.4f})"
    )


def format_steering_info(steering_result) -> str:
    """Text summary for the Gradio info panel."""
    if steering_result is None:
        return "No steering results."
    return (
        f"Layer {steering_result.config.layer_idx}, "
        f"Feature {steering_result.config.feature_idx}, "
        f"Coeff {steering_result.config.steering_coefficient} | "
        f"Steered: {len(steering_result.steered_tokens)} tokens | "
        f"Unsteered: {len(steering_result.unsteered_tokens)} tokens"
    )
