"""Gradio app for Granite 4.0 Mamba-2 Interpretability Tool."""

import os

import torch
if torch.cuda.is_available() and hasattr(torch.version, "hip"):
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import gradio as gr
import matplotlib
matplotlib.use("Agg")

from src.model_loader import load_model, get_model_info
from src.architecture_map import ArchitectureMap
from src.extraction.unified_extractor import GraniteAttentionExtractor
from src.visualization.heatmap import create_attention_heatmap, create_all_layers_overview
from src.visualization.comparison import create_comparison_view
from src.visualization.logit_lens import create_logit_lens_heatmap
from src.visualization.neuron_activation import create_neuron_activation_heatmap, format_neuron_info
from src.extraction.activation_diff import compute_activation_diff
from src.visualization.activation_diff_viz import (
    create_activation_diff_summary, format_diff_info,
    create_residual_similarity_plotly, create_attention_divergence_plotly,
    create_neuron_changes_plotly, create_logit_lens_diff_plotly,
)
from src.extraction.multistep_generator import run_multistep_generation
from src.visualization.multistep_viz import (
    create_multistep_dashboard, format_multistep_info,
    create_token_timeline_plotly, create_logit_lens_compact_plotly,
    create_mamba_attention_plotly, create_transformer_attention_plotly,
)
from src.extraction.causal_intervention import CausalInterventionEngine, InterventionType
from src.visualization.intervention_viz import (
    create_layer_sweep_dashboard,
    create_full_sweep_dashboard,
    format_intervention_info,
    create_position_layer_heatmap_plotly,
    create_layer_marginal_plotly,
    create_position_marginal_plotly,
    create_intervention_summary_markdown,
)
from src.extraction.circuit_discovery import CircuitDiscoveryEngine, SweepGranularity
from src.visualization.circuit_viz import (
    create_circuit_dashboard,
    format_circuit_info,
    create_path_matrix_plotly,
    create_layer_importance_plotly,
    create_circuit_diagram_plotly,
    create_component_importance_plotly,
    create_circuit_summary_markdown,
)

# Global state
MODEL = None
TOKENIZER = None
EXTRACTOR = None
ARCH_MAP = None
CACHE = {}
MULTISTEP_CACHE = {}
INTERVENTION_ENGINE = None
INTERVENTION_CACHE = {}
INTERVENTION_CACHE_MAX = 20
CIRCUIT_ENGINE = None
CIRCUIT_CACHE = {}
CIRCUIT_CACHE_MAX = 10


def initialize():
    """Load model and set up extractor."""
    global MODEL, TOKENIZER, EXTRACTOR, ARCH_MAP, INTERVENTION_ENGINE, CIRCUIT_ENGINE

    # Use CPU by default — .to(cuda) hangs intermittently on gfx1151.
    # 350M model runs fine on CPU. Set DEVICE=cuda to force GPU.
    device = os.environ.get("DEVICE", "cpu")
    print(f"Loading model on {device}...")
    MODEL, TOKENIZER, device = load_model(device=device)
    ARCH_MAP = ArchitectureMap(MODEL.config)
    EXTRACTOR = GraniteAttentionExtractor(MODEL, TOKENIZER, device=device)
    INTERVENTION_ENGINE = CausalInterventionEngine(MODEL, TOKENIZER, ARCH_MAP, device=device)
    CIRCUIT_ENGINE = CircuitDiscoveryEngine(MODEL, TOKENIZER, ARCH_MAP, device=device)
    print(f"Model loaded. {ARCH_MAP.summary()}")


def run_extraction(prompt: str):
    """Run extraction with caching."""
    if not prompt or not prompt.strip():
        return None

    prompt = prompt.strip()
    if prompt in CACHE:
        return CACHE[prompt]

    result = EXTRACTOR.extract(prompt)
    CACHE[prompt] = result
    return result


def single_layer_view(prompt, layer_idx, head_agg):
    """Generate single layer heatmap."""
    result = run_extraction(prompt)
    if result is None:
        return None, "Please enter a prompt."

    layer_idx = int(layer_idx)
    layer_type = ARCH_MAP.layer_type(layer_idx)

    if layer_type == "mamba":
        attn_data = result["mamba_attention"]
    else:
        attn_data = result["transformer_attention"]

    if layer_idx not in attn_data:
        return None, f"No attention data for layer {layer_idx} ({layer_type})"

    fig = create_attention_heatmap(
        attention=attn_data[layer_idx],
        tokens=result["tokens"],
        layer_idx=layer_idx,
        layer_type=layer_type,
        head_agg=head_agg,
    )

    info = (
        f"Layer {layer_idx} ({layer_type.upper()})\n"
        f"Tokens: {len(result['tokens'])}\n"
        f"Heads: {attn_data[layer_idx].shape[1]}\n"
        f"Aggregation: {head_agg}"
    )
    return fig, info


def comparison_view(prompt, head_agg):
    """Generate Mamba vs Transformer comparison."""
    result = run_extraction(prompt)
    if result is None:
        return None, "Please enter a prompt."

    # Find a mamba and transformer layer with data
    mamba_idx = None
    for idx in ARCH_MAP.mamba_indices:
        if idx in result["mamba_attention"]:
            mamba_idx = idx
            break

    attn_idx = None
    for idx in ARCH_MAP.attention_indices:
        if idx in result["transformer_attention"]:
            attn_idx = idx
            break

    if mamba_idx is None or attn_idx is None:
        return None, "Could not find both Mamba and Transformer attention data."

    fig = create_comparison_view(
        mamba_attn=result["mamba_attention"][mamba_idx],
        transformer_attn=result["transformer_attention"][attn_idx],
        tokens=result["tokens"],
        mamba_layer_idx=mamba_idx,
        transformer_layer_idx=attn_idx,
        head_agg=head_agg,
    )

    info = (
        f"Comparing Mamba-2 layer {mamba_idx} vs Transformer layer {attn_idx}\n"
        f"Tokens: {len(result['tokens'])}\n"
        f"Aggregation: {head_agg}"
    )
    return fig, info


def all_layers_view(prompt, head_agg):
    """Generate overview of all layers."""
    result = run_extraction(prompt)
    if result is None:
        return None, "Please enter a prompt."

    fig = create_all_layers_overview(
        mamba_attention=result["mamba_attention"],
        transformer_attention=result["transformer_attention"],
        tokens=result["tokens"],
        layer_types=ARCH_MAP.layer_types,
        head_agg=head_agg,
    )

    n_mamba = len(result["mamba_attention"])
    n_attn = len(result["transformer_attention"])
    info = (
        f"Showing all {ARCH_MAP.num_layers} layers\n"
        f"Mamba-2 layers extracted: {n_mamba}\n"
        f"Transformer layers extracted: {n_attn}\n"
        f"Tokens: {len(result['tokens'])}"
    )
    return fig, info


def logit_lens_view(prompt):
    """Generate logit lens prediction evolution visualization."""
    result = run_extraction(prompt)
    if result is None:
        return None, "Please enter a prompt."

    logit_lens = result.get("logit_lens")
    if not logit_lens or not logit_lens.get("layers"):
        return None, "Logit lens data not available."

    fig = create_logit_lens_heatmap(logit_lens)

    info = (
        f"Logit Lens — Prediction evolution\n"
        f"Position: last token ({logit_lens['position_token']})\n"
        f"Layers: {len(logit_lens['layers'])}\n"
        f"Top-k: 10"
    )
    return fig, info


def neuron_activation_view(prompt, layer_idx):
    """Generate neuron activation heatmap for a single layer."""
    result = run_extraction(prompt)
    if result is None:
        return None, "Please enter a prompt."

    layer_idx = int(layer_idx)
    layer_type = ARCH_MAP.layer_type(layer_idx)
    activations = result.get("neuron_activations", {})

    if layer_idx not in activations:
        return None, f"No neuron activation data for layer {layer_idx}"

    fig = create_neuron_activation_heatmap(
        activations=activations[layer_idx],
        tokens=result["tokens"],
        layer_idx=layer_idx,
        layer_type=layer_type,
    )

    info = format_neuron_info(
        activations=activations[layer_idx],
        tokens=result["tokens"],
        layer_idx=layer_idx,
        layer_type=layer_type,
    )
    return fig, info


def activation_diff_view(prompt_a, prompt_b, head_agg):
    """Generate activation diff comparison between two prompts."""
    if not prompt_b or not prompt_b.strip():
        return None, "Please enter a comparison prompt (Prompt B)."

    result_a = run_extraction(prompt_a)
    result_b = run_extraction(prompt_b)
    if result_a is None or result_b is None:
        return None, "Extraction failed for one or both prompts."

    diff = compute_activation_diff(result_a, result_b, head_agg=head_agg)
    fig = create_activation_diff_summary(diff, ARCH_MAP)
    info = format_diff_info(diff)
    return fig, info


def multistep_view(prompt, max_tokens, step_idx, head_agg):
    """Generate tokens step-by-step with full analysis at each step."""
    cache_key = (prompt.strip(), int(max_tokens))

    if cache_key not in MULTISTEP_CACHE:
        result = run_multistep_generation(
            extractor=EXTRACTOR,
            tokenizer=TOKENIZER,
            prompt=prompt,
            max_tokens=int(max_tokens),
        )
        MULTISTEP_CACHE[cache_key] = result
    else:
        result = MULTISTEP_CACHE[cache_key]

    num_steps = len(result["steps"])
    step_idx = max(0, min(int(step_idx), num_steps - 1))

    fig = create_multistep_dashboard(
        step_data=result["steps"][step_idx],
        arch_map=ARCH_MAP,
        head_agg=head_agg,
    )
    info = format_multistep_info(result, step_idx)
    return fig, info, gr.update(maximum=num_steps - 1, value=step_idx)


def intervention_view(clean_prompt, corrupted_prompt, correct_token, incorrect_token,
                      intervention_type, sweep_mode, progress_callback=None):
    """Run causal intervention experiment and visualize results."""
    if INTERVENTION_ENGINE is None:
        return None, "Intervention engine not initialized. Please restart the app."

    clean_prompt = (clean_prompt or "").strip()
    corrupted_prompt = (corrupted_prompt or "").strip()
    correct_token = (correct_token or "").strip()
    incorrect_token = (incorrect_token or "").strip()
    intervention_type = intervention_type or InterventionType.ACTIVATION_PATCH.value
    sweep_mode = sweep_mode or "Layer Sweep (all positions)"

    if not corrupted_prompt:
        return None, "Please enter a corrupted prompt."
    if not correct_token:
        return None, "Please enter a correct token (the answer the clean prompt should produce)."
    if not incorrect_token:
        return None, "Please enter an incorrect token (the answer the corrupted prompt produces)."

    int_type = InterventionType(intervention_type)

    cache_key = (
        clean_prompt, corrupted_prompt,
        correct_token, incorrect_token,
        intervention_type, sweep_mode,
    )

    if cache_key in INTERVENTION_CACHE:
        sweep_result = INTERVENTION_CACHE[cache_key]
    else:
        if sweep_mode == "Full Position × Layer Sweep":
            sweep_result = INTERVENTION_ENGINE.sweep_positions_and_layers(
                clean_prompt, corrupted_prompt,
                correct_token, incorrect_token,
                int_type,
                progress_callback=progress_callback,
            )
        else:
            positions = None  # all positions by default
            sweep_result = INTERVENTION_ENGINE.sweep_layers(
                clean_prompt, corrupted_prompt,
                correct_token, incorrect_token,
                int_type, positions=positions,
                progress_callback=progress_callback,
            )
        # Evict oldest entries if cache is full
        if len(INTERVENTION_CACHE) >= INTERVENTION_CACHE_MAX:
            oldest_key = next(iter(INTERVENTION_CACHE))
            del INTERVENTION_CACHE[oldest_key]
        INTERVENTION_CACHE[cache_key] = sweep_result

    if sweep_result.recovery_matrix.dim() == 1:
        fig = create_layer_sweep_dashboard(sweep_result, ARCH_MAP)
    else:
        fig = create_full_sweep_dashboard(sweep_result, ARCH_MAP)

    info = format_intervention_info(sweep_result)
    return fig, info


def circuit_discovery_view(clean_prompt, corrupted_prompt, correct_token, incorrect_token,
                           threshold, granularity, progress_callback=None):
    """Run circuit discovery and visualize results."""
    if CIRCUIT_ENGINE is None:
        return None, "Circuit discovery engine not initialized. Please restart the app."

    clean_prompt = (clean_prompt or "").strip()
    corrupted_prompt = (corrupted_prompt or "").strip()
    correct_token = (correct_token or "").strip()
    incorrect_token = (incorrect_token or "").strip()
    threshold = float(threshold) if threshold is not None else 0.1
    granularity = granularity or "Fast (layer paths only)"

    if not corrupted_prompt:
        return None, "Please enter a corrupted prompt."
    if not correct_token:
        return None, "Please enter a correct token (the answer the clean prompt should produce)."
    if not incorrect_token:
        return None, "Please enter an incorrect token (the answer the corrupted prompt produces)."

    gran = SweepGranularity.DETAILED if "Detailed" in granularity else SweepGranularity.FAST

    cache_key = (
        clean_prompt, corrupted_prompt,
        correct_token, incorrect_token,
        threshold, gran.value,
    )

    if cache_key in CIRCUIT_CACHE:
        circuit_result = CIRCUIT_CACHE[cache_key]
    else:
        circuit_result = CIRCUIT_ENGINE.find_circuit(
            clean_prompt, corrupted_prompt,
            correct_token, incorrect_token,
            threshold=threshold,
            granularity=gran,
            progress_callback=progress_callback,
        )
        # Evict oldest entries if cache is full
        if len(CIRCUIT_CACHE) >= CIRCUIT_CACHE_MAX:
            oldest_key = next(iter(CIRCUIT_CACHE))
            del CIRCUIT_CACHE[oldest_key]
        CIRCUIT_CACHE[cache_key] = circuit_result

    info = format_circuit_info(circuit_result)
    return circuit_result, info


def get_layer_type_label(layer_idx):
    """Return the layer type for display."""
    layer_idx = int(layer_idx)
    if ARCH_MAP:
        return f"Layer {layer_idx}: {ARCH_MAP.layer_type(layer_idx).upper()}"
    return f"Layer {layer_idx}"


def build_app():
    """Build and return the Gradio app."""
    model_info = get_model_info(MODEL)
    num_layers = model_info["num_layers"]

    example_prompts = [
        "The capital of France is",
        "def fibonacci(n):\n    if n <= 1:\n        return n",
        "In machine learning, attention mechanisms allow",
        "The quick brown fox jumps over the lazy dog",
        "Water boils at 100 degrees Celsius at sea level",
    ]

    with gr.Blocks(
        title="Granite 4.0 Mamba-2 Interpretability",
    ) as app:
        gr.Markdown(
            """
# Granite 4.0 Mamba-2 Interpretability Tool

Visualize **hidden attention** in IBM Granite 4.0's hybrid Mamba-2/Transformer architecture.

Mamba-2 layers don't have explicit attention matrices — this tool extracts implicit "hidden attention"
using the formula from [Ali et al. (2024)](https://arxiv.org/abs/2403.01590), making the black-box
SSM layers interpretable alongside standard Transformer attention.

**Architecture**: {n_layers} layers — {n_mamba} Mamba-2 (magma) + {n_attn} Transformer (viridis)
""".format(
                n_layers=num_layers,
                n_mamba=len(ARCH_MAP.mamba_indices),
                n_attn=len(ARCH_MAP.attention_indices),
            )
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Input Prompt",
                    placeholder="Type a prompt to analyze...",
                    lines=3,
                    value=example_prompts[0],
                )
                gr.Examples(
                    examples=[[p] for p in example_prompts],
                    inputs=[prompt_input],
                    label="Example Prompts",
                )

                prompt_b_input = gr.Textbox(
                    label="Comparison / Corrupted Prompt",
                    placeholder="Enter a second prompt (for Activation Diff or Causal Intervention)...",
                    lines=2,
                )

                view_mode = gr.Radio(
                    choices=["Single Layer", "Mamba vs Transformer", "All Layers", "Logit Lens", "Neuron Activation", "Activation Diff", "Causal Intervention", "Circuit Discovery", "Multi-Step Generation"],
                    value="Single Layer",
                    label="View Mode",
                )

                layer_slider = gr.Slider(
                    minimum=0,
                    maximum=num_layers - 1,
                    step=1,
                    value=0,
                    label="Layer",
                    info="Select layer to visualize",
                )

                layer_type_display = gr.Textbox(
                    label="Layer Type",
                    value=get_layer_type_label(0),
                    interactive=False,
                )

                head_agg = gr.Radio(
                    choices=["mean", "max"],
                    value="mean",
                    label="Head Aggregation",
                )

                analyze_btn = gr.Button("Analyze", variant="primary")

                # Multi-Step Generation controls
                max_tokens_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=5,
                    label="Max Tokens to Generate",
                    info="Number of tokens to generate (Multi-Step mode)",
                    visible=False,
                )
                step_nav_slider = gr.Slider(
                    minimum=0,
                    maximum=19,
                    step=1,
                    value=0,
                    label="Generation Step",
                    info="Browse through generation steps",
                    visible=False,
                )

                # Causal Intervention controls
                correct_token_input = gr.Textbox(
                    label="Correct Token",
                    placeholder='e.g. "Paris" — the answer the clean prompt should produce',
                    visible=False,
                )
                incorrect_token_input = gr.Textbox(
                    label="Incorrect Token",
                    placeholder='e.g. "Warsaw" — the answer the corrupted prompt produces',
                    visible=False,
                )
                token_info_display = gr.Textbox(
                    label="Token Resolution",
                    interactive=False,
                    visible=False,
                )
                intervention_type_radio = gr.Radio(
                    choices=[t.value for t in InterventionType],
                    value=InterventionType.ACTIVATION_PATCH.value,
                    label="Intervention Type",
                    visible=False,
                )
                sweep_mode_radio = gr.Radio(
                    choices=["Layer Sweep (all positions)", "Full Position × Layer Sweep"],
                    value="Layer Sweep (all positions)",
                    label="Sweep Mode",
                    visible=False,
                )

                # Circuit Discovery controls
                circuit_threshold_slider = gr.Slider(
                    minimum=0.01,
                    maximum=0.5,
                    step=0.01,
                    value=0.1,
                    label="Circuit Threshold",
                    info="Minimum recovery score for a node/edge to be included in the circuit",
                    visible=False,
                )
                circuit_granularity_radio = gr.Radio(
                    choices=["Fast (layer paths only)", "Detailed (+ attention heads)"],
                    value="Fast (layer paths only)",
                    label="Sweep Granularity",
                    visible=False,
                )

                # Update token resolution display when tokens are typed
                def update_token_info(correct_text, incorrect_text):
                    if INTERVENTION_ENGINE is None:
                        return "Intervention engine not initialized."
                    parts = []
                    if correct_text and correct_text.strip():
                        try:
                            _, display = INTERVENTION_ENGINE.resolve_token(correct_text.strip())
                            parts.append(f"Correct: {display}")
                        except ValueError as e:
                            parts.append(f"Correct: {e}")
                    if incorrect_text and incorrect_text.strip():
                        try:
                            _, display = INTERVENTION_ENGINE.resolve_token(incorrect_text.strip())
                            parts.append(f"Incorrect: {display}")
                        except ValueError as e:
                            parts.append(f"Incorrect: {e}")
                    return " | ".join(parts) if parts else ""

                correct_token_input.change(
                    fn=update_token_info,
                    inputs=[correct_token_input, incorrect_token_input],
                    outputs=[token_info_display],
                )
                incorrect_token_input.change(
                    fn=update_token_info,
                    inputs=[correct_token_input, incorrect_token_input],
                    outputs=[token_info_display],
                )

            with gr.Column(scale=2):
                with gr.Column(visible=True) as single_plot_wrapper:
                    output_plot = gr.Plot(label="Attention Visualization")

                with gr.Column(visible=False) as circuit_output_wrapper:
                    with gr.Tabs():
                        with gr.Tab("Path Matrix"):
                            circuit_path_plot = gr.Plot(label="Path Patching Matrix")
                        with gr.Tab("Layer Importance"):
                            circuit_layer_plot = gr.Plot(label="Layer Importance")
                        with gr.Tab("Circuit Diagram"):
                            circuit_diagram_plot = gr.Plot(label="Discovered Circuit")
                        with gr.Tab("Components"):
                            circuit_component_plot = gr.Plot(label="Head Importance")
                        with gr.Tab("Summary"):
                            circuit_summary_md = gr.Markdown("Run circuit discovery to see results.")

                with gr.Column(visible=False) as diff_output_wrapper:
                    with gr.Tabs():
                        with gr.Tab("Residual Similarity"):
                            diff_residual_plot = gr.Plot(label="Residual Stream Cosine Similarity")
                        with gr.Tab("Attention Divergence"):
                            diff_attention_plot = gr.Plot(label="Attention Pattern Divergence")
                        with gr.Tab("Top Changed Neurons"):
                            diff_neuron_plot = gr.Plot(label="Top Changed Neurons")
                        with gr.Tab("Logit Lens Diff"):
                            diff_logit_plot = gr.Plot(label="Logit Lens Prediction Diff")

                with gr.Column(visible=False) as intervention_output_wrapper:
                    with gr.Tabs():
                        with gr.Tab("Recovery Heatmap"):
                            intervention_heatmap_plot = gr.Plot(label="Position × Layer Recovery")
                        with gr.Tab("Layer Recovery"):
                            intervention_layer_plot = gr.Plot(label="Per-Layer Mean Recovery")
                        with gr.Tab("Position Recovery"):
                            intervention_position_plot = gr.Plot(label="Per-Position Mean Recovery")
                        with gr.Tab("Summary"):
                            intervention_summary_md = gr.Markdown("Run a full sweep to see results.")

                with gr.Column(visible=False) as multistep_output_wrapper:
                    with gr.Tabs():
                        with gr.Tab("Token Timeline"):
                            multistep_timeline_plot = gr.Plot(label="Token Timeline")
                        with gr.Tab("Logit Lens"):
                            multistep_logit_plot = gr.Plot(label="Logit Lens")
                        with gr.Tab("Mamba Attention"):
                            multistep_mamba_plot = gr.Plot(label="Mamba-2 Attention")
                        with gr.Tab("Transformer Attention"):
                            multistep_transformer_plot = gr.Plot(label="Transformer Attention")

                output_info = gr.Textbox(label="Info", interactive=False, lines=4)

        # Update layer type label when slider changes
        layer_slider.change(
            fn=get_layer_type_label,
            inputs=[layer_slider],
            outputs=[layer_type_display],
        )

        # Toggle visibility of controls based on view mode
        def on_view_mode_change(mode):
            is_multistep = (mode == "Multi-Step Generation")
            is_intervention = (mode == "Causal Intervention")
            is_circuit = (mode == "Circuit Discovery")
            is_diff = (mode == "Activation Diff")
            needs_tokens = is_intervention or is_circuit
            # single_plot_wrapper: visible when NOT circuit/diff/intervention/multistep
            show_single = not (is_circuit or is_diff or is_intervention or is_multistep)
            return (
                gr.update(visible=is_multistep),     # max_tokens_slider
                gr.update(visible=is_multistep),     # step_nav_slider
                gr.update(visible=needs_tokens),     # correct_token_input
                gr.update(visible=needs_tokens),     # incorrect_token_input
                gr.update(visible=needs_tokens),     # token_info_display
                gr.update(visible=is_intervention),  # intervention_type_radio
                gr.update(visible=is_intervention),  # sweep_mode_radio
                gr.update(visible=is_circuit),       # circuit_threshold_slider
                gr.update(visible=is_circuit),       # circuit_granularity_radio
                gr.update(visible=show_single),      # single_plot_wrapper
                gr.update(visible=is_circuit),       # circuit_output_wrapper
                gr.update(visible=is_diff),          # diff_output_wrapper
                gr.update(visible=is_intervention),  # intervention_output_wrapper
                gr.update(visible=is_multistep),     # multistep_output_wrapper
            )

        view_mode.change(
            fn=on_view_mode_change,
            inputs=[view_mode],
            outputs=[
                max_tokens_slider, step_nav_slider,
                correct_token_input, incorrect_token_input, token_info_display,
                intervention_type_radio, sweep_mode_radio,
                circuit_threshold_slider, circuit_granularity_radio,
                single_plot_wrapper, circuit_output_wrapper,
                diff_output_wrapper, intervention_output_wrapper,
                multistep_output_wrapper,
            ],
        )

        # Helper: default gr.update() values for tab-specific output slots
        def _default_circuit_outputs():
            return (
                gr.update(),               # circuit_path_plot
                gr.update(),               # circuit_layer_plot
                gr.update(),               # circuit_diagram_plot
                gr.update(),               # circuit_component_plot
                gr.update(),               # circuit_summary_md
            )

        def _default_diff_outputs():
            return (
                gr.update(),               # diff_residual_plot
                gr.update(),               # diff_attention_plot
                gr.update(),               # diff_neuron_plot
                gr.update(),               # diff_logit_plot
            )

        def _default_intervention_outputs():
            return (
                gr.update(),               # intervention_heatmap_plot
                gr.update(),               # intervention_layer_plot
                gr.update(),               # intervention_position_plot
                gr.update(),               # intervention_summary_md
            )

        def _default_multistep_outputs():
            return (
                gr.update(),               # multistep_timeline_plot
                gr.update(),               # multistep_logit_plot
                gr.update(),               # multistep_mamba_plot
                gr.update(),               # multistep_transformer_plot
            )

        # Main analysis function
        def analyze(prompt, prompt_b, view_mode, layer_idx, head_agg, max_tokens, step_idx,
                    correct_token, incorrect_token, intervention_type, sweep_mode,
                    circuit_threshold, circuit_granularity, progress=gr.Progress(track_tqdm=False)):

            # Common prefix for non-tabbed views: output_plot, output_info, step_nav_slider,
            # then 5 wrapper visibilities, then all tab slot updates
            def _make_return(plot, info, slider_upd, vis_mode, **tab_overrides):
                """Build the full return tuple.

                vis_mode: one of "single", "circuit", "diff", "intervention", "multistep"
                tab_overrides: dict of group -> tuple, e.g. circuit=(p,l,d,c,s)
                """
                wrappers = (
                    gr.update(visible=(vis_mode == "single")),
                    gr.update(visible=(vis_mode == "circuit")),
                    gr.update(visible=(vis_mode == "diff")),
                    gr.update(visible=(vis_mode == "intervention")),
                    gr.update(visible=(vis_mode == "multistep")),
                )

                circuit = tab_overrides.get("circuit", _default_circuit_outputs())
                diff = tab_overrides.get("diff", _default_diff_outputs())
                intervention = tab_overrides.get("intervention", _default_intervention_outputs())
                multistep = tab_overrides.get("multistep", _default_multistep_outputs())

                return (plot, info, slider_upd) + wrappers + circuit + diff + intervention + multistep

            if not prompt or not prompt.strip():
                return _make_return(None, "Please enter a prompt.", gr.update(), "single")

            try:
                if view_mode == "Circuit Discovery":
                    def cd_progress(step, total):
                        progress(step / total, desc=f"Circuit discovery: {step}/{total}")

                    circuit_result, info = circuit_discovery_view(
                        prompt, prompt_b, correct_token, incorrect_token,
                        circuit_threshold, circuit_granularity,
                        progress_callback=cd_progress,
                    )

                    path_fig = create_path_matrix_plotly(circuit_result, ARCH_MAP)
                    layer_fig = create_layer_importance_plotly(circuit_result, ARCH_MAP)
                    diagram_fig = create_circuit_diagram_plotly(circuit_result, ARCH_MAP)
                    component_fig = create_component_importance_plotly(circuit_result, ARCH_MAP)
                    summary_md = create_circuit_summary_markdown(circuit_result, ARCH_MAP)

                    return _make_return(
                        gr.update(), info, gr.update(), "circuit",
                        circuit=(path_fig, layer_fig, diagram_fig, component_fig, summary_md),
                    )

                if view_mode == "Activation Diff":
                    if not prompt_b or not prompt_b.strip():
                        return _make_return(None, "Please enter a comparison prompt (Prompt B).", gr.update(), "diff")
                    result_a = run_extraction(prompt)
                    result_b = run_extraction(prompt_b)
                    if result_a is None or result_b is None:
                        return _make_return(None, "Extraction failed for one or both prompts.", gr.update(), "diff")
                    diff_result = compute_activation_diff(result_a, result_b, head_agg=head_agg)
                    info = format_diff_info(diff_result)
                    return _make_return(
                        gr.update(), info, gr.update(), "diff",
                        diff=(
                            create_residual_similarity_plotly(diff_result, ARCH_MAP),
                            create_attention_divergence_plotly(diff_result),
                            create_neuron_changes_plotly(diff_result),
                            create_logit_lens_diff_plotly(diff_result),
                        ),
                    )

                if view_mode == "Causal Intervention":
                    def iv_progress(step, total):
                        progress(step / total, desc=f"Intervention sweep: {step}/{total}")

                    fig, info = intervention_view(
                        prompt, prompt_b, correct_token, incorrect_token,
                        intervention_type, sweep_mode,
                        progress_callback=iv_progress,
                    )

                    # Check if this is a full sweep (2D matrix) — use tabs
                    cache_key = (
                        prompt.strip(), (prompt_b or "").strip(),
                        (correct_token or "").strip(), (incorrect_token or "").strip(),
                        intervention_type, sweep_mode,
                    )
                    sweep_result = INTERVENTION_CACHE.get(cache_key)
                    is_full = (sweep_result is not None
                               and sweep_result.recovery_matrix.dim() == 2)

                    if is_full:
                        return _make_return(
                            gr.update(), info, gr.update(), "intervention",
                            intervention=(
                                create_position_layer_heatmap_plotly(sweep_result, ARCH_MAP),
                                create_layer_marginal_plotly(sweep_result, ARCH_MAP),
                                create_position_marginal_plotly(sweep_result),
                                create_intervention_summary_markdown(sweep_result),
                            ),
                        )
                    else:
                        # Layer sweep — single plot, no tabs
                        return _make_return(fig, info, gr.update(), "single")

                if view_mode == "Multi-Step Generation":
                    fig_compat, info, slider_update = multistep_view(prompt, max_tokens, step_idx, head_agg)
                    step_data = None
                    cache_key = (prompt.strip(), int(max_tokens))
                    if cache_key in MULTISTEP_CACHE:
                        result = MULTISTEP_CACHE[cache_key]
                        sidx = max(0, min(int(step_idx), len(result["steps"]) - 1))
                        step_data = result["steps"][sidx]

                    return _make_return(
                        gr.update(), info, slider_update, "multistep",
                        multistep=(
                            create_token_timeline_plotly(step_data),
                            create_logit_lens_compact_plotly(step_data),
                            create_mamba_attention_plotly(step_data, ARCH_MAP, head_agg),
                            create_transformer_attention_plotly(step_data, ARCH_MAP, head_agg),
                        ),
                    )

                # All other views — single plot
                if view_mode == "Single Layer":
                    fig, info = single_layer_view(prompt, layer_idx, head_agg)
                elif view_mode == "Mamba vs Transformer":
                    fig, info = comparison_view(prompt, head_agg)
                elif view_mode == "All Layers":
                    fig, info = all_layers_view(prompt, head_agg)
                elif view_mode == "Logit Lens":
                    fig, info = logit_lens_view(prompt)
                elif view_mode == "Neuron Activation":
                    fig, info = neuron_activation_view(prompt, layer_idx)
                else:
                    return _make_return(None, "Unknown view mode.", gr.update(), "single")

                return _make_return(fig, info, gr.update(), "single")
            except Exception as e:
                return _make_return(None, f"Error: {str(e)}", gr.update(), "single")

        all_inputs = [prompt_input, prompt_b_input, view_mode, layer_slider,
                      head_agg, max_tokens_slider, step_nav_slider,
                      correct_token_input, incorrect_token_input,
                      intervention_type_radio, sweep_mode_radio,
                      circuit_threshold_slider, circuit_granularity_radio]
        all_outputs = [
            output_plot, output_info, step_nav_slider,
            # 5 wrapper visibilities
            single_plot_wrapper, circuit_output_wrapper,
            diff_output_wrapper, intervention_output_wrapper,
            multistep_output_wrapper,
            # Circuit tabs (5)
            circuit_path_plot, circuit_layer_plot,
            circuit_diagram_plot, circuit_component_plot,
            circuit_summary_md,
            # Diff tabs (4)
            diff_residual_plot, diff_attention_plot,
            diff_neuron_plot, diff_logit_plot,
            # Intervention tabs (4)
            intervention_heatmap_plot, intervention_layer_plot,
            intervention_position_plot, intervention_summary_md,
            # Multistep tabs (4)
            multistep_timeline_plot, multistep_logit_plot,
            multistep_mamba_plot, multistep_transformer_plot,
        ]

        analyze_btn.click(fn=analyze, inputs=all_inputs, outputs=all_outputs)

        # Also trigger on Enter in prompt box
        prompt_input.submit(fn=analyze, inputs=all_inputs, outputs=all_outputs)

        # Step slider navigates between cached steps without re-generating
        def on_step_change(prompt, max_tokens, step_idx, head_agg, current_mode):
            # Must return same shape as all_outputs minus the first 3 (plot, info, slider)
            # plus the first 2 (plot, info). Total = 2 + (len(all_outputs) - 3) padding
            n_tab_slots = 5 + 4 + 4 + 4  # circuit + diff + intervention + multistep
            n_wrappers = 5
            no_change = (gr.update(), gr.update()) + tuple(gr.update() for _ in range(n_wrappers + n_tab_slots))
            if current_mode != "Multi-Step Generation":
                return no_change
            cache_key = (prompt.strip(), int(max_tokens))
            if cache_key not in MULTISTEP_CACHE:
                return no_change
            result = MULTISTEP_CACHE[cache_key]
            num_steps = len(result["steps"])
            step_idx = max(0, min(int(step_idx), num_steps - 1))
            step_data = result["steps"][step_idx]
            info = format_multistep_info(result, step_idx)

            # Build multistep Plotly figures
            timeline_fig = create_token_timeline_plotly(step_data)
            logit_fig = create_logit_lens_compact_plotly(step_data)
            mamba_fig = create_mamba_attention_plotly(step_data, ARCH_MAP, head_agg)
            transformer_fig = create_transformer_attention_plotly(step_data, ARCH_MAP, head_agg)

            # Return: output_plot (unused), output_info, 5 wrappers (no change),
            # circuit(5) no change, diff(4) no change, intervention(4) no change,
            # multistep(4) updated
            padding = tuple(gr.update() for _ in range(n_wrappers + 5 + 4 + 4))
            return (
                gr.update(),  # output_plot
                info,         # output_info
            ) + padding + (timeline_fig, logit_fig, mamba_fig, transformer_fig)

        step_nav_outputs = [
            output_plot, output_info,
            single_plot_wrapper, circuit_output_wrapper,
            diff_output_wrapper, intervention_output_wrapper,
            multistep_output_wrapper,
            circuit_path_plot, circuit_layer_plot,
            circuit_diagram_plot, circuit_component_plot,
            circuit_summary_md,
            diff_residual_plot, diff_attention_plot,
            diff_neuron_plot, diff_logit_plot,
            intervention_heatmap_plot, intervention_layer_plot,
            intervention_position_plot, intervention_summary_md,
            multistep_timeline_plot, multistep_logit_plot,
            multistep_mamba_plot, multistep_transformer_plot,
        ]

        step_nav_slider.release(
            fn=on_step_change,
            inputs=[prompt_input, max_tokens_slider, step_nav_slider, head_agg, view_mode],
            outputs=step_nav_outputs,
        )

    return app


if __name__ == "__main__":
    initialize()
    app = build_app()

    if os.environ.get("SPACE_ID"):
        # HuggingFace Spaces: bind to all interfaces, no auth
        app.launch(server_name="0.0.0.0", server_port=7860)
    else:
        # Local development: localhost only, optional auth
        auth_user = os.environ.get("GRADIO_AUTH_USERNAME")
        auth_pass = os.environ.get("GRADIO_AUTH_PASSWORD")
        auth = (auth_user, auth_pass) if auth_user and auth_pass else None
        app.launch(share=False, auth=auth, server_name="127.0.0.1", server_port=7860)
