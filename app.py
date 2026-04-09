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
from src.storage.result_store import ResultStore
from src.sae.sparse_autoencoder import SparseAutoencoder
from src.sae.trainer import SAETrainingConfig, collect_activations, train_sae
from src.sae.feature_search import (
    search_features, get_default_bee_texts, get_default_baseline_texts,
)
from src.extraction.activation_steering import generate_comparison as steering_generate_comparison
from src.visualization.steering_viz import (
    create_training_loss_plotly,
    create_feature_density_plotly,
    create_training_summary_markdown,
    create_feature_search_plotly,
    create_feature_activations_plotly,
    create_feature_search_summary_markdown,
    create_steered_comparison_plotly,
    create_steering_top5_plotly,
    create_steering_summary_markdown,
    format_training_info,
    format_feature_search_info,
    format_steering_info,
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
LAST_INTERVENTION_RESULT = None
LAST_CIRCUIT_RESULT = None
RESULT_STORE = None
SAE_MODEL = None
SAE_TRAINING_RESULT = None
SAE_FEATURE_SEARCH_RESULT = None
SAE_STEERING_RESULT = None
SAE_LAYER_IDX = 16  # Track which layer the SAE was trained on


def initialize():
    """Load model and set up extractor."""
    global MODEL, TOKENIZER, EXTRACTOR, ARCH_MAP, INTERVENTION_ENGINE, CIRCUIT_ENGINE, RESULT_STORE

    # Use CPU by default — .to(cuda) hangs intermittently on gfx1151.
    # 350M model runs fine on CPU. Set DEVICE=cuda to force GPU.
    device = os.environ.get("DEVICE", "cpu")
    print(f"Loading model on {device}...")
    MODEL, TOKENIZER, device = load_model(device=device)
    ARCH_MAP = ArchitectureMap(MODEL.config)
    EXTRACTOR = GraniteAttentionExtractor(MODEL, TOKENIZER, device=device)
    INTERVENTION_ENGINE = CausalInterventionEngine(MODEL, TOKENIZER, ARCH_MAP, device=device)
    CIRCUIT_ENGINE = CircuitDiscoveryEngine(MODEL, TOKENIZER, ARCH_MAP, device=device)
    RESULT_STORE = ResultStore()
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
                    choices=["Single Layer", "Mamba vs Transformer", "All Layers", "Logit Lens", "Neuron Activation", "Activation Diff", "Causal Intervention", "Circuit Discovery", "Multi-Step Generation", "SAE Training", "Feature Browser", "Activation Steering"],
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

                # Experiment storage controls
                save_notes_input = gr.Textbox(
                    label="Notes",
                    placeholder="Describe what you're testing...",
                    lines=2,
                    visible=False,
                )
                save_btn = gr.Button("Save Results", variant="secondary", visible=False)
                save_status = gr.Textbox(label="Save Status", interactive=False, visible=False)
                load_dropdown = gr.Dropdown(
                    label="Load Saved Experiment",
                    choices=[],
                    interactive=True,
                    visible=False,
                )
                load_btn = gr.Button("Load", variant="secondary", visible=False)

                # ── SAE Training controls ──
                sae_layer_slider = gr.Slider(
                    minimum=0, maximum=num_layers - 1, step=1, value=16,
                    label="SAE Target Layer",
                    info="Which layer's residual stream to train the SAE on",
                    visible=False,
                )
                sae_expansion_slider = gr.Slider(
                    minimum=2, maximum=16, step=1, value=8,
                    label="Expansion Factor",
                    info="Latent dim = expansion_factor × hidden_size",
                    visible=False,
                )
                sae_num_activations_slider = gr.Slider(
                    minimum=10000, maximum=500000, step=10000, value=50000,
                    label="Training Activations",
                    info="Number of activation vectors to collect from model",
                    visible=False,
                )
                sae_l1_slider = gr.Slider(
                    minimum=1e-5, maximum=1e-1, step=1e-5, value=1e-3,
                    label="L1 Coefficient",
                    info="Sparsity penalty weight",
                    visible=False,
                )
                sae_train_btn = gr.Button("Train SAE", variant="primary", visible=False)
                sae_train_status = gr.Textbox(label="Training Status", interactive=False, visible=False)

                # ── Feature Browser controls ──
                feature_concept_input = gr.Textbox(
                    label="Concept Texts (one per line, blank = default bee texts)",
                    placeholder="Honeybees are essential pollinators...\nThe queen bee lays eggs...",
                    lines=3,
                    visible=False,
                )
                feature_search_btn = gr.Button("Search Features", variant="primary", visible=False)

                # ── Activation Steering controls ──
                steering_feature_slider = gr.Slider(
                    minimum=0, maximum=6143, step=1, value=0,
                    label="Steering Feature Index",
                    info="SAE feature to steer with (use Feature Browser to find best one)",
                    visible=False,
                )
                steering_coefficient_slider = gr.Slider(
                    minimum=0.0, maximum=50.0, step=0.5, value=10.0,
                    label="Steering Coefficient",
                    info="How strongly to steer (5-20 typical sweet spot)",
                    visible=False,
                )
                steering_max_tokens_slider = gr.Slider(
                    minimum=5, maximum=100, step=5, value=50,
                    label="Max Tokens to Generate",
                    info="Maximum tokens for steered generation",
                    visible=False,
                )
                steer_btn = gr.Button("Generate with Steering", variant="primary", visible=False)

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

                with gr.Column(visible=False) as sae_training_output_wrapper:
                    with gr.Tabs():
                        with gr.Tab("Training Loss"):
                            sae_loss_plot = gr.Plot(label="Training Loss Curves")
                        with gr.Tab("Feature Density"):
                            sae_density_plot = gr.Plot(label="Feature Activation Density")
                        with gr.Tab("Summary"):
                            sae_training_summary_md = gr.Markdown("Train an SAE to see results.")

                with gr.Column(visible=False) as feature_browser_output_wrapper:
                    with gr.Tabs():
                        with gr.Tab("Feature Ranking"):
                            feature_ranking_plot = gr.Plot(label="Top Features by Selectivity")
                        with gr.Tab("Feature Activations"):
                            feature_activations_plot = gr.Plot(label="Per-Text Feature Activations")
                        with gr.Tab("Summary"):
                            feature_search_summary_md = gr.Markdown("Search for features to see results.")

                with gr.Column(visible=False) as steering_output_wrapper:
                    with gr.Tabs():
                        with gr.Tab("Comparison"):
                            steering_comparison_plot = gr.Plot(label="Steered vs Unsteered Generation")
                        with gr.Tab("Top-5 Predictions"):
                            steering_top5_plot = gr.Plot(label="Per-Token Top Predictions")
                        with gr.Tab("Summary"):
                            steering_summary_md = gr.Markdown("Run steering to see results.")

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
            is_sae_training = (mode == "SAE Training")
            is_feature_browser = (mode == "Feature Browser")
            is_steering = (mode == "Activation Steering")
            needs_tokens = is_intervention or is_circuit
            show_storage = is_intervention or is_circuit
            tabbed_modes = (is_circuit or is_diff or is_intervention or is_multistep
                            or is_sae_training or is_feature_browser or is_steering)
            show_single = not tabbed_modes

            # Refresh load dropdown when switching to a storage-enabled mode
            load_choices = []
            if show_storage and RESULT_STORE is not None:
                exps = RESULT_STORE.list_experiments()
                load_choices = [e["short_label"] for e in exps]

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
                gr.update(visible=is_sae_training),  # sae_training_output_wrapper
                gr.update(visible=is_feature_browser),  # feature_browser_output_wrapper
                gr.update(visible=is_steering),      # steering_output_wrapper
                gr.update(visible=show_storage),     # save_notes_input
                gr.update(visible=show_storage),     # save_btn
                gr.update(visible=show_storage),     # save_status
                gr.update(visible=show_storage, choices=load_choices, value=None),  # load_dropdown
                gr.update(visible=show_storage),     # load_btn
                # SAE Training controls
                gr.update(visible=is_sae_training),  # sae_layer_slider
                gr.update(visible=is_sae_training),  # sae_expansion_slider
                gr.update(visible=is_sae_training),  # sae_num_activations_slider
                gr.update(visible=is_sae_training),  # sae_l1_slider
                gr.update(visible=is_sae_training),  # sae_train_btn
                gr.update(visible=is_sae_training),  # sae_train_status
                # Feature Browser controls
                gr.update(visible=is_feature_browser),  # feature_concept_input
                gr.update(visible=is_feature_browser),  # feature_search_btn
                # Activation Steering controls
                gr.update(visible=is_steering),      # steering_feature_slider
                gr.update(visible=is_steering),      # steering_coefficient_slider
                gr.update(visible=is_steering),      # steering_max_tokens_slider
                gr.update(visible=is_steering),      # steer_btn
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
                sae_training_output_wrapper, feature_browser_output_wrapper,
                steering_output_wrapper,
                save_notes_input, save_btn, save_status,
                load_dropdown, load_btn,
                # SAE Training controls
                sae_layer_slider, sae_expansion_slider,
                sae_num_activations_slider, sae_l1_slider,
                sae_train_btn, sae_train_status,
                # Feature Browser controls
                feature_concept_input, feature_search_btn,
                # Steering controls
                steering_feature_slider, steering_coefficient_slider,
                steering_max_tokens_slider, steer_btn,
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

        def _default_sae_training_outputs():
            return (
                gr.update(),               # sae_loss_plot
                gr.update(),               # sae_density_plot
                gr.update(),               # sae_training_summary_md
            )

        def _default_feature_browser_outputs():
            return (
                gr.update(),               # feature_ranking_plot
                gr.update(),               # feature_activations_plot
                gr.update(),               # feature_search_summary_md
            )

        def _default_steering_outputs():
            return (
                gr.update(),               # steering_comparison_plot
                gr.update(),               # steering_top5_plot
                gr.update(),               # steering_summary_md
            )

        # Main analysis function
        def analyze(prompt, prompt_b, view_mode, layer_idx, head_agg, max_tokens, step_idx,
                    correct_token, incorrect_token, intervention_type, sweep_mode,
                    circuit_threshold, circuit_granularity, progress=gr.Progress(track_tqdm=False)):

            # Common prefix for non-tabbed views: output_plot, output_info, step_nav_slider,
            # then 5 wrapper visibilities, then all tab slot updates
            def _make_return(plot, info, slider_upd, vis_mode, **tab_overrides):
                """Build the full return tuple.

                vis_mode: one of "single", "circuit", "diff", "intervention", "multistep",
                          "sae_training", "feature_browser", "steering"
                tab_overrides: dict of group -> tuple, e.g. circuit=(p,l,d,c,s)
                """
                wrappers = (
                    gr.update(visible=(vis_mode == "single")),
                    gr.update(visible=(vis_mode == "circuit")),
                    gr.update(visible=(vis_mode == "diff")),
                    gr.update(visible=(vis_mode == "intervention")),
                    gr.update(visible=(vis_mode == "multistep")),
                    gr.update(visible=(vis_mode == "sae_training")),
                    gr.update(visible=(vis_mode == "feature_browser")),
                    gr.update(visible=(vis_mode == "steering")),
                )

                circuit = tab_overrides.get("circuit", _default_circuit_outputs())
                diff = tab_overrides.get("diff", _default_diff_outputs())
                intervention = tab_overrides.get("intervention", _default_intervention_outputs())
                multistep = tab_overrides.get("multistep", _default_multistep_outputs())
                sae_training = tab_overrides.get("sae_training", _default_sae_training_outputs())
                feature_browser = tab_overrides.get("feature_browser", _default_feature_browser_outputs())
                steering = tab_overrides.get("steering", _default_steering_outputs())

                return (plot, info, slider_upd) + wrappers + circuit + diff + intervention + multistep + sae_training + feature_browser + steering

            if not prompt or not prompt.strip():
                return _make_return(None, "Please enter a prompt.", gr.update(), "single")

            try:
                if view_mode == "Circuit Discovery":
                    def cd_progress(step, total):
                        progress(step / total, desc=f"Circuit discovery: {step}/{total}")

                    global LAST_CIRCUIT_RESULT
                    circuit_result, info = circuit_discovery_view(
                        prompt, prompt_b, correct_token, incorrect_token,
                        circuit_threshold, circuit_granularity,
                        progress_callback=cd_progress,
                    )
                    LAST_CIRCUIT_RESULT = circuit_result

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
                    global LAST_INTERVENTION_RESULT
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
                    LAST_INTERVENTION_RESULT = sweep_result
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

                if view_mode == "SAE Training":
                    if SAE_TRAINING_RESULT is None:
                        return _make_return(None, "No SAE trained yet. Configure settings and click 'Train SAE'.", gr.update(), "sae_training")
                    return _make_return(
                        gr.update(), format_training_info(SAE_TRAINING_RESULT), gr.update(), "sae_training",
                        sae_training=(
                            create_training_loss_plotly(SAE_TRAINING_RESULT),
                            create_feature_density_plotly(SAE_TRAINING_RESULT),
                            create_training_summary_markdown(SAE_TRAINING_RESULT),
                        ),
                    )

                if view_mode == "Feature Browser":
                    if SAE_MODEL is None:
                        return _make_return(None, "Train an SAE first, then search for features.", gr.update(), "feature_browser")
                    if SAE_FEATURE_SEARCH_RESULT is None:
                        return _make_return(None, "Click 'Search Features' to find concept-specific features.", gr.update(), "feature_browser")
                    return _make_return(
                        gr.update(), format_feature_search_info(SAE_FEATURE_SEARCH_RESULT), gr.update(), "feature_browser",
                        feature_browser=(
                            create_feature_search_plotly(SAE_FEATURE_SEARCH_RESULT),
                            create_feature_activations_plotly(SAE_FEATURE_SEARCH_RESULT, 0),
                            create_feature_search_summary_markdown(SAE_FEATURE_SEARCH_RESULT),
                        ),
                    )

                if view_mode == "Activation Steering":
                    if SAE_MODEL is None:
                        return _make_return(None, "Train an SAE first, then use steering.", gr.update(), "steering")
                    if SAE_STEERING_RESULT is None:
                        return _make_return(None, "Enter a prompt and click 'Generate with Steering'.", gr.update(), "steering")
                    return _make_return(
                        gr.update(), format_steering_info(SAE_STEERING_RESULT), gr.update(), "steering",
                        steering=(
                            create_steered_comparison_plotly(SAE_STEERING_RESULT),
                            create_steering_top5_plotly(SAE_STEERING_RESULT),
                            create_steering_summary_markdown(SAE_STEERING_RESULT),
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
            # 8 wrapper visibilities
            single_plot_wrapper, circuit_output_wrapper,
            diff_output_wrapper, intervention_output_wrapper,
            multistep_output_wrapper,
            sae_training_output_wrapper, feature_browser_output_wrapper,
            steering_output_wrapper,
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
            # SAE Training tabs (3)
            sae_loss_plot, sae_density_plot, sae_training_summary_md,
            # Feature Browser tabs (3)
            feature_ranking_plot, feature_activations_plot, feature_search_summary_md,
            # Steering tabs (3)
            steering_comparison_plot, steering_top5_plot, steering_summary_md,
        ]

        analyze_btn.click(fn=analyze, inputs=all_inputs, outputs=all_outputs)

        # Also trigger on Enter in prompt box
        prompt_input.submit(fn=analyze, inputs=all_inputs, outputs=all_outputs)

        # Step slider navigates between cached steps without re-generating
        def on_step_change(prompt, max_tokens, step_idx, head_agg, current_mode):
            # Must return same shape as step_nav_outputs
            n_tab_slots = 5 + 4 + 4 + 4 + 3 + 3 + 3  # circuit+diff+intervention+multistep+sae+feature+steering
            n_wrappers = 8
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

            # Return: output_plot (unused), output_info, 8 wrappers (no change),
            # circuit(5) no change, diff(4) no change, intervention(4) no change,
            # multistep(4) updated, sae(3)+feature(3)+steering(3) no change
            padding_before = tuple(gr.update() for _ in range(n_wrappers + 5 + 4 + 4))
            padding_after = tuple(gr.update() for _ in range(3 + 3 + 3))
            return (
                gr.update(),  # output_plot
                info,         # output_info
            ) + padding_before + (timeline_fig, logit_fig, mamba_fig, transformer_fig) + padding_after

        step_nav_outputs = [
            output_plot, output_info,
            single_plot_wrapper, circuit_output_wrapper,
            diff_output_wrapper, intervention_output_wrapper,
            multistep_output_wrapper,
            sae_training_output_wrapper, feature_browser_output_wrapper,
            steering_output_wrapper,
            circuit_path_plot, circuit_layer_plot,
            circuit_diagram_plot, circuit_component_plot,
            circuit_summary_md,
            diff_residual_plot, diff_attention_plot,
            diff_neuron_plot, diff_logit_plot,
            intervention_heatmap_plot, intervention_layer_plot,
            intervention_position_plot, intervention_summary_md,
            multistep_timeline_plot, multistep_logit_plot,
            multistep_mamba_plot, multistep_transformer_plot,
            sae_loss_plot, sae_density_plot, sae_training_summary_md,
            feature_ranking_plot, feature_activations_plot, feature_search_summary_md,
            steering_comparison_plot, steering_top5_plot, steering_summary_md,
        ]

        step_nav_slider.release(
            fn=on_step_change,
            inputs=[prompt_input, max_tokens_slider, step_nav_slider, head_agg, view_mode],
            outputs=step_nav_outputs,
        )

        # ── Save / Load handlers ───────────────────────────────────────

        def on_save_click(current_view_mode, notes):
            if RESULT_STORE is None:
                return "Storage not initialized.", gr.update()
            try:
                if current_view_mode == "Causal Intervention":
                    if LAST_INTERVENTION_RESULT is None:
                        return "No intervention results to save. Run an experiment first.", gr.update()
                    save_dir = RESULT_STORE.save_intervention(LAST_INTERVENTION_RESULT, notes=notes)
                    status = f"Saved to {save_dir.name}"
                elif current_view_mode == "Circuit Discovery":
                    if LAST_CIRCUIT_RESULT is None:
                        return "No circuit results to save. Run an experiment first.", gr.update()
                    save_dir = RESULT_STORE.save_circuit(LAST_CIRCUIT_RESULT, notes=notes)
                    status = f"Saved to {save_dir.name}"
                else:
                    return "Save is only available for Causal Intervention and Circuit Discovery.", gr.update()
                # Refresh dropdown
                exps = RESULT_STORE.list_experiments()
                choices = [e["short_label"] for e in exps]
                return status, gr.update(choices=choices, value=None)
            except Exception as e:
                return f"Save failed: {e}", gr.update()

        save_btn.click(
            fn=on_save_click,
            inputs=[view_mode, save_notes_input],
            outputs=[save_status, load_dropdown],
        )

        def on_load_click(selected_label):
            # Build a full return tuple matching load_outputs shape
            n_wrappers = 8
            n_tabs = 5 + 4 + 4 + 4 + 3 + 3 + 3  # circuit+diff+intervention+multistep+sae+feature+steering
            no_change = (gr.update(), gr.update()) + tuple(gr.update() for _ in range(n_wrappers + n_tabs))
            new_defaults = _default_sae_training_outputs() + _default_feature_browser_outputs() + _default_steering_outputs()

            if not selected_label or RESULT_STORE is None:
                return (gr.update(), "Select an experiment to load.") + tuple(
                    gr.update() for _ in range(n_wrappers + n_tabs)
                )

            exps = RESULT_STORE.list_experiments()
            match = next((e for e in exps if e["short_label"] == selected_label), None)
            if match is None:
                return (gr.update(), f"Experiment not found: {selected_label}") + tuple(
                    gr.update() for _ in range(n_wrappers + n_tabs)
                )

            try:
                if match["tool_type"] == "intervention":
                    result, meta = RESULT_STORE.load_intervention(match["path"])
                    notes_line = f"Notes: {meta.get('notes', '(none)')}"
                    info = f"[LOADED] {format_intervention_info(result)}\n{notes_line}"
                    is_full = result.recovery_matrix.dim() == 2
                    wrappers = (
                        gr.update(visible=not is_full),  # single_plot_wrapper
                        gr.update(visible=False),        # circuit_output_wrapper
                        gr.update(visible=False),        # diff_output_wrapper
                        gr.update(visible=is_full),      # intervention_output_wrapper
                        gr.update(visible=False),        # multistep_output_wrapper
                        gr.update(visible=False),        # sae_training_output_wrapper
                        gr.update(visible=False),        # feature_browser_output_wrapper
                        gr.update(visible=False),        # steering_output_wrapper
                    )
                    if is_full:
                        intervention = (
                            create_position_layer_heatmap_plotly(result, ARCH_MAP),
                            create_layer_marginal_plotly(result, ARCH_MAP),
                            create_position_marginal_plotly(result),
                            create_intervention_summary_markdown(result),
                        )
                        return (gr.update(), info) + wrappers + _default_circuit_outputs() + _default_diff_outputs() + intervention + _default_multistep_outputs() + new_defaults
                    else:
                        fig = create_layer_sweep_dashboard(result, ARCH_MAP)
                        return (fig, info) + wrappers + _default_circuit_outputs() + _default_diff_outputs() + _default_intervention_outputs() + _default_multistep_outputs() + new_defaults

                elif match["tool_type"] == "circuit":
                    result, meta = RESULT_STORE.load_circuit(match["path"])
                    notes_line = f"Notes: {meta.get('notes', '(none)')}"
                    info = f"[LOADED] {format_circuit_info(result)}\n{notes_line}"
                    wrappers = (
                        gr.update(visible=False),  # single_plot_wrapper
                        gr.update(visible=True),   # circuit_output_wrapper
                        gr.update(visible=False),  # diff_output_wrapper
                        gr.update(visible=False),  # intervention_output_wrapper
                        gr.update(visible=False),  # multistep_output_wrapper
                        gr.update(visible=False),  # sae_training_output_wrapper
                        gr.update(visible=False),  # feature_browser_output_wrapper
                        gr.update(visible=False),  # steering_output_wrapper
                    )
                    circuit = (
                        create_path_matrix_plotly(result, ARCH_MAP),
                        create_layer_importance_plotly(result, ARCH_MAP),
                        create_circuit_diagram_plotly(result, ARCH_MAP),
                        create_component_importance_plotly(result, ARCH_MAP),
                        create_circuit_summary_markdown(result, ARCH_MAP),
                    )
                    return (gr.update(), info) + wrappers + circuit + _default_diff_outputs() + _default_intervention_outputs() + _default_multistep_outputs() + new_defaults

            except Exception as e:
                return (gr.update(), f"Error loading: {e}") + tuple(
                    gr.update() for _ in range(n_wrappers + n_tabs)
                )

            return no_change

        load_outputs = [
            output_plot, output_info,
            single_plot_wrapper, circuit_output_wrapper,
            diff_output_wrapper, intervention_output_wrapper,
            multistep_output_wrapper,
            sae_training_output_wrapper, feature_browser_output_wrapper,
            steering_output_wrapper,
            circuit_path_plot, circuit_layer_plot,
            circuit_diagram_plot, circuit_component_plot,
            circuit_summary_md,
            diff_residual_plot, diff_attention_plot,
            diff_neuron_plot, diff_logit_plot,
            intervention_heatmap_plot, intervention_layer_plot,
            intervention_position_plot, intervention_summary_md,
            multistep_timeline_plot, multistep_logit_plot,
            multistep_mamba_plot, multistep_transformer_plot,
            sae_loss_plot, sae_density_plot, sae_training_summary_md,
            feature_ranking_plot, feature_activations_plot, feature_search_summary_md,
            steering_comparison_plot, steering_top5_plot, steering_summary_md,
        ]

        load_btn.click(
            fn=on_load_click,
            inputs=[load_dropdown],
            outputs=load_outputs,
        )

        # ── SAE / Feature / Steering button handlers ─────────────────

        def on_sae_train_click(sae_layer, expansion, num_acts, l1_coeff, progress=gr.Progress(track_tqdm=False)):
            global SAE_MODEL, SAE_TRAINING_RESULT, SAE_LAYER_IDX
            if MODEL is None:
                return "Model not loaded.", gr.update(), gr.update(), gr.update()

            config = SAETrainingConfig(
                layer_idx=int(sae_layer),
                expansion_factor=int(expansion),
                l1_coefficient=float(l1_coeff),
                num_activations=int(num_acts),
            )

            progress(0.0, desc="Collecting activations from model...")
            activations = collect_activations(
                MODEL, TOKENIZER, config.layer_idx, config.num_activations,
                device=str(next(MODEL.parameters()).device),
                progress_callback=lambda c, t: progress(0.4 * c / t, desc=f"Collecting activations: {c}/{t}"),
            )

            progress(0.4, desc="Training SAE...")
            result = train_sae(
                activations, config,
                progress_callback=lambda s, t: progress(0.4 + 0.6 * s / t, desc=f"Training SAE: step {s}/{t}"),
            )

            SAE_MODEL = result.sae
            SAE_TRAINING_RESULT = result
            SAE_LAYER_IDX = config.layer_idx

            return (
                format_training_info(result),
                create_training_loss_plotly(result),
                create_feature_density_plotly(result),
                create_training_summary_markdown(result),
            )

        sae_train_btn.click(
            fn=on_sae_train_click,
            inputs=[sae_layer_slider, sae_expansion_slider, sae_num_activations_slider, sae_l1_slider],
            outputs=[sae_train_status, sae_loss_plot, sae_density_plot, sae_training_summary_md],
        )

        def on_feature_search_click(concept_text, progress=gr.Progress(track_tqdm=False)):
            global SAE_FEATURE_SEARCH_RESULT
            if SAE_MODEL is None:
                return "Train an SAE first.", gr.update(), gr.update(), gr.update()

            concept_texts = (
                [t.strip() for t in concept_text.strip().split("\n") if t.strip()]
                if concept_text and concept_text.strip()
                else get_default_bee_texts()
            )
            baseline_texts = get_default_baseline_texts()

            result = search_features(
                SAE_MODEL, MODEL, TOKENIZER, SAE_LAYER_IDX,
                concept_texts, baseline_texts,
                device=str(next(MODEL.parameters()).device),
                progress_callback=lambda s, t: progress(s / t, desc=f"Searching features: {s}/{t}"),
            )
            SAE_FEATURE_SEARCH_RESULT = result

            # Update steering feature slider to best feature
            best_idx = result.top_features[0]["feature_idx"] if result.top_features else 0

            return (
                format_feature_search_info(result),
                create_feature_search_plotly(result),
                create_feature_activations_plotly(result, 0),
                create_feature_search_summary_markdown(result),
                gr.update(value=best_idx),  # steering_feature_slider
            )

        feature_search_btn.click(
            fn=on_feature_search_click,
            inputs=[feature_concept_input],
            outputs=[output_info, feature_ranking_plot, feature_activations_plot,
                     feature_search_summary_md, steering_feature_slider],
        )

        def on_steer_click(prompt, feature_idx, coefficient, steer_max_tokens, progress=gr.Progress(track_tqdm=False)):
            global SAE_STEERING_RESULT
            if SAE_MODEL is None:
                return "Train an SAE first.", gr.update(), gr.update(), gr.update()
            if not prompt or not prompt.strip():
                return "Enter a prompt to steer.", gr.update(), gr.update(), gr.update()

            steering_vector = SAE_MODEL.get_feature_direction(int(feature_idx))

            result = steering_generate_comparison(
                MODEL, TOKENIZER, prompt.strip(),
                steering_vector=steering_vector,
                layer_idx=SAE_LAYER_IDX,
                coefficient=float(coefficient),
                max_tokens=int(steer_max_tokens),
                device=str(next(MODEL.parameters()).device),
                progress_callback=lambda s, t: progress(s / t, desc=f"Generating: {s}/{t}"),
            )
            result.config.feature_idx = int(feature_idx)
            SAE_STEERING_RESULT = result

            return (
                format_steering_info(result),
                create_steered_comparison_plotly(result),
                create_steering_top5_plotly(result),
                create_steering_summary_markdown(result),
            )

        steer_btn.click(
            fn=on_steer_click,
            inputs=[prompt_input, steering_feature_slider, steering_coefficient_slider, steering_max_tokens_slider],
            outputs=[output_info, steering_comparison_plot, steering_top5_plot, steering_summary_md],
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
