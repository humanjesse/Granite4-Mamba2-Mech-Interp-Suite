"""Gradio app for Granite 4.0 Mamba-2 Interpretability Tool."""

import os

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import gradio as gr
import torch
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
from src.visualization.activation_diff_viz import create_activation_diff_summary, format_diff_info
from src.extraction.multistep_generator import run_multistep_generation
from src.visualization.multistep_viz import create_multistep_dashboard, format_multistep_info
from src.extraction.causal_intervention import CausalInterventionEngine, InterventionType, MIN_RECOVERY_DENOMINATOR
from src.visualization.intervention_viz import (
    create_layer_sweep_dashboard,
    create_full_sweep_dashboard,
    format_intervention_info,
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


def initialize():
    """Load model and set up extractor."""
    global MODEL, TOKENIZER, EXTRACTOR, ARCH_MAP, INTERVENTION_ENGINE

    # Use CPU by default — .to(cuda) hangs intermittently on gfx1151.
    # 350M model runs fine on CPU. Set DEVICE=cuda to force GPU.
    device = os.environ.get("DEVICE", "cpu")
    print(f"Loading model on {device}...")
    MODEL, TOKENIZER, device = load_model(device=device)
    ARCH_MAP = ArchitectureMap(MODEL.config)
    EXTRACTOR = GraniteAttentionExtractor(MODEL, TOKENIZER, device=device)
    INTERVENTION_ENGINE = CausalInterventionEngine(MODEL, TOKENIZER, ARCH_MAP, device=device)
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
                      intervention_type, sweep_mode):
    """Run causal intervention experiment and visualize results."""
    if INTERVENTION_ENGINE is None:
        return None, "Intervention engine not initialized. Please restart the app."

    corrupted_prompt = corrupted_prompt or ""
    correct_token = correct_token or ""
    incorrect_token = incorrect_token or ""
    intervention_type = intervention_type or InterventionType.ACTIVATION_PATCH.value
    sweep_mode = sweep_mode or "Layer Sweep (all positions)"

    if not corrupted_prompt or not corrupted_prompt.strip():
        return None, "Please enter a corrupted prompt."
    if not correct_token or not correct_token.strip():
        return None, "Please enter a correct token (the answer the clean prompt should produce)."
    if not incorrect_token or not incorrect_token.strip():
        return None, "Please enter an incorrect token (the answer the corrupted prompt produces)."

    int_type = InterventionType(intervention_type)

    cache_key = (
        clean_prompt.strip(), corrupted_prompt.strip(),
        correct_token.strip(), incorrect_token.strip(),
        intervention_type, sweep_mode,
    )

    if cache_key in INTERVENTION_CACHE:
        sweep_result = INTERVENTION_CACHE[cache_key]
    else:
        if sweep_mode == "Full Position × Layer Sweep":
            sweep_result = INTERVENTION_ENGINE.sweep_positions_and_layers(
                clean_prompt, corrupted_prompt,
                correct_token.strip(), incorrect_token.strip(),
                int_type,
            )
        else:
            positions = None  # all positions by default
            sweep_result = INTERVENTION_ENGINE.sweep_layers(
                clean_prompt, corrupted_prompt,
                correct_token.strip(), incorrect_token.strip(),
                int_type, positions=positions,
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
                    choices=["Single Layer", "Mamba vs Transformer", "All Layers", "Logit Lens", "Neuron Activation", "Activation Diff", "Causal Intervention", "Multi-Step Generation"],
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
                output_plot = gr.Plot(label="Attention Visualization")
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
            return (
                gr.update(visible=is_multistep),     # max_tokens_slider
                gr.update(visible=is_multistep),     # step_nav_slider
                gr.update(visible=is_intervention),  # correct_token_input
                gr.update(visible=is_intervention),  # incorrect_token_input
                gr.update(visible=is_intervention),  # token_info_display
                gr.update(visible=is_intervention),  # intervention_type_radio
                gr.update(visible=is_intervention),  # sweep_mode_radio
            )

        view_mode.change(
            fn=on_view_mode_change,
            inputs=[view_mode],
            outputs=[
                max_tokens_slider, step_nav_slider,
                correct_token_input, incorrect_token_input, token_info_display,
                intervention_type_radio, sweep_mode_radio,
            ],
        )

        # Main analysis function
        def analyze(prompt, prompt_b, view_mode, layer_idx, head_agg, max_tokens, step_idx,
                    correct_token, incorrect_token, intervention_type, sweep_mode):
            if not prompt or not prompt.strip():
                return None, "Please enter a prompt.", gr.update()

            try:
                if view_mode == "Multi-Step Generation":
                    return multistep_view(prompt, max_tokens, step_idx, head_agg)
                elif view_mode == "Single Layer":
                    fig, info = single_layer_view(prompt, layer_idx, head_agg)
                elif view_mode == "Mamba vs Transformer":
                    fig, info = comparison_view(prompt, head_agg)
                elif view_mode == "All Layers":
                    fig, info = all_layers_view(prompt, head_agg)
                elif view_mode == "Logit Lens":
                    fig, info = logit_lens_view(prompt)
                elif view_mode == "Neuron Activation":
                    fig, info = neuron_activation_view(prompt, layer_idx)
                elif view_mode == "Activation Diff":
                    fig, info = activation_diff_view(prompt, prompt_b, head_agg)
                elif view_mode == "Causal Intervention":
                    fig, info = intervention_view(
                        prompt, prompt_b, correct_token, incorrect_token,
                        intervention_type, sweep_mode,
                    )
                else:
                    return None, "Unknown view mode.", gr.update()
                return fig, info, gr.update()
            except Exception as e:
                return None, f"Error: {str(e)}", gr.update()

        all_inputs = [prompt_input, prompt_b_input, view_mode, layer_slider,
                      head_agg, max_tokens_slider, step_nav_slider,
                      correct_token_input, incorrect_token_input,
                      intervention_type_radio, sweep_mode_radio]
        all_outputs = [output_plot, output_info, step_nav_slider]

        analyze_btn.click(fn=analyze, inputs=all_inputs, outputs=all_outputs)

        # Also trigger on Enter in prompt box
        prompt_input.submit(fn=analyze, inputs=all_inputs, outputs=all_outputs)

        # Step slider navigates between cached steps without re-generating
        def on_step_change(prompt, max_tokens, step_idx, head_agg, current_mode):
            if current_mode != "Multi-Step Generation":
                return gr.update(), gr.update()
            cache_key = (prompt.strip(), int(max_tokens))
            if cache_key not in MULTISTEP_CACHE:
                return gr.update(), gr.update()
            result = MULTISTEP_CACHE[cache_key]
            num_steps = len(result["steps"])
            step_idx = max(0, min(int(step_idx), num_steps - 1))
            fig = create_multistep_dashboard(
                step_data=result["steps"][step_idx],
                arch_map=ARCH_MAP,
                head_agg=head_agg,
            )
            info = format_multistep_info(result, step_idx)
            return fig, info

        step_nav_slider.release(
            fn=on_step_change,
            inputs=[prompt_input, max_tokens_slider, step_nav_slider, head_agg, view_mode],
            outputs=[output_plot, output_info],
        )

    return app


if __name__ == "__main__":
    initialize()
    app = build_app()
    auth_user = os.environ.get("GRADIO_AUTH_USERNAME")
    auth_pass = os.environ.get("GRADIO_AUTH_PASSWORD")
    auth = (auth_user, auth_pass) if auth_user and auth_pass else None
    app.launch(share=False, auth=auth, server_name="127.0.0.1", server_port=7860)
