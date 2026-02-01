# Granite 4.0 Mamba-2 Interpretability Tool

The first interpretability tool for IBM Granite 4.0's hybrid Mamba-2/Transformer architecture. Extracts "hidden attention" matrices from Mamba-2 layers — making the black-box SSM layers as interpretable as standard Transformer attention.

## What This Does

Granite 4.0 uses a **hybrid architecture**: 28 Mamba-2 layers + 4 Transformer attention layers (32 total). Transformer attention is well-understood (softmax over Q·K^T), but Mamba-2 layers are opaque — there's no explicit attention matrix.

This tool extracts **implicit hidden attention** from Mamba-2 using the formula from [Ali et al. (2024)](https://arxiv.org/abs/2403.01590):

```
α̃[i,j] = exp(Σ dtA) × (C[i] · B[j]) × dt[j]
```

Mamba-2's scalar-identity A matrix makes this extraction computationally cheap (vs. Mamba-1's full matrix exponential).

## Key Features

- **Hidden attention extraction** from all 28 Mamba-2 layers
- **Standard attention extraction** from all 4 Transformer layers (via Q·K hooks, since the model doesn't support `output_attentions`)
- **Side-by-side comparison** of Mamba-2 vs Transformer attention patterns
- **All-layers overview** grid showing attention across the full 32-layer architecture
- **Logit Lens** view showing how the model's next-token predictions evolve through all 32 layers — projects each layer's residual stream through the output head to reveal prediction confidence building layer by layer
- **Neuron Activation** view showing which MLP neurons fire strongly for each input token — captures SwiGLU intermediate activations (`silu(gate) * up`) from every layer's `shared_mlp`, ranks by absolute magnitude, and displays per-token top-k neurons to identify interpretable features
- **Activation Diff** for comparing two prompts side-by-side — shows residual stream cosine similarity per layer, Jensen-Shannon divergence of attention patterns, top changed neurons, and logit lens prediction differences in a 2×2 dashboard
- **Multi-Step Generation** analysis that generates tokens one at a time and runs full interpretability extraction at each step — browse through generation steps with a slider to see how attention patterns, neuron activations, and logit lens predictions shift as the sequence grows
- **Causal Intervention** (activation patching) to identify which layers are causally responsible for a prediction — run a clean prompt and a corrupted prompt, then sweep across layers patching clean activations into the corrupted run to measure recovery. Supports activation patching, zero/mean ablation, and noise injection, with layer sweep and full position-by-layer heatmap modes
- **Interactive Gradio UI** with layer selection, head aggregation, step navigation, and example prompts

## Architecture

```
IBM Granite 4.0 h-350m (32 layers):
├── Layers 0-9:   Mamba-2 × 10
├── Layer 10:      Transformer Attention
├── Layers 11-12: Mamba-2 × 2
├── Layer 13:      Transformer Attention
├── Layers 14-16: Mamba-2 × 3
├── Layer 17:      Transformer Attention
├── Layers 18-26: Mamba-2 × 9
├── Layer 27:      Transformer Attention
└── Layers 28-31: Mamba-2 × 4
```

## Setup

```bash
# Create Python 3.12 environment (PyTorch ROCm requires <=3.12)
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install PyTorch with ROCm support
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

# Install dependencies
uv pip install transformers accelerate gradio matplotlib numpy

# Verify GPU
HSA_OVERRIDE_GFX_VERSION=11.0.0 python -c "import torch; print(torch.cuda.is_available())"
```

**Do NOT install** `mamba-ssm` or `causal-conv1d` — this forces the slow PyTorch path that exposes all SSM intermediates needed for extraction.

## Usage

```bash
# Launch Gradio app
HSA_OVERRIDE_GFX_VERSION=11.0.0 python app.py
```

The app will be available at `http://localhost:7860`.

## Running Tests

```bash
python -m pytest tests/ -v
```

## Technical Details

### Extraction Method

1. **Mamba-2 layers**: Register `forward_hook` on each Mamba layer to re-run the projection and capture dt, B, C tensors after conv1d processing. Compute hidden attention using cumulative decay (exp of cumsum), C·B dot products, and dt scaling. Uses float32 for numerical stability with clamped exponents.

2. **Transformer layers**: Hook Q and K projections, manually compute `softmax(QK^T / √d_k)` with causal mask and GQA head expansion (the model's `output_attentions` returns None for this hybrid architecture).

3. **Logit Lens**: Hook each layer's output to capture the residual stream. For the last token position, project the hidden state through `lm_head` (in float32 for numerical stability), apply softmax, and extract top-10 predicted tokens. Displayed as a heatmap with layers on the Y-axis and ranked predictions on the X-axis.

4. **Neuron Activation**: Hook `shared_mlp` on all 32 layers (both Mamba and Transformer blocks have an MLP). The Granite MLP uses SwiGLU: `input_linear` projects to 2× intermediate size (4096), the result is chunked and gated (`silu(chunk[0]) * chunk[1]`), producing 2048-dimensional intermediate activations. We capture these intermediates and rank neurons by absolute magnitude, since SwiGLU outputs both positive and negative values. Displayed as a heatmap (tokens × top-50 neurons) with per-token top-10 neuron listings. Shared neuron indices across semantically related tokens suggest interpretable features.

5. **Activation Diff**: Run full extraction on two prompts, then compare across four dimensions. Residual stream cosine similarity measures how layer representations diverge. Attention pattern divergence uses Jensen-Shannon divergence (symmetric, bounded KL-based metric) on row-normalized attention distributions. Top changed neurons are ranked by mean absolute activation delta across all tokens. Logit lens diff compares the top-1 predicted token at each layer between the two prompts.

6. **Multi-Step Generation**: Autoregressive loop that generates tokens one at a time using greedy decoding from the model's output logits. At each step, the full extraction pipeline runs on the growing sequence — capturing attention patterns, neuron activations, residual stream, and logit lens data. All step results are cached in memory so users can browse through generation steps without re-computation. The per-step dashboard shows a token timeline, compact logit lens (last 8 layers), and representative Mamba/Transformer attention heatmaps.

7. **Causal Intervention**: Activation patching engine that caches residual stream activations from a clean run, then replays the corrupted prompt with hooks that swap in clean activations at a target layer. Measures fractional recovery: `(patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)`. Supports four intervention types (activation patch, zero ablation, mean ablation, noise). Layer sweep mode runs 32 forward passes; full position-by-layer sweep produces a 2D heatmap at the cost of `num_layers × seq_len` passes. Note: Mamba layers have internal recurrent state (conv/SSM) that is not patched — recovery scores for Mamba layers should be interpreted with this caveat (see `docs/session-notes-causal-intervention.md`).

### Hardware

Developed on AMD Radeon 8060S (Strix Halo, gfx1151) with ROCm 7.1.1 and 48GB unified memory. Uses `HSA_OVERRIDE_GFX_VERSION=11.0.0` for compatibility.

## Model

- [ibm-granite/granite-4.0-h-350m](https://huggingface.co/ibm-granite/granite-4.0-h-350m) (Apache 2.0)
- 350M parameters, hybrid Mamba-2/Transformer, 9:1 ratio
