# Activation Steering Experiments on Granite 4.0 Mamba-2

This document records our attempts to steer IBM Granite 4.0 (hybrid Mamba-2/Transformer, 350M params) toward generating bee-related content, inspired by Anthropic's "Golden Gate Claude" work. We tried many approaches. Most failed. The one that worked tells us something interesting about where concepts live in hybrid architectures.

## Background

The idea: train a Sparse Autoencoder (SAE) on a model's residual stream, find a monosemantic feature that encodes "bee," then add that feature's direction vector during generation to make the model talk about bees constantly. This worked famously on Claude (the "Golden Gate Bridge" demo). We wanted to see if it works on a hybrid Mamba-2/Transformer model.

## Finding the Bee Feature

**Result: Success.** We found highly selective bee features across multiple layers and SAE configurations.

| Experiment | Layer | Best Feature | Selectivity | Concept Activation | Baseline |
|---|---|---|---|---|---|
| `run_ibeem_pipeline.py` | 16 (Mamba) | #3633 | **34,774x** | 0.0348 | 0.0000 |
| `run_ibeem_pipeline.py` | 16 (Mamba) | #3528 | 10,097x | 0.0101 | 0.0000 |
| `run_ibeem_pipeline.py` | 16 (Mamba) | #4310 | 8,991x | 0.0090 | 0.0000 |
| `run_sae_feature_search.py` | 4 (Mamba) | #2490 | 30x | 0.0525 | 0.0017 |
| `run_golden_gate_bee.py` | 17 (Attn) | #433 | 6,165x | 0.0062 | 0.0000 |

Feature #3633 at Layer 16 activates **34,774 times more strongly** on bee text than on generic text. By any standard, this is a monosemantic "bee neuron." The SAE training itself was fast (~10 seconds for 391 steps, 768->6144 latent dimensions, 50K activations, L1 coefficient 1e-3) with good reconstruction loss (0.093) and only 1 dead feature out of 6,144.

## Steering with SAE Features

**Result: Complete failure.** Despite finding extraordinarily selective features, using them to steer generation produced zero bee-related output across every configuration tested.

### SAE Feature Injection (run_ibeem_pipeline.py)

Steered with top 3 features (#3633, #3528, #4310) from Layer 16 SAE. Grid search over 4 coefficients (1.0, 2.0, 3.0, 5.0) x 2 injection modes (all-positions, last-token) = 24 configs, 192 total generations over ~48 hours of compute.

**Best result:** Feature 3633, coeff=2.0, all-positions -- score **-12**, bee keywords **0**

Every single configuration produced zero bee keywords. Negative scores indicate the steering caused repetitive/degenerate output.

### Golden Gate at Attention Layers (run_golden_gate_bee.py)

Trained SAE on Layer 17 (Transformer attention layer). Found Feature #433 with 6,165x selectivity -- confirmed as semantically bee-related via token-level validation (9/20 top-activating tokens were bee-semantic). Steered with both vector addition and SAE clamping methods.

**Result:** 0 bee keywords across all configurations.

### Multi-Layer Simultaneous Steering (run_multilayer_golden_gate_bee.py)

Hypothesis: maybe steering at a single layer is too weak. Steered all 4 Transformer attention layers (10, 13, 17, 27) simultaneously with 10 different coefficient configurations.

**Result:** 0 bee keywords.

### Norm-Scaled Steering (run_normscaled_steering.py)

Hypothesis: maybe the coefficients were too small relative to the residual stream. Measured actual residual stream norms at each attention layer (the first multi-layer experiment's coeff=15 was only 11.8% of Layer 10's stream norm). Scaled coefficients to 25%, 50%, 75%, 100%, and 150% of measured norms.

**Result:** 0 bee keywords across all 7 configurations. Even at 150% of the residual stream norm, SAE-derived directions don't produce topical steering.

### Mamba-Internal Component Steering

Tried steering through Mamba-specific parameters rather than the residual stream:

| Approach | Script | Dimensions | Result |
|---|---|---|---|
| dt (timestep scaling) | `run_dt_steering.py` | 48 | 0 bee keywords |
| conv1d output | `run_conv1d_steering.py` | 1,792 | 0 bee keywords |
| Gate (z parameter) | `run_gate_steering.py` | 1,536 | 0 bee keywords |
| SSM state injection | `run_ssm_state_injection.py` | full state | ~2 bee keywords (marginal) |

Mamba's internal components (dt, conv1d, gate) don't encode topical/semantic information in a way that's amenable to linear steering.

## What Actually Worked: Contrastive Mean-Difference Vectors

**Result: Dramatic success.** Simple mean-difference vectors from minimal pairs, bypassing the SAE entirely.

### Method (run_contrastive_steering.py)

1. **Phase 1 -- Find divergent layers:** Run 10 minimal pairs ("The bee buzzed loudly" vs "The car buzzed loudly") through the model. Measure cosine similarity of residual streams at each layer. Early Mamba layers (0, 1, 4) showed the most divergence (~0.82 cosine sim vs ~0.94 at attention layers).

2. **Phase 2 -- Compute steering vectors:** Collect residual streams from 25 bee prompts and 25 matched control prompts at the top 3 layers. Compute `mean(bee_residuals) - mean(control_residuals)`, normalize to unit vector.

3. **Phase 3 -- Grid search:** 3 layers x 5 coefficients (1, 2, 5, 10, 20) x 2 modes = 30 configs, 240 generations.

### Results

| Rank | Layer | Coefficient | Mode | Bee Keywords | Score | Repetitive? |
|---|---|---|---|---|---|---|
| #1 | **1 (Mamba)** | **20.0** | all-pos | **602** | **1142** | 7/8 prompts |
| #2 | 1 (Mamba) | 10.0 | all-pos | 400 | 748 | 6/8 |
| #3 | 0 (Mamba) | 10.0 | all-pos | 245 | 468 | 3/8 |
| #4 | 4 (Mamba) | 10.0 | all-pos | 9 | 16 | 1/8 |
| #5 | 4 (Mamba) | 5.0 | all-pos | 22 | 12 | 4/8 |

Layer 1 with coefficient 20.0 produced **602 bee keywords** across 8 test prompts. The model became obsessed with bees -- exactly the "Golden Gate" effect we were looking for.

Key observations:
- **Only early Mamba layers work** (0, 1). By Layer 4, effectiveness drops to near zero. Attention layers produce nothing.
- **All-positions mode is essential.** Last-token-only injection was consistently weaker.
- **High coefficients needed.** Coeff=1-2 had negligible effect; 10-20 was the sweet spot.
- **Repetition is a side effect.** 7/8 prompts at the best config showed some repetitive patterns, a known artifact of strong steering.

## Key Finding: Feature Selectivity != Causal Influence

The central result of these experiments:

**A feature can be extraordinarily selective for a concept (34,774x selectivity) without being a causal lever for producing that concept.**

The SAE finds features that *detect* "beeness" in the residual stream -- they fire when bee content is present. But manipulating these features doesn't *cause* the model to generate bee content. The detection direction and the generation direction are different, or the SAE's learned basis doesn't align with the model's actual causal structure.

Meanwhile, a crude mean-difference vector at Layer 1 (no SAE, no learned features, just "what's different in the residual stream when we say 'bee' vs 'car'?") works spectacularly. This suggests:

1. **Concept encoding is early and distributed** in hybrid architectures. The "bee" concept is most manipulable at Layers 0-1 (the very first Mamba layers), before the recurrent state has had time to diffuse the signal.

2. **SAE features may capture correlational, not causal, structure.** The SAE learns to reconstruct activations, and its features reflect statistical patterns in the training data. A feature that correlates with bee content during inference may not sit on the causal pathway from representation to generation.

3. **Hybrid architectures may resist single-point interventions.** Mamba layers have recurrent state (conv buffer + SSM state) that isn't captured or manipulated by residual stream steering. The information might flow through these internal channels, making residual-stream-only interventions less effective at later layers.

4. **Layer position matters more than layer type.** The successful steering happened at Mamba layers (0, 1), not because they're Mamba, but because they're *early*. Attention layers failed not because they're attention, but because they're positioned at layers 10, 13, 17, 27 -- too late in the network.

## Experiment Index

| Script | What It Does | Key Result |
|---|---|---|
| `run_ibeem_pipeline.py` | Full SAE pipeline: collect activations, train SAE, search features, steer | Found 34,774x selective feature, 0 bee keywords in steering |
| `run_sae_feature_search.py` | SAE on Layer 4, search for bee features | Found features, steering ineffective |
| `run_golden_gate_bee.py` | Golden Gate approach on Layer 17 (attention) | 6,165x selective feature, 0 bee keywords |
| `run_multilayer_golden_gate_bee.py` | Simultaneous steering at all 4 attention layers | 0 bee keywords |
| `run_normscaled_steering.py` | Norm-proportional coefficients at attention layers | 0 bee keywords even at 150% of stream norm |
| `run_mamba_sae_steering.py` | SAE steering on Mamba layers + alignment analysis | 0 bee keywords, motivated contrastive approach |
| `run_contrastive_steering.py` | Mean-difference vectors from minimal pairs | **602 bee keywords at Layer 1, coeff=20** |
| `run_dt_steering.py` | Steer via Mamba dt (timestep) parameter | 0 bee keywords |
| `run_conv1d_steering.py` | Steer via Mamba conv1d output | 0 bee keywords |
| `run_gate_steering.py` | Steer via Mamba gate (z) parameter | 0 bee keywords |
| `run_ssm_state_injection.py` | Inject contrastive vector into SSM recurrent state | ~2 bee keywords (marginal) |
| `run_bee_circuit.py` | Circuit tracing of bee steering vector propagation | Analysis of how steering signal flows through layers |
| `run_feature_analysis.py` | Feature-level analysis across layers | Diagnostic tool |
| `run_refine_layer4.py` | Quick Layer 4 steering refinement | Minimal test |

## Reproducing

All scripts are self-contained. They load the model, run the experiment, and save results to `results/`.

```bash
# The experiment that worked
HSA_OVERRIDE_GFX_VERSION=11.0.0 python run_contrastive_steering.py

# The experiment that found the monosemantic feature (but couldn't steer with it)
HSA_OVERRIDE_GFX_VERSION=11.0.0 python run_ibeem_pipeline.py
```

Results are saved to `results/` (gitignored due to size -- contains model activations and SAE checkpoints).

## Hardware

All experiments ran on AMD Radeon 8060S (Strix Halo, gfx1151) with ROCm 7.1.1, though most ran on CPU due to intermittent GPU hangs with this architecture. The ibeem pipeline (192 generations) took ~48 hours; contrastive steering (240 generations) took ~6 hours.
