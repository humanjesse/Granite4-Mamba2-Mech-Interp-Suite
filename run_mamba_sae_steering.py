#!/usr/bin/env python3
"""I-Bee-M: Mamba-Layer SAE vs Contrastive Steering Experiment.

Key question: SAE features failed to produce bee keywords at ALL layers tested
(attention L17/L27, Mamba L4/L16), yet contrastive mean-difference vectors
succeed spectacularly at early Mamba layers (L0: 245, L1: 400-602, L4: 9
bee keywords). This experiment:

1. Trains SAEs on early Mamba layers (L0, L1, L4, L9) where contrastive works
2. Computes contrastive vectors at the same layers
3. Measures cosine alignment between SAE features and contrastive vectors
4. Runs steering with both methods at same layers/coefficients
5. Produces a definitive comparison explaining WHY SAE steering fails

Architecture: Granite 4.0 hybrid — 28 Mamba-2 + 4 attention at [10, 13, 17, 27]
  L0, L1, L4, L9 are all Mamba layers (before first attention at L10)

Usage:
    source .venv/bin/activate
    python run_mamba_sae_steering.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import torch

from src.model_loader import load_model
from src.sae.trainer import SAETrainingConfig, collect_activations, train_sae
from src.sae.feature_search import (
    search_features, get_default_bee_texts, get_default_baseline_texts,
)
from src.sae.sparse_autoencoder import SparseAutoencoder
from src.extraction.activation_steering import generate_comparison
from run_contrastive_steering import (
    score_generation, TEST_PROMPTS, MAX_TOKENS, BEE_KEYWORDS,
    MINIMAL_PAIRS, EXTRA_BEE_PROMPTS, EXTRA_CONTROL_PROMPTS,
)

# ── Config ───────────────────────────────────────────────────────────────────

MAMBA_LAYERS = [0, 1, 4, 9]
STEERING_COEFFICIENTS = [5.0, 10.0, 20.0]
OUTPUT_BASE = Path("results/mamba_sae_steering")

# Known SAEs from prior runs (skip retraining)
KNOWN_SAES = {
    0: {
        "sae_path": "results/mamba_sae_steering/20260220_131932/sae_layer0.pt",
        "feature_idx": 2284,
        "selectivity": 10.55,
    },
    1: {
        "sae_path": "results/mamba_sae_steering/20260220_131932/sae_layer1.pt",
        "feature_idx": 513,
        "selectivity": 9.45,
    },
    4: {
        "sae_path": "results/sae/20260218_084436/sae_layer4.pt",
        "feature_idx": 2490,
        "selectivity": 30.19,
    },
    9: {
        "sae_path": "results/mamba_sae_steering/20260220_131932/sae_layer9.pt",
        "feature_idx": 3338,
        "selectivity": 2501.42,
    },
}

# SAE training config
SAE_CONFIG = SAETrainingConfig(
    layer_idx=0,  # overridden per layer
    expansion_factor=8,
    l1_coefficient=1e-3,
    learning_rate=3e-4,
    num_activations=50_000,
    batch_size=256,
    num_epochs=3,
)


# ── Utilities ────────────────────────────────────────────────────────────────

def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ── Phase 1: Load model ─────────────────────────────────────────────────────

def phase1_load_model():
    log("=" * 60)
    log("PHASE 1: Loading model")
    log("=" * 60)
    model, tokenizer, device = load_model(device="cuda")
    log(f"Model loaded on {device}. Hidden size: {model.config.hidden_size}")
    log(f"Target Mamba layers: {MAMBA_LAYERS}")
    for idx in MAMBA_LAYERS:
        log(f"  Layer {idx}: {model.config.layer_types[idx]}")
    return model, tokenizer, device


# ── Phase 2: Train/Load SAEs for Mamba layers ───────────────────────────────

def phase2_prepare_saes(model, tokenizer, device, output_dir):
    log("")
    log("=" * 60)
    log("PHASE 2: Training SAEs on early Mamba layers")
    log("=" * 60)

    layer_data = {}  # layer_idx -> {sae, feature_idx, selectivity, steering_vector, top_features}
    hidden_size = model.config.hidden_size

    for layer_idx in MAMBA_LAYERS:
        log(f"\n  --- Layer {layer_idx} (Mamba) ---")

        if layer_idx in KNOWN_SAES:
            info = KNOWN_SAES[layer_idx]
            log(f"  Loading existing SAE from {info['sae_path']}")
            sae = SparseAutoencoder(hidden_size, hidden_size * SAE_CONFIG.expansion_factor)
            sae.load_state_dict(torch.load(info["sae_path"], map_location="cpu", weights_only=True))
            sae = sae.to(device).eval()

            feature_idx = info["feature_idx"]
            selectivity = info["selectivity"]
            steering_vec = sae.get_feature_direction(feature_idx)

            log(f"  Known best feature: #{feature_idx} (selectivity={selectivity:.0f}x)")
            log(f"  Steering vector norm: {steering_vec.norm():.4f}")

            # Also do feature search to get top-5 for alignment analysis
            bee_texts = get_default_bee_texts()
            baseline_texts = get_default_baseline_texts()
            log(f"  Running feature search for alignment analysis...")
            search_result = search_features(
                sae, model, tokenizer,
                layer_idx=layer_idx,
                concept_texts=bee_texts,
                baseline_texts=baseline_texts,
                device=device,
                top_k=20,
                concept_name="bee",
            )

            layer_data[layer_idx] = {
                "sae": sae,
                "feature_idx": feature_idx,
                "selectivity": selectivity,
                "steering_vector": steering_vec,
                "source": "loaded",
                "top_features": search_result.top_features,
            }

        else:
            # Train new SAE
            log(f"  Training new SAE for Mamba layer {layer_idx}...")

            def act_progress(collected, total):
                if collected % 10000 == 0 or collected == total:
                    log(f"    Collected {collected}/{total} activations")

            t0 = time.time()
            activations = collect_activations(
                model, tokenizer,
                layer_idx=layer_idx,
                num_activations=SAE_CONFIG.num_activations,
                device=device,
                progress_callback=act_progress,
            )
            log(f"    Activations: {activations.shape} in {time.time() - t0:.1f}s")

            def train_progress(step, total):
                if step % 100 == 0 or step == total:
                    log(f"    Training step {step}/{total}")

            t0 = time.time()
            config = SAETrainingConfig(
                layer_idx=layer_idx,
                expansion_factor=SAE_CONFIG.expansion_factor,
                l1_coefficient=SAE_CONFIG.l1_coefficient,
                learning_rate=SAE_CONFIG.learning_rate,
                num_activations=SAE_CONFIG.num_activations,
                batch_size=SAE_CONFIG.batch_size,
                num_epochs=SAE_CONFIG.num_epochs,
            )
            sae_result = train_sae(activations, config, progress_callback=train_progress, device=device)
            train_time = time.time() - t0

            log(f"    SAE trained in {train_time:.1f}s")
            log(f"    Final recon loss: {sae_result.final_reconstruction_loss:.6f}")
            log(f"    Dead features: {sae_result.dead_features}/{sae_result.num_latent_features}")

            # Save SAE
            torch.save(sae_result.sae.state_dict(), output_dir / f"sae_layer{layer_idx}.pt")

            # Feature search
            sae = sae_result.sae.to(device).eval()
            bee_texts = get_default_bee_texts()
            baseline_texts = get_default_baseline_texts()

            log(f"    Searching for bee features...")
            t0 = time.time()
            search_result = search_features(
                sae, model, tokenizer,
                layer_idx=layer_idx,
                concept_texts=bee_texts,
                baseline_texts=baseline_texts,
                device=device,
                top_k=20,
                concept_name="bee",
            )
            log(f"    Feature search complete in {time.time() - t0:.1f}s")

            best = search_result.top_features[0]
            feature_idx = best["feature_idx"]
            selectivity = best["selectivity_score"]
            steering_vec = sae.get_feature_direction(feature_idx)

            log(f"    Best feature: #{feature_idx}")
            log(f"    Selectivity: {selectivity:.2f}x")
            log(f"    Bee activation: {best['concept_activation']:.4f}")
            log(f"    Baseline activation: {best['baseline_activation']:.6f}")
            log(f"    Steering vector norm: {steering_vec.norm():.4f}")

            # Save search results
            with open(output_dir / f"feature_search_L{layer_idx}.json", "w") as f:
                json.dump({
                    "layer_idx": layer_idx,
                    "top_features": search_result.top_features,
                }, f, indent=2)

            # Save training metrics
            with open(output_dir / f"training_metrics_L{layer_idx}.json", "w") as f:
                json.dump({
                    "layer_idx": layer_idx,
                    "input_dim": sae_result.sae.input_dim,
                    "latent_dim": sae_result.sae.latent_dim,
                    "final_loss": sae_result.final_loss,
                    "final_reconstruction_loss": sae_result.final_reconstruction_loss,
                    "final_l1_loss": sae_result.final_l1_loss,
                    "mean_active_features": sae_result.mean_active_features,
                    "dead_features": sae_result.dead_features,
                    "training_time_seconds": train_time,
                }, f, indent=2)

            layer_data[layer_idx] = {
                "sae": sae,
                "feature_idx": feature_idx,
                "selectivity": selectivity,
                "steering_vector": steering_vec,
                "source": "trained",
                "top_features": search_result.top_features,
            }

    # Summary
    log("")
    log("  Layer summary:")
    for layer_idx in MAMBA_LAYERS:
        d = layer_data[layer_idx]
        log(f"    L{layer_idx}: feature #{d['feature_idx']} "
            f"(selectivity={d['selectivity']:.1f}x, source={d['source']})")

    return layer_data


# ── Phase 3: Compute contrastive vectors ─────────────────────────────────────

def phase3_contrastive_vectors(model, tokenizer, device):
    """Compute mean-difference contrastive steering vectors at target Mamba layers."""
    from src.extraction.unified_extractor import GraniteAttentionExtractor

    log("")
    log("=" * 60)
    log("PHASE 3: Computing contrastive steering vectors")
    log("=" * 60)

    extractor = GraniteAttentionExtractor(model, tokenizer, device=device)

    # Combine minimal pair prompts with extra prompts
    bee_prompts = [p[0] for p in MINIMAL_PAIRS] + EXTRA_BEE_PROMPTS
    control_prompts = [p[1] for p in MINIMAL_PAIRS] + EXTRA_CONTROL_PROMPTS

    log(f"Bee prompts: {len(bee_prompts)}")
    log(f"Control prompts: {len(control_prompts)}")
    log(f"Target layers: {MAMBA_LAYERS}")

    # Collect residual streams for bee prompts
    log("Collecting bee prompt residual streams...")
    bee_residuals = {layer: [] for layer in MAMBA_LAYERS}
    for i, prompt in enumerate(bee_prompts):
        if (i + 1) % 5 == 0:
            log(f"  Bee prompt {i+1}/{len(bee_prompts)}")
        result = extractor.extract(prompt)
        for layer in MAMBA_LAYERS:
            rs = result["residual_stream"][layer][0].float().mean(dim=0)
            bee_residuals[layer].append(rs)

    # Collect residual streams for control prompts
    log("Collecting control prompt residual streams...")
    control_residuals = {layer: [] for layer in MAMBA_LAYERS}
    for i, prompt in enumerate(control_prompts):
        if (i + 1) % 5 == 0:
            log(f"  Control prompt {i+1}/{len(control_prompts)}")
        result = extractor.extract(prompt)
        for layer in MAMBA_LAYERS:
            rs = result["residual_stream"][layer][0].float().mean(dim=0)
            control_residuals[layer].append(rs)

    # Compute steering vectors
    contrastive_vectors = {}
    for layer in MAMBA_LAYERS:
        mean_bee = torch.stack(bee_residuals[layer]).mean(dim=0)
        mean_control = torch.stack(control_residuals[layer]).mean(dim=0)
        diff = mean_bee - mean_control
        diff_norm = diff.norm()
        steering_vec = diff / diff_norm

        contrastive_vectors[layer] = steering_vec

        cosine_sim = torch.nn.functional.cosine_similarity(
            mean_bee.unsqueeze(0), mean_control.unsqueeze(0)
        ).item()

        log(f"  Layer {layer}: diff_norm={diff_norm:.4f}, "
            f"bee_norm={mean_bee.norm():.4f}, "
            f"control_norm={mean_control.norm():.4f}, "
            f"cosine_sim={cosine_sim:.6f}")

    return contrastive_vectors


# ── Phase 4: Alignment analysis ──────────────────────────────────────────────

def phase4_alignment(layer_data, contrastive_vectors, output_dir):
    """Compute cosine similarity between SAE features and contrastive vectors."""
    log("")
    log("=" * 60)
    log("PHASE 4: SAE–Contrastive Alignment Analysis")
    log("=" * 60)

    alignment_results = {}

    for layer_idx in MAMBA_LAYERS:
        sae = layer_data[layer_idx]["sae"]
        contrastive_vec = contrastive_vectors[layer_idx]
        top_features = layer_data[layer_idx]["top_features"]

        log(f"\n  Layer {layer_idx}:")
        layer_alignment = []

        # Compute cosine similarity for top-5 SAE features
        # Move contrastive vec to CPU for comparison (SAE directions are CPU)
        contrastive_cpu = contrastive_vec.cpu().float()

        for feat in top_features[:5]:
            fidx = feat["feature_idx"]
            sae_direction = sae.get_feature_direction(fidx).cpu().float()

            cosine = torch.nn.functional.cosine_similarity(
                sae_direction.unsqueeze(0),
                contrastive_cpu.unsqueeze(0),
            ).item()

            layer_alignment.append({
                "feature_idx": fidx,
                "selectivity": feat["selectivity_score"],
                "cosine_with_contrastive": cosine,
                "concept_activation": feat["concept_activation"],
                "baseline_activation": feat["baseline_activation"],
            })

            log(f"    Feature #{fidx} (sel={feat['selectivity_score']:.1f}x): "
                f"cos(SAE, contrastive) = {cosine:+.6f}")

        # Also compute alignment of best SAE feature's steering vector
        best_sae_vec = layer_data[layer_idx]["steering_vector"].cpu().float()
        best_cosine = torch.nn.functional.cosine_similarity(
            best_sae_vec.unsqueeze(0),
            contrastive_cpu.unsqueeze(0),
        ).item()
        log(f"    Best SAE steering vec cos(SAE, contrastive) = {best_cosine:+.6f}")

        # Compute norm ratio: how big is the contrastive vector component along the SAE direction?
        projection = torch.dot(contrastive_cpu, best_sae_vec).item()
        log(f"    Projection of contrastive onto SAE direction = {projection:+.6f}")

        alignment_results[layer_idx] = {
            "feature_alignments": layer_alignment,
            "best_sae_cosine": best_cosine,
            "contrastive_projection_onto_sae": projection,
        }

    # Save alignment results
    save_data = {}
    for layer_idx, data in alignment_results.items():
        save_data[str(layer_idx)] = data
    with open(output_dir / "alignment_analysis.json", "w") as f:
        json.dump(save_data, f, indent=2)

    return alignment_results


# ── Phase 5: SAE-based steering ──────────────────────────────────────────────

def phase5_sae_steering(model, tokenizer, layer_data, device, output_dir):
    """Test SAE feature directions for steering at each Mamba layer."""
    log("")
    log("=" * 60)
    log("PHASE 5: SAE-based steering at Mamba layers")
    log("=" * 60)

    total_gens = len(MAMBA_LAYERS) * len(STEERING_COEFFICIENTS) * len(TEST_PROMPTS)
    log(f"  {len(MAMBA_LAYERS)} layers × {len(STEERING_COEFFICIENTS)} coefficients "
        f"× {len(TEST_PROMPTS)} prompts = {total_gens} generations")

    all_results = []
    gen_count = 0

    # Generate unsteered baselines once
    log("  Generating unsteered baselines...")
    unsteered_cache = {}
    for prompt in TEST_PROMPTS:
        result = generate_comparison(
            model, tokenizer, prompt,
            steering_vector=torch.zeros(model.config.hidden_size),
            layer_idx=0, coefficient=0.0,
            max_tokens=MAX_TOKENS, device=device,
        )
        unsteered_cache[prompt] = result.unsteered_text

    for layer_idx in MAMBA_LAYERS:
        vec = layer_data[layer_idx]["steering_vector"]
        feature_idx = layer_data[layer_idx]["feature_idx"]
        selectivity = layer_data[layer_idx]["selectivity"]
        log(f"\n  Layer {layer_idx} — SAE feature #{feature_idx} (sel={selectivity:.1f}x)")

        for coeff in STEERING_COEFFICIENTS:
            log(f"    coeff={coeff}")

            for prompt in TEST_PROMPTS:
                gen_count += 1
                result = generate_comparison(
                    model, tokenizer, prompt, vec,
                    layer_idx=layer_idx,
                    coefficient=coeff,
                    max_tokens=MAX_TOKENS,
                    device=device,
                    last_token_only=False,  # all-positions mode
                )

                scoring = score_generation(result.steered_text, unsteered_cache[prompt])

                all_results.append({
                    "method": "sae",
                    "layer_idx": layer_idx,
                    "coefficient": coeff,
                    "feature_idx": feature_idx,
                    "selectivity": selectivity,
                    "prompt": prompt,
                    "steered_text": result.steered_text,
                    "unsteered_text": unsteered_cache[prompt],
                    "feature_direction_norm": result.feature_direction_norm,
                    **scoring,
                })

            log(f"      {gen_count}/{total_gens} done")

    # Save results
    with open(output_dir / "sae_steering_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


# ── Phase 6: Contrastive steering ────────────────────────────────────────────

def phase6_contrastive_steering(model, tokenizer, contrastive_vectors, device, output_dir):
    """Test contrastive steering vectors at the same layers/coefficients."""
    log("")
    log("=" * 60)
    log("PHASE 6: Contrastive steering at Mamba layers")
    log("=" * 60)

    total_gens = len(MAMBA_LAYERS) * len(STEERING_COEFFICIENTS) * len(TEST_PROMPTS)
    log(f"  {len(MAMBA_LAYERS)} layers × {len(STEERING_COEFFICIENTS)} coefficients "
        f"× {len(TEST_PROMPTS)} prompts = {total_gens} generations")

    all_results = []
    gen_count = 0

    # Reuse unsteered baselines (same prompts, same model)
    log("  Generating unsteered baselines...")
    unsteered_cache = {}
    for prompt in TEST_PROMPTS:
        result = generate_comparison(
            model, tokenizer, prompt,
            steering_vector=torch.zeros(model.config.hidden_size),
            layer_idx=0, coefficient=0.0,
            max_tokens=MAX_TOKENS, device=device,
        )
        unsteered_cache[prompt] = result.unsteered_text

    for layer_idx in MAMBA_LAYERS:
        vec = contrastive_vectors[layer_idx]
        log(f"\n  Layer {layer_idx} — contrastive mean-diff vector")

        for coeff in STEERING_COEFFICIENTS:
            log(f"    coeff={coeff}")

            for prompt in TEST_PROMPTS:
                gen_count += 1
                result = generate_comparison(
                    model, tokenizer, prompt, vec,
                    layer_idx=layer_idx,
                    coefficient=coeff,
                    max_tokens=MAX_TOKENS,
                    device=device,
                    last_token_only=False,  # all-positions mode
                )

                scoring = score_generation(result.steered_text, unsteered_cache[prompt])

                all_results.append({
                    "method": "contrastive",
                    "layer_idx": layer_idx,
                    "coefficient": coeff,
                    "prompt": prompt,
                    "steered_text": result.steered_text,
                    "unsteered_text": unsteered_cache[prompt],
                    "feature_direction_norm": result.feature_direction_norm,
                    **scoring,
                })

            log(f"      {gen_count}/{total_gens} done")

    # Save results
    with open(output_dir / "contrastive_steering_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


# ── Phase 7: Report ──────────────────────────────────────────────────────────

def phase7_report(layer_data, contrastive_vectors, alignment_results,
                  sae_results, contrastive_results, output_dir):
    """Generate comprehensive comparison report."""
    log("")
    log("=" * 60)
    log("PHASE 7: Generating report")
    log("=" * 60)

    lines = []
    lines.append("=" * 70)
    lines.append("I-BEE-M: MAMBA-LAYER SAE vs CONTRASTIVE STEERING REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Architecture
    lines.append("ARCHITECTURE:")
    lines.append("  Granite 4.0 hybrid: 28 Mamba-2 + 4 Attention at [10, 13, 17, 27]")
    lines.append(f"  Target layers: {MAMBA_LAYERS} (all Mamba, before first attention at L10)")
    lines.append("")

    # Context
    lines.append("CONTEXT & MOTIVATION:")
    lines.append("  All prior SAE steering experiments produced 0 bee keywords:")
    lines.append("    - L17 attention (sel=6,164x): 0 bee kw")
    lines.append("    - L27 attention (sel=6,128x): 2 bee kw (marginal)")
    lines.append("    - L4 Mamba SAE (sel=30x): 0 bee kw")
    lines.append("    - L16 Mamba SAE (sel=34,773x): 0 bee kw")
    lines.append("  But contrastive mean-diff vectors succeed at early Mamba layers:")
    lines.append("    - L0 contrastive coeff=10: 245 bee kw")
    lines.append("    - L1 contrastive coeff=10: 400 bee kw")
    lines.append("    - L4 contrastive coeff=10: 9 bee kw (coherent!)")
    lines.append("")
    lines.append("  KEY QUESTION: Are SAE features orthogonal to the causal bee direction?")
    lines.append("")

    # ── Per-layer SAE feature summary ────────────────────────────────────
    lines.append("=" * 70)
    lines.append("SAE FEATURE SUMMARY (per layer)")
    lines.append("=" * 70)
    lines.append("")

    for layer_idx in MAMBA_LAYERS:
        d = layer_data[layer_idx]
        lines.append(f"  L{layer_idx} (Mamba): feature #{d['feature_idx']} "
                     f"(selectivity={d['selectivity']:.1f}x, source={d['source']})")
        # Show top-3 features
        for feat in d["top_features"][:3]:
            lines.append(f"    Feature #{feat['feature_idx']}: sel={feat['selectivity_score']:.1f}x, "
                         f"bee_act={feat['concept_activation']:.4f}, "
                         f"baseline_act={feat['baseline_activation']:.6f}")
    lines.append("")

    # ── Alignment analysis ───────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("SAE-CONTRASTIVE ALIGNMENT ANALYSIS")
    lines.append("  cos(SAE_feature_direction, contrastive_mean_diff_vector)")
    lines.append("  Near 0 = orthogonal (SAE feature misses causal direction)")
    lines.append("  Near 1 = aligned (SAE feature captures causal direction)")
    lines.append("=" * 70)
    lines.append("")

    for layer_idx in MAMBA_LAYERS:
        align = alignment_results[layer_idx]
        lines.append(f"  Layer {layer_idx}:")
        lines.append(f"    Best SAE feature cosine: {align['best_sae_cosine']:+.6f}")
        lines.append(f"    Contrastive projection onto SAE: {align['contrastive_projection_onto_sae']:+.6f}")
        for fa in align["feature_alignments"]:
            lines.append(f"    Feature #{fa['feature_idx']} (sel={fa['selectivity']:.1f}x): "
                         f"cos={fa['cosine_with_contrastive']:+.6f}")
        lines.append("")

    # ── Steering results comparison ──────────────────────────────────────
    lines.append("=" * 70)
    lines.append("STEERING RESULTS: SAE vs CONTRASTIVE")
    lines.append("=" * 70)
    lines.append("")

    # Aggregate by (method, layer, coeff)
    def aggregate_results(results):
        groups = {}
        for r in results:
            key = (r["method"], r["layer_idx"], r["coefficient"])
            if key not in groups:
                groups[key] = []
            groups[key].append(r)
        return groups

    sae_groups = aggregate_results(sae_results)
    contrastive_groups = aggregate_results(contrastive_results)

    # Side-by-side table
    lines.append(f"{'Layer':>5} {'Coeff':>6} │ {'SAE bee_kw':>10} {'SAE rep':>8} {'SAE score':>10} │ "
                 f"{'CTR bee_kw':>10} {'CTR rep':>8} {'CTR score':>10}")
    lines.append("─" * 85)

    total_sae_bee = 0
    total_ctr_bee = 0

    for layer_idx in MAMBA_LAYERS:
        for coeff in STEERING_COEFFICIENTS:
            sae_key = ("sae", layer_idx, coeff)
            ctr_key = ("contrastive", layer_idx, coeff)

            sae_group = sae_groups.get(sae_key, [])
            ctr_group = contrastive_groups.get(ctr_key, [])

            sae_bee = sum(r["bee_keyword_count"] for r in sae_group)
            sae_rep = sum(1 for r in sae_group if r["has_repetition"])
            sae_score = sum(r["quality_score"] for r in sae_group)
            ctr_bee = sum(r["bee_keyword_count"] for r in ctr_group)
            ctr_rep = sum(1 for r in ctr_group if r["has_repetition"])
            ctr_score = sum(r["quality_score"] for r in ctr_group)

            total_sae_bee += sae_bee
            total_ctr_bee += ctr_bee

            n = len(sae_group)
            lines.append(
                f"L{layer_idx:>3} {coeff:>6.1f} │ "
                f"{sae_bee:>10} {sae_rep:>4}/{n:<3} {sae_score:>10} │ "
                f"{ctr_bee:>10} {ctr_rep:>4}/{n:<3} {ctr_score:>10}"
            )
        lines.append("")

    lines.append(f"TOTALS:  SAE bee keywords = {total_sae_bee}    "
                 f"Contrastive bee keywords = {total_ctr_bee}")
    lines.append("")

    # ── Detailed output for best contrastive configs ─────────────────────
    lines.append("=" * 70)
    lines.append("DETAILED OUTPUT: BEST CONTRASTIVE RESULTS")
    lines.append("=" * 70)
    lines.append("")

    # Sort contrastive results by bee_keyword_count descending
    ctr_sorted = sorted(contrastive_results, key=lambda r: r["bee_keyword_count"], reverse=True)
    for r in ctr_sorted[:10]:
        lines.append(f"  L{r['layer_idx']} coeff={r['coefficient']} | bee_kw={r['bee_keyword_count']} "
                     f"rep={r['has_repetition']}")
        lines.append(f"  Prompt: \"{r['prompt']}\"")
        lines.append(f"  Steered:   {r['steered_text'][:300]}")
        lines.append(f"  Unsteered: {r['unsteered_text'][:300]}")
        lines.append("")

    # ── Detailed output for best SAE results ─────────────────────────────
    lines.append("=" * 70)
    lines.append("DETAILED OUTPUT: BEST SAE RESULTS")
    lines.append("=" * 70)
    lines.append("")

    sae_sorted = sorted(sae_results, key=lambda r: r["bee_keyword_count"], reverse=True)
    for r in sae_sorted[:10]:
        lines.append(f"  L{r['layer_idx']} coeff={r['coefficient']} feature=#{r['feature_idx']} "
                     f"(sel={r['selectivity']:.1f}x) | bee_kw={r['bee_keyword_count']} "
                     f"rep={r['has_repetition']}")
        lines.append(f"  Prompt: \"{r['prompt']}\"")
        lines.append(f"  Steered:   {r['steered_text'][:300]}")
        lines.append(f"  Unsteered: {r['unsteered_text'][:300]}")
        lines.append("")

    # ── Conclusion ───────────────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("CONCLUSION")
    lines.append("=" * 70)
    lines.append("")

    # Compute average alignment
    avg_cosine = sum(
        abs(alignment_results[l]["best_sae_cosine"]) for l in MAMBA_LAYERS
    ) / len(MAMBA_LAYERS)

    lines.append(f"Average |cos(SAE, contrastive)|: {avg_cosine:.6f}")
    lines.append(f"Total SAE bee keywords: {total_sae_bee}")
    lines.append(f"Total contrastive bee keywords: {total_ctr_bee}")
    lines.append("")

    if avg_cosine < 0.1:
        lines.append("FINDING: SAE features are ORTHOGONAL to the contrastive bee direction.")
        lines.append("The SAE learns statistically prominent features from the training corpus,")
        lines.append("but these features do NOT align with the causal direction that actually")
        lines.append("steers generation toward bee content. The contrastive mean-diff vector")
        lines.append("captures a direction in residual stream space that SAE decomposition misses.")
        lines.append("")
        lines.append("This is a METHOD-LEVEL failure: SAE feature directions are fundamentally")
        lines.append("different from the causal steering directions identified by contrastive")
        lines.append("analysis. High selectivity scores are misleading — they measure")
        lines.append("statistical association, not causal influence on generation.")
    elif avg_cosine < 0.3:
        lines.append("FINDING: SAE features have WEAK alignment with contrastive direction.")
        lines.append("There is some overlap, but the SAE feature directions capture only a small")
        lines.append("component of the full causal steering direction. The remaining orthogonal")
        lines.append("components in the contrastive vector are what drive bee keyword generation.")
    else:
        lines.append("FINDING: SAE features show MODERATE-TO-STRONG alignment with contrastive")
        lines.append("direction. The failure may be due to coefficient scaling or the specific")
        lines.append("feature selected, rather than fundamental directional mismatch.")

    lines.append("")

    if total_ctr_bee > 0 and total_sae_bee == 0:
        lines.append("DEFINITIVE: The steering failure is METHOD-LEVEL (SAE), not LAYER-LEVEL.")
        lines.append("Contrastive vectors succeed at the exact same layers where SAE features fail,")
        lines.append("proving that early Mamba layers CAN be steered — just not with SAE directions.")
    elif total_ctr_bee > total_sae_bee * 5:
        lines.append("Contrastive steering dramatically outperforms SAE steering at Mamba layers.")
        lines.append("This strongly suggests a method-level mismatch between SAE features and")
        lines.append("the causal bee direction.")
    elif total_sae_bee > 0:
        lines.append("SAE steering produced some bee keywords at Mamba layers — this is new!")
        lines.append("Early Mamba layers may be more amenable to SAE steering than attention layers.")
    lines.append("")

    report_text = "\n".join(lines)
    with open(output_dir / "report.txt", "w") as f:
        f.write(report_text)

    log(f"Report saved to {output_dir / 'report.txt'}")

    return {
        "total_sae_bee": total_sae_bee,
        "total_ctr_bee": total_ctr_bee,
        "avg_alignment_cosine": avg_cosine,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")
    log(f"Mamba layers: {MAMBA_LAYERS}")
    log(f"Steering coefficients: {STEERING_COEFFICIENTS}")
    log(f"Test prompts: {len(TEST_PROMPTS)}")

    # Save config
    config = {
        "mamba_layers": MAMBA_LAYERS,
        "steering_coefficients": STEERING_COEFFICIENTS,
        "test_prompts": TEST_PROMPTS,
        "max_tokens": MAX_TOKENS,
        "sae_expansion_factor": SAE_CONFIG.expansion_factor,
        "sae_num_activations": SAE_CONFIG.num_activations,
        "sae_num_epochs": SAE_CONFIG.num_epochs,
        "known_saes": {str(k): v for k, v in KNOWN_SAES.items()},
        "bee_keywords": BEE_KEYWORDS,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Phase 1: Load model
    model, tokenizer, device = phase1_load_model()

    # Phase 2: Train/load SAEs for Mamba layers
    layer_data = phase2_prepare_saes(model, tokenizer, device, output_dir)

    # Save layer summary
    layer_summary = {}
    for layer_idx in MAMBA_LAYERS:
        d = layer_data[layer_idx]
        layer_summary[str(layer_idx)] = {
            "feature_idx": d["feature_idx"],
            "selectivity": d["selectivity"],
            "source": d["source"],
            "steering_vector_norm": d["steering_vector"].norm().item(),
        }
    with open(output_dir / "layer_summary.json", "w") as f:
        json.dump(layer_summary, f, indent=2)

    # Phase 3: Compute contrastive vectors
    contrastive_vectors = phase3_contrastive_vectors(model, tokenizer, device)

    # Save contrastive vectors
    for layer_idx, vec in contrastive_vectors.items():
        torch.save(vec, output_dir / f"contrastive_vector_L{layer_idx}.pt")

    # Phase 4: Alignment analysis
    alignment_results = phase4_alignment(layer_data, contrastive_vectors, output_dir)

    # Phase 5: SAE steering
    sae_results = phase5_sae_steering(model, tokenizer, layer_data, device, output_dir)

    # Phase 6: Contrastive steering
    contrastive_results = phase6_contrastive_steering(
        model, tokenizer, contrastive_vectors, device, output_dir,
    )

    # Phase 7: Report
    report_stats = phase7_report(
        layer_data, contrastive_vectors, alignment_results,
        sae_results, contrastive_results, output_dir,
    )

    # ── Final summary ────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    summary = {
        "run_id": run_id,
        "elapsed_seconds": elapsed,
        "elapsed_human": f"{minutes}m {seconds}s",
        "approach": "mamba_sae_vs_contrastive",
        "mamba_layers": MAMBA_LAYERS,
        "coefficients_tested": STEERING_COEFFICIENTS,
        "total_sae_generations": len(sae_results),
        "total_contrastive_generations": len(contrastive_results),
        "total_sae_bee_keywords": report_stats["total_sae_bee"],
        "total_contrastive_bee_keywords": report_stats["total_ctr_bee"],
        "avg_alignment_cosine": report_stats["avg_alignment_cosine"],
        "per_layer": layer_summary,
        "alignment": {
            str(l): {
                "best_sae_cosine": alignment_results[l]["best_sae_cosine"],
                "projection": alignment_results[l]["contrastive_projection_onto_sae"],
            }
            for l in MAMBA_LAYERS
        },
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log("")
    log("=" * 60)
    log("MAMBA SAE vs CONTRASTIVE STEERING COMPLETE!")
    log(f"  Time: {minutes}m {seconds}s")
    log(f"  SAE steering: {len(sae_results)} generations, "
        f"{report_stats['total_sae_bee']} bee keywords")
    log(f"  Contrastive steering: {len(contrastive_results)} generations, "
        f"{report_stats['total_ctr_bee']} bee keywords")
    log(f"  Avg |cos(SAE, contrastive)|: {report_stats['avg_alignment_cosine']:.6f}")
    for l in MAMBA_LAYERS:
        log(f"    L{l}: cos={alignment_results[l]['best_sae_cosine']:+.6f}")
    log(f"  Report: {output_dir / 'report.txt'}")
    log("=" * 60)


if __name__ == "__main__":
    main()
