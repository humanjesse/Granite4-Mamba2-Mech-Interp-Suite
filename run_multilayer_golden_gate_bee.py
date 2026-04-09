#!/usr/bin/env python3
"""I-Bee-M: Multi-Layer Golden Gate Bee — All 4 Attention Layers.

Hypothesis: The model's 28 Mamba layers act as "resistors" that smooth out
single-layer perturbations. Steering at ALL 4 attention layers simultaneously
should overwhelm this resistance — every attention checkpoint reinforces the
bee direction, and perturbations at earlier layers compound through later
attention layers.

Architecture: Granite 4.0 hybrid Mamba-2/Transformer
  - 28 Mamba layers + 4 attention layers at indices [10, 13, 17, 27]

Prior results (single-layer):
  - L17 attention: selectivity 6,164x, 0 bee keywords
  - L27 attention: selectivity 6,128x, 2 bee keywords (marginal)

This script steers at ALL 4 attention layers simultaneously with 10 different
coefficient configurations to test the multi-layer hypothesis.

Usage:
    source .venv/bin/activate
    python run_multilayer_golden_gate_bee.py
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
from src.extraction.activation_steering import generate_multi_layer_comparison
from run_contrastive_steering import score_generation, TEST_PROMPTS, MAX_TOKENS

# ── Config ───────────────────────────────────────────────────────────────────

ATTENTION_LAYERS = [10, 13, 17, 27]
OUTPUT_BASE = Path("results/multilayer_golden_gate_bee")

# Known SAE results from prior single-layer runs
KNOWN_SAES = {
    17: {
        "sae_path": "results/golden_gate_bee/20260219_134642/sae_layer17.pt",
        "feature_idx": 433,
        "selectivity": 6164.0,
    },
    27: {
        "sae_path": "results/golden_gate_bee/20260219_171531/sae_layer17.pt",
        "feature_idx": 5074,
        "selectivity": 6128.0,
    },
}

# SAE training config for new layers (L10, L13)
NEW_SAE_CONFIG = SAETrainingConfig(
    layer_idx=0,  # overridden per layer
    expansion_factor=8,
    l1_coefficient=1e-3,
    learning_rate=3e-4,
    num_activations=50_000,
    batch_size=256,
    num_epochs=3,
)

# Coefficient test matrix: 10 configurations
COEFF_CONFIGS = {
    "uniform_3.0":     {10: 3,  13: 3,  17: 3,  27: 3},
    "uniform_5.0":     {10: 5,  13: 5,  17: 5,  27: 5},
    "uniform_7.0":     {10: 7,  13: 7,  17: 7,  27: 7},
    "uniform_10.0":    {10: 10, 13: 10, 17: 10, 27: 10},
    "uniform_15.0":    {10: 15, 13: 15, 17: 15, 27: 15},
    "graduated_asc":   {10: 2,  13: 3,  17: 5,  27: 10},
    "graduated_desc":  {10: 10, 13: 5,  17: 3,  27: 2},
    "deep_only":       {10: 0,  13: 0,  17: 5,  27: 10},
    "shallow_only":    {10: 10, 13: 5,  17: 0,  27: 0},
    "deepest_boost":   {10: 3,  13: 3,  17: 3,  27: 15},
}


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
    log(f"Attention layers: {ATTENTION_LAYERS}")
    for idx in ATTENTION_LAYERS:
        log(f"  Layer {idx}: {model.config.layer_types[idx]}")
    return model, tokenizer, device


# ── Phase 2: Train/Load SAEs for all 4 attention layers ─────────────────────

def phase2_prepare_saes(model, tokenizer, device, output_dir):
    log("")
    log("=" * 60)
    log("PHASE 2: Preparing SAEs for all 4 attention layers")
    log("=" * 60)

    layer_data = {}  # layer_idx -> {sae, feature_idx, selectivity, steering_vector}
    hidden_size = model.config.hidden_size

    for layer_idx in ATTENTION_LAYERS:
        log(f"\n  --- Layer {layer_idx} ---")

        if layer_idx in KNOWN_SAES:
            # Load existing SAE and known best feature
            info = KNOWN_SAES[layer_idx]
            log(f"  Loading existing SAE from {info['sae_path']}")
            sae = SparseAutoencoder(hidden_size, hidden_size * NEW_SAE_CONFIG.expansion_factor)
            sae.load_state_dict(torch.load(info["sae_path"], map_location="cpu", weights_only=True))
            sae = sae.to(device).eval()

            feature_idx = info["feature_idx"]
            selectivity = info["selectivity"]
            steering_vec = sae.get_feature_direction(feature_idx)

            log(f"  Known best feature: #{feature_idx} (selectivity={selectivity:.0f}x)")
            log(f"  Steering vector norm: {steering_vec.norm():.4f}")

            layer_data[layer_idx] = {
                "sae": sae,
                "feature_idx": feature_idx,
                "selectivity": selectivity,
                "steering_vector": steering_vec,
                "source": "loaded",
            }

        else:
            # Train new SAE and search for bee feature
            log(f"  Training new SAE for layer {layer_idx}...")

            # Collect activations
            def act_progress(collected, total):
                if collected % 10000 == 0 or collected == total:
                    log(f"    Collected {collected}/{total} activations")

            t0 = time.time()
            activations = collect_activations(
                model, tokenizer,
                layer_idx=layer_idx,
                num_activations=NEW_SAE_CONFIG.num_activations,
                device=device,
                progress_callback=act_progress,
            )
            log(f"    Activations: {activations.shape} in {time.time() - t0:.1f}s")

            # Train SAE
            def train_progress(step, total):
                if step % 100 == 0 or step == total:
                    log(f"    Training step {step}/{total}")

            t0 = time.time()
            config = SAETrainingConfig(
                layer_idx=layer_idx,
                expansion_factor=NEW_SAE_CONFIG.expansion_factor,
                l1_coefficient=NEW_SAE_CONFIG.l1_coefficient,
                learning_rate=NEW_SAE_CONFIG.learning_rate,
                num_activations=NEW_SAE_CONFIG.num_activations,
                batch_size=NEW_SAE_CONFIG.batch_size,
                num_epochs=NEW_SAE_CONFIG.num_epochs,
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
            feature_results = {
                "layer_idx": layer_idx,
                "top_features": search_result.top_features,
            }
            with open(output_dir / f"feature_search_L{layer_idx}.json", "w") as f:
                json.dump(feature_results, f, indent=2)

            # Save training metrics
            training_metrics = {
                "layer_idx": layer_idx,
                "input_dim": sae_result.sae.input_dim,
                "latent_dim": sae_result.sae.latent_dim,
                "final_loss": sae_result.final_loss,
                "final_reconstruction_loss": sae_result.final_reconstruction_loss,
                "final_l1_loss": sae_result.final_l1_loss,
                "mean_active_features": sae_result.mean_active_features,
                "dead_features": sae_result.dead_features,
                "training_time_seconds": train_time,
            }
            with open(output_dir / f"training_metrics_L{layer_idx}.json", "w") as f:
                json.dump(training_metrics, f, indent=2)

            layer_data[layer_idx] = {
                "sae": sae,
                "feature_idx": feature_idx,
                "selectivity": selectivity,
                "steering_vector": steering_vec,
                "source": "trained",
                "sae_result": sae_result,
                "search_result": search_result,
            }

    # Summary
    log("")
    log("  Layer summary:")
    for layer_idx in ATTENTION_LAYERS:
        d = layer_data[layer_idx]
        log(f"    L{layer_idx}: feature #{d['feature_idx']} "
            f"(selectivity={d['selectivity']:.0f}x, "
            f"source={d['source']})")

    return layer_data


# ── Phase 3: Multi-layer steering ───────────────────────────────────────────

def phase3_steering(model, tokenizer, layer_data, device, output_dir):
    log("")
    log("=" * 60)
    log("PHASE 3: Multi-layer steering — 10 configs × 8 prompts")
    log("=" * 60)

    num_configs = len(COEFF_CONFIGS)
    num_prompts = len(TEST_PROMPTS)
    total_gens = num_configs * num_prompts
    log(f"  {num_configs} configs × {num_prompts} prompts = {total_gens} steered generations")
    log(f"  + {num_prompts} unsteered baselines (cached and reused)")

    # Generate unsteered baselines first and cache them
    log("")
    log("  Generating unsteered baselines...")
    unsteered_cache = {}
    for i, prompt in enumerate(TEST_PROMPTS):
        result = generate_multi_layer_comparison(
            model, tokenizer, prompt,
            layer_steering=[],  # empty = unsteered only
            max_tokens=MAX_TOKENS,
            device=device,
            unsteered_cache=unsteered_cache,
        )
        log(f"    Baseline {i+1}/{num_prompts}: \"{prompt[:40]}...\"")

    all_results = []
    gen_count = 0

    for config_name, coeffs in COEFF_CONFIGS.items():
        log(f"\n  Config: {config_name}")
        coeff_str = " ".join(f"L{l}={c}" for l, c in coeffs.items())
        log(f"    Coefficients: {coeff_str}")

        # Build layer_steering list (skip layers with coefficient=0)
        layer_steering = []
        for layer_idx in ATTENTION_LAYERS:
            coeff = coeffs[layer_idx]
            if coeff > 0:
                layer_steering.append({
                    "layer_idx": layer_idx,
                    "steering_vector": layer_data[layer_idx]["steering_vector"],
                    "coefficient": coeff,
                    "feature_idx": layer_data[layer_idx]["feature_idx"],
                })

        active_layers = [ls["layer_idx"] for ls in layer_steering]
        log(f"    Active layers: {active_layers}")

        for prompt in TEST_PROMPTS:
            gen_count += 1

            result = generate_multi_layer_comparison(
                model, tokenizer, prompt,
                layer_steering=layer_steering,
                max_tokens=MAX_TOKENS,
                device=device,
                config_name=config_name,
                unsteered_cache=unsteered_cache,
            )

            scoring = score_generation(result.steered_text, result.unsteered_text)

            all_results.append({
                "config_name": config_name,
                "coefficients": coeffs,
                "active_layers": active_layers,
                "prompt": prompt,
                "steered_text": result.steered_text,
                "unsteered_text": result.unsteered_text,
                **scoring,
            })

        log(f"    {gen_count}/{total_gens} done")

    # Save raw results
    results_save = []
    for r in all_results:
        results_save.append({
            k: v for k, v in r.items()
        })
    with open(output_dir / "steering_results.json", "w") as f:
        json.dump(results_save, f, indent=2)

    return all_results


# ── Phase 4: Report ──────────────────────────────────────────────────────────

def phase4_report(layer_data, all_results, output_dir):
    log("")
    log("=" * 60)
    log("PHASE 4: Generating report")
    log("=" * 60)

    # Group results by config
    configs = {}
    for r in all_results:
        name = r["config_name"]
        if name not in configs:
            configs[name] = {
                "config_name": name,
                "coefficients": r["coefficients"],
                "active_layers": r["active_layers"],
                "results": [],
            }
        configs[name]["results"].append(r)

    # Rank configs
    ranked = []
    for name, cfg in configs.items():
        total_score = sum(r["quality_score"] for r in cfg["results"])
        total_bee = sum(r["bee_keyword_count"] for r in cfg["results"])
        total_rep = sum(1 for r in cfg["results"] if r["has_repetition"])
        total_diff = sum(1 for r in cfg["results"] if r["differs_from_unsteered"])
        ranked.append({
            "config_name": name,
            "coefficients": cfg["coefficients"],
            "active_layers": cfg["active_layers"],
            "total_quality_score": total_score,
            "total_bee_keywords": total_bee,
            "repetition_count": total_rep,
            "differs_count": total_diff,
            "num_prompts": len(cfg["results"]),
            "details": cfg["results"],
        })
    ranked.sort(key=lambda x: x["total_quality_score"], reverse=True)

    # ── Build report text ────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("I-BEE-M MULTI-LAYER GOLDEN GATE BEE REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Architecture summary
    lines.append("ARCHITECTURE:")
    lines.append("  Granite 4.0 hybrid: 28 Mamba-2 + 4 Attention layers")
    lines.append(f"  Attention layer indices: {ATTENTION_LAYERS}")
    lines.append("")

    # Per-layer feature summary
    lines.append("PER-LAYER FEATURES:")
    for layer_idx in ATTENTION_LAYERS:
        d = layer_data[layer_idx]
        lines.append(f"  L{layer_idx}: feature #{d['feature_idx']} "
                     f"(selectivity={d['selectivity']:.0f}x, "
                     f"source={d['source']})")
    lines.append("")

    # Hypothesis
    lines.append("HYPOTHESIS:")
    lines.append("  Single-layer steering fails because Mamba layers act as 'resistors'")
    lines.append("  that smooth out perturbations. Multi-layer steering at ALL 4 attention")
    lines.append("  layers should overwhelm this resistance by reinforcing the bee direction")
    lines.append("  at every attention checkpoint.")
    lines.append("")

    # Prior single-layer results
    lines.append("PRIOR SINGLE-LAYER RESULTS:")
    lines.append("  L17 attention: selectivity 6,164x, 0 bee keywords")
    lines.append("  L27 attention: selectivity 6,128x, 2 bee keywords (marginal)")
    lines.append("")

    # Ranking
    lines.append("=" * 70)
    lines.append("MULTI-LAYER STEERING RESULTS — RANKED BY QUALITY SCORE")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Configurations tested: {len(ranked)}")
    lines.append(f"Prompts per config: {len(TEST_PROMPTS)}")
    lines.append(f"Total steered generations: {len(all_results)}")
    lines.append(f"Max tokens per generation: {MAX_TOKENS}")
    lines.append("")

    for i, cfg in enumerate(ranked):
        coeffs = cfg["coefficients"]
        coeff_str = " ".join(f"L{l}={coeffs[l]}" for l in ATTENTION_LAYERS)
        lines.append(
            f"#{i+1:2d}  {cfg['config_name']:20s}  [{coeff_str}]  "
            f"score={cfg['total_quality_score']:4d}  "
            f"bee_kw={cfg['total_bee_keywords']:2d}  "
            f"rep={cfg['repetition_count']}/{cfg['num_prompts']}  "
            f"diff={cfg['differs_count']}/{cfg['num_prompts']}"
        )
    lines.append("")

    # Comparison: multi-layer vs single-layer
    lines.append("=" * 70)
    lines.append("MULTI-LAYER vs SINGLE-LAYER COMPARISON")
    lines.append("=" * 70)
    lines.append("")

    best = ranked[0]
    total_multi_bee = sum(cfg["total_bee_keywords"] for cfg in ranked)
    best_multi_bee = best["total_bee_keywords"]

    lines.append(f"Best multi-layer config: {best['config_name']}")
    lines.append(f"  Bee keywords: {best_multi_bee} (across {best['num_prompts']} prompts)")
    lines.append(f"  Quality score: {best['total_quality_score']}")
    lines.append(f"  Repetition: {best['repetition_count']}/{best['num_prompts']} prompts")
    lines.append("")
    lines.append(f"Total bee keywords across ALL multi-layer configs: {total_multi_bee}")
    lines.append(f"Prior best single-layer (L27): 2 bee keywords (across 8 prompts)")
    lines.append("")

    # deep_only vs shallow_only comparison
    deep_cfg = next((c for c in ranked if c["config_name"] == "deep_only"), None)
    shallow_cfg = next((c for c in ranked if c["config_name"] == "shallow_only"), None)
    if deep_cfg and shallow_cfg:
        lines.append("DEEP (L17+L27) vs SHALLOW (L10+L13):")
        lines.append(f"  deep_only:    score={deep_cfg['total_quality_score']}  "
                     f"bee_kw={deep_cfg['total_bee_keywords']}")
        lines.append(f"  shallow_only: score={shallow_cfg['total_quality_score']}  "
                     f"bee_kw={shallow_cfg['total_bee_keywords']}")
        lines.append("")

    # graduated_asc vs graduated_desc
    asc_cfg = next((c for c in ranked if c["config_name"] == "graduated_asc"), None)
    desc_cfg = next((c for c in ranked if c["config_name"] == "graduated_desc"), None)
    if asc_cfg and desc_cfg:
        lines.append("GRADUATED ASC (higher deeper) vs DESC (higher earlier):")
        lines.append(f"  graduated_asc:  score={asc_cfg['total_quality_score']}  "
                     f"bee_kw={asc_cfg['total_bee_keywords']}")
        lines.append(f"  graduated_desc: score={desc_cfg['total_quality_score']}  "
                     f"bee_kw={desc_cfg['total_bee_keywords']}")
        lines.append("")

    # Detailed output for top 3 configs
    lines.append("=" * 70)
    lines.append("DETAILED OUTPUT: TOP 3 CONFIGURATIONS")
    lines.append("=" * 70)

    for i, cfg in enumerate(ranked[:3]):
        coeffs = cfg["coefficients"]
        coeff_str = " ".join(f"L{l}={coeffs[l]}" for l in ATTENTION_LAYERS)
        lines.append("")
        lines.append(f"--- #{i+1}: {cfg['config_name']} [{coeff_str}] ---")
        lines.append(f"    score={cfg['total_quality_score']}  "
                     f"bee_kw={cfg['total_bee_keywords']}  "
                     f"rep={cfg['repetition_count']}/{cfg['num_prompts']}")
        lines.append("")

        for r in cfg["details"]:
            lines.append(f"  Prompt: \"{r['prompt']}\"")
            lines.append(f"  Steered:   {r['steered_text'][:300]}")
            lines.append(f"  Unsteered: {r['unsteered_text'][:300]}")
            lines.append(f"  [bee_kw={r['bee_keyword_count']} rep={r['has_repetition']} "
                         f"diff={r['differs_from_unsteered']} score={r['quality_score']}]")
            lines.append("")

    # Conclusion
    lines.append("=" * 70)
    lines.append("CONCLUSION")
    lines.append("=" * 70)
    lines.append("")

    if total_multi_bee > 2:
        lines.append("BREAKTHROUGH: Multi-layer steering produces more bee keywords than")
        lines.append("single-layer steering! The hypothesis that Mamba layers act as")
        lines.append("'resistors' is supported — simultaneous steering at all attention")
        lines.append("layers overwhelms this resistance.")
    elif total_multi_bee > 0:
        lines.append("PARTIAL SUCCESS: Some bee keywords detected in multi-layer steering,")
        lines.append("but not dramatically more than single-layer L27 (2 keywords).")
        lines.append("The multi-layer approach may need higher coefficients or additional")
        lines.append("techniques to fully overcome Mamba resistance.")
    else:
        lines.append("NEGATIVE RESULT: Multi-layer steering did not produce bee keywords.")
        lines.append("This suggests the hybrid Mamba-Transformer architecture has")
        lines.append("fundamental resistance to activation steering that cannot be")
        lines.append("overcome by simply steering at more layers simultaneously.")
        lines.append("")
        lines.append("Possible explanations:")
        lines.append("  1. Mamba layers actively counteract steering at ALL attention layers")
        lines.append("  2. The SAE features, while selective, don't correspond to generation-")
        lines.append("     time bee representations")
        lines.append("  3. Higher coefficients are needed (risk of gibberish)")
        lines.append("  4. Different steering method needed (e.g., Mamba layer steering)")
    lines.append("")

    report_text = "\n".join(lines)
    with open(output_dir / "report.txt", "w") as f:
        f.write(report_text)

    # Save ranked summary
    ranked_summary = []
    for cfg in ranked:
        ranked_summary.append({k: v for k, v in cfg.items() if k != "details"})
    with open(output_dir / "report_ranked.json", "w") as f:
        json.dump(ranked_summary, f, indent=2)

    return ranked


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")
    log(f"Attention layers: {ATTENTION_LAYERS}")
    log(f"Coefficient configs: {len(COEFF_CONFIGS)}")
    log(f"Test prompts: {len(TEST_PROMPTS)}")

    # Save config
    config = {
        "attention_layers": ATTENTION_LAYERS,
        "known_saes": {str(k): v for k, v in KNOWN_SAES.items()},
        "coeff_configs": {k: {str(l): c for l, c in v.items()} for k, v in COEFF_CONFIGS.items()},
        "test_prompts": TEST_PROMPTS,
        "max_tokens": MAX_TOKENS,
        "sae_expansion_factor": NEW_SAE_CONFIG.expansion_factor,
        "sae_num_activations": NEW_SAE_CONFIG.num_activations,
        "sae_num_epochs": NEW_SAE_CONFIG.num_epochs,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Phase 1: Load model
    model, tokenizer, device = phase1_load_model()

    # Phase 2: Prepare SAEs for all 4 attention layers
    layer_data = phase2_prepare_saes(model, tokenizer, device, output_dir)

    # Save layer summary
    layer_summary = {}
    for layer_idx in ATTENTION_LAYERS:
        d = layer_data[layer_idx]
        layer_summary[str(layer_idx)] = {
            "feature_idx": d["feature_idx"],
            "selectivity": d["selectivity"],
            "source": d["source"],
            "steering_vector_norm": d["steering_vector"].norm().item(),
        }
    with open(output_dir / "layer_summary.json", "w") as f:
        json.dump(layer_summary, f, indent=2)

    # Phase 3: Multi-layer steering
    all_results = phase3_steering(model, tokenizer, layer_data, device, output_dir)

    # Phase 4: Report
    ranked = phase4_report(layer_data, all_results, output_dir)

    # Log top 5
    log("")
    log("TOP 5 MULTI-LAYER CONFIGURATIONS:")
    for i, cfg in enumerate(ranked[:5]):
        coeffs = cfg["coefficients"]
        coeff_str = " ".join(f"L{l}={coeffs[l]}" for l in ATTENTION_LAYERS)
        log(f"  #{i+1}: {cfg['config_name']:20s}  [{coeff_str}]  "
            f"score={cfg['total_quality_score']}  "
            f"bee_kw={cfg['total_bee_keywords']}")

    # Final summary
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    total_bee = sum(cfg["total_bee_keywords"] for cfg in ranked)

    summary = {
        "run_id": run_id,
        "elapsed_seconds": elapsed,
        "elapsed_human": f"{minutes}m {seconds}s",
        "approach": "multilayer_golden_gate_bee",
        "attention_layers": ATTENTION_LAYERS,
        "num_configs": len(COEFF_CONFIGS),
        "num_prompts": len(TEST_PROMPTS),
        "total_generations": len(all_results),
        "total_bee_keywords_all_configs": total_bee,
        "best_config": {
            "name": ranked[0]["config_name"],
            "coefficients": ranked[0]["coefficients"],
            "quality_score": ranked[0]["total_quality_score"],
            "bee_keywords": ranked[0]["total_bee_keywords"],
        } if ranked else None,
        "per_layer": layer_summary,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log("")
    log("=" * 60)
    log("MULTI-LAYER GOLDEN GATE BEE COMPLETE!")
    log(f"  Time: {minutes}m {seconds}s")
    log(f"  Total steered generations: {len(all_results)}")
    log(f"  Total bee keywords (all configs): {total_bee}")
    if ranked:
        log(f"  Best config: {ranked[0]['config_name']} "
            f"(score={ranked[0]['total_quality_score']}, "
            f"bee_kw={ranked[0]['total_bee_keywords']})")
    log(f"  Report: {output_dir / 'report.txt'}")
    log("=" * 60)


if __name__ == "__main__":
    main()
