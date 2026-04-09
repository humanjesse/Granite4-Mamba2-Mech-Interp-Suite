#!/usr/bin/env python3
"""I-Bee-M: SAE feature search pipeline.

The proper Golden Gate Claude approach:
1. Collect residual stream activations from Layer 4
2. Train a Sparse Autoencoder to decompose them into monosemantic features
3. Search for features that selectively activate on bee-related text
4. Extract the decoder column as a steering vector for the best bee feature

Usage:
    source .venv/bin/activate
    python run_sae_feature_search.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import torch

from src.model_loader import load_model
from src.sae.trainer import SAETrainingConfig, collect_activations, train_sae
from src.sae.feature_search import search_features, get_default_bee_texts, get_default_baseline_texts
from src.extraction.activation_steering import generate_comparison
from run_contrastive_steering import score_generation, TEST_PROMPTS, MAX_TOKENS

# ── Config ───────────────────────────────────────────────────────────────────

LAYER_IDX = 4
OUTPUT_BASE = Path("results/sae")

SAE_CONFIG = SAETrainingConfig(
    layer_idx=LAYER_IDX,
    expansion_factor=8,       # 768 * 8 = 6144 latent features
    l1_coefficient=1e-3,
    learning_rate=3e-4,
    num_activations=50_000,
    batch_size=256,
    num_epochs=3,
)

# Coefficients to test with the best SAE bee feature
STEERING_COEFFICIENTS = [3.0, 5.0, 7.0, 10.0, 15.0]


# ── Utilities ────────────────────────────────────────────────────────────────

def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    # ── Load model ───────────────────────────────────────────────────────
    log("Loading model...")
    model, tokenizer, device = load_model(device="cuda")
    log(f"Model loaded on {device}. Hidden size: {model.config.hidden_size}")

    # ── Phase 1: Collect activations ─────────────────────────────────────
    log("")
    log("=" * 60)
    log("PHASE 1: Collecting activations from Layer 4")
    log("=" * 60)

    def collect_progress(collected, total):
        if collected % 5000 == 0 or collected == total:
            log(f"  Collected {collected}/{total} activations")

    t0 = time.time()
    activations = collect_activations(
        model, tokenizer,
        layer_idx=LAYER_IDX,
        num_activations=SAE_CONFIG.num_activations,
        device=device,
        progress_callback=collect_progress,
    )
    log(f"  Done: {activations.shape} in {time.time() - t0:.1f}s")

    # Save activation stats
    act_stats = {
        "shape": list(activations.shape),
        "mean": activations.mean().item(),
        "std": activations.std().item(),
        "min": activations.min().item(),
        "max": activations.max().item(),
    }
    with open(output_dir / "activation_stats.json", "w") as f:
        json.dump(act_stats, f, indent=2)

    # ── Phase 2: Train SAE ───────────────────────────────────────────────
    log("")
    log("=" * 60)
    log("PHASE 2: Training Sparse Autoencoder")
    log("=" * 60)
    log(f"  Input dim: {activations.shape[1]}")
    log(f"  Latent dim: {activations.shape[1] * SAE_CONFIG.expansion_factor}")
    log(f"  Activations: {activations.shape[0]}")
    log(f"  Epochs: {SAE_CONFIG.num_epochs}")
    log(f"  L1 coefficient: {SAE_CONFIG.l1_coefficient}")

    last_logged_step = [0]

    def train_progress(step, total):
        if step % 50 == 0 or step == total:
            log(f"  Step {step}/{total}  "
                f"({100 * step / total:.0f}%)")
            last_logged_step[0] = step

    t0 = time.time()
    result = train_sae(activations, SAE_CONFIG, progress_callback=train_progress, device=device)
    train_time = time.time() - t0

    log(f"  Training complete in {train_time:.1f}s")
    log(f"  Final loss: {result.final_loss:.6f}")
    log(f"  Reconstruction loss: {result.final_reconstruction_loss:.6f}")
    log(f"  L1 loss: {result.final_l1_loss:.6f}")
    log(f"  Mean active features: {result.mean_active_features:.1f}")
    log(f"  Dead features: {result.dead_features}/{result.num_latent_features}")

    # Save trained SAE
    torch.save(result.sae.state_dict(), output_dir / "sae_layer4.pt")

    # Save training metrics
    training_metrics = {
        "layer_idx": LAYER_IDX,
        "input_dim": result.sae.input_dim,
        "latent_dim": result.sae.latent_dim,
        "expansion_factor": SAE_CONFIG.expansion_factor,
        "l1_coefficient": SAE_CONFIG.l1_coefficient,
        "learning_rate": SAE_CONFIG.learning_rate,
        "num_activations": SAE_CONFIG.num_activations,
        "num_epochs": SAE_CONFIG.num_epochs,
        "final_loss": result.final_loss,
        "final_reconstruction_loss": result.final_reconstruction_loss,
        "final_l1_loss": result.final_l1_loss,
        "mean_active_features": result.mean_active_features,
        "dead_features": result.dead_features,
        "num_latent_features": result.num_latent_features,
        "training_time_seconds": train_time,
    }
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)

    # Save loss curves
    with open(output_dir / "loss_history.json", "w") as f:
        json.dump({
            "loss": result.loss_history,
            "reconstruction": result.reconstruction_history,
            "l1": result.l1_history,
        }, f)

    # ── Phase 3: Search for bee features ─────────────────────────────────
    log("")
    log("=" * 60)
    log("PHASE 3: Searching for bee features")
    log("=" * 60)

    bee_texts = get_default_bee_texts()
    baseline_texts = get_default_baseline_texts()
    log(f"  Bee texts: {len(bee_texts)}")
    log(f"  Baseline texts: {len(baseline_texts)}")

    # Move SAE to the right device for feature search
    sae = result.sae.to(device)

    t0 = time.time()
    search_result = search_features(
        sae, model, tokenizer,
        layer_idx=LAYER_IDX,
        concept_texts=bee_texts,
        baseline_texts=baseline_texts,
        device=device,
        top_k=20,
        concept_name="bee",
    )
    log(f"  Feature search complete in {time.time() - t0:.1f}s")

    log("")
    log("TOP 20 BEE-SELECTIVE FEATURES:")
    log(f"  {'Rank':>4}  {'Feature':>8}  {'Bee Act':>10}  {'Base Act':>10}  {'Selectivity':>12}")
    for i, feat in enumerate(search_result.top_features):
        log(f"  {i+1:4d}  {feat['feature_idx']:8d}  "
            f"{feat['concept_activation']:10.4f}  "
            f"{feat['baseline_activation']:10.4f}  "
            f"{feat['selectivity_score']:12.2f}")

    # Save feature search results
    feature_results = {
        "concept": "bee",
        "top_features": search_result.top_features,
        "bee_texts": bee_texts,
        "baseline_texts": baseline_texts,
    }
    with open(output_dir / "feature_search.json", "w") as f:
        json.dump(feature_results, f, indent=2)

    # ── Phase 4: Steer with top bee feature ──────────────────────────────
    log("")
    log("=" * 60)
    log("PHASE 4: Steering with top bee feature")
    log("=" * 60)

    best_feature = search_result.top_features[0]
    best_idx = best_feature["feature_idx"]
    log(f"  Best feature: #{best_idx}")
    log(f"  Selectivity: {best_feature['selectivity_score']:.2f}")
    log(f"  Bee activation: {best_feature['concept_activation']:.4f}")
    log(f"  Baseline activation: {best_feature['baseline_activation']:.4f}")

    # Get the feature direction (decoder column)
    steering_vec = sae.get_feature_direction(best_idx).to(device)
    log(f"  Steering vector shape: {steering_vec.shape}")
    log(f"  Steering vector norm: {steering_vec.norm():.4f}")

    # Save the steering vector
    torch.save(steering_vec.cpu(), output_dir / f"bee_feature_{best_idx}_direction.pt")

    # Also save top 5 feature directions
    for i, feat in enumerate(search_result.top_features[:5]):
        fidx = feat["feature_idx"]
        fvec = sae.get_feature_direction(fidx)
        torch.save(fvec, output_dir / f"feature_{fidx}_direction.pt")

    all_steering_results = []
    gen_count = 0
    total_gens = len(STEERING_COEFFICIENTS) * len(TEST_PROMPTS)

    for coeff in STEERING_COEFFICIENTS:
        log(f"  coeff={coeff}")
        for prompt in TEST_PROMPTS:
            gen_count += 1
            result_gen = generate_comparison(
                model, tokenizer, prompt, steering_vec,
                layer_idx=LAYER_IDX,
                coefficient=coeff,
                max_tokens=MAX_TOKENS,
                device=device,
            )

            scoring = score_generation(result_gen.steered_text, result_gen.unsteered_text)

            all_steering_results.append({
                "prompt": prompt,
                "steered_text": result_gen.steered_text,
                "unsteered_text": result_gen.unsteered_text,
                "layer_idx": LAYER_IDX,
                "feature_idx": best_idx,
                "coefficient": coeff,
                "feature_direction_norm": result_gen.feature_direction_norm,
                **scoring,
            })

        log(f"    {gen_count}/{total_gens} done")

    # Save steering results
    with open(output_dir / "steering_results.json", "w") as f:
        json.dump(all_steering_results, f, indent=2)

    # ── Generate report ──────────────────────────────────────────────────
    log("")
    log("Generating report...")

    # Rank configs
    configs = {}
    for r in all_steering_results:
        key = r["coefficient"]
        if key not in configs:
            configs[key] = {"coefficient": r["coefficient"], "results": []}
        configs[key]["results"].append(r)

    ranked = []
    for key, cfg in configs.items():
        total_score = sum(r["quality_score"] for r in cfg["results"])
        total_bee = sum(r["bee_keyword_count"] for r in cfg["results"])
        total_rep = sum(1 for r in cfg["results"] if r["has_repetition"])
        total_diff = sum(1 for r in cfg["results"] if r["differs_from_unsteered"])
        ranked.append({
            "coefficient": cfg["coefficient"],
            "total_quality_score": total_score,
            "total_bee_keywords": total_bee,
            "repetition_count": total_rep,
            "differs_count": total_diff,
            "num_prompts": len(cfg["results"]),
            "details": cfg["results"],
        })
    ranked.sort(key=lambda x: x["total_quality_score"], reverse=True)

    lines = []
    lines.append("=" * 70)
    lines.append("I-BEE-M SAE FEATURE SEARCH REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Layer:              {LAYER_IDX}")
    lines.append(f"SAE latent dim:     {result.sae.latent_dim}")
    lines.append(f"Training samples:   {SAE_CONFIG.num_activations}")
    lines.append(f"Training epochs:    {SAE_CONFIG.num_epochs}")
    lines.append(f"Dead features:      {result.dead_features}/{result.num_latent_features}")
    lines.append(f"Mean active:        {result.mean_active_features:.1f}")
    lines.append("")
    lines.append("TOP 20 BEE-SELECTIVE FEATURES:")
    lines.append(f"  {'Rank':>4}  {'Feature':>8}  {'Bee Act':>10}  {'Base Act':>10}  {'Selectivity':>12}")
    for i, feat in enumerate(search_result.top_features):
        lines.append(f"  {i+1:4d}  {feat['feature_idx']:8d}  "
                     f"{feat['concept_activation']:10.4f}  "
                     f"{feat['baseline_activation']:10.4f}  "
                     f"{feat['selectivity_score']:12.2f}")
    lines.append("")
    lines.append(f"Best bee feature: #{best_idx} "
                 f"(selectivity={best_feature['selectivity_score']:.2f})")
    lines.append("")
    lines.append("STEERING RESULTS (using best bee feature):")
    lines.append("")

    for i, cfg in enumerate(ranked):
        lines.append(
            f"#{i+1:2d}  coeff={cfg['coefficient']:5.1f}  "
            f"score={cfg['total_quality_score']:3d}  "
            f"bee_kw={cfg['total_bee_keywords']}  "
            f"rep={cfg['repetition_count']}/{cfg['num_prompts']}  "
            f"diff={cfg['differs_count']}/{cfg['num_prompts']}"
        )

    lines.append("")
    lines.append("=" * 70)
    lines.append("DETAILED OUTPUT: ALL CONFIGS")
    lines.append("=" * 70)

    for cfg in ranked:
        lines.append("")
        lines.append(f"--- coeff={cfg['coefficient']} ---")
        lines.append("")
        for r in cfg["details"]:
            lines.append(f"  Prompt: \"{r['prompt']}\"")
            lines.append(f"  Steered:   {r['steered_text'][:300]}")
            lines.append(f"  Unsteered: {r['unsteered_text'][:300]}")
            lines.append(f"  [bee_kw={r['bee_keyword_count']} rep={r['has_repetition']} "
                         f"diff={r['differs_from_unsteered']} score={r['quality_score']}]")
            lines.append("")

    report_text = "\n".join(lines)
    with open(output_dir / "report.txt", "w") as f:
        f.write(report_text)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    log("")
    log("TOP STEERING CONFIGS:")
    for i, cfg in enumerate(ranked[:3]):
        log(f"  #{i+1}: coeff={cfg['coefficient']} "
            f"→ score={cfg['total_quality_score']} bee_kw={cfg['total_bee_keywords']} "
            f"rep={cfg['repetition_count']}/{cfg['num_prompts']}")

    summary = {
        "run_id": run_id,
        "elapsed_seconds": elapsed,
        "elapsed_human": f"{minutes}m {seconds}s",
        "approach": "sae_feature_search",
        "layer_idx": LAYER_IDX,
        "sae_latent_dim": result.sae.latent_dim,
        "best_bee_feature": best_feature,
        "steering_coefficients_tested": STEERING_COEFFICIENTS,
        "total_generations": len(all_steering_results),
        "best_steering_config": {
            "coefficient": ranked[0]["coefficient"],
            "quality_score": ranked[0]["total_quality_score"],
            "bee_keywords": ranked[0]["total_bee_keywords"],
        } if ranked else None,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log("")
    log("=" * 60)
    log("SAE FEATURE SEARCH COMPLETE!")
    log(f"  Time: {minutes}m {seconds}s")
    log(f"  Best bee feature: #{best_idx} (selectivity={best_feature['selectivity_score']:.2f})")
    log(f"  Total generations: {len(all_steering_results)}")
    log(f"  Report: {output_dir / 'report.txt'}")
    log(f"  Results: {output_dir}")
    log("=" * 60)


if __name__ == "__main__":
    main()
