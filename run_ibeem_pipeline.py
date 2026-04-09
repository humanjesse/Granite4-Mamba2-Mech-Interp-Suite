#!/usr/bin/env python3
"""I-Bee-M Granite4: Full overnight pipeline (Run 2).

Runs the complete SAE training → feature search → activation steering pipeline
with a grid search over features, coefficients, and steering modes.

Usage:
    source .venv/bin/activate
    python run_ibeem_pipeline.py

Results are saved to results/ibeem/ with timestamps.
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# ── Configuration ────────────────────────────────────────────────────────────

LAYER_IDX = 16                # Residual stream layer to train SAE on
EXPANSION_FACTOR = 8          # SAE latent dim = hidden_size * expansion_factor
NUM_ACTIVATIONS = 100_000     # Number of activation vectors to collect
L1_COEFFICIENT = 1e-3         # Sparsity penalty weight
LEARNING_RATE = 3e-4          # Adam LR for SAE training
BATCH_SIZE = 256              # SAE training batch size
NUM_EPOCHS = 1                # SAE training epochs

TOP_K_FEATURES = 20           # Number of top features to report
NUM_STEERING_FEATURES = 3     # Top N features to steer with
STEERING_COEFFICIENTS = [1.0, 2.0, 3.0, 5.0]
STEERING_MODES = [False, True]  # [all-positions, last-token-only]
MAX_TOKENS = 100              # Tokens to generate per prompt

TEST_PROMPTS = [
    "The weather today is",
    "I went to the store to buy",
    "The most important thing in life is",
    "Scientists recently discovered that",
    "My favorite hobby is",
    "The CEO announced that the company will",
    "In the garden, I noticed",
    "The recipe calls for",
]

BEE_KEYWORDS = [
    "bee", "bees", "hive", "hives", "honey", "honeybee", "honeycomb",
    "buzz", "buzzing", "pollen", "pollinator", "pollination", "nectar",
    "beekeeper", "beekeeping", "apiary", "swarm", "colony", "queen bee",
    "worker bee", "drone", "wax", "beeswax", "sting", "stinger",
]

# ── Setup ────────────────────────────────────────────────────────────────────


def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ── Quality scoring ──────────────────────────────────────────────────────────


def detect_repetition(text: str) -> bool:
    """Check if text contains repeated phrases (3+ word ngrams repeated 3+ times)."""
    words = text.lower().split()
    if len(words) < 9:
        return False
    for n in range(3, min(8, len(words) // 3 + 1)):
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
        from collections import Counter
        counts = Counter(ngrams)
        if counts.most_common(1)[0][1] >= 3:
            return True
    return False


def count_bee_keywords(text: str) -> int:
    """Count how many bee-related keywords appear in the text."""
    text_lower = text.lower()
    count = 0
    for kw in BEE_KEYWORDS:
        count += len(re.findall(r'\b' + re.escape(kw) + r'\b', text_lower))
    return count


def text_differs(steered: str, unsteered: str) -> bool:
    """Check if steered output meaningfully differs from unsteered."""
    # Strip the prompt (shared prefix) and compare the generated parts
    # Simple heuristic: more than 20% of words differ
    s_words = steered.lower().split()
    u_words = unsteered.lower().split()
    if not s_words or not u_words:
        return True
    # Compare the last N words (generated portion)
    min_len = min(len(s_words), len(u_words))
    diffs = sum(1 for a, b in zip(s_words[-min_len:], u_words[-min_len:]) if a != b)
    return diffs / min_len > 0.2 if min_len > 0 else True


def score_generation(steered_text: str, unsteered_text: str) -> dict:
    """Score a single generation for quality."""
    has_repetition = detect_repetition(steered_text)
    bee_count = count_bee_keywords(steered_text)
    differs = text_differs(steered_text, unsteered_text)

    # Quality score: higher is better
    # -10 for repetition, +2 per bee keyword, +1 if differs from unsteered
    score = 0
    if has_repetition:
        score -= 10
    score += 2 * bee_count
    if differs:
        score += 1

    return {
        "has_repetition": has_repetition,
        "bee_keyword_count": bee_count,
        "differs_from_unsteered": differs,
        "quality_score": score,
    }


def generate_report(all_results: list, output_dir: Path):
    """Generate a report.txt ranking all configurations by quality."""
    # Group results by configuration
    configs = {}
    for r in all_results:
        key = (r["feature_idx"], r["coefficient"], r["last_token_only"])
        if key not in configs:
            configs[key] = {
                "feature_idx": r["feature_idx"],
                "coefficient": r["coefficient"],
                "last_token_only": r["last_token_only"],
                "results": [],
            }
        scoring = score_generation(r["steered_text"], r["unsteered_text"])
        configs[key]["results"].append({**r, **scoring})

    # Compute aggregate scores per config
    ranked = []
    for key, cfg in configs.items():
        total_score = sum(r["quality_score"] for r in cfg["results"])
        total_bee = sum(r["bee_keyword_count"] for r in cfg["results"])
        total_rep = sum(1 for r in cfg["results"] if r["has_repetition"])
        total_diff = sum(1 for r in cfg["results"] if r["differs_from_unsteered"])
        ranked.append({
            "feature_idx": cfg["feature_idx"],
            "coefficient": cfg["coefficient"],
            "last_token_only": cfg["last_token_only"],
            "total_quality_score": total_score,
            "total_bee_keywords": total_bee,
            "repetition_count": total_rep,
            "differs_count": total_diff,
            "num_prompts": len(cfg["results"]),
            "details": cfg["results"],
        })

    # Sort by total quality score (descending)
    ranked.sort(key=lambda x: x["total_quality_score"], reverse=True)

    # Write report
    lines = []
    lines.append("=" * 70)
    lines.append("I-BEE-M RUN 2: ACTIVATION STEERING GRID SEARCH REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Configurations tested: {len(ranked)}")
    lines.append(f"Prompts per config: {len(TEST_PROMPTS)}")
    lines.append(f"Total generations: {len(all_results)}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("RANKING (best to worst)")
    lines.append("-" * 70)
    lines.append("")

    for i, cfg in enumerate(ranked):
        mode = "last-token" if cfg["last_token_only"] else "all-positions"
        lines.append(f"#{i+1}  Feature {cfg['feature_idx']}  coeff={cfg['coefficient']}  mode={mode}")
        lines.append(f"     Score: {cfg['total_quality_score']}  "
                     f"Bee keywords: {cfg['total_bee_keywords']}  "
                     f"Repetitions: {cfg['repetition_count']}/{cfg['num_prompts']}  "
                     f"Changed: {cfg['differs_count']}/{cfg['num_prompts']}")
        lines.append("")

    # Detailed output for top 3 configs
    lines.append("")
    lines.append("=" * 70)
    lines.append("DETAILED OUTPUT: TOP 3 CONFIGURATIONS")
    lines.append("=" * 70)

    for i, cfg in enumerate(ranked[:3]):
        mode = "last-token" if cfg["last_token_only"] else "all-positions"
        lines.append("")
        lines.append(f"--- #{i+1}: Feature {cfg['feature_idx']}  coeff={cfg['coefficient']}  mode={mode} ---")
        lines.append("")
        for r in cfg["details"]:
            lines.append(f"  Prompt: \"{r['prompt']}\"")
            lines.append(f"  Steered:   {r['steered_text'][:200]}")
            lines.append(f"  Unsteered: {r['unsteered_text'][:200]}")
            lines.append(f"  [bee_kw={r['bee_keyword_count']} rep={r['has_repetition']} diff={r['differs_from_unsteered']} score={r['quality_score']}]")
            lines.append("")

    report_text = "\n".join(lines)
    with open(output_dir / "report.txt", "w") as f:
        f.write(report_text)

    # Also save the full ranked data as JSON
    # Strip 'details' with full text for the JSON summary
    ranked_summary = []
    for cfg in ranked:
        ranked_summary.append({k: v for k, v in cfg.items() if k != "details"})
    with open(output_dir / "report_ranked.json", "w") as f:
        json.dump(ranked_summary, f, indent=2)

    return ranked


def main():
    start_time = time.time()

    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/ibeem") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    # Save config
    config = {
        "layer_idx": LAYER_IDX,
        "expansion_factor": EXPANSION_FACTOR,
        "num_activations": NUM_ACTIVATIONS,
        "l1_coefficient": L1_COEFFICIENT,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "top_k_features": TOP_K_FEATURES,
        "num_steering_features": NUM_STEERING_FEATURES,
        "steering_coefficients": STEERING_COEFFICIENTS,
        "steering_modes": ["all-positions", "last-token-only"],
        "max_tokens": MAX_TOKENS,
        "test_prompts": TEST_PROMPTS,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Step 1: Load model ───────────────────────────────────────────────

    log("Step 1/5: Loading model...")
    from src.model_loader import load_model
    model, tokenizer, device = load_model(device="cpu")
    log(f"Model loaded on {device}. Hidden size: {model.config.hidden_size}")

    # ── Step 2: Collect activations ──────────────────────────────────────

    log(f"Step 2/5: Collecting {NUM_ACTIVATIONS:,} activations from layer {LAYER_IDX}...")
    from src.sae.trainer import SAETrainingConfig, collect_activations, train_sae

    last_pct = [0]
    def collect_progress(current, total):
        pct = int(100 * current / total)
        if pct >= last_pct[0] + 10:
            last_pct[0] = pct
            log(f"  Collecting activations: {current:,}/{total:,} ({pct}%)")

    activations = collect_activations(
        model, tokenizer, LAYER_IDX, NUM_ACTIVATIONS,
        device=device, progress_callback=collect_progress,
    )
    log(f"Collected activations: {activations.shape}")

    # Save raw activations
    torch.save(activations, output_dir / "activations.pt")
    log(f"Saved activations to {output_dir / 'activations.pt'}")

    # ── Step 3: Train SAE ────────────────────────────────────────────────

    log("Step 3/5: Training SAE...")
    sae_config = SAETrainingConfig(
        layer_idx=LAYER_IDX,
        expansion_factor=EXPANSION_FACTOR,
        l1_coefficient=L1_COEFFICIENT,
        learning_rate=LEARNING_RATE,
        num_activations=NUM_ACTIVATIONS,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
    )

    last_pct[0] = 0
    def train_progress(step, total):
        pct = int(100 * step / total)
        if pct >= last_pct[0] + 10:
            last_pct[0] = pct
            log(f"  Training: step {step}/{total} ({pct}%)")

    training_result = train_sae(activations, sae_config, progress_callback=train_progress)

    log(f"SAE trained: {training_result.sae.input_dim} → {training_result.sae.latent_dim}")
    log(f"  Final loss: {training_result.final_loss:.6f} "
        f"(recon: {training_result.final_reconstruction_loss:.6f}, "
        f"L1: {training_result.final_l1_loss:.6f})")
    log(f"  Active features: {training_result.mean_active_features:.0f}/{training_result.num_latent_features}")
    log(f"  Dead features: {training_result.dead_features}/{training_result.num_latent_features}")

    # Save SAE weights
    torch.save(training_result.sae.state_dict(), output_dir / "sae_weights.pt")

    # Save training metrics
    training_metrics = {
        "input_dim": training_result.sae.input_dim,
        "latent_dim": training_result.sae.latent_dim,
        "final_loss": training_result.final_loss,
        "final_reconstruction_loss": training_result.final_reconstruction_loss,
        "final_l1_loss": training_result.final_l1_loss,
        "mean_active_features": training_result.mean_active_features,
        "dead_features": training_result.dead_features,
        "num_latent_features": training_result.num_latent_features,
        "loss_history": training_result.loss_history,
        "reconstruction_history": training_result.reconstruction_history,
        "l1_history": training_result.l1_history,
    }
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)
    log(f"Saved SAE weights and metrics to {output_dir}")

    # ── Step 4: Feature search ───────────────────────────────────────────

    log("Step 4/5: Searching for bee features...")
    from src.sae.feature_search import search_features, get_default_bee_texts, get_default_baseline_texts

    bee_texts = get_default_bee_texts()
    baseline_texts = get_default_baseline_texts()

    def search_progress(step, total):
        log(f"  Feature search: {step}/{total} texts processed")

    search_result = search_features(
        training_result.sae, model, tokenizer, LAYER_IDX,
        bee_texts, baseline_texts,
        device=device, top_k=TOP_K_FEATURES,
        concept_name="bee",
        progress_callback=search_progress,
    )

    log(f"Feature search complete. Top {len(search_result.top_features)} features:")
    for i, f in enumerate(search_result.top_features[:5]):
        log(f"  #{i+1}: Feature {f['feature_idx']} "
            f"(selectivity: {f['selectivity_score']:.1f}x, "
            f"concept: {f['concept_activation']:.4f}, "
            f"baseline: {f['baseline_activation']:.4f})")

    # Save feature search results
    feature_results = {
        "concept": search_result.concept,
        "concept_texts": search_result.concept_texts,
        "baseline_texts": search_result.baseline_texts,
        "top_features": search_result.top_features,
    }
    with open(output_dir / "feature_search.json", "w") as f:
        json.dump(feature_results, f, indent=2)
    log(f"Saved feature search results to {output_dir}")

    # ── Step 5: Grid search activation steering ──────────────────────────

    log("Step 5/5: Running activation steering grid search...")
    from src.extraction.activation_steering import generate_comparison

    # Select top N features for steering
    top_features = search_result.top_features[:NUM_STEERING_FEATURES]
    feature_indices = [f["feature_idx"] for f in top_features]
    log(f"Steering with top {NUM_STEERING_FEATURES} features: {feature_indices}")

    total_configs = len(feature_indices) * len(STEERING_COEFFICIENTS) * len(STEERING_MODES)
    total_generations = total_configs * len(TEST_PROMPTS)
    log(f"Grid: {len(feature_indices)} features x {len(STEERING_COEFFICIENTS)} coefficients "
        f"x {len(STEERING_MODES)} modes = {total_configs} configs, "
        f"{total_generations} total generations")

    all_steering_results = []
    gen_count = 0

    for feat_idx in feature_indices:
        steering_vector = training_result.sae.get_feature_direction(feat_idx)
        vec_norm = steering_vector.norm().item()
        log(f"  Feature {feat_idx}: direction norm = {vec_norm:.4f}")

        for coeff in STEERING_COEFFICIENTS:
            for last_token_only in STEERING_MODES:
                mode_str = "last-token" if last_token_only else "all-positions"
                log(f"    Config: feature={feat_idx} coeff={coeff} mode={mode_str}")

                for i, prompt in enumerate(TEST_PROMPTS):
                    gen_count += 1
                    result = generate_comparison(
                        model, tokenizer, prompt, steering_vector,
                        layer_idx=LAYER_IDX,
                        coefficient=coeff,
                        max_tokens=MAX_TOKENS,
                        device=device,
                        last_token_only=last_token_only,
                    )

                    steering_data = {
                        "prompt": prompt,
                        "steered_text": result.steered_text,
                        "unsteered_text": result.unsteered_text,
                        "steered_tokens": result.steered_tokens,
                        "unsteered_tokens": result.unsteered_tokens,
                        "feature_idx": feat_idx,
                        "coefficient": coeff,
                        "last_token_only": last_token_only,
                        "feature_direction_norm": result.feature_direction_norm,
                    }
                    all_steering_results.append(steering_data)

                log(f"      {gen_count}/{total_generations} generations done")

    # Save all steering results
    with open(output_dir / "steering_results.json", "w") as f:
        json.dump(all_steering_results, f, indent=2)
    log(f"Saved {len(all_steering_results)} steering results to {output_dir}")

    # ── Generate quality report ──────────────────────────────────────────

    log("Generating quality report...")
    ranked = generate_report(all_steering_results, output_dir)

    # Log top 3
    log("")
    log("TOP 3 CONFIGURATIONS:")
    for i, cfg in enumerate(ranked[:3]):
        mode = "last-token" if cfg["last_token_only"] else "all-positions"
        log(f"  #{i+1}: Feature {cfg['feature_idx']} coeff={cfg['coefficient']} mode={mode} "
            f"→ score={cfg['total_quality_score']} bee_kw={cfg['total_bee_keywords']} "
            f"rep={cfg['repetition_count']}/{cfg['num_prompts']}")

    # ── Summary ──────────────────────────────────────────────────────────

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    summary = {
        "run_id": run_id,
        "elapsed_seconds": elapsed,
        "elapsed_human": f"{minutes}m {seconds}s",
        "layer_idx": LAYER_IDX,
        "sae_dims": f"{training_result.sae.input_dim} → {training_result.sae.latent_dim}",
        "final_loss": training_result.final_loss,
        "num_features_steered": NUM_STEERING_FEATURES,
        "feature_indices": feature_indices,
        "coefficients_tested": STEERING_COEFFICIENTS,
        "modes_tested": ["all-positions", "last-token-only"],
        "total_configs": total_configs,
        "total_generations": len(all_steering_results),
        "best_config": {
            "feature_idx": ranked[0]["feature_idx"],
            "coefficient": ranked[0]["coefficient"],
            "last_token_only": ranked[0]["last_token_only"],
            "quality_score": ranked[0]["total_quality_score"],
        } if ranked else None,
        "output_dir": str(output_dir),
        "files": [
            "config.json",
            "activations.pt",
            "sae_weights.pt",
            "training_metrics.json",
            "feature_search.json",
            "steering_results.json",
            "report.txt",
            "report_ranked.json",
            "summary.json",
        ],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log("=" * 60)
    log("I-BEE-M PIPELINE RUN 2 COMPLETE!")
    log(f"  Time: {minutes}m {seconds}s")
    log(f"  SAE: {training_result.sae.input_dim} → {training_result.sae.latent_dim}")
    log(f"  Features steered: {feature_indices}")
    log(f"  Total generations: {len(all_steering_results)}")
    if ranked:
        mode = "last-token" if ranked[0]["last_token_only"] else "all-positions"
        log(f"  Best config: Feature {ranked[0]['feature_idx']} "
            f"coeff={ranked[0]['coefficient']} mode={mode} "
            f"(score={ranked[0]['total_quality_score']})")
    log(f"  Report: {output_dir / 'report.txt'}")
    log(f"  Results: {output_dir}")
    log("=" * 60)


if __name__ == "__main__":
    main()
