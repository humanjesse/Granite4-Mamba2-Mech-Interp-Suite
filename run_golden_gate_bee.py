#!/usr/bin/env python3
"""I-Bee-M: Golden Gate Bee — Anthropic-style SAE steering pipeline.

Replicates Anthropic's Golden Gate Claude approach:
1. Train SAE on Layer 27 (last attention layer, ~84% depth)
2. Search for bee features via contrastive activation
3. Validate features via token-level analysis
4. Steer with BOTH methods: vector addition AND encode-clamp-decode
5. Compare results to determine which method produces bee content

Key insight: Previous attempts steered at Layer 4 (Mamba) and Layer 17 (attention,
53% depth) — both produced 0 bee keywords. Layer 27 is the last attention layer,
with only 5 Mamba layers between it and output, giving the shortest path for
the perturbation to affect generation.

Usage:
    source .venv/bin/activate
    python run_golden_gate_bee.py
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
from src.extraction.activation_steering import (
    generate_comparison, generate_clamp_comparison,
)
from run_contrastive_steering import score_generation, TEST_PROMPTS, MAX_TOKENS
from run_feature_analysis import (
    collect_per_token_activations, analyze_feature, determine_verdict,
    TOP_N_TOKENS,
)

# ── Config ───────────────────────────────────────────────────────────────────

LAYER_IDX = 27  # Last attention layer at ~84% depth
OUTPUT_BASE = Path("results/golden_gate_bee")

SAE_CONFIG = SAETrainingConfig(
    layer_idx=LAYER_IDX,
    expansion_factor=8,       # 768 * 8 = 6144 latent features
    l1_coefficient=1e-3,
    learning_rate=3e-4,
    num_activations=50_000,
    batch_size=256,
    num_epochs=3,
)

# Vector addition coefficients (Method A)
VECTOR_COEFFICIENTS = [3.0, 5.0, 7.0, 10.0, 15.0, 20.0]

# SAE clamp values (Method B — Anthropic's approach)
CLAMP_VALUES = [5.0, 10.0, 20.0, 50.0, 100.0]

NUM_FEATURES_TO_ANALYZE = 5


# ── Utilities ────────────────────────────────────────────────────────────────

def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ── Phase 1: Load model ─────────────────────────────────────────────────────

def phase1_load_model():
    log("Loading model...")
    model, tokenizer, device = load_model(device="cuda")
    log(f"Model loaded on {device}. Hidden size: {model.config.hidden_size}")
    log(f"Layer {LAYER_IDX} type: {model.config.layer_types[LAYER_IDX]}")
    return model, tokenizer, device


# ── Phase 2: Collect activations ─────────────────────────────────────────────

def phase2_collect_activations(model, tokenizer, device, output_dir):
    log("")
    log("=" * 60)
    log(f"PHASE 2: Collecting activations from Layer {LAYER_IDX}")
    log("=" * 60)

    def progress(collected, total):
        if collected % 5000 == 0 or collected == total:
            log(f"  Collected {collected}/{total} activations")

    t0 = time.time()
    activations = collect_activations(
        model, tokenizer,
        layer_idx=LAYER_IDX,
        num_activations=SAE_CONFIG.num_activations,
        device=device,
        progress_callback=progress,
    )
    log(f"  Done: {activations.shape} in {time.time() - t0:.1f}s")

    act_stats = {
        "shape": list(activations.shape),
        "mean": activations.mean().item(),
        "std": activations.std().item(),
        "min": activations.min().item(),
        "max": activations.max().item(),
    }
    with open(output_dir / "activation_stats.json", "w") as f:
        json.dump(act_stats, f, indent=2)

    return activations


# ── Phase 3: Train SAE ──────────────────────────────────────────────────────

def phase3_train_sae(activations, device, output_dir):
    log("")
    log("=" * 60)
    log("PHASE 3: Training Sparse Autoencoder")
    log("=" * 60)
    log(f"  Input dim: {activations.shape[1]}")
    log(f"  Latent dim: {activations.shape[1] * SAE_CONFIG.expansion_factor}")
    log(f"  Activations: {activations.shape[0]}")
    log(f"  Epochs: {SAE_CONFIG.num_epochs}")
    log(f"  L1 coefficient: {SAE_CONFIG.l1_coefficient}")

    def progress(step, total):
        if step % 50 == 0 or step == total:
            log(f"  Step {step}/{total} ({100 * step / total:.0f}%)")

    t0 = time.time()
    result = train_sae(activations, SAE_CONFIG, progress_callback=progress, device=device)
    train_time = time.time() - t0

    log(f"  Training complete in {train_time:.1f}s")
    log(f"  Final loss: {result.final_loss:.6f}")
    log(f"  Reconstruction loss: {result.final_reconstruction_loss:.6f}")
    log(f"  L1 loss: {result.final_l1_loss:.6f}")
    log(f"  Mean active features: {result.mean_active_features:.1f}")
    log(f"  Dead features: {result.dead_features}/{result.num_latent_features}")

    torch.save(result.sae.state_dict(), output_dir / "sae_layer17.pt")

    training_metrics = {
        "layer_idx": LAYER_IDX,
        "layer_type": "attention",
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

    with open(output_dir / "loss_history.json", "w") as f:
        json.dump({
            "loss": result.loss_history,
            "reconstruction": result.reconstruction_history,
            "l1": result.l1_history,
        }, f)

    return result


# ── Phase 4: Feature search ─────────────────────────────────────────────────

def phase4_feature_search(sae, model, tokenizer, device, output_dir):
    log("")
    log("=" * 60)
    log("PHASE 4: Searching for bee features")
    log("=" * 60)

    bee_texts = get_default_bee_texts()
    baseline_texts = get_default_baseline_texts()
    log(f"  Bee texts: {len(bee_texts)}")
    log(f"  Baseline texts: {len(baseline_texts)}")

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

    feature_results = {
        "concept": "bee",
        "layer_idx": LAYER_IDX,
        "layer_type": "attention",
        "top_features": search_result.top_features,
        "bee_texts": bee_texts,
        "baseline_texts": baseline_texts,
    }
    with open(output_dir / "feature_search.json", "w") as f:
        json.dump(feature_results, f, indent=2)

    # Save top 5 feature directions
    for feat in search_result.top_features[:5]:
        fidx = feat["feature_idx"]
        fvec = sae.get_feature_direction(fidx)
        torch.save(fvec, output_dir / f"feature_{fidx}_direction.pt")

    return search_result


# ── Phase 5: Token-level validation ──────────────────────────────────────────

def phase5_token_analysis(sae, model, tokenizer, device, top_features, output_dir):
    log("")
    log("=" * 60)
    log("PHASE 5: Token-level feature validation")
    log("=" * 60)

    bee_texts = get_default_bee_texts()
    baseline_texts = get_default_baseline_texts()
    feature_indices = [f["feature_idx"] for f in top_features]

    log(f"  Analyzing {len(feature_indices)} features across "
        f"{len(bee_texts)} bee + {len(baseline_texts)} baseline texts...")

    log("  Processing bee texts...")
    bee_records = collect_per_token_activations(
        sae, model, tokenizer, LAYER_IDX,
        bee_texts, "bee", feature_indices, device,
    )

    log("  Processing baseline texts...")
    base_records = collect_per_token_activations(
        sae, model, tokenizer, LAYER_IDX,
        baseline_texts, "baseline", feature_indices, device,
    )

    all_records = bee_records + base_records

    log("")
    verdicts = []
    for feat in top_features:
        analysis = analyze_feature(all_records, feat["feature_idx"], TOP_N_TOKENS)
        verdict = determine_verdict(analysis)
        cats = analysis["top_n_categories"]
        log(f"  Feature #{feat['feature_idx']}: {verdict}  "
            f"(bee-sem={cats.get('bee-semantic', 0)}, "
            f"struct={cats.get('structural', 0)}, "
            f"content={cats.get('content', 0)})")
        verdicts.append({
            "feature_idx": feat["feature_idx"],
            "selectivity": feat["selectivity_score"],
            "verdict": verdict,
            "top_n_categories": dict(cats),
        })

    with open(output_dir / "feature_verdicts.json", "w") as f:
        json.dump(verdicts, f, indent=2)

    return verdicts


# ── Phase 6: Steering comparison ─────────────────────────────────────────────

def phase6_steering(model, tokenizer, sae, best_feature, device, output_dir):
    log("")
    log("=" * 60)
    log("PHASE 6: Steering comparison — vector addition vs SAE clamping")
    log("=" * 60)

    best_idx = best_feature["feature_idx"]
    steering_vec = sae.get_feature_direction(best_idx).to(device)
    log(f"  Best feature: #{best_idx} (selectivity={best_feature['selectivity_score']:.2f})")
    log(f"  Bee activation: {best_feature['concept_activation']:.4f}")
    log(f"  Steering vector norm: {steering_vec.norm():.4f}")

    # ── 6a: Vector addition ───────────────────────────────────────────────
    log("")
    log("METHOD A: Vector addition (x' = x + alpha * decoder_column)")
    vector_results = []
    gen_count = 0
    total_gens = len(VECTOR_COEFFICIENTS) * len(TEST_PROMPTS)

    for coeff in VECTOR_COEFFICIENTS:
        log(f"  coeff={coeff}")
        for prompt in TEST_PROMPTS:
            gen_count += 1
            result = generate_comparison(
                model, tokenizer, prompt, steering_vec,
                layer_idx=LAYER_IDX,
                coefficient=coeff,
                max_tokens=MAX_TOKENS,
                device=device,
            )
            scoring = score_generation(result.steered_text, result.unsteered_text)
            vector_results.append({
                "method": "vector_addition",
                "prompt": prompt,
                "steered_text": result.steered_text,
                "unsteered_text": result.unsteered_text,
                "coefficient": coeff,
                "feature_idx": best_idx,
                **scoring,
            })
        log(f"    {gen_count}/{total_gens} done")

    with open(output_dir / "steering_vector_results.json", "w") as f:
        json.dump(vector_results, f, indent=2)

    # ── 6b: SAE clamping ─────────────────────────────────────────────────
    log("")
    log("METHOD B: SAE clamping (encode -> clamp -> decode -> replace)")
    clamp_results = []
    gen_count = 0
    total_gens = len(CLAMP_VALUES) * len(TEST_PROMPTS)

    for clamp_val in CLAMP_VALUES:
        log(f"  clamp={clamp_val}")
        for prompt in TEST_PROMPTS:
            gen_count += 1
            result = generate_clamp_comparison(
                model, tokenizer, prompt, sae,
                layer_idx=LAYER_IDX,
                feature_idx=best_idx,
                clamp_value=clamp_val,
                max_tokens=MAX_TOKENS,
                device=device,
            )
            scoring = score_generation(result.steered_text, result.unsteered_text)
            clamp_results.append({
                "method": "sae_clamp",
                "prompt": prompt,
                "steered_text": result.steered_text,
                "unsteered_text": result.unsteered_text,
                "clamp_value": clamp_val,
                "feature_idx": best_idx,
                **scoring,
            })
        log(f"    {gen_count}/{total_gens} done")

    with open(output_dir / "steering_clamp_results.json", "w") as f:
        json.dump(clamp_results, f, indent=2)

    return vector_results, clamp_results


# ── Phase 7: Report ──────────────────────────────────────────────────────────

def phase7_report(sae_result, search_result, verdicts,
                  vector_results, clamp_results, output_dir):
    log("")
    log("=" * 60)
    log("PHASE 7: Generating report")
    log("=" * 60)

    best_feature = search_result.top_features[0]
    best_idx = best_feature["feature_idx"]

    # Rank vector addition configs
    vec_configs = {}
    for r in vector_results:
        key = r["coefficient"]
        if key not in vec_configs:
            vec_configs[key] = {"coefficient": key, "results": []}
        vec_configs[key]["results"].append(r)

    vec_ranked = []
    for key, cfg in vec_configs.items():
        total_score = sum(r["quality_score"] for r in cfg["results"])
        total_bee = sum(r["bee_keyword_count"] for r in cfg["results"])
        total_rep = sum(1 for r in cfg["results"] if r["has_repetition"])
        total_diff = sum(1 for r in cfg["results"] if r["differs_from_unsteered"])
        vec_ranked.append({
            "coefficient": cfg["coefficient"],
            "total_quality_score": total_score,
            "total_bee_keywords": total_bee,
            "repetition_count": total_rep,
            "differs_count": total_diff,
            "num_prompts": len(cfg["results"]),
            "details": cfg["results"],
        })
    vec_ranked.sort(key=lambda x: x["total_quality_score"], reverse=True)

    # Rank clamp configs
    clamp_configs = {}
    for r in clamp_results:
        key = r["clamp_value"]
        if key not in clamp_configs:
            clamp_configs[key] = {"clamp_value": key, "results": []}
        clamp_configs[key]["results"].append(r)

    clamp_ranked = []
    for key, cfg in clamp_configs.items():
        total_score = sum(r["quality_score"] for r in cfg["results"])
        total_bee = sum(r["bee_keyword_count"] for r in cfg["results"])
        total_rep = sum(1 for r in cfg["results"] if r["has_repetition"])
        total_diff = sum(1 for r in cfg["results"] if r["differs_from_unsteered"])
        clamp_ranked.append({
            "clamp_value": cfg["clamp_value"],
            "total_quality_score": total_score,
            "total_bee_keywords": total_bee,
            "repetition_count": total_rep,
            "differs_count": total_diff,
            "num_prompts": len(cfg["results"]),
            "details": cfg["results"],
        })
    clamp_ranked.sort(key=lambda x: x["total_quality_score"], reverse=True)

    # ── Build report ──────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("I-BEE-M GOLDEN GATE BEE REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Approach: Anthropic Golden Gate Claude method at Layer 17")
    lines.append(f"Layer:              {LAYER_IDX} (attention)")
    lines.append(f"SAE latent dim:     {sae_result.sae.latent_dim}")
    lines.append(f"Training samples:   {SAE_CONFIG.num_activations}")
    lines.append(f"Training epochs:    {SAE_CONFIG.num_epochs}")
    lines.append(f"Final recon loss:   {sae_result.final_reconstruction_loss:.6f}")
    lines.append(f"Dead features:      {sae_result.dead_features}/{sae_result.num_latent_features}")
    lines.append(f"Mean active:        {sae_result.mean_active_features:.1f}")
    lines.append("")

    # Top 20 features
    lines.append("TOP 20 BEE-SELECTIVE FEATURES:")
    lines.append(f"  {'Rank':>4}  {'Feature':>8}  {'Bee Act':>10}  {'Base Act':>10}  {'Selectivity':>12}")
    for i, feat in enumerate(search_result.top_features):
        lines.append(f"  {i+1:4d}  {feat['feature_idx']:8d}  "
                     f"{feat['concept_activation']:10.4f}  "
                     f"{feat['baseline_activation']:10.4f}  "
                     f"{feat['selectivity_score']:12.2f}")
    lines.append("")

    # Token-level verdicts
    lines.append("TOKEN-LEVEL FEATURE VALIDATION:")
    for v in verdicts:
        cats = v["top_n_categories"]
        lines.append(f"  #{v['feature_idx']:>4d} (sel={v['selectivity']:.2f}): "
                     f"{v['verdict']}  "
                     f"[bee-sem={cats.get('bee-semantic', 0)} "
                     f"struct={cats.get('structural', 0)} "
                     f"content={cats.get('content', 0)}]")
    lines.append("")

    # Method A: Vector addition
    lines.append("=" * 70)
    lines.append(f"METHOD A: VECTOR ADDITION (feature #{best_idx})")
    lines.append("=" * 70)
    lines.append("")
    for i, cfg in enumerate(vec_ranked):
        lines.append(
            f"  #{i+1:2d}  coeff={cfg['coefficient']:5.1f}  "
            f"score={cfg['total_quality_score']:3d}  "
            f"bee_kw={cfg['total_bee_keywords']}  "
            f"rep={cfg['repetition_count']}/{cfg['num_prompts']}  "
            f"diff={cfg['differs_count']}/{cfg['num_prompts']}"
        )
    lines.append("")

    # Method B: SAE clamping
    lines.append("=" * 70)
    lines.append(f"METHOD B: SAE CLAMPING (feature #{best_idx})")
    lines.append("=" * 70)
    lines.append("")
    for i, cfg in enumerate(clamp_ranked):
        lines.append(
            f"  #{i+1:2d}  clamp={cfg['clamp_value']:6.1f}  "
            f"score={cfg['total_quality_score']:3d}  "
            f"bee_kw={cfg['total_bee_keywords']}  "
            f"rep={cfg['repetition_count']}/{cfg['num_prompts']}  "
            f"diff={cfg['differs_count']}/{cfg['num_prompts']}"
        )
    lines.append("")

    # Head-to-head
    lines.append("=" * 70)
    lines.append("HEAD-TO-HEAD COMPARISON")
    lines.append("=" * 70)
    lines.append("")
    best_vec = vec_ranked[0] if vec_ranked else None
    best_clamp = clamp_ranked[0] if clamp_ranked else None

    if best_vec:
        lines.append(f"Best vector addition: coeff={best_vec['coefficient']}  "
                     f"score={best_vec['total_quality_score']}  "
                     f"bee_kw={best_vec['total_bee_keywords']}")
    if best_clamp:
        lines.append(f"Best SAE clamping:    clamp={best_clamp['clamp_value']}  "
                     f"score={best_clamp['total_quality_score']}  "
                     f"bee_kw={best_clamp['total_bee_keywords']}")
    lines.append("")

    if best_vec and best_clamp:
        if best_clamp["total_quality_score"] > best_vec["total_quality_score"]:
            winner = "SAE CLAMPING (Anthropic method)"
        elif best_vec["total_quality_score"] > best_clamp["total_quality_score"]:
            winner = "VECTOR ADDITION"
        else:
            winner = "TIE"
        lines.append(f"Winner: {winner}")
    lines.append("")

    # Detailed output — best config from each method
    for method_label, ranked in [("VECTOR ADDITION", vec_ranked),
                                  ("SAE CLAMPING", clamp_ranked)]:
        if not ranked:
            continue
        best = ranked[0]
        lines.append("=" * 70)
        lines.append(f"BEST {method_label} — DETAILED OUTPUT")
        lines.append("=" * 70)
        lines.append("")
        for r in best["details"]:
            lines.append(f"  Prompt: \"{r['prompt']}\"")
            lines.append(f"  Steered:   {r['steered_text'][:300]}")
            lines.append(f"  Unsteered: {r['unsteered_text'][:300]}")
            lines.append(f"  [bee_kw={r['bee_keyword_count']} "
                         f"rep={r['has_repetition']} "
                         f"diff={r['differs_from_unsteered']} "
                         f"score={r['quality_score']}]")
            lines.append("")

    # Overall conclusion
    lines.append("=" * 70)
    lines.append("CONCLUSION")
    lines.append("=" * 70)
    lines.append("")

    total_bee_vec = sum(r["bee_keyword_count"] for r in vector_results)
    total_bee_clamp = sum(r["bee_keyword_count"] for r in clamp_results)
    any_bee = total_bee_vec + total_bee_clamp

    if any_bee > 0:
        lines.append("SUCCESS: Bee content was generated via activation steering!")
        lines.append(f"  Vector addition: {total_bee_vec} total bee keywords")
        lines.append(f"  SAE clamping:    {total_bee_clamp} total bee keywords")
        lines.append("")
        lines.append("The Golden Gate Claude approach works: training an SAE on an")
        lines.append("attention layer (Layer 17) and steering there produces bee")
        lines.append("content, confirming that the Layer 4 Mamba failure was due")
        lines.append("to the layer type, not the feature quality.")
    else:
        lines.append("Neither method produced bee keywords at Layer 17.")
        lines.append("Possible next steps:")
        lines.append("  - Try Layer 27 (last attention layer, closer to output)")
        lines.append("  - Increase clamp values / coefficients further")
        lines.append("  - Try steering at multiple attention layers simultaneously")
        lines.append("  - Check if the model's hybrid architecture fundamentally")
        lines.append("    resists single-layer steering interventions")
    lines.append("")

    report_text = "\n".join(lines)
    with open(output_dir / "report.txt", "w") as f:
        f.write(report_text)

    return report_text


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")
    log(f"Target layer: {LAYER_IDX} (attention)")

    # Phase 1: Load model
    model, tokenizer, device = phase1_load_model()

    # Phase 2: Collect activations from Layer 17
    activations = phase2_collect_activations(model, tokenizer, device, output_dir)

    # Phase 3: Train SAE
    sae_result = phase3_train_sae(activations, device, output_dir)
    sae = sae_result.sae.to(device).eval()

    # Phase 4: Feature search
    search_result = phase4_feature_search(sae, model, tokenizer, device, output_dir)

    # Phase 5: Token-level validation
    top_features = search_result.top_features[:NUM_FEATURES_TO_ANALYZE]
    verdicts = phase5_token_analysis(
        sae, model, tokenizer, device, top_features, output_dir,
    )

    # Phase 6: Steering comparison
    best_feature = search_result.top_features[0]
    vector_results, clamp_results = phase6_steering(
        model, tokenizer, sae, best_feature, device, output_dir,
    )

    # Phase 7: Report
    phase7_report(
        sae_result, search_result, verdicts,
        vector_results, clamp_results, output_dir,
    )

    # Summary
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    total_bee_vec = sum(r["bee_keyword_count"] for r in vector_results)
    total_bee_clamp = sum(r["bee_keyword_count"] for r in clamp_results)

    summary = {
        "run_id": run_id,
        "elapsed_seconds": elapsed,
        "elapsed_human": f"{minutes}m {seconds}s",
        "approach": "golden_gate_bee",
        "layer_idx": LAYER_IDX,
        "layer_type": "attention",
        "sae_latent_dim": sae_result.sae.latent_dim,
        "best_bee_feature": search_result.top_features[0],
        "feature_verdicts": verdicts,
        "vector_addition_total_bee_keywords": total_bee_vec,
        "sae_clamp_total_bee_keywords": total_bee_clamp,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log("")
    log("=" * 60)
    log("GOLDEN GATE BEE COMPLETE!")
    log(f"  Time: {minutes}m {seconds}s")
    log(f"  Best feature: #{best_feature['feature_idx']} "
        f"(selectivity={best_feature['selectivity_score']:.2f})")
    log(f"  Vector addition bee keywords: {total_bee_vec}")
    log(f"  SAE clamping bee keywords:    {total_bee_clamp}")
    log(f"  Report: {output_dir / 'report.txt'}")
    log("=" * 60)


if __name__ == "__main__":
    main()
