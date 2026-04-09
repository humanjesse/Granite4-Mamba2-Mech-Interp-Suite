#!/usr/bin/env python3
"""I-Bee-M: Norm-Scaled Multi-Layer Steering — Follow-up Experiment.

The first multi-layer experiment (run 20260219_214401) produced 0 bee keywords
across all 80 generations. Code audit confirmed the implementation was correct
but coefficients were too weak relative to residual stream magnitudes.

Measured residual stream norms at each attention layer:
  L10: ~127.5  (coeff=15 was only 11.8% of stream)
  L13: ~21.9   (coeff=15 was 68.6% of stream)
  L17: ~40.5   (coeff=15 was 37.0% of stream)
  L27: ~65.5   (coeff=15 was 22.9% of stream)

Golden Gate Claude used perturbations of ~50-200% of stream norm.

This script uses norm-scaled coefficients that target a consistent percentage
of stream magnitude at each layer, ensuring all layers receive proportionally
equivalent perturbations.

Usage:
    source .venv/bin/activate
    python run_normscaled_steering.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import torch

from src.model_loader import load_model
from src.sae.sparse_autoencoder import SparseAutoencoder
from src.extraction.activation_steering import generate_multi_layer_comparison
from run_contrastive_steering import score_generation, TEST_PROMPTS, MAX_TOKENS

# -- Config ---------------------------------------------------------------

ATTENTION_LAYERS = [10, 13, 17, 27]
OUTPUT_BASE = Path("results/normscaled_steering")

# All 4 SAEs from prior runs (no retraining needed)
SAE_SOURCES = {
    10: {
        "sae_path": "results/multilayer_golden_gate_bee/20260219_214401/sae_layer10.pt",
        "feature_idx": 611,
        "selectivity": 104.0,
    },
    13: {
        "sae_path": "results/multilayer_golden_gate_bee/20260219_214401/sae_layer13.pt",
        "feature_idx": 1351,
        "selectivity": 6000.0,
    },
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

# SAE expansion factor (needed to reconstruct latent_dim)
SAE_EXPANSION_FACTOR = 8

# Target percentages of stream norm for coefficient scaling
TARGET_PCTS = {
    "norm_25pct":          {10: 0.25, 13: 0.25, 17: 0.25, 27: 0.25},
    "norm_50pct":          {10: 0.50, 13: 0.50, 17: 0.50, 27: 0.50},
    "norm_75pct":          {10: 0.75, 13: 0.75, 17: 0.75, 27: 0.75},
    "norm_100pct":         {10: 1.00, 13: 1.00, 17: 1.00, 27: 1.00},
    "norm_150pct":         {10: 1.50, 13: 1.50, 17: 1.50, 27: 1.50},
    "norm_50_deep_only":   {10: 0.00, 13: 0.00, 17: 0.50, 27: 0.50},
    "norm_100_deep_only":  {10: 0.00, 13: 0.00, 17: 1.00, 27: 1.00},
}

# Sample texts for measuring residual stream norms
NORM_SAMPLE_TEXTS = [
    "The weather today is warm and sunny with clear skies.",
    "Scientists recently discovered a new species of butterfly.",
    "The stock market experienced significant volatility this quarter.",
    "I went to the grocery store to buy some fresh vegetables.",
    "The history of ancient Rome is full of fascinating stories.",
    "My favorite hobby is reading books about science and nature.",
    "The recipe calls for two cups of flour and one egg.",
    "In the garden, I noticed several colorful flowers blooming.",
]


# -- Utilities ------------------------------------------------------------

def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# -- Phase 1: Load model --------------------------------------------------

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


# -- Phase 2: Load all 4 SAEs --------------------------------------------

def phase2_load_saes(model, device):
    log("")
    log("=" * 60)
    log("PHASE 2: Loading SAEs for all 4 attention layers")
    log("=" * 60)

    hidden_size = model.config.hidden_size
    latent_dim = hidden_size * SAE_EXPANSION_FACTOR
    layer_data = {}

    for layer_idx in ATTENTION_LAYERS:
        info = SAE_SOURCES[layer_idx]
        log(f"  L{layer_idx}: Loading {info['sae_path']}")

        sae = SparseAutoencoder(hidden_size, latent_dim)
        sae.load_state_dict(torch.load(info["sae_path"], map_location="cpu", weights_only=True))
        sae = sae.to(device).eval()

        feature_idx = info["feature_idx"]
        steering_vec = sae.get_feature_direction(feature_idx)

        log(f"    Feature #{feature_idx} (selectivity={info['selectivity']:.0f}x)")
        log(f"    Steering vector norm: {steering_vec.norm():.4f}")

        layer_data[layer_idx] = {
            "sae": sae,
            "feature_idx": feature_idx,
            "selectivity": info["selectivity"],
            "steering_vector": steering_vec,
        }

    return layer_data


# -- Phase 2b: Measure residual stream norms ------------------------------

def phase2b_measure_norms(model, tokenizer, device):
    log("")
    log("=" * 60)
    log("PHASE 2b: Measuring residual stream norms at attention layers")
    log("=" * 60)
    log(f"  Using {len(NORM_SAMPLE_TEXTS)} sample texts")

    model_dtype = next(model.parameters()).dtype
    stream_norms = {layer_idx: [] for layer_idx in ATTENTION_LAYERS}

    for i, text in enumerate(NORM_SAMPLE_TEXTS):
        input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device)

        # Hook each attention layer to capture hidden state norms
        captured = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # Mean norm across all positions in the sequence
                norms = hidden.float().norm(dim=-1)  # [batch, seq_len]
                captured[layer_idx] = norms.mean().item()
            return hook_fn

        hooks = []
        for layer_idx in ATTENTION_LAYERS:
            layer = model.model.layers[layer_idx]
            hooks.append(layer.register_forward_hook(make_hook(layer_idx)))

        try:
            with torch.no_grad():
                model(input_ids=input_ids)
        finally:
            for h in hooks:
                h.remove()

        for layer_idx in ATTENTION_LAYERS:
            if layer_idx in captured:
                stream_norms[layer_idx].append(captured[layer_idx])

        if (i + 1) % 4 == 0:
            log(f"    Processed {i + 1}/{len(NORM_SAMPLE_TEXTS)} texts")

    # Average across all samples
    avg_norms = {}
    for layer_idx in ATTENTION_LAYERS:
        vals = stream_norms[layer_idx]
        avg_norms[layer_idx] = sum(vals) / len(vals) if vals else 0.0

    log("")
    log("  Measured residual stream norms:")
    for layer_idx in ATTENTION_LAYERS:
        log(f"    L{layer_idx}: {avg_norms[layer_idx]:.1f}")

    return avg_norms


# -- Phase 3: Norm-scaled steering ----------------------------------------

def phase3_steering(model, tokenizer, layer_data, stream_norms, device, output_dir):
    log("")
    log("=" * 60)
    log("PHASE 3: Norm-scaled multi-layer steering")
    log("=" * 60)

    # Compute actual coefficients from target percentages
    coeff_configs = {}
    log("")
    log("  Coefficient matrix (target_pct * stream_norm):")
    for config_name, pcts in TARGET_PCTS.items():
        coeffs = {}
        parts = []
        for layer_idx in ATTENTION_LAYERS:
            coeff = pcts[layer_idx] * stream_norms[layer_idx]
            coeffs[layer_idx] = round(coeff, 1)
            parts.append(f"L{layer_idx}={coeffs[layer_idx]:.1f}")
        coeff_configs[config_name] = coeffs
        pct_str = " ".join(f"L{l}={pcts[l]*100:.0f}%" for l in ATTENTION_LAYERS)
        log(f"    {config_name:25s}  {pct_str}  ->  {' '.join(parts)}")

    # Save coefficient configs
    coeff_save = {
        name: {str(l): c for l, c in coeffs.items()}
        for name, coeffs in coeff_configs.items()
    }
    with open(output_dir / "coeff_configs.json", "w") as f:
        json.dump(coeff_save, f, indent=2)

    num_configs = len(coeff_configs)
    num_prompts = len(TEST_PROMPTS)
    total_gens = num_configs * num_prompts
    log("")
    log(f"  {num_configs} configs x {num_prompts} prompts = {total_gens} steered generations")
    log(f"  + {num_prompts} unsteered baselines (cached and reused)")

    # Generate unsteered baselines first
    log("")
    log("  Generating unsteered baselines...")
    unsteered_cache = {}
    for i, prompt in enumerate(TEST_PROMPTS):
        generate_multi_layer_comparison(
            model, tokenizer, prompt,
            layer_steering=[],
            max_tokens=MAX_TOKENS,
            device=device,
            unsteered_cache=unsteered_cache,
        )
        log(f"    Baseline {i + 1}/{num_prompts}: \"{prompt[:40]}...\"")

    all_results = []
    gen_count = 0

    for config_name, coeffs in coeff_configs.items():
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
                "coefficients": {str(l): c for l, c in coeffs.items()},
                "target_pcts": {str(l): p for l, p in TARGET_PCTS[config_name].items()},
                "active_layers": active_layers,
                "prompt": prompt,
                "steered_text": result.steered_text,
                "unsteered_text": result.unsteered_text,
                **scoring,
            })

        log(f"    {gen_count}/{total_gens} done")

    # Save raw results
    with open(output_dir / "steering_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results, coeff_configs


# -- Phase 4: Report ------------------------------------------------------

def phase4_report(layer_data, stream_norms, all_results, coeff_configs, output_dir):
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
                "target_pcts": r["target_pcts"],
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
            "target_pcts": cfg["target_pcts"],
            "active_layers": cfg["active_layers"],
            "total_quality_score": total_score,
            "total_bee_keywords": total_bee,
            "repetition_count": total_rep,
            "differs_count": total_diff,
            "num_prompts": len(cfg["results"]),
            "details": cfg["results"],
        })
    ranked.sort(key=lambda x: x["total_quality_score"], reverse=True)

    # -- Build report text ------------------------------------------------
    lines = []
    lines.append("=" * 70)
    lines.append("I-BEE-M NORM-SCALED MULTI-LAYER STEERING REPORT")
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
                     f"(selectivity={d['selectivity']:.0f}x)")
    lines.append("")

    # Stream norms
    lines.append("RESIDUAL STREAM NORMS (measured):")
    for layer_idx in ATTENTION_LAYERS:
        lines.append(f"  L{layer_idx}: {stream_norms[layer_idx]:.1f}")
    lines.append("")

    # Motivation
    lines.append("MOTIVATION:")
    lines.append("  First multi-layer experiment (run 20260219_214401) used uniform")
    lines.append("  coefficients up to 15.0, but L10 stream norm is ~127.5, so")
    lines.append("  coeff=15 was only 11.8% of stream — far below the 50-200%")
    lines.append("  perturbation used in Golden Gate Claude.")
    lines.append("")
    lines.append("  This experiment uses norm-scaled coefficients:")
    lines.append("    actual_coeff = target_percentage * stream_norm[layer]")
    lines.append("  ensuring proportionally equivalent perturbations at every layer.")
    lines.append("")

    # Prior results
    lines.append("PRIOR RESULTS:")
    lines.append("  Single-layer L17 (coeff=15): 0 bee keywords")
    lines.append("  Single-layer L27 (coeff=15): 2 bee keywords (marginal)")
    lines.append("  Multi-layer uniform_15 (first run): 0 bee keywords")
    lines.append("")

    # Coefficient matrix
    lines.append("COEFFICIENT MATRIX:")
    lines.append(f"  {'Config':<25s}  {'L10':>8s}  {'L13':>8s}  {'L17':>8s}  {'L27':>8s}")
    lines.append("  " + "-" * 60)
    for config_name in TARGET_PCTS:
        coeffs = coeff_configs[config_name]
        parts = []
        for l in ATTENTION_LAYERS:
            pct = TARGET_PCTS[config_name][l]
            c = coeffs[l]
            parts.append(f"{c:>5.1f}({pct*100:>3.0f}%)")
        lines.append(f"  {config_name:<25s}  {'  '.join(parts)}")
    lines.append("")

    # Ranking
    lines.append("=" * 70)
    lines.append("RESULTS — RANKED BY QUALITY SCORE")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Configurations tested: {len(ranked)}")
    lines.append(f"Prompts per config: {len(TEST_PROMPTS)}")
    lines.append(f"Total steered generations: {len(all_results)}")
    lines.append(f"Max tokens per generation: {MAX_TOKENS}")
    lines.append("")

    for i, cfg in enumerate(ranked):
        coeffs = cfg["coefficients"]
        coeff_str = " ".join(f"L{l}={coeffs[str(l)]}" for l in ATTENTION_LAYERS)
        lines.append(
            f"#{i+1:2d}  {cfg['config_name']:25s}  [{coeff_str}]  "
            f"score={cfg['total_quality_score']:4d}  "
            f"bee_kw={cfg['total_bee_keywords']:2d}  "
            f"rep={cfg['repetition_count']}/{cfg['num_prompts']}  "
            f"diff={cfg['differs_count']}/{cfg['num_prompts']}"
        )
    lines.append("")

    # Comparison: norm-scaled vs prior experiments
    lines.append("=" * 70)
    lines.append("NORM-SCALED vs PRIOR EXPERIMENTS")
    lines.append("=" * 70)
    lines.append("")

    best = ranked[0]
    total_norm_bee = sum(cfg["total_bee_keywords"] for cfg in ranked)
    best_norm_bee = best["total_bee_keywords"]

    lines.append(f"Best norm-scaled config: {best['config_name']}")
    lines.append(f"  Bee keywords: {best_norm_bee} (across {best['num_prompts']} prompts)")
    lines.append(f"  Quality score: {best['total_quality_score']}")
    lines.append(f"  Repetition: {best['repetition_count']}/{best['num_prompts']} prompts")
    lines.append("")
    lines.append(f"Total bee keywords across ALL norm-scaled configs: {total_norm_bee}")
    lines.append(f"Prior best single-layer (L27, coeff=15): 2 bee keywords")
    lines.append(f"Prior multi-layer uniform_15: 0 bee keywords")
    lines.append("")

    # deep_only vs all-layers comparison
    deep_50 = next((c for c in ranked if c["config_name"] == "norm_50_deep_only"), None)
    deep_100 = next((c for c in ranked if c["config_name"] == "norm_100_deep_only"), None)
    norm_50 = next((c for c in ranked if c["config_name"] == "norm_50pct"), None)
    norm_100 = next((c for c in ranked if c["config_name"] == "norm_100pct"), None)

    if deep_50 and norm_50:
        lines.append("DEEP-ONLY (L17+L27) vs ALL 4 LAYERS at 50%:")
        lines.append(f"  norm_50_deep_only:  score={deep_50['total_quality_score']}  "
                     f"bee_kw={deep_50['total_bee_keywords']}")
        lines.append(f"  norm_50pct (all 4): score={norm_50['total_quality_score']}  "
                     f"bee_kw={norm_50['total_bee_keywords']}")
        lines.append("")

    if deep_100 and norm_100:
        lines.append("DEEP-ONLY (L17+L27) vs ALL 4 LAYERS at 100%:")
        lines.append(f"  norm_100_deep_only:  score={deep_100['total_quality_score']}  "
                     f"bee_kw={deep_100['total_bee_keywords']}")
        lines.append(f"  norm_100pct (all 4): score={norm_100['total_quality_score']}  "
                     f"bee_kw={norm_100['total_bee_keywords']}")
        lines.append("")

    # Coherence threshold analysis
    lines.append("COHERENCE vs STRENGTH:")
    for cfg in ranked:
        pcts = cfg["target_pcts"]
        # Use the first non-zero layer's target pct as the label
        pct_label = next(
            (f"{p*100:.0f}%" for l, p in sorted(pcts.items()) if p > 0),
            "0%"
        )
        lines.append(
            f"  {cfg['config_name']:25s}  target={pct_label:>5s}  "
            f"bee_kw={cfg['total_bee_keywords']:2d}  "
            f"rep={cfg['repetition_count']}/{cfg['num_prompts']}  "
            f"diff={cfg['differs_count']}/{cfg['num_prompts']}"
        )
    lines.append("")

    # Detailed output for top 3 configs
    lines.append("=" * 70)
    lines.append("DETAILED OUTPUT: TOP 3 CONFIGURATIONS")
    lines.append("=" * 70)

    for i, cfg in enumerate(ranked[:3]):
        coeffs = cfg["coefficients"]
        coeff_str = " ".join(f"L{l}={coeffs[str(l)]}" for l in ATTENTION_LAYERS)
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

    if total_norm_bee > 10:
        lines.append("BREAKTHROUGH: Norm-scaled coefficients produce significant bee keywords!")
        lines.append("The key insight is that steering strength must be proportional to")
        lines.append("residual stream magnitude. The first multi-layer experiment failed")
        lines.append("because coefficients were too small relative to stream norms,")
        lines.append("especially at L10 (127.5 norm vs coeff=15).")
    elif total_norm_bee > 2:
        lines.append("PARTIAL SUCCESS: Norm-scaled coefficients produce more bee keywords")
        lines.append("than the first multi-layer experiment (0 keywords) and match or")
        lines.append("exceed single-layer L27 (2 keywords). Norm-scaling is the right")
        lines.append("direction but may need further tuning.")
    elif total_norm_bee > 0:
        lines.append("MARGINAL IMPROVEMENT: Some bee keywords detected with norm-scaled")
        lines.append("coefficients, but not dramatically more than prior experiments.")
        lines.append("The hybrid Mamba-Transformer architecture may require even stronger")
        lines.append("perturbations or fundamentally different steering approaches.")
    else:
        lines.append("NEGATIVE RESULT: Even with norm-scaled coefficients targeting up to")
        lines.append("150% of stream norm, no bee keywords were produced.")
        lines.append("")
        lines.append("This is a strong negative result suggesting that:")
        lines.append("  1. Mamba layers actively counteract steering at attention layers,")
        lines.append("     regardless of perturbation strength")
        lines.append("  2. The SAE features, while selective in analysis, don't correspond")
        lines.append("     to generation-time bee representations")
        lines.append("  3. The hybrid architecture may need Mamba-layer steering (not just")
        lines.append("     attention-layer steering) to achieve behavioral modification")
        lines.append("  4. A different steering paradigm (e.g., representation engineering,")
        lines.append("     prompt-based activation injection) may be needed")
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


# -- Main -----------------------------------------------------------------

def main():
    start_time = time.time()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")
    log(f"Attention layers: {ATTENTION_LAYERS}")
    log(f"Target % configs: {len(TARGET_PCTS)}")
    log(f"Test prompts: {len(TEST_PROMPTS)}")

    # Save config
    config = {
        "attention_layers": ATTENTION_LAYERS,
        "sae_sources": {str(k): v for k, v in SAE_SOURCES.items()},
        "sae_expansion_factor": SAE_EXPANSION_FACTOR,
        "target_pcts": {
            name: {str(l): p for l, p in pcts.items()}
            for name, pcts in TARGET_PCTS.items()
        },
        "norm_sample_texts": NORM_SAMPLE_TEXTS,
        "test_prompts": TEST_PROMPTS,
        "max_tokens": MAX_TOKENS,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Phase 1: Load model
    model, tokenizer, device = phase1_load_model()

    # Phase 2: Load all 4 SAEs
    layer_data = phase2_load_saes(model, device)

    # Phase 2b: Measure residual stream norms
    stream_norms = phase2b_measure_norms(model, tokenizer, device)

    # Save layer summary with norms
    layer_summary = {}
    for layer_idx in ATTENTION_LAYERS:
        d = layer_data[layer_idx]
        layer_summary[str(layer_idx)] = {
            "feature_idx": d["feature_idx"],
            "selectivity": d["selectivity"],
            "steering_vector_norm": d["steering_vector"].norm().item(),
            "stream_norm": stream_norms[layer_idx],
        }
    with open(output_dir / "layer_summary.json", "w") as f:
        json.dump(layer_summary, f, indent=2)

    # Phase 3: Norm-scaled steering
    all_results, coeff_configs = phase3_steering(
        model, tokenizer, layer_data, stream_norms, device, output_dir,
    )

    # Phase 4: Report
    ranked = phase4_report(layer_data, stream_norms, all_results, coeff_configs, output_dir)

    # Log top 5
    log("")
    log("TOP NORM-SCALED CONFIGURATIONS:")
    for i, cfg in enumerate(ranked[:5]):
        coeffs = cfg["coefficients"]
        coeff_str = " ".join(f"L{l}={coeffs[str(l)]}" for l in ATTENTION_LAYERS)
        log(f"  #{i+1}: {cfg['config_name']:25s}  [{coeff_str}]  "
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
        "approach": "normscaled_multilayer_steering",
        "attention_layers": ATTENTION_LAYERS,
        "stream_norms": {str(l): n for l, n in stream_norms.items()},
        "num_configs": len(TARGET_PCTS),
        "num_prompts": len(TEST_PROMPTS),
        "total_generations": len(all_results),
        "total_bee_keywords_all_configs": total_bee,
        "best_config": {
            "name": ranked[0]["config_name"],
            "coefficients": ranked[0]["coefficients"],
            "target_pcts": ranked[0]["target_pcts"],
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
    log("NORM-SCALED MULTI-LAYER STEERING COMPLETE!")
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
