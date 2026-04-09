#!/usr/bin/env python3
"""dt Steering: Mamba-Specific Timestep Manipulation.

Steers Granite 4.0 by modifying the dt (timestep/delta) parameter in Mamba layers.
dt controls how much each token updates the SSM hidden state — it's Mamba's
analog of attention strength.

Usage:
    source .venv/bin/activate
    python run_dt_steering.py
"""

import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

# ── Reuse from contrastive steering ──────────────────────────────────────────

MINIMAL_PAIRS = [
    ("The bee buzzed loudly", "The car buzzed loudly"),
    ("The bee landed on the flower", "The bird landed on the flower"),
    ("The bee was in the garden", "The cat was in the garden"),
    ("I saw a bee today", "I saw a dog today"),
    ("There is a bee here", "There is a rock here"),
    ("Look at the bee", "Look at the fish"),
    ("A bee appeared suddenly", "A fox appeared suddenly"),
    ("The bee flew past", "The fly flew past"),
    ("The bee was tiny", "The ant was tiny"),
    ("The bee is important", "The wasp is important"),
]

EXTRA_BEE_PROMPTS = [
    "The bee collected pollen",
    "A bee flew by the window",
    "The bee returned to the hive",
    "I heard a bee nearby",
    "The bee sat on the leaf",
    "A small bee hovered above",
    "The bee moved between flowers",
    "One bee escaped the jar",
    "The bee circled the tree",
    "A golden bee appeared",
    "The bee rested on my hand",
    "That bee is very fast",
    "The bee found the nectar",
    "A lone bee crossed the field",
    "The bee hummed softly",
]

EXTRA_CONTROL_PROMPTS = [
    "The ball rolled down the hill",
    "A cloud drifted past slowly",
    "The child ran across the yard",
    "I heard a bell nearby",
    "The leaf fell from the tree",
    "A small bird perched above",
    "The ship moved between waves",
    "One coin escaped the bag",
    "The kite circled the park",
    "A golden ring appeared",
    "The snow rested on the roof",
    "That car is very fast",
    "The deer found the stream",
    "A lone wolf crossed the field",
    "The wind hummed softly",
]

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

# ── Config ───────────────────────────────────────────────────────────────────

# Top-5 layers from circuit tracing (most important)
TARGET_LAYERS = [0, 1, 6, 9, 11]

COEFFICIENTS = [1.0, 5.0, 10.0, 20.0, 50.0]

MAX_TOKENS = 100


# ── Utilities ────────────────────────────────────────────────────────────────

def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def count_bee_keywords(text: str) -> int:
    text_lower = text.lower()
    count = 0
    for kw in BEE_KEYWORDS:
        count += len(re.findall(r'\b' + re.escape(kw) + r'\b', text_lower))
    return count


def detect_repetition(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 9:
        return False
    for n in range(3, min(8, len(words) // 3 + 1)):
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        if counts.most_common(1)[0][1] >= 3:
            return True
    return False


# ── Phases ───────────────────────────────────────────────────────────────────

def run_phase1(device):
    """Load model, tokenizer, arch_map."""
    log("=" * 60)
    log("PHASE 1: Loading Model & Architecture")
    log("=" * 60)

    from src.model_loader import load_model
    model, tokenizer, device = load_model(device=device)

    from src.architecture_map import ArchitectureMap
    arch_map = ArchitectureMap(model.config)

    log(f"Architecture: {arch_map.summary()}")
    log(f"Mamba num_heads: {model.config.mamba_n_heads}")
    log(f"Device: {device}")

    return model, tokenizer, arch_map, device


def run_phase2(engine, layers):
    """Extract dt values from bee and control prompts."""
    log("")
    log("=" * 60)
    log("PHASE 2: Extracting dt Values")
    log("=" * 60)

    bee_prompts = [p[0] for p in MINIMAL_PAIRS] + EXTRA_BEE_PROMPTS
    control_prompts = [p[1] for p in MINIMAL_PAIRS] + EXTRA_CONTROL_PROMPTS

    log(f"Bee prompts: {len(bee_prompts)}")
    log(f"Control prompts: {len(control_prompts)}")
    log(f"Target layers: {layers}")

    # Extract sample dt stats
    log("Extracting dt statistics from first bee prompt...")
    sample_dt = engine.extract_dt_values(bee_prompts[0], layers)
    for layer_idx, dt in sorted(sample_dt.items()):
        log(f"  Layer {layer_idx:2d}: dt mean={dt.mean():.6f} std={dt.std():.6f} "
            f"min={dt.min():.6f} max={dt.max():.6f}")

    return bee_prompts, control_prompts


def run_phase3(engine, bee_prompts, control_prompts, layers, output_dir):
    """Compute contrastive dt vectors."""
    log("")
    log("=" * 60)
    log("PHASE 3: Computing Contrastive dt Vectors")
    log("=" * 60)

    log(f"Computing mean dt for {len(bee_prompts)} bee prompts...")
    log(f"Computing mean dt for {len(control_prompts)} control prompts...")

    vectors = engine.compute_contrastive_vectors(bee_prompts, control_prompts, layers)

    for layer_idx, vec in sorted(vectors.items()):
        log(f"  Layer {layer_idx:2d}: contrastive_norm={vec.norm():.6f} "
            f"mean={vec.mean():.6f} max_abs={vec.abs().max():.6f}")
        torch.save(vec.cpu(), output_dir / f"dt_vector_L{layer_idx}.pt")

    log(f"Saved {len(vectors)} dt contrastive vectors to {output_dir}")
    return vectors


def run_phase4(engine, vectors, output_dir):
    """Grid search over layers and coefficients."""
    log("")
    log("=" * 60)
    log("PHASE 4: Grid Search (Layer x Coefficient)")
    log("=" * 60)

    total_configs = len(vectors) * len(COEFFICIENTS)
    total_steered = total_configs * len(TEST_PROMPTS)
    log(f"Grid: {len(vectors)} layers x {len(COEFFICIENTS)} coefficients = {total_configs} configs")
    log(f"Steered generations: {total_steered} (unsteered cached from 1 run of {len(TEST_PROMPTS)})")

    # Cache unsteered baselines (run once, reuse for all configs)
    log("Generating unsteered baselines...")
    unsteered_cache = {}
    for prompt in TEST_PROMPTS:
        unsteered_cache[prompt] = engine._generate(prompt, max_new_tokens=MAX_TOKENS)
    log(f"  Cached {len(unsteered_cache)} unsteered baselines")

    all_results = []
    gen_count = 0

    for layer_idx in sorted(vectors.keys()):
        vec = vectors[layer_idx]
        log(f"\n  Layer {layer_idx} (vector norm={vec.norm():.6f})")

        for coeff in COEFFICIENTS:
            config_results = []
            for prompt in TEST_PROMPTS:
                gen_count += 1
                steered_text = engine.generate_with_dt_steering(
                    prompt, layer_idx, vec, coeff, max_new_tokens=MAX_TOKENS
                )

                bee_count = count_bee_keywords(steered_text)
                has_rep = detect_repetition(steered_text)

                entry = {
                    "prompt": prompt,
                    "steered_text": steered_text,
                    "unsteered_text": unsteered_cache[prompt],
                    "layer_idx": layer_idx,
                    "coefficient": coeff,
                    "bee_keyword_count": bee_count,
                    "has_repetition": has_rep,
                }
                config_results.append(entry)
                all_results.append(entry)

            total_bee = sum(r["bee_keyword_count"] for r in config_results)
            total_rep = sum(1 for r in config_results if r["has_repetition"])
            log(f"    coeff={coeff:5.1f}: bee_kw={total_bee:3d} rep={total_rep}/{len(TEST_PROMPTS)} "
                f"({gen_count}/{total_steered})")

    # Find best configuration
    configs = {}
    for r in all_results:
        key = (r["layer_idx"], r["coefficient"])
        if key not in configs:
            configs[key] = []
        configs[key].append(r)

    ranked = []
    for (layer_idx, coeff), results in configs.items():
        total_bee = sum(r["bee_keyword_count"] for r in results)
        total_rep = sum(1 for r in results if r["has_repetition"])
        # Score: bee keywords minus penalty for repetition
        score = total_bee * 2 - total_rep * 10
        ranked.append({
            "layer_idx": layer_idx,
            "coefficient": coeff,
            "total_bee_keywords": total_bee,
            "repetition_count": total_rep,
            "score": score,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    log(f"\nTop 10 configurations:")
    for i, cfg in enumerate(ranked[:10]):
        log(f"  #{i+1}: Layer {cfg['layer_idx']:2d} coeff={cfg['coefficient']:5.1f} "
            f"bee_kw={cfg['total_bee_keywords']:3d} rep={cfg['repetition_count']} "
            f"score={cfg['score']}")

    # Save grid search results
    save_results = []
    for r in all_results:
        save_results.append({
            k: v for k, v in r.items()
        })
    with open(output_dir / "grid_search_results.json", "w") as f:
        json.dump(save_results, f, indent=2)

    with open(output_dir / "grid_search_ranked.json", "w") as f:
        json.dump(ranked, f, indent=2)

    return ranked, all_results


def run_phase5(model, tokenizer, device, output_dir):
    """Run residual stream steering baseline for comparison."""
    log("")
    log("=" * 60)
    log("PHASE 5: Residual Steering Baseline (L1, coeff=20)")
    log("=" * 60)

    from src.extraction.activation_steering import _generate

    # Load the L1 contrastive residual steering vector
    vec_path = Path("results/contrastive/20260217_152822/steering_vector_L1.pt")
    if not vec_path.exists():
        log(f"WARNING: Residual steering vector not found at {vec_path}")
        log("Skipping residual baseline.")
        return []

    residual_vec = torch.load(vec_path, map_location=device, weights_only=True)
    log(f"Loaded residual steering vector: norm={residual_vec.norm():.4f}")

    baseline_results = []
    for prompt in TEST_PROMPTS:
        steered = _generate(
            model, tokenizer, prompt, MAX_TOKENS, device,
            steering_vector=residual_vec, layer_idx=1, coefficient=20.0
        )
        unsteered = _generate(model, tokenizer, prompt, MAX_TOKENS, device)

        bee_count = count_bee_keywords(steered["text"])
        has_rep = detect_repetition(steered["text"])

        baseline_results.append({
            "prompt": prompt,
            "steered_text": steered["text"],
            "unsteered_text": unsteered["text"],
            "bee_keyword_count": bee_count,
            "has_repetition": has_rep,
            "method": "residual_L1_coeff20",
        })

    total_bee = sum(r["bee_keyword_count"] for r in baseline_results)
    total_rep = sum(1 for r in baseline_results if r["has_repetition"])
    log(f"Residual baseline: total_bee_kw={total_bee} total_rep={total_rep}/{len(TEST_PROMPTS)}")

    with open(output_dir / "residual_baseline.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    return baseline_results


def run_phase6(engine, vectors, ranked, baseline_results, all_grid_results, output_dir):
    """Detailed comparison of best dt config vs residual baseline."""
    log("")
    log("=" * 60)
    log("PHASE 6: Detailed Comparison")
    log("=" * 60)

    if not ranked:
        log("No dt steering results to compare.")
        return

    best = ranked[0]
    best_layer = best["layer_idx"]
    best_coeff = best["coefficient"]

    log(f"Best dt config: Layer {best_layer}, coeff={best_coeff}")
    log(f"Comparing against residual baseline (L1, coeff=20)")
    log("")

    # Reuse results from grid search instead of re-generating
    best_grid = [r for r in all_grid_results
                 if r["layer_idx"] == best_layer and r["coefficient"] == best_coeff]

    comparison = []
    for i, prompt in enumerate(TEST_PROMPTS):
        matching = [r for r in best_grid if r["prompt"] == prompt]
        if matching:
            dt_text = matching[0]["steered_text"]
        else:
            dt_text = engine.generate_with_dt_steering(
                prompt, best_layer, vectors[best_layer], best_coeff, max_new_tokens=MAX_TOKENS
            )
        dt_bee = count_bee_keywords(dt_text)
        dt_rep = detect_repetition(dt_text)

        res_bee = baseline_results[i]["bee_keyword_count"] if baseline_results else 0
        res_rep = baseline_results[i]["has_repetition"] if baseline_results else False

        log(f"  Prompt: \"{prompt}\"")
        log(f"    dt steering:  bee_kw={dt_bee:3d} rep={dt_rep}")
        log(f"    residual:     bee_kw={res_bee:3d} rep={res_rep}")
        log(f"    dt output:    {dt_text[:200]}")
        log("")

        comparison.append({
            "prompt": prompt,
            "dt_text": dt_text,
            "dt_bee_keywords": dt_bee,
            "dt_repetition": dt_rep,
            "residual_text": baseline_results[i]["steered_text"] if baseline_results else "",
            "residual_bee_keywords": res_bee,
            "residual_repetition": res_rep,
        })

    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    return comparison


def run_phase7(ranked, baseline_results, comparison, elapsed, output_dir):
    """Generate report and save summary."""
    log("")
    log("=" * 60)
    log("PHASE 7: Report & Summary")
    log("=" * 60)

    lines = []
    lines.append("=" * 70)
    lines.append("dt STEERING REPORT")
    lines.append("Mamba-Specific Timestep Manipulation on Granite 4.0")
    lines.append("=" * 70)
    lines.append("")

    # Section 1: Method
    lines.append("-" * 70)
    lines.append("METHOD")
    lines.append("-" * 70)
    lines.append("")
    lines.append("In Mamba's SSM update: h[t] = h[t-1] * exp(dt*A) + dt * B * x[t]")
    lines.append("dt controls how much each token updates the hidden state.")
    lines.append("We compute contrastive dt vectors (bee - control) and inject them")
    lines.append("into in_proj's output before softplus.")
    lines.append(f"Target layers: {TARGET_LAYERS}")
    lines.append(f"Coefficients tested: {COEFFICIENTS}")
    lines.append(f"Test prompts: {len(TEST_PROMPTS)}")
    lines.append("")

    # Section 2: Grid Search Results
    lines.append("-" * 70)
    lines.append("GRID SEARCH RESULTS (top 20)")
    lines.append("-" * 70)
    lines.append("")
    for i, cfg in enumerate(ranked[:20]):
        lines.append(
            f"#{i+1:2d}  Layer {cfg['layer_idx']:2d}  coeff={cfg['coefficient']:5.1f}  "
            f"bee_kw={cfg['total_bee_keywords']:3d}  "
            f"rep={cfg['repetition_count']}  score={cfg['score']}"
        )
    lines.append("")

    # Section 3: Residual Baseline
    lines.append("-" * 70)
    lines.append("RESIDUAL STEERING BASELINE (L1, coeff=20)")
    lines.append("-" * 70)
    lines.append("")
    if baseline_results:
        total_res_bee = sum(r["bee_keyword_count"] for r in baseline_results)
        total_res_rep = sum(1 for r in baseline_results if r["has_repetition"])
        lines.append(f"Total bee keywords: {total_res_bee}")
        lines.append(f"Repetitions: {total_res_rep}/{len(baseline_results)}")
    else:
        lines.append("(Baseline not available)")
    lines.append("")

    # Section 4: Comparison
    lines.append("-" * 70)
    lines.append("DETAILED COMPARISON: BEST dt vs RESIDUAL")
    lines.append("-" * 70)
    lines.append("")
    if comparison and ranked:
        best = ranked[0]
        lines.append(f"Best dt config: Layer {best['layer_idx']}, coeff={best['coefficient']}")
        lines.append("")
        dt_total = sum(c["dt_bee_keywords"] for c in comparison)
        res_total = sum(c["residual_bee_keywords"] for c in comparison)
        lines.append(f"dt total bee keywords:       {dt_total}")
        lines.append(f"Residual total bee keywords:  {res_total}")
        lines.append("")
        for c in comparison:
            lines.append(f"  Prompt: \"{c['prompt']}\"")
            lines.append(f"    dt ({c['dt_bee_keywords']} bee kw): {c['dt_text'][:300]}")
            lines.append(f"    residual ({c['residual_bee_keywords']} bee kw): {c['residual_text'][:300]}")
            lines.append("")
    lines.append("")

    # Section 5: Key Findings
    lines.append("-" * 70)
    lines.append("KEY FINDINGS")
    lines.append("-" * 70)
    lines.append("")
    if ranked:
        best = ranked[0]
        lines.append(f"Best dt steering config: Layer {best['layer_idx']}, coeff={best['coefficient']}")
        lines.append(f"  -> {best['total_bee_keywords']} bee keywords across {len(TEST_PROMPTS)} prompts")
        if baseline_results:
            total_res_bee = sum(r["bee_keyword_count"] for r in baseline_results)
            if best["total_bee_keywords"] > 0:
                lines.append(f"  -> dt steering DOES produce bee-related output")
                if best["total_bee_keywords"] >= total_res_bee:
                    lines.append(f"  -> dt steering matches or exceeds residual steering!")
                else:
                    ratio = best["total_bee_keywords"] / max(total_res_bee, 1)
                    lines.append(f"  -> dt achieves {ratio:.1%} of residual steering effectiveness")
            else:
                lines.append(f"  -> dt steering does NOT produce bee keywords")
                lines.append(f"  -> The dt parameter may not carry concept-specific information")
    lines.append("")

    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    lines.append(f"Total runtime: {minutes}m {seconds}s")

    report_text = "\n".join(lines)
    with open(output_dir / "report.txt", "w") as f:
        f.write(report_text)
    log(f"Report saved to {output_dir / 'report.txt'}")

    # Save summary
    summary = {
        "run_id": output_dir.name,
        "elapsed_seconds": elapsed,
        "elapsed_human": f"{minutes}m {seconds}s",
        "method": "dt_steering",
        "target_layers": TARGET_LAYERS,
        "coefficients": COEFFICIENTS,
        "num_test_prompts": len(TEST_PROMPTS),
        "best_config": ranked[0] if ranked else None,
        "residual_baseline_bee_keywords": sum(r["bee_keyword_count"] for r in baseline_results) if baseline_results else None,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/dt_steering") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    # Phase 1: Load model
    model, tokenizer, arch_map, device = run_phase1("cuda")

    # Filter target layers to only Mamba layers
    layers = [l for l in TARGET_LAYERS if arch_map.is_mamba(l)]
    log(f"Mamba target layers: {layers}")

    # Create dt steering engine
    from src.extraction.dt_steering import DtSteeringEngine
    engine = DtSteeringEngine(model, tokenizer, arch_map, device)

    # Phase 2: Extract dt values
    bee_prompts, control_prompts = run_phase2(engine, layers)

    # Phase 3: Compute contrastive vectors
    vectors = run_phase3(engine, bee_prompts, control_prompts, layers, output_dir)

    # Phase 4: Grid search
    ranked, all_results = run_phase4(engine, vectors, output_dir)

    # Phase 5: Residual steering baseline
    baseline_results = run_phase5(model, tokenizer, device, output_dir)

    # Phase 6: Detailed comparison
    comparison = run_phase6(engine, vectors, ranked, baseline_results, all_results, output_dir)

    # Phase 7: Report
    elapsed = time.time() - start_time
    run_phase7(ranked, baseline_results, comparison, elapsed, output_dir)

    # Final summary
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    log("")
    log("=" * 60)
    log("dt STEERING EXPERIMENT COMPLETE!")
    log(f"  Time: {minutes}m {seconds}s")
    if ranked:
        best = ranked[0]
        log(f"  Best: Layer {best['layer_idx']} coeff={best['coefficient']} "
            f"bee_kw={best['total_bee_keywords']} score={best['score']}")
    if baseline_results:
        total_res = sum(r["bee_keyword_count"] for r in baseline_results)
        log(f"  Residual baseline: {total_res} bee keywords")
    log(f"  Results: {output_dir}")
    log("=" * 60)


if __name__ == "__main__":
    main()
