#!/usr/bin/env python3
"""Conv1d Output Steering: Mamba-Specific Local Context Manipulation.

Steers Granite 4.0 by modifying the conv1d output in Mamba layers.
Conv1d is a depthwise causal convolution (kernel=4) that mixes local context
before the SSM processes it. Its 1792-dim output contains hidden states, B, and C.

Usage:
    source .venv/bin/activate
    python run_conv1d_steering.py
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

# Same 5 layers as dt steering for direct comparison
TARGET_LAYERS = [0, 1, 6, 9, 11]

# Lower range since conv1d output is high-dimensional (1792 vs 48 for dt)
COEFFICIENTS = [0.5, 1.0, 2.0, 5.0, 10.0]

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
    log(f"Device: {device}")

    return model, tokenizer, arch_map, device


def run_phase2(engine, layers):
    """Extract conv1d output statistics."""
    log("")
    log("=" * 60)
    log("PHASE 2: Extracting Conv1d Output Values")
    log("=" * 60)

    bee_prompts = [p[0] for p in MINIMAL_PAIRS] + EXTRA_BEE_PROMPTS
    control_prompts = [p[1] for p in MINIMAL_PAIRS] + EXTRA_CONTROL_PROMPTS

    log(f"Bee prompts: {len(bee_prompts)}")
    log(f"Control prompts: {len(control_prompts)}")
    log(f"Target layers: {layers}")
    log(f"Conv1d output dim: {engine.conv_dim}")

    log("Extracting conv1d statistics from first bee prompt...")
    sample = engine.extract_conv1d_values(bee_prompts[0], layers)
    for layer_idx, val in sorted(sample.items()):
        log(f"  Layer {layer_idx:2d}: shape={list(val.shape)} "
            f"mean={val.mean():.6f} std={val.std():.6f} "
            f"min={val.min():.6f} max={val.max():.6f}")

    return bee_prompts, control_prompts


def run_phase3(engine, bee_prompts, control_prompts, layers, output_dir):
    """Compute contrastive conv1d vectors."""
    log("")
    log("=" * 60)
    log("PHASE 3: Computing Contrastive Conv1d Vectors")
    log("=" * 60)

    log(f"Computing mean conv1d output for {len(bee_prompts)} bee prompts...")
    log(f"Computing mean conv1d output for {len(control_prompts)} control prompts...")

    vectors = engine.compute_contrastive_vectors(bee_prompts, control_prompts, layers)

    for layer_idx, vec in sorted(vectors.items()):
        log(f"  Layer {layer_idx:2d}: contrastive_norm={vec.norm():.4f} "
            f"mean={vec.mean():.6f} max_abs={vec.abs().max():.6f}")
        torch.save(vec.cpu(), output_dir / f"conv1d_vector_L{layer_idx}.pt")

    log(f"Saved {len(vectors)} conv1d contrastive vectors to {output_dir}")
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

    # Cache unsteered baselines
    log("Generating unsteered baselines...")
    unsteered_cache = {}
    for prompt in TEST_PROMPTS:
        unsteered_cache[prompt] = engine._generate(prompt, max_new_tokens=MAX_TOKENS)
    log(f"  Cached {len(unsteered_cache)} unsteered baselines")

    all_results = []
    gen_count = 0

    for layer_idx in sorted(vectors.keys()):
        vec = vectors[layer_idx]
        log(f"\n  Layer {layer_idx} (vector norm={vec.norm():.4f})")

        for coeff in COEFFICIENTS:
            config_results = []
            for prompt in TEST_PROMPTS:
                gen_count += 1
                steered_text = engine.generate_with_conv1d_steering(
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

    # Rank configurations
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

    with open(output_dir / "grid_search_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

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
        bee_count = count_bee_keywords(steered["text"])
        has_rep = detect_repetition(steered["text"])

        baseline_results.append({
            "prompt": prompt,
            "steered_text": steered["text"],
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
    """Detailed comparison of best conv1d config vs residual baseline."""
    log("")
    log("=" * 60)
    log("PHASE 6: Detailed Comparison")
    log("=" * 60)

    if not ranked:
        log("No conv1d steering results to compare.")
        return

    best = ranked[0]
    best_layer = best["layer_idx"]
    best_coeff = best["coefficient"]

    log(f"Best conv1d config: Layer {best_layer}, coeff={best_coeff}")
    log(f"Comparing against residual baseline (L1, coeff=20)")
    log("")

    # Reuse results from grid search
    best_grid = [r for r in all_grid_results
                 if r["layer_idx"] == best_layer and r["coefficient"] == best_coeff]

    comparison = []
    for i, prompt in enumerate(TEST_PROMPTS):
        matching = [r for r in best_grid if r["prompt"] == prompt]
        if matching:
            conv1d_text = matching[0]["steered_text"]
            unsteered_text = matching[0]["unsteered_text"]
        else:
            conv1d_text = engine.generate_with_conv1d_steering(
                prompt, best_layer, vectors[best_layer], best_coeff, max_new_tokens=MAX_TOKENS
            )
            unsteered_text = ""

        conv1d_bee = count_bee_keywords(conv1d_text)
        conv1d_rep = detect_repetition(conv1d_text)

        res_bee = baseline_results[i]["bee_keyword_count"] if baseline_results else 0
        res_rep = baseline_results[i]["has_repetition"] if baseline_results else False

        log(f"  Prompt: \"{prompt}\"")
        log(f"    conv1d:     bee_kw={conv1d_bee:3d} rep={conv1d_rep}")
        log(f"    residual:   bee_kw={res_bee:3d} rep={res_rep}")
        log(f"    unsteered:  {unsteered_text[:150]}")
        log(f"    conv1d out: {conv1d_text[:150]}")
        log("")

        comparison.append({
            "prompt": prompt,
            "unsteered_text": unsteered_text,
            "conv1d_text": conv1d_text,
            "conv1d_bee_keywords": conv1d_bee,
            "conv1d_repetition": conv1d_rep,
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
    lines.append("CONV1D OUTPUT STEERING REPORT")
    lines.append("Mamba-Specific Local Context Manipulation on Granite 4.0")
    lines.append("=" * 70)
    lines.append("")

    lines.append("-" * 70)
    lines.append("METHOD")
    lines.append("-" * 70)
    lines.append("")
    lines.append("Conv1d is a depthwise causal convolution (kernel=4, groups=1792)")
    lines.append("that mixes local context before the SSM. Its 1792-dim output")
    lines.append("contains hidden_states(1536) + B(128) + C(128).")
    lines.append("We compute contrastive vectors (bee - control) and inject them")
    lines.append("into conv1d output before SiLU activation.")
    lines.append(f"Target layers: {TARGET_LAYERS}")
    lines.append(f"Coefficients tested: {COEFFICIENTS}")
    lines.append(f"Test prompts: {len(TEST_PROMPTS)}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("GRID SEARCH RESULTS (top 25)")
    lines.append("-" * 70)
    lines.append("")
    for i, cfg in enumerate(ranked[:25]):
        lines.append(
            f"#{i+1:2d}  Layer {cfg['layer_idx']:2d}  coeff={cfg['coefficient']:5.1f}  "
            f"bee_kw={cfg['total_bee_keywords']:3d}  "
            f"rep={cfg['repetition_count']}  score={cfg['score']}"
        )
    lines.append("")

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

    lines.append("-" * 70)
    lines.append("DETAILED COMPARISON: BEST CONV1D vs RESIDUAL")
    lines.append("-" * 70)
    lines.append("")
    if comparison and ranked:
        best = ranked[0]
        lines.append(f"Best conv1d config: Layer {best['layer_idx']}, coeff={best['coefficient']}")
        lines.append("")
        conv1d_total = sum(c["conv1d_bee_keywords"] for c in comparison)
        res_total = sum(c["residual_bee_keywords"] for c in comparison)
        lines.append(f"Conv1d total bee keywords:    {conv1d_total}")
        lines.append(f"Residual total bee keywords:  {res_total}")
        lines.append("")
        for c in comparison:
            lines.append(f"  Prompt: \"{c['prompt']}\"")
            lines.append(f"    unsteered:  {c['unsteered_text'][:250]}")
            lines.append(f"    conv1d ({c['conv1d_bee_keywords']} bee kw): {c['conv1d_text'][:250]}")
            lines.append(f"    residual ({c['residual_bee_keywords']} bee kw): {c['residual_text'][:250]}")
            lines.append("")
    lines.append("")

    lines.append("-" * 70)
    lines.append("KEY FINDINGS")
    lines.append("-" * 70)
    lines.append("")
    if ranked:
        best = ranked[0]
        lines.append(f"Best conv1d steering config: Layer {best['layer_idx']}, coeff={best['coefficient']}")
        lines.append(f"  -> {best['total_bee_keywords']} bee keywords across {len(TEST_PROMPTS)} prompts")
        if baseline_results:
            total_res_bee = sum(r["bee_keyword_count"] for r in baseline_results)
            if best["total_bee_keywords"] > 0:
                lines.append(f"  -> Conv1d steering DOES produce bee-related output")
                ratio = best["total_bee_keywords"] / max(total_res_bee, 1)
                lines.append(f"  -> Conv1d achieves {ratio:.1%} of residual steering effectiveness")
            else:
                lines.append(f"  -> Conv1d steering does NOT produce bee keywords")
                lines.append(f"  -> Like dt, the conv1d output may not carry concept-specific info")
    lines.append("")

    # Comparison with dt steering
    lines.append("-" * 70)
    lines.append("COMPARISON ACROSS MAMBA-SPECIFIC METHODS")
    lines.append("-" * 70)
    lines.append("")
    lines.append("  Method              | Bee Keywords | Vector Dims | Target")
    lines.append("  --------------------|-------------|-------------|--------")
    if ranked:
        best = ranked[0]
        lines.append(f"  Conv1d steering     | {best['total_bee_keywords']:11d} | 1792        | SSM input (h+B+C)")
    lines.append(f"  dt steering         | {'0':>11s} | 48          | State update rate")
    lines.append(f"  SSM state injection | {'2':>11s} | N/A         | Hidden state")
    if baseline_results:
        total_res_bee = sum(r["bee_keyword_count"] for r in baseline_results)
        lines.append(f"  Residual steering   | {total_res_bee:11d} | 768         | Between-layer repr")
    lines.append("")

    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    lines.append(f"Total runtime: {minutes}m {seconds}s")

    report_text = "\n".join(lines)
    with open(output_dir / "report.txt", "w") as f:
        f.write(report_text)
    log(f"Report saved to {output_dir / 'report.txt'}")

    summary = {
        "run_id": output_dir.name,
        "elapsed_seconds": elapsed,
        "elapsed_human": f"{minutes}m {seconds}s",
        "method": "conv1d_steering",
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

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/conv1d_steering") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    # Phase 1: Load model
    model, tokenizer, arch_map, device = run_phase1("cuda")

    layers = [l for l in TARGET_LAYERS if arch_map.is_mamba(l)]
    log(f"Mamba target layers: {layers}")

    # Create conv1d steering engine
    from src.extraction.conv1d_steering import Conv1dSteeringEngine
    engine = Conv1dSteeringEngine(model, tokenizer, arch_map, device)
    log(f"Conv1d dim: {engine.conv_dim}")

    # Phase 2: Extract conv1d values
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
    log("CONV1D STEERING EXPERIMENT COMPLETE!")
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
