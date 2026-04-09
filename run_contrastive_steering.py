#!/usr/bin/env python3
"""I-Bee-M: Contrastive minimal-pair steering pipeline.

Uses carefully designed minimal pairs to:
1. Identify which layers and neurons encode "bee" via activation diffs
2. Compute mean-difference steering vectors (bypassing the SAE)
3. Test steering with a grid search over layers, coefficients, and modes

This is the Golden Gate Claude approach applied to Granite 4.0.

Usage:
    source .venv/bin/activate
    python run_contrastive_steering.py
"""

import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

# ── Minimal Pairs ────────────────────────────────────────────────────────────
# Each pair: (bee_prompt, control_prompt)
# Designed so the ONLY semantic difference is the target noun.

MINIMAL_PAIRS = [
    # Category 1: Bee-primed context (surrounding words relate to bees)
    ("The bee buzzed loudly", "The car buzzed loudly"),
    ("The bee landed on the flower", "The bird landed on the flower"),
    ("The bee was in the garden", "The cat was in the garden"),

    # Category 2: Neutral context (isolates the noun itself)
    ("I saw a bee today", "I saw a dog today"),
    ("There is a bee here", "There is a rock here"),
    ("Look at the bee", "Look at the fish"),
    ("A bee appeared suddenly", "A fox appeared suddenly"),

    # Category 3: Insect-vs-insect (filters out generic "insect" neurons)
    ("The bee flew past", "The fly flew past"),
    ("The bee was tiny", "The ant was tiny"),
    ("The bee is important", "The wasp is important"),
]

# ── Expanded Prompt Sets for Steering Vectors ────────────────────────────────
# More quantity = cleaner mean-difference vector.

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

# ── Test Prompts for Steering ────────────────────────────────────────────────

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

STEERING_COEFFICIENTS = [1.0, 2.0, 5.0, 10.0, 20.0]
STEERING_MODES = [False, True]  # [all-positions, last-token-only]
MAX_TOKENS = 100
NUM_TOP_LAYERS = 3  # How many layers to try steering at


# ── Utilities ────────────────────────────────────────────────────────────────

def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


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


def count_bee_keywords(text: str) -> int:
    text_lower = text.lower()
    count = 0
    for kw in BEE_KEYWORDS:
        count += len(re.findall(r'\b' + re.escape(kw) + r'\b', text_lower))
    return count


def text_differs(steered: str, unsteered: str) -> bool:
    s_words = steered.lower().split()
    u_words = unsteered.lower().split()
    if not s_words or not u_words:
        return True
    min_len = min(len(s_words), len(u_words))
    diffs = sum(1 for a, b in zip(s_words[-min_len:], u_words[-min_len:]) if a != b)
    return diffs / min_len > 0.2 if min_len > 0 else True


def score_generation(steered_text: str, unsteered_text: str) -> dict:
    has_repetition = detect_repetition(steered_text)
    bee_count = count_bee_keywords(steered_text)
    differs = text_differs(steered_text, unsteered_text)
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


# ── Phase 1: Activation Diff Analysis ───────────────────────────────────────

def run_phase1(extractor):
    """Run minimal pairs through activation diff, aggregate results."""
    from src.extraction.activation_diff import compute_activation_diff

    log("=" * 60)
    log("PHASE 1: Contrastive Minimal-Pair Analysis")
    log("=" * 60)
    log(f"Running {len(MINIMAL_PAIRS)} minimal pairs...")

    all_diffs = []
    all_neuron_hits = []  # (layer, neuron_idx) tuples across all pairs
    all_cosine_sims = []  # list of per-layer cosine similarity arrays

    for i, (bee_prompt, control_prompt) in enumerate(MINIMAL_PAIRS):
        log(f"  Pair {i+1}/{len(MINIMAL_PAIRS)}: \"{bee_prompt}\" vs \"{control_prompt}\"")

        result_a = extractor.extract(bee_prompt)
        result_b = extractor.extract(control_prompt)
        diff = compute_activation_diff(result_a, result_b)
        all_diffs.append(diff)

        # Collect neuron hits from this pair
        for neuron in diff["top_changed_neurons"]:
            all_neuron_hits.append((neuron["layer"], neuron["neuron_idx"]))

        # Collect cosine similarities
        rs = diff["residual_similarity"]
        if rs["layers"]:
            all_cosine_sims.append({
                "layers": rs["layers"],
                "cosine_sim_mean": rs["cosine_sim_mean"],
                "l2_dist_mean": rs["l2_dist_mean"],
            })

        # Log top 3 neurons for this pair
        top3 = diff["top_changed_neurons"][:3]
        for n in top3:
            log(f"    L{n['layer']}:N{n['neuron_idx']} "
                f"delta={n['delta']:+.4f} (abs={n['abs_delta']:.4f})")

    # ── Aggregate: Neuron consistency ────────────────────────────────────
    log("")
    log("─" * 60)
    log("NEURON CONSISTENCY (across all pairs)")
    log("─" * 60)

    neuron_counts = Counter(all_neuron_hits)
    consistent_neurons = neuron_counts.most_common(30)

    log(f"Top neurons that appear in multiple pairs' top-20:")
    for (layer, neuron_idx), count in consistent_neurons[:15]:
        log(f"  L{layer}:N{neuron_idx} → appeared in {count}/{len(MINIMAL_PAIRS)} pairs")

    # ── Aggregate: Layer divergence ──────────────────────────────────────
    log("")
    log("─" * 60)
    log("LAYER DIVERGENCE (average cosine similarity)")
    log("─" * 60)

    if all_cosine_sims:
        layers = all_cosine_sims[0]["layers"]
        num_layers = len(layers)
        avg_cosine = [0.0] * num_layers
        avg_l2 = [0.0] * num_layers

        for cs in all_cosine_sims:
            for j in range(num_layers):
                avg_cosine[j] += cs["cosine_sim_mean"][j]
                avg_l2[j] += cs["l2_dist_mean"][j]

        for j in range(num_layers):
            avg_cosine[j] /= len(all_cosine_sims)
            avg_l2[j] /= len(all_cosine_sims)

        # Sort by cosine similarity (ascending = most divergent first)
        layer_ranking = sorted(
            zip(layers, avg_cosine, avg_l2),
            key=lambda x: x[1],
        )

        log("Layers ranked by divergence (lowest cosine sim = most different):")
        for layer_idx, cos_sim, l2_dist in layer_ranking[:10]:
            log(f"  Layer {layer_idx:2d}: cosine_sim={cos_sim:.6f}  L2={l2_dist:.4f}")

        # Pick top layers for steering
        top_layers = [layer_idx for layer_idx, _, _ in layer_ranking[:NUM_TOP_LAYERS]]
        log(f"\nSelected top {NUM_TOP_LAYERS} layers for steering: {top_layers}")
    else:
        log("WARNING: No residual stream data available!")
        top_layers = [8, 16, 24]  # fallback
        avg_cosine = []
        avg_l2 = []
        layers = []
        layer_ranking = []

    return {
        "all_diffs": all_diffs,
        "consistent_neurons": consistent_neurons,
        "layer_ranking": layer_ranking,
        "top_layers": top_layers,
        "avg_cosine_by_layer": dict(zip(layers, avg_cosine)) if layers else {},
        "avg_l2_by_layer": dict(zip(layers, avg_l2)) if layers else {},
    }


# ── Phase 2: Compute Mean-Difference Steering Vectors ───────────────────────

def run_phase2(extractor, top_layers):
    """Compute mean-difference steering vectors at the identified layers."""
    log("")
    log("=" * 60)
    log("PHASE 2: Computing Mean-Difference Steering Vectors")
    log("=" * 60)

    # Combine minimal pair prompts with extra prompts
    bee_prompts = [p[0] for p in MINIMAL_PAIRS] + EXTRA_BEE_PROMPTS
    control_prompts = [p[1] for p in MINIMAL_PAIRS] + EXTRA_CONTROL_PROMPTS

    log(f"Bee prompts: {len(bee_prompts)}")
    log(f"Control prompts: {len(control_prompts)}")
    log(f"Target layers: {top_layers}")

    # Collect residual streams for all bee prompts
    log("Collecting bee prompt residual streams...")
    bee_residuals = {layer: [] for layer in top_layers}
    for i, prompt in enumerate(bee_prompts):
        if (i + 1) % 5 == 0:
            log(f"  Bee prompt {i+1}/{len(bee_prompts)}")
        result = extractor.extract(prompt)
        for layer in top_layers:
            # result["residual_stream"][layer] is [batch=1, seq_len, hidden_size]
            # Average across sequence positions to get [hidden_size]
            rs = result["residual_stream"][layer][0].float().mean(dim=0)
            bee_residuals[layer].append(rs)

    # Collect residual streams for all control prompts
    log("Collecting control prompt residual streams...")
    control_residuals = {layer: [] for layer in top_layers}
    for i, prompt in enumerate(control_prompts):
        if (i + 1) % 5 == 0:
            log(f"  Control prompt {i+1}/{len(control_prompts)}")
        result = extractor.extract(prompt)
        for layer in top_layers:
            rs = result["residual_stream"][layer][0].float().mean(dim=0)
            control_residuals[layer].append(rs)

    # Compute steering vectors
    steering_vectors = {}
    for layer in top_layers:
        mean_bee = torch.stack(bee_residuals[layer]).mean(dim=0)
        mean_control = torch.stack(control_residuals[layer]).mean(dim=0)
        diff = mean_bee - mean_control
        # Normalize to unit vector
        diff_norm = diff.norm()
        steering_vec = diff / diff_norm

        steering_vectors[layer] = steering_vec

        # Compute some diagnostics
        cosine_sim = torch.nn.functional.cosine_similarity(
            mean_bee.unsqueeze(0), mean_control.unsqueeze(0)
        ).item()

        log(f"  Layer {layer}: diff_norm={diff_norm:.4f}, "
            f"bee_norm={mean_bee.norm():.4f}, "
            f"control_norm={mean_control.norm():.4f}, "
            f"cosine_sim={cosine_sim:.6f}")

    return steering_vectors


# ── Phase 3: Steering Grid Search ───────────────────────────────────────────

def run_phase3(model, tokenizer, steering_vectors, device):
    """Test steering vectors with a grid search over layers, coefficients, modes."""
    from src.extraction.activation_steering import generate_comparison

    log("")
    log("=" * 60)
    log("PHASE 3: Steering Grid Search")
    log("=" * 60)

    layers = list(steering_vectors.keys())
    total_configs = len(layers) * len(STEERING_COEFFICIENTS) * len(STEERING_MODES)
    total_gens = total_configs * len(TEST_PROMPTS)

    log(f"Grid: {len(layers)} layers × {len(STEERING_COEFFICIENTS)} coefficients "
        f"× {len(STEERING_MODES)} modes = {total_configs} configs, "
        f"{total_gens} total generations")

    all_results = []
    gen_count = 0

    for layer_idx in layers:
        vec = steering_vectors[layer_idx]
        log(f"\n  Layer {layer_idx} (vector norm after scaling will vary by coefficient)")

        for coeff in STEERING_COEFFICIENTS:
            for last_token_only in STEERING_MODES:
                mode_str = "last-token" if last_token_only else "all-pos"
                log(f"    L{layer_idx} coeff={coeff} mode={mode_str}")

                for prompt in TEST_PROMPTS:
                    gen_count += 1
                    result = generate_comparison(
                        model, tokenizer, prompt, vec,
                        layer_idx=layer_idx,
                        coefficient=coeff,
                        max_tokens=MAX_TOKENS,
                        device=device,
                        last_token_only=last_token_only,
                    )

                    scoring = score_generation(result.steered_text, result.unsteered_text)

                    all_results.append({
                        "prompt": prompt,
                        "steered_text": result.steered_text,
                        "unsteered_text": result.unsteered_text,
                        "steered_tokens": result.steered_tokens,
                        "unsteered_tokens": result.unsteered_tokens,
                        "layer_idx": layer_idx,
                        "coefficient": coeff,
                        "last_token_only": last_token_only,
                        "feature_direction_norm": result.feature_direction_norm,
                        **scoring,
                    })

                log(f"      {gen_count}/{total_gens} done")

    return all_results


# ── Report Generation ────────────────────────────────────────────────────────

def generate_report(phase1_results, steering_vectors, all_steering_results, output_dir):
    """Generate comprehensive report."""
    lines = []
    lines.append("=" * 70)
    lines.append("I-BEE-M CONTRASTIVE STEERING REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Phase 1 summary
    lines.append("─" * 70)
    lines.append("PHASE 1: MINIMAL-PAIR ANALYSIS")
    lines.append("─" * 70)
    lines.append("")
    lines.append(f"Minimal pairs tested: {len(MINIMAL_PAIRS)}")
    lines.append("")

    lines.append("Top consistent neurons (appear across multiple pairs):")
    for (layer, neuron_idx), count in phase1_results["consistent_neurons"][:15]:
        lines.append(f"  L{layer}:N{neuron_idx} → {count}/{len(MINIMAL_PAIRS)} pairs")

    lines.append("")
    lines.append("Layer divergence ranking (lowest cosine sim = most different):")
    for layer_idx, cos_sim, l2_dist in phase1_results["layer_ranking"][:10]:
        lines.append(f"  Layer {layer_idx:2d}: cosine_sim={cos_sim:.6f}  L2={l2_dist:.4f}")

    lines.append("")
    lines.append(f"Selected steering layers: {phase1_results['top_layers']}")

    # Phase 2 summary
    lines.append("")
    lines.append("─" * 70)
    lines.append("PHASE 2: STEERING VECTORS")
    lines.append("─" * 70)
    lines.append("")
    lines.append(f"Bee prompts used: {len(MINIMAL_PAIRS) + len(EXTRA_BEE_PROMPTS)}")
    lines.append(f"Control prompts used: {len(MINIMAL_PAIRS) + len(EXTRA_CONTROL_PROMPTS)}")

    # Phase 3 summary — group by config and rank
    lines.append("")
    lines.append("─" * 70)
    lines.append("PHASE 3: STEERING GRID SEARCH RESULTS")
    lines.append("─" * 70)
    lines.append("")

    configs = {}
    for r in all_steering_results:
        key = (r["layer_idx"], r["coefficient"], r["last_token_only"])
        if key not in configs:
            configs[key] = {
                "layer_idx": r["layer_idx"],
                "coefficient": r["coefficient"],
                "last_token_only": r["last_token_only"],
                "results": [],
            }
        configs[key]["results"].append(r)

    ranked = []
    for key, cfg in configs.items():
        total_score = sum(r["quality_score"] for r in cfg["results"])
        total_bee = sum(r["bee_keyword_count"] for r in cfg["results"])
        total_rep = sum(1 for r in cfg["results"] if r["has_repetition"])
        total_diff = sum(1 for r in cfg["results"] if r["differs_from_unsteered"])
        ranked.append({
            "layer_idx": cfg["layer_idx"],
            "coefficient": cfg["coefficient"],
            "last_token_only": cfg["last_token_only"],
            "total_quality_score": total_score,
            "total_bee_keywords": total_bee,
            "repetition_count": total_rep,
            "differs_count": total_diff,
            "num_prompts": len(cfg["results"]),
            "details": cfg["results"],
        })

    ranked.sort(key=lambda x: x["total_quality_score"], reverse=True)

    lines.append(f"Configurations tested: {len(ranked)}")
    lines.append(f"Prompts per config: {len(TEST_PROMPTS)}")
    lines.append(f"Total generations: {len(all_steering_results)}")
    lines.append("")
    lines.append("RANKING (best to worst):")
    lines.append("")

    for i, cfg in enumerate(ranked):
        mode = "last-token" if cfg["last_token_only"] else "all-pos"
        lines.append(
            f"#{i+1:2d}  Layer {cfg['layer_idx']:2d}  coeff={cfg['coefficient']:5.1f}  "
            f"mode={mode:10s}  "
            f"score={cfg['total_quality_score']:3d}  "
            f"bee_kw={cfg['total_bee_keywords']}  "
            f"rep={cfg['repetition_count']}/{cfg['num_prompts']}  "
            f"diff={cfg['differs_count']}/{cfg['num_prompts']}"
        )

    # Detailed output for top 5
    lines.append("")
    lines.append("=" * 70)
    lines.append("DETAILED OUTPUT: TOP 5 CONFIGURATIONS")
    lines.append("=" * 70)

    for i, cfg in enumerate(ranked[:5]):
        mode = "last-token" if cfg["last_token_only"] else "all-pos"
        lines.append("")
        lines.append(f"--- #{i+1}: Layer {cfg['layer_idx']}  coeff={cfg['coefficient']}  mode={mode} ---")
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

    # Save ranked summary as JSON (without full text details)
    ranked_summary = []
    for cfg in ranked:
        ranked_summary.append({k: v for k, v in cfg.items() if k != "details"})
    with open(output_dir / "report_ranked.json", "w") as f:
        json.dump(ranked_summary, f, indent=2)

    return ranked


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/contrastive") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    # Save config
    config = {
        "minimal_pairs": MINIMAL_PAIRS,
        "num_extra_bee": len(EXTRA_BEE_PROMPTS),
        "num_extra_control": len(EXTRA_CONTROL_PROMPTS),
        "steering_coefficients": STEERING_COEFFICIENTS,
        "steering_modes": ["all-positions", "last-token-only"],
        "max_tokens": MAX_TOKENS,
        "num_top_layers": NUM_TOP_LAYERS,
        "test_prompts": TEST_PROMPTS,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Load model ───────────────────────────────────────────────────────
    log("Loading model...")
    from src.model_loader import load_model
    model, tokenizer, device = load_model(device="cuda")
    log(f"Model loaded on {device}. Hidden size: {model.config.hidden_size}")

    # Create extractor
    from src.extraction.unified_extractor import GraniteAttentionExtractor
    extractor = GraniteAttentionExtractor(model, tokenizer, device=device)

    # ── Phase 1 ──────────────────────────────────────────────────────────
    phase1_results = run_phase1(extractor)

    # Save Phase 1 results
    phase1_save = {
        "consistent_neurons": [
            {"layer": l, "neuron_idx": n, "count": c}
            for (l, n), c in phase1_results["consistent_neurons"]
        ],
        "layer_ranking": [
            {"layer": int(l), "cosine_sim": float(c), "l2_dist": float(d)}
            for l, c, d in phase1_results["layer_ranking"]
        ],
        "top_layers": phase1_results["top_layers"],
    }
    with open(output_dir / "phase1_results.json", "w") as f:
        json.dump(phase1_save, f, indent=2)
    log(f"Phase 1 results saved to {output_dir / 'phase1_results.json'}")

    # ── Phase 2 ──────────────────────────────────────────────────────────
    top_layers = phase1_results["top_layers"]
    steering_vectors = run_phase2(extractor, top_layers)

    # Save steering vectors
    for layer_idx, vec in steering_vectors.items():
        torch.save(vec, output_dir / f"steering_vector_L{layer_idx}.pt")
    log(f"Steering vectors saved to {output_dir}")

    # ── Phase 3 ──────────────────────────────────────────────────────────
    all_steering_results = run_phase3(model, tokenizer, steering_vectors, device)

    # Save raw steering results
    # Strip tensors for JSON serialization
    steering_save = []
    for r in all_steering_results:
        steering_save.append({
            k: v for k, v in r.items()
            if k not in ("steered_tokens", "unsteered_tokens")
        })
    with open(output_dir / "steering_results.json", "w") as f:
        json.dump(steering_save, f, indent=2)

    # ── Generate report ──────────────────────────────────────────────────
    log("")
    log("Generating report...")
    ranked = generate_report(phase1_results, steering_vectors, all_steering_results, output_dir)

    # Log top 5
    log("")
    log("TOP 5 CONFIGURATIONS:")
    for i, cfg in enumerate(ranked[:5]):
        mode = "last-token" if cfg["last_token_only"] else "all-pos"
        log(f"  #{i+1}: Layer {cfg['layer_idx']} coeff={cfg['coefficient']} mode={mode} "
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
        "approach": "contrastive_minimal_pairs",
        "num_minimal_pairs": len(MINIMAL_PAIRS),
        "total_bee_prompts": len(MINIMAL_PAIRS) + len(EXTRA_BEE_PROMPTS),
        "total_control_prompts": len(MINIMAL_PAIRS) + len(EXTRA_CONTROL_PROMPTS),
        "top_layers": top_layers,
        "coefficients_tested": STEERING_COEFFICIENTS,
        "modes_tested": ["all-positions", "last-token-only"],
        "total_configs": len(ranked),
        "total_generations": len(all_steering_results),
        "best_config": {
            "layer_idx": ranked[0]["layer_idx"],
            "coefficient": ranked[0]["coefficient"],
            "last_token_only": ranked[0]["last_token_only"],
            "quality_score": ranked[0]["total_quality_score"],
            "bee_keywords": ranked[0]["total_bee_keywords"],
        } if ranked else None,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log("")
    log("=" * 60)
    log("I-BEE-M CONTRASTIVE STEERING COMPLETE!")
    log(f"  Time: {minutes}m {seconds}s")
    log(f"  Top layers: {top_layers}")
    log(f"  Total generations: {len(all_steering_results)}")
    if ranked:
        mode = "last-token" if ranked[0]["last_token_only"] else "all-pos"
        log(f"  Best config: Layer {ranked[0]['layer_idx']} "
            f"coeff={ranked[0]['coefficient']} mode={mode} "
            f"(score={ranked[0]['total_quality_score']}, "
            f"bee_kw={ranked[0]['total_bee_keywords']})")
    log(f"  Report: {output_dir / 'report.txt'}")
    log(f"  Results: {output_dir}")
    log("=" * 60)


if __name__ == "__main__":
    main()
