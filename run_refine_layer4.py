#!/usr/bin/env python3
"""I-Bee-M: Refined Layer 4 steering coefficient search.

Reuses the pre-computed steering vector from the contrastive run to do a
finer-grained grid search over coefficients at Layer 4 only.

Skips Phases 1 & 2 entirely — just loads the saved vector and runs generation.

Usage:
    source .venv/bin/activate
    python run_refine_layer4.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import torch

from src.extraction.activation_steering import generate_comparison
from src.model_loader import load_model
from run_contrastive_steering import (
    score_generation,
    TEST_PROMPTS,
    BEE_KEYWORDS,
    MAX_TOKENS,
)

# ── Config ───────────────────────────────────────────────────────────────────

VECTOR_PATH = Path("results/contrastive/20260217_152822/steering_vector_L4.pt")
LAYER_IDX = 4
COEFFICIENTS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]
MODES = [False, True]  # [all-positions, last-token-only]
OUTPUT_BASE = Path("results/contrastive_refined")


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

    # ── Load steering vector ─────────────────────────────────────────────
    log(f"Loading steering vector from {VECTOR_PATH}")
    steering_vec = torch.load(VECTOR_PATH, weights_only=True)
    log(f"  Shape: {steering_vec.shape}  Norm: {steering_vec.norm():.4f}")

    # ── Load model ───────────────────────────────────────────────────────
    log("Loading model...")
    model, tokenizer, device = load_model(device="cuda")
    log(f"Model loaded on {device}. Hidden size: {model.config.hidden_size}")

    # ── Save config ──────────────────────────────────────────────────────
    config = {
        "vector_path": str(VECTOR_PATH),
        "layer_idx": LAYER_IDX,
        "coefficients": COEFFICIENTS,
        "modes": ["all-positions", "last-token-only"],
        "max_tokens": MAX_TOKENS,
        "test_prompts": TEST_PROMPTS,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Grid search ──────────────────────────────────────────────────────
    total_configs = len(COEFFICIENTS) * len(MODES)
    total_gens = total_configs * len(TEST_PROMPTS)
    log(f"Grid: {len(COEFFICIENTS)} coefficients × {len(MODES)} modes "
        f"= {total_configs} configs, {total_gens} total generations")

    all_results = []
    gen_count = 0

    for coeff in COEFFICIENTS:
        for last_token_only in MODES:
            mode_str = "last-token" if last_token_only else "all-pos"
            log(f"  L{LAYER_IDX} coeff={coeff} mode={mode_str}")

            for prompt in TEST_PROMPTS:
                gen_count += 1
                result = generate_comparison(
                    model, tokenizer, prompt, steering_vec,
                    layer_idx=LAYER_IDX,
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
                    "layer_idx": LAYER_IDX,
                    "coefficient": coeff,
                    "last_token_only": last_token_only,
                    "feature_direction_norm": result.feature_direction_norm,
                    **scoring,
                })

            log(f"    {gen_count}/{total_gens} done")

    # ── Save raw results ─────────────────────────────────────────────────
    with open(output_dir / "steering_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"Raw results saved ({len(all_results)} entries)")

    # ── Rank configs ─────────────────────────────────────────────────────
    configs = {}
    for r in all_results:
        key = (r["coefficient"], r["last_token_only"])
        if key not in configs:
            configs[key] = {"coefficient": r["coefficient"],
                            "last_token_only": r["last_token_only"],
                            "results": []}
        configs[key]["results"].append(r)

    ranked = []
    for key, cfg in configs.items():
        total_score = sum(r["quality_score"] for r in cfg["results"])
        total_bee = sum(r["bee_keyword_count"] for r in cfg["results"])
        total_rep = sum(1 for r in cfg["results"] if r["has_repetition"])
        total_diff = sum(1 for r in cfg["results"] if r["differs_from_unsteered"])
        ranked.append({
            "layer_idx": LAYER_IDX,
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

    # ── Generate report ──────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("I-BEE-M REFINED LAYER 4 STEERING REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Source vector: {VECTOR_PATH}")
    lines.append(f"Vector shape:  {steering_vec.shape}")
    lines.append(f"Vector norm:   {steering_vec.norm():.4f}")
    lines.append(f"Layer:         {LAYER_IDX}")
    lines.append(f"Coefficients:  {COEFFICIENTS}")
    lines.append(f"Modes:         all-positions, last-token-only")
    lines.append(f"Prompts/config:{len(TEST_PROMPTS)}")
    lines.append(f"Total gens:    {len(all_results)}")
    lines.append("")
    lines.append("RANKING (best to worst):")
    lines.append("")

    for i, cfg in enumerate(ranked):
        mode = "last-token" if cfg["last_token_only"] else "all-pos"
        lines.append(
            f"#{i+1:2d}  Layer {LAYER_IDX}  coeff={cfg['coefficient']:5.1f}  "
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
        lines.append(f"--- #{i+1}: Layer {LAYER_IDX}  coeff={cfg['coefficient']}  mode={mode} ---")
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

    # Save ranked summary as JSON
    ranked_summary = [{k: v for k, v in cfg.items() if k != "details"}
                      for cfg in ranked]
    with open(output_dir / "report_ranked.json", "w") as f:
        json.dump(ranked_summary, f, indent=2)

    # ── Log top 5 ────────────────────────────────────────────────────────
    log("")
    log("TOP 5 CONFIGURATIONS:")
    for i, cfg in enumerate(ranked[:5]):
        mode = "last-token" if cfg["last_token_only"] else "all-pos"
        log(f"  #{i+1}: coeff={cfg['coefficient']} mode={mode} "
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
        "approach": "contrastive_refined_layer4",
        "vector_source": str(VECTOR_PATH),
        "layer_idx": LAYER_IDX,
        "coefficients_tested": COEFFICIENTS,
        "modes_tested": ["all-positions", "last-token-only"],
        "total_configs": len(ranked),
        "total_generations": len(all_results),
        "best_config": {
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
    log("REFINED LAYER 4 SEARCH COMPLETE!")
    log(f"  Time: {minutes}m {seconds}s")
    log(f"  Total generations: {len(all_results)}")
    if ranked:
        mode = "last-token" if ranked[0]["last_token_only"] else "all-pos"
        log(f"  Best config: coeff={ranked[0]['coefficient']} mode={mode} "
            f"(score={ranked[0]['total_quality_score']}, "
            f"bee_kw={ranked[0]['total_bee_keywords']})")
    log(f"  Report: {output_dir / 'report.txt'}")
    log(f"  Results: {output_dir}")
    log("=" * 60)


if __name__ == "__main__":
    main()
