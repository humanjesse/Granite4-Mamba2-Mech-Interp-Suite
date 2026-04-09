#!/usr/bin/env python3
"""SSM State Injection Steering Experiment.

Tests a novel steering approach: inject contrastive difference vectors
directly into Mamba's SSM recurrent state during generation.

Previous findings:
- Contrastive residual stream steering: 602 bee keywords (L1, coeff=20)
- SAE steering: 0 bee keywords (all configs)
- SAE features are orthogonal to the actual causal direction (cos=0.042)

Hypothesis: Operating in Mamba's native SSM state space may be even
more effective because it targets the recurrent dynamics that propagate
information across tokens.

Phases:
1. Load model & architecture
2. Extract SSM states for bee/control prompts
3. Compute contrastive SSM vectors + alignment analysis
4. SSM injection steering grid search
5. Residual stream baseline (for head-to-head comparison)
6. Comparison report
7. Save & log

Usage:
    source .venv/bin/activate
    python run_ssm_state_injection.py
"""

import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

# ── Reuse prompts and scoring from contrastive steering ──────────────────────

from run_contrastive_steering import (
    MINIMAL_PAIRS,
    EXTRA_BEE_PROMPTS,
    EXTRA_CONTROL_PROMPTS,
    TEST_PROMPTS,
    BEE_KEYWORDS,
    count_bee_keywords,
    detect_repetition,
    text_differs,
    score_generation,
)

# ── Configuration ────────────────────────────────────────────────────────────

TARGET_LAYERS = [0, 1, 4, 9]
SSM_COEFFICIENTS = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0]
MAX_TOKENS = 100


# ── Utilities ────────────────────────────────────────────────────────────────

def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ── Phase 1: Load Model & Architecture ──────────────────────────────────────

def run_phase1():
    """Load model and report architecture details."""
    log("=" * 70)
    log("PHASE 1: Load Model & Architecture")
    log("=" * 70)

    from src.model_loader import load_model, get_model_info
    model, tokenizer, device = load_model(device="cuda")

    info = get_model_info(model)
    log(f"Model type: {info['model_type']}")
    log(f"Layers: {info['num_layers']} (hidden_size={info['hidden_size']})")
    log(f"Mamba heads: {info['mamba_n_heads']}, d_head: {info['mamba_d_head']}, "
        f"d_state: {info['mamba_d_state']}, n_groups: {info['mamba_n_groups']}")
    log(f"SSM state shape per layer: [1, {info['mamba_n_heads']}, "
        f"{info['mamba_d_head']}, {info['mamba_d_state']}]")
    log(f"Layer types: {info['layer_types']}")
    log(f"Device: {device}")

    return model, tokenizer, device, info


# ── Phase 2: Extract SSM States ─────────────────────────────────────────────

def run_phase2(model, tokenizer, device):
    """Extract SSM states for bee and control prompts."""
    from src.extraction.ssm_steering import extract_ssm_states

    log("")
    log("=" * 70)
    log("PHASE 2: Extract SSM States")
    log("=" * 70)

    bee_prompts = [p[0] for p in MINIMAL_PAIRS] + EXTRA_BEE_PROMPTS
    control_prompts = [p[1] for p in MINIMAL_PAIRS] + EXTRA_CONTROL_PROMPTS

    log(f"Bee prompts: {len(bee_prompts)}")
    log(f"Control prompts: {len(control_prompts)}")
    log(f"Target layers: {TARGET_LAYERS}")

    log("Extracting bee SSM states...")
    bee_states = extract_ssm_states(model, tokenizer, bee_prompts, TARGET_LAYERS, device)
    log(f"  Done. Shape per state: {bee_states[TARGET_LAYERS[0]][0].shape}")

    log("Extracting control SSM states...")
    control_states = extract_ssm_states(model, tokenizer, control_prompts, TARGET_LAYERS, device)
    log(f"  Done.")

    # Measure SSM state norms per layer
    ssm_norms = {}
    for layer_idx in TARGET_LAYERS:
        all_norms = []
        for s in bee_states[layer_idx] + control_states[layer_idx]:
            all_norms.append(s.norm().item())
        avg_norm = sum(all_norms) / len(all_norms)
        ssm_norms[layer_idx] = avg_norm
        log(f"  Layer {layer_idx} avg SSM state Frobenius norm: {avg_norm:.4f} "
            f"(min={min(all_norms):.4f}, max={max(all_norms):.4f})")

    return bee_states, control_states, ssm_norms


# ── Phase 3: Compute Contrastive SSM Vectors ────────────────────────────────

def run_phase3(model, tokenizer, device, bee_states, control_states, ssm_norms):
    """Compute contrastive SSM vectors and analyze alignment with residual stream."""
    from src.extraction.ssm_steering import compute_ssm_contrastive_vectors

    log("")
    log("=" * 70)
    log("PHASE 3: Compute Contrastive SSM Vectors")
    log("=" * 70)

    ssm_vectors = compute_ssm_contrastive_vectors(bee_states, control_states, normalize=True)

    for layer_idx in TARGET_LAYERS:
        vec = ssm_vectors[layer_idx]
        log(f"  Layer {layer_idx}: shape={vec.shape}, "
            f"norm={vec.norm().item():.6f} (normalized)")

    # Also compute residual stream contrastive vectors for alignment analysis
    log("")
    log("Computing residual stream contrastive vectors for alignment analysis...")
    from src.extraction.unified_extractor import GraniteAttentionExtractor

    extractor = GraniteAttentionExtractor(model, tokenizer, device=device)

    bee_prompts = [p[0] for p in MINIMAL_PAIRS] + EXTRA_BEE_PROMPTS
    control_prompts = [p[1] for p in MINIMAL_PAIRS] + EXTRA_CONTROL_PROMPTS

    residual_vectors = {}
    for layer_idx in TARGET_LAYERS:
        bee_residuals = []
        for prompt in bee_prompts:
            result = extractor.extract(prompt)
            rs = result["residual_stream"][layer_idx][0].float().mean(dim=0)
            bee_residuals.append(rs)

        control_residuals = []
        for prompt in control_prompts:
            result = extractor.extract(prompt)
            rs = result["residual_stream"][layer_idx][0].float().mean(dim=0)
            control_residuals.append(rs)

        mean_bee = torch.stack(bee_residuals).mean(dim=0)
        mean_control = torch.stack(control_residuals).mean(dim=0)
        diff = mean_bee - mean_control
        diff = diff / diff.norm()
        residual_vectors[layer_idx] = diff

    log("")
    log("Alignment Analysis (SSM flattened vs Residual stream):")
    log("  (Low cosine = different directions, which is expected since")
    log("   SSM state and residual stream are fundamentally different spaces)")
    for layer_idx in TARGET_LAYERS:
        ssm_flat = ssm_vectors[layer_idx].flatten()
        res_flat = residual_vectors[layer_idx].flatten()
        # They have different dimensions, so we can't directly compare
        log(f"  Layer {layer_idx}: SSM vec dims={ssm_flat.shape[0]}, "
            f"Residual vec dims={res_flat.shape[0]}")
        log(f"    SSM state norm (pre-normalize): {ssm_norms[layer_idx]:.4f}")

    return ssm_vectors, residual_vectors


# ── Phase 4: SSM Injection Steering ─────────────────────────────────────────

def run_phase4(model, tokenizer, device, ssm_vectors, ssm_norms):
    """Run SSM state injection steering with grid search."""
    from src.extraction.ssm_steering import generate_ssm_comparison

    log("")
    log("=" * 70)
    log("PHASE 4: SSM Injection Steering Grid Search")
    log("=" * 70)

    total_configs = len(TARGET_LAYERS) * len(SSM_COEFFICIENTS)
    total_gens = total_configs * len(TEST_PROMPTS)

    log(f"Grid: {len(TARGET_LAYERS)} layers × {len(SSM_COEFFICIENTS)} coefficients "
        f"= {total_configs} configs, {total_gens} total generations")
    log(f"Coefficients: {SSM_COEFFICIENTS}")
    log(f"Max tokens: {MAX_TOKENS}")

    all_results = []
    unsteered_cache = {}  # Cache unsteered baselines
    gen_count = 0

    for layer_idx in TARGET_LAYERS:
        vec = ssm_vectors[layer_idx]
        log(f"\n  Layer {layer_idx} (SSM state norm: {ssm_norms[layer_idx]:.4f})")

        for coeff in SSM_COEFFICIENTS:
            config_name = f"L{layer_idx}_coeff{coeff}"
            log(f"    {config_name}")

            ssm_injections = {
                layer_idx: {"vector": vec, "coefficient": coeff}
            }

            config_bee_kw = 0
            for prompt in TEST_PROMPTS:
                gen_count += 1
                result = generate_ssm_comparison(
                    model, tokenizer, prompt,
                    ssm_injections=ssm_injections,
                    max_tokens=MAX_TOKENS,
                    device=device,
                    config_name=config_name,
                    unsteered_cache=unsteered_cache,
                )

                scoring = score_generation(result.steered_text, result.unsteered_text)
                config_bee_kw += scoring["bee_keyword_count"]

                all_results.append({
                    "method": "ssm_injection",
                    "prompt": prompt,
                    "steered_text": result.steered_text,
                    "unsteered_text": result.unsteered_text,
                    "layer_idx": layer_idx,
                    "coefficient": coeff,
                    "config_name": config_name,
                    **scoring,
                })

            log(f"      {gen_count}/{total_gens} done | "
                f"bee_kw this config: {config_bee_kw}")

    return all_results


# ── Phase 5: Residual Stream Baseline ────────────────────────────────────────

def run_phase5(model, tokenizer, device, residual_vectors):
    """Run residual stream contrastive steering as baseline comparison."""
    from src.extraction.activation_steering import generate_comparison

    log("")
    log("=" * 70)
    log("PHASE 5: Residual Stream Contrastive Baseline")
    log("=" * 70)

    # Use same coefficient range for fair comparison
    residual_coefficients = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0]

    total_configs = len(TARGET_LAYERS) * len(residual_coefficients)
    total_gens = total_configs * len(TEST_PROMPTS)

    log(f"Grid: {len(TARGET_LAYERS)} layers × {len(residual_coefficients)} coefficients "
        f"= {total_configs} configs, {total_gens} total generations")

    all_results = []
    gen_count = 0

    for layer_idx in TARGET_LAYERS:
        vec = residual_vectors[layer_idx]
        log(f"\n  Layer {layer_idx}")

        for coeff in residual_coefficients:
            config_name = f"residual_L{layer_idx}_coeff{coeff}"
            log(f"    {config_name}")

            config_bee_kw = 0
            for prompt in TEST_PROMPTS:
                gen_count += 1
                result = generate_comparison(
                    model, tokenizer, prompt, vec,
                    layer_idx=layer_idx,
                    coefficient=coeff,
                    max_tokens=MAX_TOKENS,
                    device=device,
                    last_token_only=False,
                )

                scoring = score_generation(result.steered_text, result.unsteered_text)
                config_bee_kw += scoring["bee_keyword_count"]

                all_results.append({
                    "method": "residual_contrastive",
                    "prompt": prompt,
                    "steered_text": result.steered_text,
                    "unsteered_text": result.unsteered_text,
                    "layer_idx": layer_idx,
                    "coefficient": coeff,
                    "config_name": config_name,
                    **scoring,
                })

            log(f"      {gen_count}/{total_gens} done | "
                f"bee_kw this config: {config_bee_kw}")

    return all_results


# ── Phase 6: Comparison Report ───────────────────────────────────────────────

def generate_report(
    model_info,
    ssm_norms,
    ssm_results,
    residual_results,
    output_dir,
):
    """Generate comprehensive comparison report."""
    lines = []
    lines.append("=" * 70)
    lines.append("SSM STATE INJECTION vs RESIDUAL STREAM CONTRASTIVE STEERING")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Architecture
    lines.append("─" * 70)
    lines.append("ARCHITECTURE")
    lines.append("─" * 70)
    lines.append(f"Model: {model_info['model_type']}")
    lines.append(f"Layers: {model_info['num_layers']} (hidden_size={model_info['hidden_size']})")
    lines.append(f"Mamba SSM: heads={model_info['mamba_n_heads']}, "
                 f"d_head={model_info['mamba_d_head']}, "
                 f"d_state={model_info['mamba_d_state']}")
    lines.append(f"SSM state shape: [1, {model_info['mamba_n_heads']}, "
                 f"{model_info['mamba_d_head']}, {model_info['mamba_d_state']}]")
    lines.append(f"SSM state size: {model_info['mamba_n_heads'] * model_info['mamba_d_head'] * model_info['mamba_d_state']} floats per layer")
    lines.append("")
    lines.append("SSM state norms by layer:")
    for layer_idx in TARGET_LAYERS:
        lines.append(f"  Layer {layer_idx}: avg Frobenius norm = {ssm_norms[layer_idx]:.4f}")

    # Helper to rank results by config
    def rank_results(results, method_name):
        configs = {}
        for r in results:
            key = (r["layer_idx"], r["coefficient"])
            if key not in configs:
                configs[key] = {
                    "layer_idx": r["layer_idx"],
                    "coefficient": r["coefficient"],
                    "results": [],
                }
            configs[key]["results"].append(r)

        ranked = []
        for key, cfg in configs.items():
            total_bee = sum(r["bee_keyword_count"] for r in cfg["results"])
            total_score = sum(r["quality_score"] for r in cfg["results"])
            total_rep = sum(1 for r in cfg["results"] if r["has_repetition"])
            total_diff = sum(1 for r in cfg["results"] if r["differs_from_unsteered"])
            ranked.append({
                "layer_idx": cfg["layer_idx"],
                "coefficient": cfg["coefficient"],
                "total_bee_keywords": total_bee,
                "total_quality_score": total_score,
                "repetition_count": total_rep,
                "differs_count": total_diff,
                "num_prompts": len(cfg["results"]),
                "details": cfg["results"],
            })

        ranked.sort(key=lambda x: x["total_bee_keywords"], reverse=True)
        return ranked

    ssm_ranked = rank_results(ssm_results, "SSM")
    residual_ranked = rank_results(residual_results, "Residual")

    # SSM results
    lines.append("")
    lines.append("─" * 70)
    lines.append("SSM STATE INJECTION RESULTS")
    lines.append("─" * 70)
    lines.append(f"Configurations tested: {len(ssm_ranked)}")
    lines.append(f"Total generations: {len(ssm_results)}")
    lines.append("")
    lines.append("RANKING (by bee keywords, best to worst):")
    lines.append("")
    for i, cfg in enumerate(ssm_ranked):
        lines.append(
            f"#{i+1:2d}  Layer {cfg['layer_idx']:2d}  coeff={cfg['coefficient']:6.1f}  "
            f"bee_kw={cfg['total_bee_keywords']:4d}  "
            f"score={cfg['total_quality_score']:4d}  "
            f"rep={cfg['repetition_count']}/{cfg['num_prompts']}  "
            f"diff={cfg['differs_count']}/{cfg['num_prompts']}"
        )

    # Residual results
    lines.append("")
    lines.append("─" * 70)
    lines.append("RESIDUAL STREAM CONTRASTIVE RESULTS")
    lines.append("─" * 70)
    lines.append(f"Configurations tested: {len(residual_ranked)}")
    lines.append(f"Total generations: {len(residual_results)}")
    lines.append("")
    lines.append("RANKING (by bee keywords, best to worst):")
    lines.append("")
    for i, cfg in enumerate(residual_ranked):
        lines.append(
            f"#{i+1:2d}  Layer {cfg['layer_idx']:2d}  coeff={cfg['coefficient']:6.1f}  "
            f"bee_kw={cfg['total_bee_keywords']:4d}  "
            f"score={cfg['total_quality_score']:4d}  "
            f"rep={cfg['repetition_count']}/{cfg['num_prompts']}  "
            f"diff={cfg['differs_count']}/{cfg['num_prompts']}"
        )

    # Head-to-head comparison
    lines.append("")
    lines.append("─" * 70)
    lines.append("HEAD-TO-HEAD COMPARISON")
    lines.append("─" * 70)
    lines.append("")

    # Build lookup for residual results
    residual_lookup = {}
    for cfg in residual_ranked:
        residual_lookup[(cfg["layer_idx"], cfg["coefficient"])] = cfg

    lines.append(f"{'Layer':>5} {'Coeff':>7} {'SSM bee_kw':>11} {'Residual bee_kw':>15} {'Winner':>10}")
    lines.append("-" * 55)

    for cfg in ssm_ranked:
        key = (cfg["layer_idx"], cfg["coefficient"])
        res_cfg = residual_lookup.get(key)
        ssm_kw = cfg["total_bee_keywords"]
        res_kw = res_cfg["total_bee_keywords"] if res_cfg else "N/A"

        if res_cfg:
            if ssm_kw > res_cfg["total_bee_keywords"]:
                winner = "SSM"
            elif ssm_kw < res_cfg["total_bee_keywords"]:
                winner = "Residual"
            else:
                winner = "Tie"
        else:
            winner = "?"

        lines.append(f"L{cfg['layer_idx']:>4} {cfg['coefficient']:>7.1f} "
                     f"{ssm_kw:>11} {str(res_kw):>15} {winner:>10}")

    # Summary statistics
    lines.append("")
    lines.append("─" * 70)
    lines.append("SUMMARY")
    lines.append("─" * 70)
    lines.append("")

    best_ssm = ssm_ranked[0] if ssm_ranked else None
    best_res = residual_ranked[0] if residual_ranked else None

    if best_ssm:
        lines.append(f"Best SSM config: Layer {best_ssm['layer_idx']}, "
                     f"coeff={best_ssm['coefficient']} → "
                     f"{best_ssm['total_bee_keywords']} bee keywords")
    if best_res:
        lines.append(f"Best Residual config: Layer {best_res['layer_idx']}, "
                     f"coeff={best_res['coefficient']} → "
                     f"{best_res['total_bee_keywords']} bee keywords")

    total_ssm_kw = sum(r["bee_keyword_count"] for r in ssm_results)
    total_res_kw = sum(r["bee_keyword_count"] for r in residual_results)
    lines.append("")
    lines.append(f"Total SSM bee keywords (all configs): {total_ssm_kw}")
    lines.append(f"Total Residual bee keywords (all configs): {total_res_kw}")

    if total_ssm_kw > total_res_kw:
        lines.append("VERDICT: SSM state injection produces MORE bee content overall")
    elif total_ssm_kw < total_res_kw:
        lines.append("VERDICT: Residual stream contrastive produces MORE bee content overall")
    else:
        lines.append("VERDICT: Both methods produce equal bee content")

    # Detailed outputs for top configs
    lines.append("")
    lines.append("=" * 70)
    lines.append("DETAILED OUTPUT: TOP 3 SSM CONFIGS")
    lines.append("=" * 70)

    for i, cfg in enumerate(ssm_ranked[:3]):
        lines.append("")
        lines.append(f"--- SSM #{i+1}: Layer {cfg['layer_idx']} coeff={cfg['coefficient']} "
                     f"(bee_kw={cfg['total_bee_keywords']}) ---")
        lines.append("")
        for r in cfg["details"]:
            lines.append(f"  Prompt: \"{r['prompt']}\"")
            lines.append(f"  Steered:   {r['steered_text'][:300]}")
            lines.append(f"  Unsteered: {r['unsteered_text'][:300]}")
            lines.append(f"  [bee_kw={r['bee_keyword_count']} rep={r['has_repetition']} "
                         f"diff={r['differs_from_unsteered']} score={r['quality_score']}]")
            lines.append("")

    lines.append("")
    lines.append("=" * 70)
    lines.append("DETAILED OUTPUT: TOP 3 RESIDUAL CONFIGS")
    lines.append("=" * 70)

    for i, cfg in enumerate(residual_ranked[:3]):
        lines.append("")
        lines.append(f"--- Residual #{i+1}: Layer {cfg['layer_idx']} coeff={cfg['coefficient']} "
                     f"(bee_kw={cfg['total_bee_keywords']}) ---")
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

    log(f"Report written to {output_dir / 'report.txt'}")

    return ssm_ranked, residual_ranked


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/ssm_state_injection") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    # Save config
    config = {
        "target_layers": TARGET_LAYERS,
        "ssm_coefficients": SSM_COEFFICIENTS,
        "max_tokens": MAX_TOKENS,
        "test_prompts": TEST_PROMPTS,
        "num_bee_prompts": len(MINIMAL_PAIRS) + len(EXTRA_BEE_PROMPTS),
        "num_control_prompts": len(MINIMAL_PAIRS) + len(EXTRA_CONTROL_PROMPTS),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Phase 1 ──────────────────────────────────────────────────────────
    model, tokenizer, device, model_info = run_phase1()

    # ── Phase 2 ──────────────────────────────────────────────────────────
    bee_states, control_states, ssm_norms = run_phase2(model, tokenizer, device)

    # ── Phase 3 ──────────────────────────────────────────────────────────
    ssm_vectors, residual_vectors = run_phase3(
        model, tokenizer, device, bee_states, control_states, ssm_norms
    )

    # Save vectors
    for layer_idx in TARGET_LAYERS:
        torch.save(ssm_vectors[layer_idx], output_dir / f"ssm_vector_L{layer_idx}.pt")
        torch.save(residual_vectors[layer_idx], output_dir / f"residual_vector_L{layer_idx}.pt")
    log(f"Vectors saved to {output_dir}")

    # Free extraction memory
    del bee_states, control_states
    torch.cuda.empty_cache() if device != "cpu" else None

    # ── Phase 4 ──────────────────────────────────────────────────────────
    ssm_results = run_phase4(model, tokenizer, device, ssm_vectors, ssm_norms)

    # Save SSM results incrementally
    ssm_save = []
    for r in ssm_results:
        ssm_save.append({k: v for k, v in r.items()})
    with open(output_dir / "ssm_results.json", "w") as f:
        json.dump(ssm_save, f, indent=2)
    log(f"SSM results saved ({len(ssm_results)} generations)")

    # ── Phase 5 ──────────────────────────────────────────────────────────
    residual_results = run_phase5(model, tokenizer, device, residual_vectors)

    # Save residual results
    residual_save = []
    for r in residual_results:
        residual_save.append({k: v for k, v in r.items()})
    with open(output_dir / "residual_results.json", "w") as f:
        json.dump(residual_save, f, indent=2)
    log(f"Residual results saved ({len(residual_results)} generations)")

    # ── Phase 6 ──────────────────────────────────────────────────────────
    log("")
    log("Generating comparison report...")
    ssm_ranked, residual_ranked = generate_report(
        model_info, ssm_norms, ssm_results, residual_results, output_dir
    )

    # Log top results
    log("")
    log("=" * 70)
    log("TOP SSM CONFIGS:")
    for i, cfg in enumerate(ssm_ranked[:5]):
        log(f"  #{i+1}: Layer {cfg['layer_idx']} coeff={cfg['coefficient']} → "
            f"bee_kw={cfg['total_bee_keywords']} score={cfg['total_quality_score']}")

    log("")
    log("TOP RESIDUAL CONFIGS:")
    for i, cfg in enumerate(residual_ranked[:5]):
        log(f"  #{i+1}: Layer {cfg['layer_idx']} coeff={cfg['coefficient']} → "
            f"bee_kw={cfg['total_bee_keywords']} score={cfg['total_quality_score']}")

    # ── Phase 7 ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    log("")
    log("=" * 70)
    log(f"EXPERIMENT COMPLETE in {hours}h {minutes}m {seconds}s")
    log(f"Results: {output_dir}")
    log("=" * 70)

    # Save summary
    summary = {
        "elapsed_seconds": elapsed,
        "ssm_total_generations": len(ssm_results),
        "residual_total_generations": len(residual_results),
        "ssm_total_bee_keywords": sum(r["bee_keyword_count"] for r in ssm_results),
        "residual_total_bee_keywords": sum(r["bee_keyword_count"] for r in residual_results),
        "best_ssm": {
            "layer": ssm_ranked[0]["layer_idx"],
            "coefficient": ssm_ranked[0]["coefficient"],
            "bee_keywords": ssm_ranked[0]["total_bee_keywords"],
        } if ssm_ranked else None,
        "best_residual": {
            "layer": residual_ranked[0]["layer_idx"],
            "coefficient": residual_ranked[0]["coefficient"],
            "bee_keywords": residual_ranked[0]["total_bee_keywords"],
        } if residual_ranked else None,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
