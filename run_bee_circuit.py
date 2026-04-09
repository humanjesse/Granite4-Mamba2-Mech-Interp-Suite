#!/usr/bin/env python3
"""Circuit tracing: how the bee steering vector propagates through Granite 4.0.

Traces the causal pathway of the bee concept through all 32 layers
(28 Mamba-2 + 4 Transformer) using:
  Phase 1: Load model, architecture map, saved steering vector
  Phase 2: Multi-pair circuit discovery (path patching in DETAILED mode)
  Phase 3: Layer importance profiling (activation patching + zero ablation)
  Phase 4: Logit lens analysis (prediction evolution across layers)
  Phase 5: Steering vector propagation (how L1 injection ripples through layers)

Usage:
    source .venv/bin/activate
    python run_bee_circuit.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────

# Minimal pairs for circuit discovery.
# Each needs clean/corrupted prompts + correct/incorrect next-token predictions.
# The "correct" token is what the model predicts after the clean prompt;
# "incorrect" is what it predicts after the corrupted prompt.
# We use the subject noun itself as the token to trace, with leading space
# for proper tokenization.
CIRCUIT_PAIRS = [
    {
        "clean": "The bee buzzed loudly",
        "corrupted": "The car buzzed loudly",
        "correct": " bee",
        "incorrect": " car",
    },
    {
        "clean": "The bee landed on the flower",
        "corrupted": "The bird landed on the flower",
        "correct": " bee",
        "incorrect": " bird",
    },
    {
        "clean": "I saw a bee today",
        "corrupted": "I saw a dog today",
        "correct": " bee",
        "incorrect": " dog",
    },
    {
        "clean": "The bee flew past",
        "corrupted": "The fly flew past",
        "correct": " bee",
        "incorrect": " fly",
    },
    {
        "clean": "A bee appeared suddenly",
        "corrupted": "A fox appeared suddenly",
        "correct": " bee",
        "incorrect": " fox",
    },
]

CIRCUIT_THRESHOLD = 0.05  # Lower threshold to catch subtle paths

# Prompts for steering propagation analysis (Phase 5)
PROPAGATION_PROMPTS = [
    "The weather today is",
    "I went to the store to buy",
    "In the garden, I noticed",
]

STEERING_VECTOR_PATH = Path("results/contrastive/20260217_152822/steering_vector_L1.pt")
STEERING_LAYER = 1
STEERING_COEFFICIENT = 20.0  # Best coefficient from prior experiments


# ── Utilities ────────────────────────────────────────────────────────────────

def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ── Phase 1: Load Model & Architecture ───────────────────────────────────────

def run_phase1():
    """Load model, create architecture map, load steering vector."""
    log("=" * 70)
    log("PHASE 1: Load Model & Architecture")
    log("=" * 70)

    from src.model_loader import load_model
    model, tokenizer, device = load_model(device="cuda")

    from src.architecture_map import ArchitectureMap
    arch_map = ArchitectureMap(model.config)

    log(f"Model loaded on {device}")
    log(f"Architecture: {arch_map.summary()}")
    log(f"Hidden size: {model.config.hidden_size}")

    # Load saved steering vector
    bee_vector = torch.load(STEERING_VECTOR_PATH, map_location="cpu", weights_only=True)
    log(f"Loaded bee steering vector from {STEERING_VECTOR_PATH}")
    log(f"  Shape: {bee_vector.shape}, Norm: {bee_vector.norm():.4f}")

    return model, tokenizer, device, arch_map, bee_vector


# ── Phase 2: Multi-Pair Circuit Discovery ────────────────────────────────────

def run_phase2(model, tokenizer, arch_map, device, output_dir):
    """Run path patching circuit discovery on all minimal pairs."""
    from src.extraction.circuit_discovery import CircuitDiscoveryEngine, SweepGranularity

    log("")
    log("=" * 70)
    log("PHASE 2: Multi-Pair Circuit Discovery (Path Patching)")
    log("=" * 70)

    engine = CircuitDiscoveryEngine(model, tokenizer, arch_map, device)
    num_layers = arch_map.num_layers
    all_results = []
    all_path_matrices = []
    all_layer_importance = []

    for i, pair in enumerate(CIRCUIT_PAIRS):
        log(f"\n  Pair {i+1}/{len(CIRCUIT_PAIRS)}: "
            f"\"{pair['clean']}\" vs \"{pair['corrupted']}\"")
        log(f"  Tracing: {pair['correct']} vs {pair['incorrect']}")

        def progress(step, total):
            if step % 50 == 0 or step == total:
                log(f"    [{step}/{total}]")

        try:
            result = engine.find_circuit(
                clean_prompt=pair["clean"],
                corrupted_prompt=pair["corrupted"],
                correct_token=pair["correct"],
                incorrect_token=pair["incorrect"],
                threshold=CIRCUIT_THRESHOLD,
                granularity=SweepGranularity.DETAILED,
                progress_callback=progress,
            )

            all_results.append(result)
            all_path_matrices.append(result.path_matrix)
            all_layer_importance.append(result.layer_importance)

            # Log highlights
            log(f"  Clean logit diff: {result.clean_logit_diff:+.3f}, "
                f"Corrupted: {result.corrupted_logit_diff:+.3f}")
            log(f"  Circuit: {len(result.circuit_nodes)} nodes, "
                f"{len(result.circuit_edges)} edges")

            top_nodes = sorted(result.circuit_nodes,
                               key=lambda n: abs(n.importance), reverse=True)[:3]
            for n in top_nodes:
                log(f"    Node L{n.layer_idx} ({n.layer_type}): {n.importance:+.3f}")

            # Save individual result
            save_circuit_result(result, output_dir / f"circuit_pair_{i+1}.json", arch_map)

            # Clear engine cache between pairs to save memory
            engine.clear_cache()

        except Exception as e:
            log(f"  ERROR on pair {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        log("WARNING: No circuit results obtained!")
        return None, None, all_results

    # Aggregate across pairs
    log("\n" + "─" * 70)
    log("AGGREGATION: Consensus Circuit")
    log("─" * 70)

    avg_path_matrix = torch.stack(all_path_matrices).mean(dim=0)
    avg_layer_importance = torch.stack(all_layer_importance).mean(dim=0)

    # Identify consistent layers (above threshold in ≥3/5 pairs)
    consistency_count = torch.zeros(num_layers)
    for imp in all_layer_importance:
        consistency_count += (imp.abs() >= CIRCUIT_THRESHOLD).float()

    consistent_layers = [
        i for i in range(num_layers) if consistency_count[i] >= 3
    ]

    log(f"Layers consistently important (≥3/5 pairs): {consistent_layers}")

    # Top average importance
    sorted_layers = sorted(range(num_layers),
                           key=lambda i: abs(avg_layer_importance[i].item()),
                           reverse=True)
    log("Top 10 layers by average importance:")
    for idx in sorted_layers[:10]:
        lt = arch_map.layer_type(idx)
        log(f"  Layer {idx:2d} ({lt:>5s}): {avg_layer_importance[idx]:+.4f} "
            f"(consistent in {int(consistency_count[idx])}/5 pairs)")

    # Top average paths
    log("\nTop 10 paths by average recovery:")
    upper_indices = torch.triu_indices(num_layers, num_layers, offset=1)
    path_values = avg_path_matrix[upper_indices[0], upper_indices[1]]
    top_path_idx = path_values.abs().argsort(descending=True)[:10]
    for fi in top_path_idx:
        s = upper_indices[0][fi].item()
        t = upper_indices[1][fi].item()
        val = path_values[fi].item()
        s_type = "M" if arch_map.layer_type(s) == "mamba" else "T"
        t_type = "M" if arch_map.layer_type(t) == "mamba" else "T"
        log(f"  L{s}{s_type} → L{t}{t_type}: {val:+.4f}")

    # Save aggregated data
    torch.save(avg_path_matrix, output_dir / "avg_path_matrix.pt")
    torch.save(avg_layer_importance, output_dir / "avg_layer_importance.pt")

    return avg_path_matrix, avg_layer_importance, all_results


# ── Phase 3: Layer Importance Profiling ──────────────────────────────────────

def run_phase3(model, tokenizer, arch_map, device, output_dir):
    """Activation patching + zero ablation to profile layer importance."""
    from src.extraction.causal_intervention import CausalInterventionEngine, InterventionType

    log("")
    log("=" * 70)
    log("PHASE 3: Layer Importance Profiling (Activation Patching)")
    log("=" * 70)

    engine = CausalInterventionEngine(model, tokenizer, arch_map, device)
    num_layers = arch_map.num_layers

    # Activation patching sweep
    log("\n--- Activation Patching (which layers carry the bee signal?) ---")
    all_patch_recovery = []

    for i, pair in enumerate(CIRCUIT_PAIRS):
        log(f"  Pair {i+1}/{len(CIRCUIT_PAIRS)}: \"{pair['clean'][:40]}...\"")

        def progress(step, total):
            if step % 8 == 0 or step == total:
                log(f"    [{step}/{total}]")

        result = engine.sweep_layers(
            clean_prompt=pair["clean"],
            corrupted_prompt=pair["corrupted"],
            correct_token=pair["correct"],
            incorrect_token=pair["incorrect"],
            intervention_type=InterventionType.ACTIVATION_PATCH,
            progress_callback=progress,
        )
        all_patch_recovery.append(result.recovery_matrix)
        engine.clear_cache()

    avg_patch = torch.stack(all_patch_recovery).mean(dim=0)

    # Zero ablation sweep
    log("\n--- Zero Ablation (which layers are necessary?) ---")
    all_ablation_recovery = []

    for i, pair in enumerate(CIRCUIT_PAIRS):
        log(f"  Pair {i+1}/{len(CIRCUIT_PAIRS)}: \"{pair['clean'][:40]}...\"")

        def progress(step, total):
            if step % 8 == 0 or step == total:
                log(f"    [{step}/{total}]")

        result = engine.sweep_layers(
            clean_prompt=pair["clean"],
            corrupted_prompt=pair["corrupted"],
            correct_token=pair["correct"],
            incorrect_token=pair["incorrect"],
            intervention_type=InterventionType.ZERO_ABLATION,
            progress_callback=progress,
        )
        all_ablation_recovery.append(result.recovery_matrix)
        engine.clear_cache()

    avg_ablation = torch.stack(all_ablation_recovery).mean(dim=0)

    # Report
    log("\n" + "─" * 70)
    log("LAYER IMPORTANCE RANKING")
    log("─" * 70)

    log("\nActivation Patching (recovery = layer carries bee signal):")
    sorted_patch = sorted(range(num_layers),
                          key=lambda i: abs(avg_patch[i].item()), reverse=True)
    for idx in sorted_patch[:10]:
        lt = arch_map.layer_type(idx)
        log(f"  Layer {idx:2d} ({lt:>5s}): recovery={avg_patch[idx]:+.4f}")

    log("\nZero Ablation (negative = layer is necessary for prediction):")
    sorted_ablation = sorted(range(num_layers),
                             key=lambda i: avg_ablation[i].item())
    for idx in sorted_ablation[:10]:
        lt = arch_map.layer_type(idx)
        log(f"  Layer {idx:2d} ({lt:>5s}): recovery={avg_ablation[idx]:+.4f}")

    # Layer type breakdown
    log("\nMamba vs Transformer importance:")
    mamba_patch = [avg_patch[i].item() for i in arch_map.mamba_indices]
    attn_patch = [avg_patch[i].item() for i in arch_map.attention_indices]
    log(f"  Mamba   avg patch recovery: {np.mean(np.abs(mamba_patch)):.4f} "
        f"(max: {max(np.abs(mamba_patch)):.4f})")
    log(f"  Transf. avg patch recovery: {np.mean(np.abs(attn_patch)):.4f} "
        f"(max: {max(np.abs(attn_patch)):.4f})")

    # Save
    torch.save(avg_patch, output_dir / "avg_patch_recovery.pt")
    torch.save(avg_ablation, output_dir / "avg_ablation_recovery.pt")

    return avg_patch, avg_ablation


# ── Phase 4: Logit Lens Analysis ─────────────────────────────────────────────

def run_phase4(model, tokenizer, arch_map, device, output_dir):
    """Track how predictions evolve across layers for clean vs corrupted."""
    from src.extraction.unified_extractor import GraniteAttentionExtractor

    log("")
    log("=" * 70)
    log("PHASE 4: Logit Lens Analysis")
    log("=" * 70)

    extractor = GraniteAttentionExtractor(model, tokenizer, device=device)
    lm_head = model.lm_head
    num_layers = arch_map.num_layers

    all_logit_lens = []

    for i, pair in enumerate(CIRCUIT_PAIRS):
        log(f"\n  Pair {i+1}/{len(CIRCUIT_PAIRS)}")

        # Extract residual streams
        clean_result = extractor.extract(pair["clean"])
        corrupted_result = extractor.extract(pair["corrupted"])

        pair_data = {
            "clean_prompt": pair["clean"],
            "corrupted_prompt": pair["corrupted"],
            "correct": pair["correct"],
            "incorrect": pair["incorrect"],
            "layers": [],
        }

        # Resolve token IDs
        correct_ids = tokenizer.encode(pair["correct"], add_special_tokens=False)
        incorrect_ids = tokenizer.encode(pair["incorrect"], add_special_tokens=False)
        correct_id = correct_ids[-1] if correct_ids else -1
        incorrect_id = incorrect_ids[-1] if incorrect_ids else -1

        with torch.no_grad():
            for layer_idx in range(num_layers):
                # Project each layer's residual through lm_head
                # Residual streams are on CPU from extractor; move to model device
                clean_hidden = clean_result["residual_stream"][layer_idx][:, -1, :].to(device)
                corrupted_hidden = corrupted_result["residual_stream"][layer_idx][:, -1, :].to(device)

                clean_logits = clean_hidden.float() @ lm_head.weight.T.float()
                corrupted_logits = corrupted_hidden.float() @ lm_head.weight.T.float()
                if lm_head.bias is not None:
                    clean_logits = clean_logits + lm_head.bias.float()
                    corrupted_logits = corrupted_logits + lm_head.bias.float()

                clean_probs = torch.softmax(clean_logits, dim=-1)[0]
                corrupted_probs = torch.softmax(corrupted_logits, dim=-1)[0]

                # Top prediction
                clean_top_id = clean_probs.argmax().item()
                corrupted_top_id = corrupted_probs.argmax().item()
                clean_top_token = tokenizer.decode(clean_top_id).strip() or repr(
                    tokenizer.convert_ids_to_tokens(clean_top_id))
                corrupted_top_token = tokenizer.decode(corrupted_top_id).strip() or repr(
                    tokenizer.convert_ids_to_tokens(corrupted_top_id))

                # Probability of correct/incorrect tokens
                clean_p_correct = clean_probs[correct_id].item() if correct_id >= 0 else 0
                clean_p_incorrect = clean_probs[incorrect_id].item() if incorrect_id >= 0 else 0
                corrupted_p_correct = corrupted_probs[correct_id].item() if correct_id >= 0 else 0
                corrupted_p_incorrect = corrupted_probs[incorrect_id].item() if incorrect_id >= 0 else 0

                pair_data["layers"].append({
                    "layer": layer_idx,
                    "layer_type": arch_map.layer_type(layer_idx),
                    "clean_top1": clean_top_token,
                    "corrupted_top1": corrupted_top_token,
                    "clean_p_correct": clean_p_correct,
                    "clean_p_incorrect": clean_p_incorrect,
                    "corrupted_p_correct": corrupted_p_correct,
                    "corrupted_p_incorrect": corrupted_p_incorrect,
                })

        all_logit_lens.append(pair_data)

        # Log key transitions
        log(f"  '{pair['correct'].strip()}' first appears in clean top-1 at:")
        for ld in pair_data["layers"]:
            if pair["correct"].strip().lower() in ld["clean_top1"].lower():
                log(f"    Layer {ld['layer']} ({ld['layer_type']})")
                break
        else:
            log(f"    Never appears as top-1")

    # Aggregate: at which layer does bee probability peak?
    log("\n" + "─" * 70)
    log("LOGIT LENS: Prediction Evolution")
    log("─" * 70)

    avg_clean_p_correct = np.zeros(num_layers)
    avg_corrupted_p_correct = np.zeros(num_layers)
    for pair_data in all_logit_lens:
        for ld in pair_data["layers"]:
            avg_clean_p_correct[ld["layer"]] += ld["clean_p_correct"]
            avg_corrupted_p_correct[ld["layer"]] += ld["corrupted_p_correct"]
    avg_clean_p_correct /= len(all_logit_lens)
    avg_corrupted_p_correct /= len(all_logit_lens)

    log("\nAvg P(correct) by layer (clean | corrupted):")
    for layer_idx in range(num_layers):
        lt = arch_map.layer_type(layer_idx)
        marker = " ***" if abs(avg_clean_p_correct[layer_idx] - avg_corrupted_p_correct[layer_idx]) > 0.05 else ""
        log(f"  L{layer_idx:2d} ({lt[0].upper()}): "
            f"{avg_clean_p_correct[layer_idx]:.4f} | "
            f"{avg_corrupted_p_correct[layer_idx]:.4f}{marker}")

    # Find divergence point
    for layer_idx in range(num_layers):
        diff = avg_clean_p_correct[layer_idx] - avg_corrupted_p_correct[layer_idx]
        if diff > 0.01:
            log(f"\nDivergence point: Layer {layer_idx} ({arch_map.layer_type(layer_idx)}) "
                f"— clean P(correct) first exceeds corrupted by >0.01")
            break

    # Save
    with open(output_dir / "logit_lens.json", "w") as f:
        json.dump(all_logit_lens, f, indent=2)

    return all_logit_lens, avg_clean_p_correct, avg_corrupted_p_correct


# ── Phase 5: Steering Vector Propagation ─────────────────────────────────────

def run_phase5(model, tokenizer, arch_map, device, bee_vector, output_dir):
    """Trace how the L1 bee vector perturbation propagates through layers."""
    log("")
    log("=" * 70)
    log("PHASE 5: Steering Vector Propagation Analysis")
    log("=" * 70)

    num_layers = arch_map.num_layers
    lm_head = model.lm_head
    bee_vec = bee_vector.to(device)

    all_propagation = []

    for prompt in PROPAGATION_PROMPTS:
        log(f"\n  Prompt: \"{prompt}\"")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

        # --- Clean forward pass ---
        clean_residuals = {}
        clean_hooks = []

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]

            def hook_fn(module, input, output, _idx=layer_idx):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                clean_residuals[_idx] = hidden.detach().clone()

            h = layer.register_forward_hook(hook_fn)
            clean_hooks.append(h)

        with torch.no_grad():
            model(**inputs)

        for h in clean_hooks:
            h.remove()

        # --- Steered forward pass (inject bee vector at L1) ---
        steered_residuals = {}
        steered_hooks = []

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]

            def capture_hook(module, input, output, _idx=layer_idx):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                steered_residuals[_idx] = hidden.detach().clone()

            h = layer.register_forward_hook(capture_hook)
            steered_hooks.append(h)

        # Injection hook at steering layer
        steering_layer = model.model.layers[STEERING_LAYER]

        def inject_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Add steering vector scaled by coefficient at all positions
            modified = hidden + STEERING_COEFFICIENT * bee_vec.to(hidden.device, hidden.dtype)
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        inject_h = steering_layer.register_forward_hook(inject_hook)
        steered_hooks.append(inject_h)

        with torch.no_grad():
            model(**inputs)

        for h in steered_hooks:
            h.remove()

        # --- Analyze perturbation at each layer ---
        prompt_data = {
            "prompt": prompt,
            "layers": [],
        }

        with torch.no_grad():
            for layer_idx in range(num_layers):
                clean_r = clean_residuals[layer_idx][:, -1, :].float()  # [1, hidden]
                steered_r = steered_residuals[layer_idx][:, -1, :].float()

                delta = steered_r - clean_r  # [1, hidden]
                delta_norm = delta.norm().item()

                # Cosine similarity with bee vector
                cos_sim = torch.nn.functional.cosine_similarity(
                    delta, bee_vec.float().unsqueeze(0)
                ).item()

                # Project delta through lm_head to see what tokens it boosts
                delta_logits = delta @ lm_head.weight.T.float()
                if lm_head.bias is not None:
                    # Don't add bias for delta — we want the *change* in logits
                    pass

                top_boost_probs, top_boost_ids = delta_logits[0].topk(5)
                top_boosted = []
                for tid, tp in zip(top_boost_ids, top_boost_probs):
                    tok = tokenizer.decode(tid.item()).strip() or repr(
                        tokenizer.convert_ids_to_tokens(tid.item()))
                    top_boosted.append({"token": tok, "logit_boost": tp.item()})

                prompt_data["layers"].append({
                    "layer": layer_idx,
                    "layer_type": arch_map.layer_type(layer_idx),
                    "delta_norm": delta_norm,
                    "cosine_with_bee": cos_sim,
                    "top_boosted_tokens": top_boosted,
                })

        all_propagation.append(prompt_data)

        # Log summary
        log(f"  Layer | δ norm  | cos(δ,bee) | Top boosted token")
        log(f"  ------|---------|------------|------------------")
        for ld in prompt_data["layers"]:
            top_tok = ld["top_boosted_tokens"][0]["token"] if ld["top_boosted_tokens"] else "?"
            log(f"  L{ld['layer']:2d} {ld['layer_type'][0].upper()} | "
                f"{ld['delta_norm']:7.3f} | {ld['cosine_with_bee']:+.4f}   | {top_tok}")

    # Save
    with open(output_dir / "propagation.json", "w") as f:
        json.dump(all_propagation, f, indent=2)

    return all_propagation


# ── Report Generation ────────────────────────────────────────────────────────

def generate_report(
    avg_path_matrix, avg_layer_importance, circuit_results,
    avg_patch, avg_ablation,
    logit_lens_data, avg_clean_p, avg_corrupted_p,
    propagation_data,
    arch_map, output_dir,
):
    """Generate comprehensive text report."""
    lines = []
    num_layers = arch_map.num_layers

    lines.append("=" * 70)
    lines.append("BEE CIRCUIT TRACING REPORT")
    lines.append("How the L1 bee steering vector propagates through Granite 4.0")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Architecture: {arch_map.summary()}")
    lines.append(f"Steering: Layer {STEERING_LAYER}, coefficient {STEERING_COEFFICIENT}")
    lines.append(f"Circuit pairs tested: {len(CIRCUIT_PAIRS)}")
    lines.append(f"Circuit threshold: {CIRCUIT_THRESHOLD}")
    lines.append("")

    # Section 1: Consensus Circuit
    lines.append("─" * 70)
    lines.append("SECTION 1: CONSENSUS CIRCUIT (Path Patching)")
    lines.append("─" * 70)
    lines.append("")

    if avg_path_matrix is not None:
        lines.append("Top 15 paths by average recovery:")
        upper_indices = torch.triu_indices(num_layers, num_layers, offset=1)
        path_values = avg_path_matrix[upper_indices[0], upper_indices[1]]
        top_path_idx = path_values.abs().argsort(descending=True)[:15]
        for fi in top_path_idx:
            s = upper_indices[0][fi].item()
            t = upper_indices[1][fi].item()
            val = path_values[fi].item()
            s_type = "M" if arch_map.layer_type(s) == "mamba" else "T"
            t_type = "M" if arch_map.layer_type(t) == "mamba" else "T"
            lines.append(f"  L{s:2d}{s_type} → L{t:2d}{t_type}: {val:+.4f}")
        lines.append("")

    if avg_layer_importance is not None:
        lines.append("Layer importance (average across pairs):")
        sorted_layers = sorted(range(num_layers),
                               key=lambda i: abs(avg_layer_importance[i].item()),
                               reverse=True)
        for idx in sorted_layers[:15]:
            lt = arch_map.layer_type(idx)
            lines.append(f"  Layer {idx:2d} ({lt:>5s}): {avg_layer_importance[idx]:+.4f}")
        lines.append("")

    # Per-pair circuit summaries
    for i, result in enumerate(circuit_results):
        lines.append(f"  Pair {i+1}: {len(result.circuit_nodes)} nodes, "
                     f"{len(result.circuit_edges)} edges, "
                     f"clean_diff={result.clean_logit_diff:+.3f}, "
                     f"corrupted_diff={result.corrupted_logit_diff:+.3f}")

    # Section 2: Layer Importance Profile
    lines.append("")
    lines.append("─" * 70)
    lines.append("SECTION 2: LAYER IMPORTANCE PROFILE")
    lines.append("─" * 70)
    lines.append("")

    if avg_patch is not None:
        lines.append("Activation Patching (higher = layer carries more bee signal):")
        sorted_patch = sorted(range(num_layers),
                              key=lambda i: abs(avg_patch[i].item()), reverse=True)
        for idx in sorted_patch[:15]:
            lt = arch_map.layer_type(idx)
            lines.append(f"  Layer {idx:2d} ({lt:>5s}): {avg_patch[idx]:+.4f}")
        lines.append("")

    if avg_ablation is not None:
        lines.append("Zero Ablation (most negative = most necessary):")
        sorted_abl = sorted(range(num_layers),
                            key=lambda i: avg_ablation[i].item())
        for idx in sorted_abl[:15]:
            lt = arch_map.layer_type(idx)
            lines.append(f"  Layer {idx:2d} ({lt:>5s}): {avg_ablation[idx]:+.4f}")
        lines.append("")

    # Transformer layer analysis
    lines.append("Transformer layer analysis (L10, L13, L17, L27):")
    for idx in arch_map.attention_indices:
        patch_val = avg_patch[idx].item() if avg_patch is not None else 0
        abl_val = avg_ablation[idx].item() if avg_ablation is not None else 0
        lines.append(f"  L{idx}: patch_recovery={patch_val:+.4f}, "
                     f"ablation_recovery={abl_val:+.4f}")
    lines.append("")

    # Section 3: Logit Lens
    lines.append("─" * 70)
    lines.append("SECTION 3: LOGIT LENS (Prediction Evolution)")
    lines.append("─" * 70)
    lines.append("")

    if avg_clean_p is not None:
        lines.append("Avg P(correct_token) by layer:")
        lines.append(f"  {'Layer':>5s} {'Type':>5s} {'Clean':>8s} {'Corrupted':>10s} {'Delta':>8s}")
        for layer_idx in range(num_layers):
            lt = arch_map.layer_type(layer_idx)[0].upper()
            clean_p = avg_clean_p[layer_idx]
            corrupt_p = avg_corrupted_p[layer_idx]
            delta = clean_p - corrupt_p
            marker = " ***" if abs(delta) > 0.05 else ""
            lines.append(f"  L{layer_idx:2d}     {lt}  {clean_p:8.4f} {corrupt_p:10.4f} {delta:+8.4f}{marker}")
        lines.append("")

    # Section 4: Steering Propagation
    lines.append("─" * 70)
    lines.append("SECTION 4: STEERING VECTOR PROPAGATION")
    lines.append("─" * 70)
    lines.append("")

    if propagation_data:
        # Average across prompts
        avg_delta_norm = np.zeros(num_layers)
        avg_cos_sim = np.zeros(num_layers)
        for pdata in propagation_data:
            for ld in pdata["layers"]:
                avg_delta_norm[ld["layer"]] += ld["delta_norm"]
                avg_cos_sim[ld["layer"]] += ld["cosine_with_bee"]
        avg_delta_norm /= len(propagation_data)
        avg_cos_sim /= len(propagation_data)

        lines.append("Average perturbation across prompts:")
        lines.append(f"  {'Layer':>5s} {'Type':>5s} {'δ norm':>8s} {'cos(δ,bee)':>11s} {'Top boosted':>15s}")
        for layer_idx in range(num_layers):
            lt = arch_map.layer_type(layer_idx)[0].upper()
            # Get most common top-boosted token across prompts
            top_tokens = []
            for pdata in propagation_data:
                ld = pdata["layers"][layer_idx]
                if ld["top_boosted_tokens"]:
                    top_tokens.append(ld["top_boosted_tokens"][0]["token"])
            common_tok = max(set(top_tokens), key=top_tokens.count) if top_tokens else "?"
            lines.append(f"  L{layer_idx:2d}     {lt}  "
                         f"{avg_delta_norm[layer_idx]:8.3f} "
                         f"{avg_cos_sim[layer_idx]:+11.4f} "
                         f"{common_tok:>15s}")
        lines.append("")

        # Highlight key observations
        lines.append("Key observations:")
        max_norm_layer = int(np.argmax(avg_delta_norm))
        lines.append(f"  Peak perturbation norm: Layer {max_norm_layer} "
                     f"({arch_map.layer_type(max_norm_layer)}) "
                     f"δ={avg_delta_norm[max_norm_layer]:.3f}")
        max_cos_layer = int(np.argmax(avg_cos_sim))
        lines.append(f"  Highest bee alignment: Layer {max_cos_layer} "
                     f"({arch_map.layer_type(max_cos_layer)}) "
                     f"cos={avg_cos_sim[max_cos_layer]:+.4f}")

        # Where does cosine similarity drop below 0.5?
        for layer_idx in range(STEERING_LAYER + 1, num_layers):
            if avg_cos_sim[layer_idx] < 0.5:
                lines.append(f"  Bee direction decays below cos=0.5 at Layer {layer_idx} "
                             f"({arch_map.layer_type(layer_idx)})")
                break

        lines.append("")

    # Section 5: Key Findings
    lines.append("─" * 70)
    lines.append("SECTION 5: KEY FINDINGS")
    lines.append("─" * 70)
    lines.append("")
    lines.append("(Analysis based on aggregated results above)")
    lines.append("")

    # Determine if Transformer layers are relay points
    if avg_patch is not None:
        attn_max = max(abs(avg_patch[i].item()) for i in arch_map.attention_indices)
        mamba_max = max(abs(avg_patch[i].item()) for i in arch_map.mamba_indices)
        if attn_max > mamba_max * 0.5:
            lines.append("• Transformer layers appear to be significant relay points for the bee signal")
        else:
            lines.append("• Mamba layers dominate the bee circuit; Transformer layers play a lesser role")

    if propagation_data:
        # Does perturbation amplify or decay?
        early_norm = avg_delta_norm[STEERING_LAYER + 1] if STEERING_LAYER + 1 < num_layers else 0
        late_norm = avg_delta_norm[-1]
        if late_norm > early_norm * 1.5:
            lines.append("• The steering perturbation AMPLIFIES as it propagates (multiplicative dynamics)")
        elif late_norm < early_norm * 0.5:
            lines.append("• The steering perturbation DECAYS as it propagates (the model partially corrects)")
        else:
            lines.append("• The steering perturbation maintains roughly constant magnitude through layers")

    lines.append("")

    report_text = "\n".join(lines)
    with open(output_dir / "report.txt", "w") as f:
        f.write(report_text)

    log(f"Report saved to {output_dir / 'report.txt'}")
    return report_text


# ── Visualization ────────────────────────────────────────────────────────────

def save_visualizations(
    avg_path_matrix, avg_layer_importance, circuit_results,
    avg_patch, avg_ablation,
    avg_clean_p, avg_corrupted_p,
    propagation_data,
    arch_map, output_dir,
):
    """Generate and save matplotlib/plotly visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_layers = arch_map.num_layers

    # 1. Path matrix heatmap (average)
    if avg_path_matrix is not None:
        from src.visualization.circuit_viz import _plot_path_matrix
        # Create a minimal CircuitResult-like object for the visualization
        from src.extraction.circuit_discovery import CircuitResult, SweepGranularity

        # Use the first circuit result as template, replace matrices
        if circuit_results:
            template = circuit_results[0]
            avg_result = CircuitResult(
                clean_tokens=template.clean_tokens,
                corrupted_tokens=template.corrupted_tokens,
                correct_token=template.correct_token,
                incorrect_token=template.incorrect_token,
                correct_token_id=template.correct_token_id,
                incorrect_token_id=template.incorrect_token_id,
                correct_token_display="(averaged across pairs)",
                incorrect_token_display="(averaged across pairs)",
                path_matrix=avg_path_matrix,
                layer_indices=list(range(num_layers)),
                layer_importance=avg_layer_importance,
                component_results=[],
                circuit_nodes=[],
                circuit_edges=[],
                clean_logit_diff=0,
                corrupted_logit_diff=0,
                clean_prob_correct=0,
                corrupted_prob_correct=0,
                threshold=CIRCUIT_THRESHOLD,
                granularity=SweepGranularity.DETAILED,
            )

            from src.visualization.circuit_viz import (
                create_path_matrix_plotly,
                create_layer_importance_plotly,
            )

            fig = create_path_matrix_plotly(avg_result, arch_map)
            fig.update_layout(title="Average Path Patching Matrix (5 bee pairs)")
            fig.write_html(str(output_dir / "avg_path_matrix.html"))
            log("Saved avg_path_matrix.html")

            fig = create_layer_importance_plotly(avg_result, arch_map)
            fig.update_layout(title="Average Layer Importance (5 bee pairs)")
            fig.write_html(str(output_dir / "avg_layer_importance.html"))
            log("Saved avg_layer_importance.html")

    # 2. Layer importance comparison (patch vs ablation)
    if avg_patch is not None and avg_ablation is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        colors = ["#ab47bc" if arch_map.layer_type(i) == "mamba" else "#26a69a"
                  for i in range(num_layers)]

        ax1.bar(range(num_layers), avg_patch.numpy(), color=colors, width=0.8)
        ax1.axhline(y=CIRCUIT_THRESHOLD, color="gold", linestyle="--", alpha=0.6)
        ax1.set_ylabel("Recovery")
        ax1.set_title("Activation Patching (which layers CARRY the bee signal)")
        ax1.grid(axis="y", alpha=0.3)

        ax2.bar(range(num_layers), avg_ablation.numpy(), color=colors, width=0.8)
        ax2.axhline(y=0, color="gray", linewidth=0.5)
        ax2.set_ylabel("Recovery")
        ax2.set_xlabel("Layer")
        ax2.set_title("Zero Ablation (which layers are NECESSARY)")
        ax2.grid(axis="y", alpha=0.3)

        # Mark transformer layers
        for idx in arch_map.attention_indices:
            ax1.axvline(x=idx, color="#26a69a", linewidth=0.5, alpha=0.3)
            ax2.axvline(x=idx, color="#26a69a", linewidth=0.5, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / "layer_importance_comparison.png", dpi=150)
        plt.close(fig)
        log("Saved layer_importance_comparison.png")

    # 3. Logit lens heatmap
    if avg_clean_p is not None:
        fig, ax = plt.subplots(figsize=(14, 4))
        data = np.stack([avg_clean_p, avg_corrupted_p, avg_clean_p - avg_corrupted_p])
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", interpolation="nearest")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Clean P(correct)", "Corrupted P(correct)", "Delta"])
        ax.set_xlabel("Layer")
        ax.set_title("Logit Lens: P(correct token) Evolution Across Layers")
        plt.colorbar(im, ax=ax, shrink=0.8)
        for idx in arch_map.attention_indices:
            ax.axvline(x=idx, color="white", linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        fig.savefig(output_dir / "logit_lens.png", dpi=150)
        plt.close(fig)
        log("Saved logit_lens.png")

    # 4. Steering propagation curves
    if propagation_data:
        avg_delta_norm = np.zeros(num_layers)
        avg_cos_sim = np.zeros(num_layers)
        for pdata in propagation_data:
            for ld in pdata["layers"]:
                avg_delta_norm[ld["layer"]] += ld["delta_norm"]
                avg_cos_sim[ld["layer"]] += ld["cosine_with_bee"]
        avg_delta_norm /= len(propagation_data)
        avg_cos_sim /= len(propagation_data)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        colors = ["#ab47bc" if arch_map.layer_type(i) == "mamba" else "#26a69a"
                  for i in range(num_layers)]

        ax1.bar(range(num_layers), avg_delta_norm, color=colors, width=0.8)
        ax1.set_ylabel("‖δ‖ (perturbation norm)")
        ax1.set_title("Steering Perturbation Magnitude Across Layers")
        ax1.axvline(x=STEERING_LAYER, color="red", linewidth=2, linestyle="--",
                    label=f"Injection (L{STEERING_LAYER})")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        ax2.bar(range(num_layers), avg_cos_sim, color=colors, width=0.8)
        ax2.set_ylabel("cos(δ, bee_vector)")
        ax2.set_xlabel("Layer")
        ax2.set_title("Alignment of Perturbation with Bee Direction")
        ax2.axhline(y=0, color="gray", linewidth=0.5)
        ax2.axvline(x=STEERING_LAYER, color="red", linewidth=2, linestyle="--",
                    label=f"Injection (L{STEERING_LAYER})")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        for idx in arch_map.attention_indices:
            ax1.axvline(x=idx, color="#26a69a", linewidth=0.5, alpha=0.3)
            ax2.axvline(x=idx, color="#26a69a", linewidth=0.5, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / "steering_propagation.png", dpi=150)
        plt.close(fig)
        log("Saved steering_propagation.png")


# ── Helpers ──────────────────────────────────────────────────────────────────

def save_circuit_result(result, path, arch_map):
    """Save a CircuitResult to JSON (tensor-safe)."""
    data = {
        "correct_token": result.correct_token,
        "incorrect_token": result.incorrect_token,
        "correct_token_display": result.correct_token_display,
        "incorrect_token_display": result.incorrect_token_display,
        "clean_logit_diff": result.clean_logit_diff,
        "corrupted_logit_diff": result.corrupted_logit_diff,
        "clean_prob_correct": result.clean_prob_correct,
        "corrupted_prob_correct": result.corrupted_prob_correct,
        "threshold": result.threshold,
        "granularity": result.granularity.value,
        "num_nodes": len(result.circuit_nodes),
        "num_edges": len(result.circuit_edges),
        "nodes": [
            {"layer": n.layer_idx, "type": n.layer_type, "importance": n.importance}
            for n in result.circuit_nodes
        ],
        "edges": [
            {"source": e.source_layer, "target": e.target_layer,
             "importance": e.importance, "source_type": e.source_type,
             "target_type": e.target_type}
            for e in result.circuit_edges
        ],
        "component_results": [
            {"layer": c.layer_idx, "head": c.component_idx,
             "recovery": c.recovery_score}
            for c in result.component_results
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/bee_circuit") / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    # Save config
    config = {
        "circuit_pairs": CIRCUIT_PAIRS,
        "circuit_threshold": CIRCUIT_THRESHOLD,
        "steering_vector_path": str(STEERING_VECTOR_PATH),
        "steering_layer": STEERING_LAYER,
        "steering_coefficient": STEERING_COEFFICIENT,
        "propagation_prompts": PROPAGATION_PROMPTS,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Phase 1
    model, tokenizer, device, arch_map, bee_vector = run_phase1()

    # Phase 2
    avg_path_matrix, avg_layer_importance, circuit_results = run_phase2(
        model, tokenizer, arch_map, device, output_dir
    )

    # Phase 3
    avg_patch, avg_ablation = run_phase3(
        model, tokenizer, arch_map, device, output_dir
    )

    # Phase 4
    logit_lens_data, avg_clean_p, avg_corrupted_p = run_phase4(
        model, tokenizer, arch_map, device, output_dir
    )

    # Phase 5
    propagation_data = run_phase5(
        model, tokenizer, arch_map, device, bee_vector, output_dir
    )

    # Report
    log("")
    log("=" * 70)
    log("GENERATING REPORT & VISUALIZATIONS")
    log("=" * 70)

    report = generate_report(
        avg_path_matrix, avg_layer_importance, circuit_results,
        avg_patch, avg_ablation,
        logit_lens_data, avg_clean_p, avg_corrupted_p,
        propagation_data,
        arch_map, output_dir,
    )

    save_visualizations(
        avg_path_matrix, avg_layer_importance, circuit_results,
        avg_patch, avg_ablation,
        avg_clean_p, avg_corrupted_p,
        propagation_data,
        arch_map, output_dir,
    )

    # Summary
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    summary = {
        "run_id": run_id,
        "elapsed_seconds": elapsed,
        "elapsed_human": f"{minutes}m {seconds}s",
        "num_pairs": len(CIRCUIT_PAIRS),
        "num_circuit_results": len(circuit_results),
        "steering_layer": STEERING_LAYER,
        "steering_coefficient": STEERING_COEFFICIENT,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log("")
    log("=" * 70)
    log("BEE CIRCUIT TRACING COMPLETE!")
    log(f"  Time: {minutes}m {seconds}s")
    log(f"  Circuit results: {len(circuit_results)}/{len(CIRCUIT_PAIRS)} pairs")
    log(f"  Report: {output_dir / 'report.txt'}")
    log(f"  Results: {output_dir}")
    log("=" * 70)


if __name__ == "__main__":
    main()
