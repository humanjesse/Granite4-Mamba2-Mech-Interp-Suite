#!/usr/bin/env python3
"""Token-level SAE feature activation analysis.

Determines whether the top bee-selective SAE features fire on bee-semantic
tokens ("bee", "honey", "hive") or structural tokens (".", ",", "the").
This answers: is feature #2490 a real bee feature, or a statistical artifact?

Usage:
    source .venv/bin/activate
    python run_feature_analysis.py
"""

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

from src.model_loader import load_model
from src.sae.sparse_autoencoder import SparseAutoencoder
from src.sae.feature_search import get_default_bee_texts, get_default_baseline_texts

# ── Config ───────────────────────────────────────────────────────────────────

LAYER_IDX = 4
SAE_PATH = Path("results/sae/20260218_084436/sae_layer4.pt")
FEATURE_SEARCH_PATH = Path("results/sae/20260218_084436/feature_search.json")
OUTPUT_BASE = Path("results/sae_analysis")

# Top 5 bee-selective features from previous run
TOP_FEATURES = [
    {"idx": 2490, "selectivity": 30.19},
    {"idx": 18, "selectivity": 16.15},
    {"idx": 2497, "selectivity": 12.51},
    {"idx": 4104, "selectivity": 8.02},
    {"idx": 2249, "selectivity": 7.49},
]

# Number of top activating tokens to show per feature
TOP_N_TOKENS = 20

# Bee-semantic keywords (lowercased, checked as substrings)
BEE_SEMANTIC_WORDS = {
    "bee", "honey", "hive", "pollen", "pollinator", "pollination",
    "nectar", "swarm", "colony", "queen", "drone", "worker",
    "wax", "sting", "buzz", "keeper", "comb", "propolis",
    "jelly", "melittin", "waggle", "flower", "apiar",
}

# Structural tokens
STRUCTURAL_TOKENS = {
    ".", ",", "!", "?", ":", ";", "-", "(", ")", "'", '"',
    "the", "a", "an", "is", "are", "was", "were", "of", "in",
    "to", "and", "for", "that", "this", "with", "by", "on", "at",
    "from", "it", "its", "has", "have", "had", "be", "been",
}


def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def classify_token(token_str: str) -> str:
    """Classify a token as 'bee-semantic', 'structural', or 'content'."""
    # Clean up tokenizer artifacts (leading Ġ/▁ = space prefix in many tokenizers)
    cleaned = token_str.lstrip("ĠĊ▁Ġ").strip().lower()

    if not cleaned:
        return "structural"

    # Check bee-semantic: any bee keyword is a substring of the token
    for word in BEE_SEMANTIC_WORDS:
        if word in cleaned or cleaned in word:
            return "bee-semantic"

    # Check structural
    if cleaned in STRUCTURAL_TOKENS:
        return "structural"

    return "content"


def collect_per_token_activations(
    sae: SparseAutoencoder,
    model,
    tokenizer,
    layer_idx: int,
    texts: list[str],
    corpus_label: str,
    feature_indices: list[int],
    device: str,
) -> list[dict]:
    """Run texts through model, get per-token SAE feature activations.

    Returns a list of records:
        {token_str, activation, feature_idx, text_idx, corpus, position}
    """
    layer = model.model.layers[layer_idx]
    storage = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        storage["hidden"] = hidden.detach()

    hook = layer.register_forward_hook(hook_fn)
    records = []

    try:
        for text_idx, text in enumerate(texts):
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128,
            )
            input_ids = inputs["input_ids"].to(device)

            with torch.no_grad():
                model(input_ids=input_ids)

            hidden = storage["hidden"][0].float()  # [seq_len, hidden_size]

            with torch.no_grad():
                z = sae.encode(hidden)  # [seq_len, latent_dim]

            # Get token strings
            token_ids = input_ids[0].tolist()
            token_strs = tokenizer.convert_ids_to_tokens(token_ids)

            for feat_idx in feature_indices:
                activations = z[:, feat_idx]  # [seq_len]

                for pos, (tok_str, act_val) in enumerate(
                    zip(token_strs, activations.tolist())
                ):
                    records.append({
                        "token_str": tok_str,
                        "activation": act_val,
                        "feature_idx": feat_idx,
                        "text_idx": text_idx,
                        "corpus": corpus_label,
                        "position": pos,
                        "text_snippet": text[:80],
                    })
    finally:
        hook.remove()

    return records


def analyze_feature(records: list[dict], feature_idx: int, top_n: int) -> dict:
    """Analyze activation patterns for a single feature."""
    feat_records = [r for r in records if r["feature_idx"] == feature_idx]

    # Split by corpus
    bee_records = [r for r in feat_records if r["corpus"] == "bee"]
    base_records = [r for r in feat_records if r["corpus"] == "baseline"]

    # Sort all by activation (descending)
    feat_records.sort(key=lambda r: r["activation"], reverse=True)

    # Top N tokens
    top_tokens = feat_records[:top_n]

    # Classify top tokens
    categories = defaultdict(int)
    for r in top_tokens:
        cat = classify_token(r["token_str"])
        categories[cat] += 1

    # Also compute stats for tokens with non-zero activation
    active_records = [r for r in feat_records if r["activation"] > 0]
    active_bee = [r for r in active_records if r["corpus"] == "bee"]
    active_base = [r for r in active_records if r["corpus"] == "baseline"]

    # Top tokens from bee corpus only
    bee_sorted = sorted(bee_records, key=lambda r: r["activation"], reverse=True)
    top_bee_tokens = bee_sorted[:top_n]
    bee_categories = defaultdict(int)
    for r in top_bee_tokens:
        cat = classify_token(r["token_str"])
        bee_categories[cat] += 1

    # Compute mean activation by category across all tokens
    cat_activations = defaultdict(list)
    for r in feat_records:
        cat = classify_token(r["token_str"])
        cat_activations[cat].append(r["activation"])

    cat_means = {}
    for cat, vals in cat_activations.items():
        cat_means[cat] = sum(vals) / len(vals) if vals else 0.0

    return {
        "feature_idx": feature_idx,
        "top_tokens": top_tokens,
        "top_n_categories": dict(categories),
        "top_bee_tokens": top_bee_tokens,
        "top_bee_categories": dict(bee_categories),
        "total_active_tokens": len(active_records),
        "active_from_bee": len(active_bee),
        "active_from_baseline": len(active_base),
        "category_mean_activations": cat_means,
    }


def determine_verdict(analysis: dict) -> str:
    """Classify a feature based on its activation patterns."""
    cats = analysis["top_n_categories"]
    bee_sem = cats.get("bee-semantic", 0)
    structural = cats.get("structural", 0)
    content = cats.get("content", 0)
    total = bee_sem + structural + content

    if total == 0:
        return "inactive"

    bee_pct = bee_sem / total
    struct_pct = structural / total

    if bee_pct >= 0.3:
        return "semantic bee feature"
    elif struct_pct >= 0.5:
        return "structural feature"
    elif bee_pct >= 0.15:
        return "weak bee signal"
    else:
        return "mixed/unclear"


def generate_report(analyses: list[dict], feature_info: list[dict]) -> str:
    """Generate human-readable report."""
    lines = []
    lines.append("=" * 70)
    lines.append("TOKEN-LEVEL SAE FEATURE ACTIVATION ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Question: Do top bee-selective features fire on bee-semantic")
    lines.append("tokens or structural tokens?")
    lines.append("")
    lines.append(f"SAE:       {SAE_PATH}")
    lines.append(f"Layer:     {LAYER_IDX}")
    lines.append(f"Features:  {len(analyses)} (top bee-selective)")
    lines.append(f"Bee texts: {len(get_default_bee_texts())}")
    lines.append(f"Baseline:  {len(get_default_baseline_texts())}")
    lines.append("")

    # ── Per-feature analysis ──────────────────────────────────────────────
    verdicts = []

    for analysis, info in zip(analyses, feature_info):
        fidx = analysis["feature_idx"]
        verdict = determine_verdict(analysis)
        verdicts.append((fidx, info["selectivity"], verdict))

        lines.append("=" * 70)
        lines.append(
            f"FEATURE #{fidx}  (selectivity={info['selectivity']:.2f})"
        )
        lines.append(f"VERDICT: {verdict.upper()}")
        lines.append("=" * 70)
        lines.append("")

        # Top token table
        cats = analysis["top_n_categories"]
        total_top = sum(cats.values())
        lines.append(f"Top {TOP_N_TOKENS} highest-activating tokens (all texts):")
        lines.append(
            f"  bee-semantic: {cats.get('bee-semantic', 0)}/{total_top} "
            f"({100*cats.get('bee-semantic', 0)/max(total_top,1):.0f}%)"
        )
        lines.append(
            f"  structural:   {cats.get('structural', 0)}/{total_top} "
            f"({100*cats.get('structural', 0)/max(total_top,1):.0f}%)"
        )
        lines.append(
            f"  content:      {cats.get('content', 0)}/{total_top} "
            f"({100*cats.get('content', 0)/max(total_top,1):.0f}%)"
        )
        lines.append("")

        lines.append(
            f"  {'Rank':>4}  {'Token':<20}  {'Activation':>10}  "
            f"{'Category':<14}  {'Corpus':<8}  Source text"
        )
        lines.append("  " + "-" * 100)
        for rank, r in enumerate(analysis["top_tokens"], 1):
            cat = classify_token(r["token_str"])
            tok_display = repr(r["token_str"])
            lines.append(
                f"  {rank:4d}  {tok_display:<20}  {r['activation']:10.4f}  "
                f"{cat:<14}  {r['corpus']:<8}  "
                f"{r['text_snippet'][:50]}..."
            )
        lines.append("")

        # Mean activation by category
        lines.append("Mean activation by token category:")
        for cat in ["bee-semantic", "structural", "content"]:
            mean_val = analysis["category_mean_activations"].get(cat, 0.0)
            lines.append(f"  {cat:<14}: {mean_val:.6f}")
        lines.append("")

        # Active token stats
        lines.append(
            f"Active tokens (activation > 0): {analysis['total_active_tokens']}"
        )
        lines.append(f"  From bee corpus:      {analysis['active_from_bee']}")
        lines.append(f"  From baseline corpus: {analysis['active_from_baseline']}")
        lines.append("")

        # Top bee-corpus tokens
        bee_cats = analysis["top_bee_categories"]
        total_bee_top = sum(bee_cats.values())
        lines.append(f"Top {TOP_N_TOKENS} tokens from BEE corpus only:")
        lines.append(
            f"  bee-semantic: {bee_cats.get('bee-semantic', 0)}/{total_bee_top} "
            f"({100*bee_cats.get('bee-semantic', 0)/max(total_bee_top,1):.0f}%)"
        )
        lines.append(
            f"  structural:   {bee_cats.get('structural', 0)}/{total_bee_top} "
            f"({100*bee_cats.get('structural', 0)/max(total_bee_top,1):.0f}%)"
        )
        lines.append(
            f"  content:      {bee_cats.get('content', 0)}/{total_bee_top} "
            f"({100*bee_cats.get('content', 0)/max(total_bee_top,1):.0f}%)"
        )
        lines.append("")
        for rank, r in enumerate(analysis["top_bee_tokens"][:10], 1):
            cat = classify_token(r["token_str"])
            tok_display = repr(r["token_str"])
            lines.append(
                f"  {rank:4d}  {tok_display:<20}  {r['activation']:10.4f}  {cat}"
            )
        lines.append("")

    # ── Summary ───────────────────────────────────────────────────────────
    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        f"  {'Feature':>8}  {'Selectivity':>12}  Verdict"
    )
    lines.append("  " + "-" * 50)
    for fidx, sel, verdict in verdicts:
        lines.append(f"  {fidx:8d}  {sel:12.2f}  {verdict}")
    lines.append("")

    # Overall conclusion
    semantic_count = sum(1 for _, _, v in verdicts if "semantic" in v)
    structural_count = sum(1 for _, _, v in verdicts if "structural" in v)

    lines.append("OVERALL CONCLUSION:")
    if semantic_count >= 3:
        lines.append(
            "  Strong evidence of monosemantic bee features. The SAE has found")
        lines.append(
            "  features that genuinely respond to bee-related tokens. The steering")
        lines.append(
            "  failure is likely a separate issue (wrong layer, coefficient, etc).")
    elif semantic_count >= 1:
        lines.append(
            "  Mixed evidence. Some features show bee-semantic activation, but")
        lines.append(
            "  others fire on structural tokens. The SAE partially captures bee")
        lines.append(
            "  semantics but may need more training data or different hyperparams.")
    elif structural_count >= 3:
        lines.append(
            "  Features are primarily structural — they fire on punctuation,")
        lines.append(
            "  articles, and common words that happen to correlate with bee text")
        lines.append(
            "  patterns. The high selectivity scores are statistical artifacts,")
        lines.append(
            "  NOT genuine bee features. This explains why steering produced")
        lines.append(
            "  zero bee content.")
    else:
        lines.append(
            "  Unclear results. Features show mixed activation patterns without")
        lines.append(
            "  a clear signal for either bee semantics or structural tokens.")
        lines.append(
            "  Further investigation needed (more training data, different layer).")
    lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    # ── Load model ────────────────────────────────────────────────────────
    log("Loading model...")
    model, tokenizer, device = load_model(device="cuda")
    log(f"Model loaded on {device}")

    # ── Load SAE ──────────────────────────────────────────────────────────
    log(f"Loading SAE from {SAE_PATH}...")
    sae = SparseAutoencoder(input_dim=768, latent_dim=6144)
    sae.load_state_dict(torch.load(SAE_PATH, map_location="cpu", weights_only=True))
    sae = sae.to(device).eval()
    log(f"SAE loaded: {sae.input_dim} → {sae.latent_dim}")

    # ── Collect per-token activations ─────────────────────────────────────
    bee_texts = get_default_bee_texts()
    baseline_texts = get_default_baseline_texts()
    feature_indices = [f["idx"] for f in TOP_FEATURES]

    log(f"Analyzing {len(feature_indices)} features across "
        f"{len(bee_texts)} bee + {len(baseline_texts)} baseline texts...")
    log("")

    log("Processing bee texts...")
    bee_records = collect_per_token_activations(
        sae, model, tokenizer, LAYER_IDX,
        bee_texts, "bee", feature_indices, device,
    )
    log(f"  {len(bee_records)} token-feature records from bee corpus")

    log("Processing baseline texts...")
    base_records = collect_per_token_activations(
        sae, model, tokenizer, LAYER_IDX,
        baseline_texts, "baseline", feature_indices, device,
    )
    log(f"  {len(base_records)} token-feature records from baseline corpus")

    all_records = bee_records + base_records

    # ── Analyze each feature ──────────────────────────────────────────────
    log("")
    log("Analyzing per-feature activation patterns...")
    analyses = []
    for info in TOP_FEATURES:
        analysis = analyze_feature(all_records, info["idx"], TOP_N_TOKENS)
        verdict = determine_verdict(analysis)
        cats = analysis["top_n_categories"]
        log(f"  Feature #{info['idx']}: {verdict}  "
            f"(bee-sem={cats.get('bee-semantic', 0)}, "
            f"struct={cats.get('structural', 0)}, "
            f"content={cats.get('content', 0)})")
        analyses.append(analysis)

    # ── Generate report ───────────────────────────────────────────────────
    log("")
    log("Generating report...")
    report = generate_report(analyses, TOP_FEATURES)

    report_path = output_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Save raw data as JSON for further analysis
    json_records = []
    for r in all_records:
        json_records.append({
            "token_str": r["token_str"],
            "activation": r["activation"],
            "feature_idx": r["feature_idx"],
            "text_idx": r["text_idx"],
            "corpus": r["corpus"],
            "position": r["position"],
            "category": classify_token(r["token_str"]),
        })
    with open(output_dir / "token_activations.json", "w") as f:
        json.dump(json_records, f, indent=2)

    # Save summary
    elapsed = time.time() - start_time
    summary = {
        "run_id": run_id,
        "elapsed_seconds": elapsed,
        "sae_path": str(SAE_PATH),
        "layer_idx": LAYER_IDX,
        "features_analyzed": [f["idx"] for f in TOP_FEATURES],
        "verdicts": {
            str(a["feature_idx"]): determine_verdict(a) for a in analyses
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────
    log("")
    log("=" * 60)
    log("FEATURE ANALYSIS COMPLETE")
    log("=" * 60)
    for analysis, info in zip(analyses, TOP_FEATURES):
        verdict = determine_verdict(analysis)
        log(f"  #{info['idx']:>4d} (sel={info['selectivity']:>5.2f}): {verdict}")
    log("")
    log(f"Report: {report_path}")
    log(f"Time:   {elapsed:.1f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
