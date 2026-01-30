"""Compute meaningful differences between two extraction results."""

import torch
import torch.nn.functional as F


def compute_activation_diff(result_a: dict, result_b: dict, head_agg: str = "mean") -> dict:
    """Compare two extraction results from GraniteAttentionExtractor.extract().

    Args:
        result_a: Extraction dict for prompt A.
        result_b: Extraction dict for prompt B.
        head_agg: How to aggregate attention heads ("mean" or "max").

    Returns:
        Dict with comparison metrics across residual streams, attention patterns,
        neuron activations, and logit lens predictions.
    """
    tokens_a = result_a["tokens"]
    tokens_b = result_b["tokens"]
    shared_len = min(len(tokens_a), len(tokens_b))

    return {
        "tokens_a": tokens_a,
        "tokens_b": tokens_b,
        "shared_len": shared_len,
        "residual_similarity": _residual_stream_similarity(result_a, result_b, shared_len),
        "attention_divergence": _attention_divergence(result_a, result_b, shared_len, head_agg),
        "top_changed_neurons": _top_changed_neurons(result_a, result_b, shared_len),
        "logit_lens_diff": _logit_lens_diff(result_a, result_b),
    }


def _residual_stream_similarity(result_a: dict, result_b: dict, shared_len: int) -> dict:
    """Cosine similarity and L2 distance between residual streams at each layer."""
    rs_a = result_a.get("residual_stream", {})
    rs_b = result_b.get("residual_stream", {})

    common_layers = sorted(set(rs_a.keys()) & set(rs_b.keys()))
    if not common_layers:
        return {"layers": [], "cosine_sim_mean": [], "l2_dist_mean": []}

    cosine_means = []
    l2_means = []

    for layer_idx in common_layers:
        a = rs_a[layer_idx][:, :shared_len, :].float()
        b = rs_b[layer_idx][:, :shared_len, :].float()

        cos_sim = F.cosine_similarity(a, b, dim=-1)  # [batch, shared_len]
        cosine_means.append(cos_sim.mean().item())

        l2 = torch.norm(a - b, dim=-1).mean().item()
        l2_means.append(l2)

    return {
        "layers": common_layers,
        "cosine_sim_mean": cosine_means,
        "l2_dist_mean": l2_means,
    }


def _attention_divergence(result_a: dict, result_b: dict, shared_len: int, head_agg: str) -> dict:
    """Jensen-Shannon divergence between attention distributions at each layer."""
    combined = []

    for attn_key, layer_type in [("mamba_attention", "mamba"), ("transformer_attention", "attention")]:
        attn_a = result_a.get(attn_key, {})
        attn_b = result_b.get(attn_key, {})
        common = sorted(set(attn_a.keys()) & set(attn_b.keys()))

        for layer_idx in common:
            a = attn_a[layer_idx].float()
            b = attn_b[layer_idx].float()

            # Aggregate heads: [batch, heads, seq, seq] -> [seq, seq]
            if head_agg == "mean":
                a = a[0].mean(dim=0)
                b = b[0].mean(dim=0)
            else:
                a = a[0].max(dim=0).values
                b = b[0].max(dim=0).values

            # Truncate to shared length
            a = a[:shared_len, :shared_len]
            b = b[:shared_len, :shared_len]

            # Normalize rows to probability distributions
            a = a / (a.sum(dim=-1, keepdim=True) + 1e-10)
            b = b / (b.sum(dim=-1, keepdim=True) + 1e-10)

            # JSD per row, then average
            m = 0.5 * (a + b)
            kl_am = (a * torch.log(a.clamp(min=1e-10) / m.clamp(min=1e-10))).sum(dim=-1)
            kl_bm = (b * torch.log(b.clamp(min=1e-10) / m.clamp(min=1e-10))).sum(dim=-1)
            jsd = (0.5 * kl_am + 0.5 * kl_bm).mean().item()

            combined.append((layer_idx, jsd, layer_type))

    combined.sort(key=lambda x: x[0])
    return {"combined": combined}


def _top_changed_neurons(result_a: dict, result_b: dict, shared_len: int, top_k: int = 20) -> list:
    """Find neurons with the largest activation change between prompts."""
    na_a = result_a.get("neuron_activations", {})
    na_b = result_b.get("neuron_activations", {})
    common = sorted(set(na_a.keys()) & set(na_b.keys()))

    all_changes = []
    for layer_idx in common:
        a = na_a[layer_idx][:shared_len, :].float().abs().mean(dim=0)
        b = na_b[layer_idx][:shared_len, :].float().abs().mean(dim=0)
        delta = b - a

        for neuron_idx in range(len(delta)):
            all_changes.append({
                "layer": layer_idx,
                "neuron_idx": neuron_idx,
                "mean_act_a": a[neuron_idx].item(),
                "mean_act_b": b[neuron_idx].item(),
                "delta": delta[neuron_idx].item(),
                "abs_delta": abs(delta[neuron_idx].item()),
            })

    all_changes.sort(key=lambda x: x["abs_delta"], reverse=True)
    return all_changes[:top_k]


def _logit_lens_diff(result_a: dict, result_b: dict) -> dict:
    """Compare logit lens predictions between two prompts."""
    ll_a = result_a.get("logit_lens", {})
    ll_b = result_b.get("logit_lens", {})

    layers_a = {d["layer"]: d for d in ll_a.get("layers", [])}
    layers_b = {d["layer"]: d for d in ll_b.get("layers", [])}
    common = sorted(set(layers_a.keys()) & set(layers_b.keys()))

    layers = []
    for layer_idx in common:
        da = layers_a[layer_idx]
        db = layers_b[layer_idx]
        top1_a = da["tokens"][0] if da["tokens"] else ""
        prob_a = da["probs"][0] if da["probs"] else 0.0
        top1_b = db["tokens"][0] if db["tokens"] else ""
        prob_b = db["probs"][0] if db["probs"] else 0.0

        layers.append({
            "layer": layer_idx,
            "top1_a": top1_a,
            "prob_a": prob_a,
            "top1_b": top1_b,
            "prob_b": prob_b,
            "same_prediction": top1_a == top1_b,
        })

    return {"layers": layers}
