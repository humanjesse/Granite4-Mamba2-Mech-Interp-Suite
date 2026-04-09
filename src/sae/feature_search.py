"""Contrastive feature search: find SAE features associated with specific concepts.

Method:
1. Run concept texts and baseline texts through the model
2. Collect residual stream activations at the SAE's target layer
3. Encode through the SAE to get feature activations
4. Compare mean feature activations: concept vs baseline
5. Rank by selectivity score = concept_mean / (baseline_mean + epsilon)
"""

import torch
from dataclasses import dataclass

from .sparse_autoencoder import SparseAutoencoder


@dataclass
class FeatureSearchResult:
    """Result of searching for concept-associated features."""

    concept: str
    top_features: list[dict]
    # Each dict: {feature_idx, concept_activation, baseline_activation, selectivity_score}
    concept_texts: list[str]
    baseline_texts: list[str]
    # Per-text, per-feature activations for the top features
    concept_activations: torch.Tensor  # [num_concept_texts, num_top_features]
    baseline_activations: torch.Tensor  # [num_baseline_texts, num_top_features]


def _collect_mean_feature_activations(
    sae: SparseAutoencoder,
    model,
    tokenizer,
    layer_idx: int,
    texts: list[str],
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run texts through model, encode residual stream through SAE.

    Returns:
        (per_text_means, per_text_per_feature):
        - per_text_means: [num_texts, latent_dim] mean activation per text
        - per_text_per_feature: same as per_text_means (kept for per-text viz)
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
    per_text_means = []

    try:
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128,
            )
            input_ids = inputs["input_ids"].to(device)

            with torch.no_grad():
                model(input_ids=input_ids)

            hidden = storage["hidden"][0].float()  # [seq_len, hidden_size]

            with torch.no_grad():
                z = sae.encode(hidden)  # [seq_len, latent_dim]

            # Mean activation across all token positions for this text
            per_text_means.append(z.mean(dim=0))
    finally:
        hook.remove()

    return torch.stack(per_text_means)  # [num_texts, latent_dim]


def search_features(
    sae: SparseAutoencoder,
    model,
    tokenizer,
    layer_idx: int,
    concept_texts: list[str],
    baseline_texts: list[str],
    device: str = "cpu",
    top_k: int = 20,
    concept_name: str = "bee",
    progress_callback=None,
) -> FeatureSearchResult:
    """Find features that activate selectively on concept texts vs baseline.

    Args:
        sae: Trained sparse autoencoder.
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        layer_idx: Layer the SAE was trained on.
        concept_texts: Texts containing the target concept.
        baseline_texts: Generic texts for comparison.
        device: Inference device.
        top_k: Number of top features to return.
        concept_name: Label for the concept (used in results).
        progress_callback: Called with (step, total).

    Returns:
        FeatureSearchResult with ranked features and activation data.
    """
    total_steps = len(concept_texts) + len(baseline_texts)

    if progress_callback:
        progress_callback(0, total_steps)

    # Collect concept activations
    concept_acts = _collect_mean_feature_activations(
        sae, model, tokenizer, layer_idx, concept_texts, device,
    )
    if progress_callback:
        progress_callback(len(concept_texts), total_steps)

    # Collect baseline activations
    baseline_acts = _collect_mean_feature_activations(
        sae, model, tokenizer, layer_idx, baseline_texts, device,
    )
    if progress_callback:
        progress_callback(total_steps, total_steps)

    # Mean across texts: [latent_dim]
    concept_mean = concept_acts.mean(dim=0)
    baseline_mean = baseline_acts.mean(dim=0)

    # Selectivity: how much more does this feature activate on concept vs baseline
    epsilon = 1e-6
    selectivity = concept_mean / (baseline_mean + epsilon)

    # Get top-k most selective features
    top_indices = selectivity.topk(top_k).indices.tolist()

    top_features = []
    for idx in top_indices:
        top_features.append({
            "feature_idx": idx,
            "concept_activation": concept_mean[idx].item(),
            "baseline_activation": baseline_mean[idx].item(),
            "selectivity_score": selectivity[idx].item(),
        })

    # Per-text activations for top features (for visualization)
    top_idx_tensor = torch.tensor(top_indices)
    concept_top = concept_acts[:, top_idx_tensor]    # [num_concept, top_k]
    baseline_top = baseline_acts[:, top_idx_tensor]  # [num_baseline, top_k]

    return FeatureSearchResult(
        concept=concept_name,
        top_features=top_features,
        concept_texts=concept_texts,
        baseline_texts=baseline_texts,
        concept_activations=concept_top,
        baseline_activations=baseline_top,
    )


def get_default_bee_texts() -> list[str]:
    """Built-in bee-related texts for the I-Bee-M feature search."""
    return [
        "Honeybees are essential pollinators for agriculture and food production.",
        "The queen bee lays thousands of eggs per day inside the hive.",
        "Bees communicate through waggle dances to indicate flower locations.",
        "Bee colonies are organized into workers, drones, and a single queen.",
        "The buzz of bees is a familiar sound in summer gardens and meadows.",
        "Beekeeping has been practiced by humans for thousands of years.",
        "Colony collapse disorder threatens bee populations around the world.",
        "A single bee can visit up to 5,000 flowers in one day.",
        "Bees produce honey, beeswax, royal jelly, and propolis.",
        "The honeycomb is a hexagonal structure built by bees from wax.",
        "Bee stings inject venom containing melittin and other compounds.",
        "Swarms of bees travel together when a colony splits to form a new hive.",
        "Pollination by bees is critical for growing fruits, vegetables, and nuts.",
        "The beekeeper carefully inspected each frame of the buzzing hive.",
        "Wild bees nest in hollow trees, underground burrows, and rock crevices.",
    ]


def get_default_baseline_texts() -> list[str]:
    """Generic baseline texts for contrastive feature search."""
    return [
        "The capital of France is Paris, a major European city.",
        "Machine learning algorithms process large datasets efficiently.",
        "The ocean covers approximately 71 percent of Earth's surface.",
        "Classical music evolved through several distinct historical periods.",
        "Python is a popular programming language used in data science.",
        "The human heart beats approximately 100,000 times per day.",
        "Mountains form through tectonic plate movements over millions of years.",
        "Electric vehicles are becoming more affordable and widely available.",
        "The printing press revolutionized the spread of written information.",
        "Basketball was invented by James Naismith in 1891 in Massachusetts.",
        "Satellites orbit the Earth providing communications and weather data.",
        "The stock market fluctuates based on supply and demand dynamics.",
        "Ancient Rome was one of the most powerful civilizations in history.",
        "Renewable energy sources include solar, wind, and hydroelectric power.",
        "The periodic table organizes chemical elements by atomic number.",
    ]
