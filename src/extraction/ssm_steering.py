"""SSM state injection steering for Mamba-2 layers.

Implements a novel steering approach that operates in Mamba's native SSM
(State Space Model) recurrent state space rather than the residual stream.

The approach:
1. Extract SSM states from Mamba layers after processing bee/control prompts
2. Compute mean-difference contrastive vectors in SSM state space
3. During generation, inject the contrastive vector into the SSM state cache
   after each forward pass, steering the model's recurrent dynamics

The SSM state update equation is:
    h[t] = h[t-1] * dA + dBx
    y[t] = C @ h[t] + D * x[t]

Injection happens between steps: after the model updates h[t], we add
coefficient * contrastive_vector to h[t]. On the next step, the model
reads our modified state, maintaining persistent steering pressure.
"""

import torch
from dataclasses import dataclass, field


@dataclass
class SSMSteeringConfig:
    """Configuration for SSM state injection steering."""

    layers: list[dict] = field(default_factory=list)
    # Each dict: {layer_idx: int, coefficient: float}
    max_tokens: int = 50
    config_name: str = ""


@dataclass
class SSMSteeringResult:
    """Result of SSM steered vs unsteered generation."""

    prompt: str
    steered_text: str
    unsteered_text: str
    config: SSMSteeringConfig
    steered_tokens: list[str] = field(default_factory=list)
    unsteered_tokens: list[str] = field(default_factory=list)
    steered_top5: list[list[tuple[str, float]]] = field(default_factory=list)
    unsteered_top5: list[list[tuple[str, float]]] = field(default_factory=list)


def _get_cache_class(model):
    """Import the cache class from the model's module."""
    # The cache class lives in the same module as the model
    import importlib
    module = importlib.import_module(model.__class__.__module__)
    # Walk up to find the modeling module (CausalLM wraps the base model)
    base_module = importlib.import_module(model.model.__class__.__module__)
    if hasattr(base_module, "HybridMambaAttentionDynamicCache"):
        return base_module.HybridMambaAttentionDynamicCache
    if hasattr(module, "HybridMambaAttentionDynamicCache"):
        return module.HybridMambaAttentionDynamicCache
    raise RuntimeError("Could not find HybridMambaAttentionDynamicCache class")


def extract_ssm_states(
    model,
    tokenizer,
    prompts: list[str],
    target_layers: list[int],
    device: str,
) -> dict[int, list[torch.Tensor]]:
    """Extract SSM states from Mamba layers after processing prompts.

    For each prompt, creates a fresh cache, runs a prefill forward pass,
    then reads the SSM state that the model built up.

    Args:
        model: HuggingFace causal LM (GraniteMoeHybrid).
        tokenizer: Corresponding tokenizer.
        prompts: List of text prompts to process.
        target_layers: Mamba layer indices to extract SSM states from.
        device: Inference device.

    Returns:
        Dict mapping layer_idx -> list of SSM state tensors.
        Each tensor has shape [n_heads, d_head, d_state] (batch dim squeezed).
    """
    CacheClass = _get_cache_class(model)
    states = {layer: [] for layer in target_layers}

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

        # Fresh cache for each prompt
        cache = CacheClass(
            config=model.config,
            batch_size=1,
            dtype=next(model.parameters()).dtype,
            device=device,
        )

        with torch.no_grad():
            model(input_ids=input_ids, past_key_values=cache)

        # Extract SSM states (squeeze batch dimension)
        for layer_idx in target_layers:
            ssm_state = cache.ssm_states[layer_idx].detach().clone().squeeze(0)
            # Shape: [n_heads, d_head, d_state]
            states[layer_idx].append(ssm_state.float())

        # Free cache memory
        del cache

    return states


def compute_ssm_contrastive_vectors(
    bee_states: dict[int, list[torch.Tensor]],
    control_states: dict[int, list[torch.Tensor]],
    normalize: bool = True,
) -> dict[int, torch.Tensor]:
    """Compute mean-difference contrastive vectors in SSM state space.

    Args:
        bee_states: {layer_idx: [tensor, ...]} from bee prompts.
        control_states: {layer_idx: [tensor, ...]} from control prompts.
        normalize: If True, L2-normalize the difference vector.

    Returns:
        Dict mapping layer_idx -> contrastive vector [n_heads, d_head, d_state].
    """
    vectors = {}
    for layer_idx in bee_states:
        mean_bee = torch.stack(bee_states[layer_idx]).mean(dim=0)
        mean_control = torch.stack(control_states[layer_idx]).mean(dim=0)
        diff = mean_bee - mean_control

        if normalize:
            norm = diff.norm()
            if norm > 0:
                diff = diff / norm

        vectors[layer_idx] = diff

    return vectors


def _generate_with_ssm_injection(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    device: str,
    ssm_injections: dict[int, dict] | None = None,
    progress_callback=None,
) -> dict:
    """Autoregressive generation with SSM state injection.

    Uses HybridMambaAttentionDynamicCache for stateful generation.
    After each forward pass, injects contrastive vectors into the
    SSM state for configured layers.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: Input text.
        max_tokens: Max tokens to generate.
        device: Inference device.
        ssm_injections: Dict of {layer_idx: {"vector": Tensor, "coefficient": float}}.
            If None, runs unsteered generation (still uses cache for consistency).
        progress_callback: Called with (step, total).

    Returns:
        Dict with: text, tokens, top5 (per-step top-5 predictions).
    """
    CacheClass = _get_cache_class(model)
    model_dtype = next(model.parameters()).dtype

    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    # Create fresh cache
    cache = CacheClass(
        config=model.config,
        batch_size=1,
        dtype=model_dtype,
        device=device,
    )

    # Prefill: process entire prompt, populate cache
    with torch.no_grad():
        outputs = model(input_ids=input_ids, past_key_values=cache)

    # Get first generated token from prefill logits
    logits = outputs.logits[0, -1, :].float()
    next_id = logits.argmax().item()

    # Track outputs
    all_ids = input_ids[0].tolist()
    generated_tokens = []
    top5_per_step = []

    # Record first token
    probs = torch.softmax(logits, dim=-1)
    top5_probs, top5_ids = probs.topk(5)
    top5 = [
        (
            tokenizer.decode(tid.item()).strip()
            or repr(tokenizer.convert_ids_to_tokens(tid.item())),
            tp.item(),
        )
        for tid, tp in zip(top5_ids, top5_probs)
    ]
    top5_per_step.append(top5)
    generated_tokens.append(tokenizer.decode(next_id))
    all_ids.append(next_id)

    # Inject after prefill (before first generation step)
    if ssm_injections:
        for layer_idx, cfg in ssm_injections.items():
            vec = cfg["vector"].to(device=device, dtype=model_dtype)
            coeff = cfg["coefficient"]
            cache.ssm_states[layer_idx] = cache.ssm_states[layer_idx] + coeff * vec.unsqueeze(0)

    if progress_callback:
        progress_callback(1, max_tokens)

    if next_id == tokenizer.eos_token_id:
        full_text = tokenizer.decode(all_ids, skip_special_tokens=False)
        return {"text": full_text, "tokens": generated_tokens, "top5": top5_per_step}

    # Generation loop (token by token with cache)
    for step in range(1, max_tokens):
        next_input = torch.tensor([[next_id]], device=device)

        with torch.no_grad():
            outputs = model(input_ids=next_input, past_key_values=cache)

        # Inject SSM state after forward pass
        if ssm_injections:
            for layer_idx, cfg in ssm_injections.items():
                vec = cfg["vector"].to(device=device, dtype=model_dtype)
                coeff = cfg["coefficient"]
                cache.ssm_states[layer_idx] = cache.ssm_states[layer_idx] + coeff * vec.unsqueeze(0)

        logits = outputs.logits[0, -1, :].float()
        next_id = logits.argmax().item()

        # Top-5 predictions
        probs = torch.softmax(logits, dim=-1)
        top5_probs, top5_ids = probs.topk(5)
        top5 = [
            (
                tokenizer.decode(tid.item()).strip()
                or repr(tokenizer.convert_ids_to_tokens(tid.item())),
                tp.item(),
            )
            for tid, tp in zip(top5_ids, top5_probs)
        ]
        top5_per_step.append(top5)

        token_str = tokenizer.decode(next_id)
        generated_tokens.append(token_str)
        all_ids.append(next_id)

        if progress_callback:
            progress_callback(step + 1, max_tokens)

        if next_id == tokenizer.eos_token_id:
            break

    full_text = tokenizer.decode(all_ids, skip_special_tokens=False)
    del cache

    return {
        "text": full_text,
        "tokens": generated_tokens,
        "top5": top5_per_step,
    }


def generate_ssm_comparison(
    model,
    tokenizer,
    prompt: str,
    ssm_injections: dict[int, dict],
    max_tokens: int = 50,
    device: str = "cpu",
    config_name: str = "",
    progress_callback=None,
    unsteered_cache: dict | None = None,
) -> SSMSteeringResult:
    """Run SSM-steered and unsteered generation for comparison.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: Input text.
        ssm_injections: {layer_idx: {"vector": Tensor, "coefficient": float}}.
        max_tokens: Max tokens to generate.
        device: Inference device.
        config_name: Label for this configuration.
        progress_callback: Called with (step, total) across both runs.
        unsteered_cache: If provided and prompt is a key, skip unsteered generation.

    Returns:
        SSMSteeringResult with steered and unsteered outputs.
    """
    total = max_tokens * 2

    def steered_progress(step, _total):
        if progress_callback:
            progress_callback(step, total)

    def unsteered_progress(step, _total):
        if progress_callback:
            progress_callback(max_tokens + step, total)

    # Steered generation
    steered = _generate_with_ssm_injection(
        model, tokenizer, prompt, max_tokens, device,
        ssm_injections=ssm_injections,
        progress_callback=steered_progress,
    )

    # Unsteered baseline (use cache if available)
    if unsteered_cache is not None and prompt in unsteered_cache:
        unsteered = unsteered_cache[prompt]
    else:
        unsteered = _generate_with_ssm_injection(
            model, tokenizer, prompt, max_tokens, device,
            ssm_injections=None,
            progress_callback=unsteered_progress,
        )
        if unsteered_cache is not None:
            unsteered_cache[prompt] = unsteered

    layers_info = [
        {"layer_idx": li, "coefficient": cfg["coefficient"]}
        for li, cfg in ssm_injections.items()
    ]

    config = SSMSteeringConfig(
        layers=layers_info,
        max_tokens=max_tokens,
        config_name=config_name,
    )

    return SSMSteeringResult(
        prompt=prompt,
        steered_text=steered["text"],
        unsteered_text=unsteered["text"],
        config=config,
        steered_tokens=steered["tokens"],
        unsteered_tokens=unsteered["tokens"],
        steered_top5=steered["top5"],
        unsteered_top5=unsteered["top5"],
    )
