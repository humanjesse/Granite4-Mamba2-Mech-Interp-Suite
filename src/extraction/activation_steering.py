"""Activation steering engine: add SAE feature directions to the residual stream.

Implements the Golden Gate Claude technique:
1. Get a feature direction from a trained SAE (decoder column)
2. During autoregressive generation, hook a target layer
3. Add coefficient * direction to the residual stream at every position
4. The model becomes "obsessed" with the concept encoded by that feature

The steering hook is: hidden = hidden + alpha * steering_vector
Applied at the target layer's output (post-layer residual stream).
"""

import torch
from dataclasses import dataclass, field


@dataclass
class SteeringConfig:
    """Configuration for activation steering."""

    layer_idx: int
    feature_idx: int
    steering_coefficient: float = 10.0
    max_tokens: int = 50


@dataclass
class SteeringResult:
    """Result of steered vs unsteered generation."""

    prompt: str
    steered_text: str
    unsteered_text: str
    config: SteeringConfig
    feature_direction_norm: float
    steered_tokens: list[str] = field(default_factory=list)
    unsteered_tokens: list[str] = field(default_factory=list)
    steered_top5: list[list[tuple[str, float]]] = field(default_factory=list)
    unsteered_top5: list[list[tuple[str, float]]] = field(default_factory=list)


def _generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    device: str,
    steering_vector: torch.Tensor | None = None,
    layer_idx: int | None = None,
    coefficient: float = 0.0,
    progress_callback=None,
    last_token_only: bool = False,
) -> dict:
    """Autoregressive generation, optionally with a steering hook.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: Input text.
        max_tokens: Max tokens to generate.
        device: Inference device.
        steering_vector: If provided, add this direction to residual stream.
        layer_idx: Which layer to hook for steering.
        coefficient: Steering strength (alpha).
        progress_callback: Called with (step, total).
        last_token_only: If True, only steer the last token position (less aggressive).

    Returns:
        Dict with: text, tokens, top5 (per-step top-5 predictions).
    """
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    all_ids = list(input_ids)
    generated_tokens = []
    top5_per_step = []

    for step in range(max_tokens):
        ids_tensor = torch.tensor([all_ids], device=device)

        hook = None
        if steering_vector is not None and layer_idx is not None and coefficient != 0.0:
            layer = model.model.layers[layer_idx]
            vec = steering_vector.to(device, dtype=next(model.parameters()).dtype)

            def steering_hook(module, input, output, _vec=vec, _coeff=coefficient, _last_only=last_token_only):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                if _last_only:
                    modified = hidden.clone()
                    modified[:, -1, :] = modified[:, -1, :] + _coeff * _vec
                else:
                    modified = hidden + _coeff * _vec
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified

            hook = layer.register_forward_hook(steering_hook)

        try:
            with torch.no_grad():
                outputs = model(input_ids=ids_tensor)
        finally:
            if hook is not None:
                hook.remove()

        logits = outputs.logits[0, -1, :].float()

        # Greedy decoding
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

    return {
        "text": full_text,
        "tokens": generated_tokens,
        "top5": top5_per_step,
    }


def generate_comparison(
    model,
    tokenizer,
    prompt: str,
    steering_vector: torch.Tensor,
    layer_idx: int,
    coefficient: float = 10.0,
    max_tokens: int = 50,
    device: str = "cpu",
    progress_callback=None,
    last_token_only: bool = False,
) -> SteeringResult:
    """Run both steered and unsteered generation for side-by-side comparison.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: Input text.
        steering_vector: SAE feature direction [hidden_size].
        layer_idx: Layer to hook for steering.
        coefficient: Steering strength (5-20 is typical sweet spot).
        max_tokens: Max tokens to generate.
        device: Inference device.
        progress_callback: Called with (step, total) across both runs.
        last_token_only: If True, only steer the last token position (less aggressive).

    Returns:
        SteeringResult with steered and unsteered outputs.
    """
    total = max_tokens * 2

    def steered_progress(step, _total):
        if progress_callback:
            progress_callback(step, total)

    def unsteered_progress(step, _total):
        if progress_callback:
            progress_callback(max_tokens + step, total)

    # Steered generation
    steered = _generate(
        model, tokenizer, prompt, max_tokens, device,
        steering_vector=steering_vector,
        layer_idx=layer_idx,
        coefficient=coefficient,
        progress_callback=steered_progress,
        last_token_only=last_token_only,
    )

    # Unsteered baseline
    unsteered = _generate(
        model, tokenizer, prompt, max_tokens, device,
        progress_callback=unsteered_progress,
    )

    config = SteeringConfig(
        layer_idx=layer_idx,
        feature_idx=-1,  # Set by caller if known
        steering_coefficient=coefficient,
        max_tokens=max_tokens,
    )

    return SteeringResult(
        prompt=prompt,
        steered_text=steered["text"],
        unsteered_text=unsteered["text"],
        config=config,
        feature_direction_norm=steering_vector.norm().item(),
        steered_tokens=steered["tokens"],
        unsteered_tokens=unsteered["tokens"],
        steered_top5=steered["top5"],
        unsteered_top5=unsteered["top5"],
    )


def _generate_with_clamp(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    device: str,
    sae,
    layer_idx: int,
    feature_idx: int,
    clamp_value: float = 10.0,
    progress_callback=None,
) -> dict:
    """Autoregressive generation with SAE encode→clamp→decode→replace steering.

    This is Anthropic's Golden Gate Claude method: intercept the residual stream,
    encode through the SAE, clamp the target feature to a fixed value, decode back,
    and replace the original residual stream with the modified reconstruction.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: Input text.
        max_tokens: Max tokens to generate.
        device: Inference device.
        sae: Trained SparseAutoencoder (must be on same device, eval mode).
        layer_idx: Which layer to hook.
        feature_idx: SAE feature index to clamp.
        clamp_value: Value to clamp the feature activation to.
        progress_callback: Called with (step, total).

    Returns:
        Dict with: text, tokens, top5 (per-step top-5 predictions).
    """
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    all_ids = list(input_ids)
    generated_tokens = []
    top5_per_step = []

    for step in range(max_tokens):
        ids_tensor = torch.tensor([all_ids], device=device)

        layer = model.model.layers[layer_idx]

        def clamp_hook(module, input, output,
                       _sae=sae, _fidx=feature_idx, _clamp=clamp_value):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            original_dtype = hidden.dtype
            h_flat = hidden.float()[0]       # [seq_len, hidden_size]

            z = _sae.encode(h_flat)          # [seq_len, latent_dim]
            z[:, _fidx] = _clamp             # clamp target feature
            x_hat = _sae.decode(z)           # [seq_len, hidden_size]

            modified = x_hat.unsqueeze(0).to(original_dtype)

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        hook = layer.register_forward_hook(clamp_hook)

        try:
            with torch.no_grad():
                outputs = model(input_ids=ids_tensor)
        finally:
            hook.remove()

        logits = outputs.logits[0, -1, :].float()

        # Greedy decoding
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

    return {
        "text": full_text,
        "tokens": generated_tokens,
        "top5": top5_per_step,
    }


@dataclass
class MultiLayerSteeringConfig:
    """Configuration for multi-layer activation steering."""

    layers: list[dict] = field(default_factory=list)
    # Each dict: {layer_idx: int, feature_idx: int, coefficient: float}
    max_tokens: int = 50
    config_name: str = ""


@dataclass
class MultiLayerSteeringResult:
    """Result of multi-layer steered vs unsteered generation."""

    prompt: str
    steered_text: str
    unsteered_text: str
    config: MultiLayerSteeringConfig
    steered_tokens: list[str] = field(default_factory=list)
    unsteered_tokens: list[str] = field(default_factory=list)
    steered_top5: list[list[tuple[str, float]]] = field(default_factory=list)
    unsteered_top5: list[list[tuple[str, float]]] = field(default_factory=list)


def _generate_multi_layer(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    device: str,
    layer_steering: list[dict] | None = None,
    progress_callback=None,
    last_token_only: bool = False,
) -> dict:
    """Autoregressive generation with steering hooks on multiple layers simultaneously.

    Unlike _generate() which registers/removes a hook per step, this registers
    all hooks ONCE before the generation loop for efficiency.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: Input text.
        max_tokens: Max tokens to generate.
        device: Inference device.
        layer_steering: List of dicts, each with:
            - layer_idx: int — which layer to hook
            - steering_vector: Tensor [hidden_size] — direction to add
            - coefficient: float — steering strength
        progress_callback: Called with (step, total).
        last_token_only: If True, only steer the last token position.

    Returns:
        Dict with: text, tokens, top5 (per-step top-5 predictions).
    """
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    all_ids = list(input_ids)
    generated_tokens = []
    top5_per_step = []

    # Register all hooks ONCE before the generation loop
    hooks = []
    if layer_steering:
        model_dtype = next(model.parameters()).dtype

        def make_hook(vec, coeff, last_only):
            """Factory to avoid late-binding closure bugs."""
            def steering_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                if last_only:
                    modified = hidden.clone()
                    modified[:, -1, :] = modified[:, -1, :] + coeff * vec
                else:
                    modified = hidden + coeff * vec
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified
            return steering_hook

        for ls in layer_steering:
            layer = model.model.layers[ls["layer_idx"]]
            vec = ls["steering_vector"].to(device, dtype=model_dtype)
            coeff = ls["coefficient"]
            hook = layer.register_forward_hook(make_hook(vec, coeff, last_token_only))
            hooks.append(hook)

    try:
        for step in range(max_tokens):
            ids_tensor = torch.tensor([all_ids], device=device)

            with torch.no_grad():
                outputs = model(input_ids=ids_tensor)

            logits = outputs.logits[0, -1, :].float()

            # Greedy decoding
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
    finally:
        for hook in hooks:
            hook.remove()

    full_text = tokenizer.decode(all_ids, skip_special_tokens=False)

    return {
        "text": full_text,
        "tokens": generated_tokens,
        "top5": top5_per_step,
    }


def generate_multi_layer_comparison(
    model,
    tokenizer,
    prompt: str,
    layer_steering: list[dict],
    max_tokens: int = 50,
    device: str = "cpu",
    config_name: str = "",
    progress_callback=None,
    last_token_only: bool = False,
    unsteered_cache: dict | None = None,
) -> MultiLayerSteeringResult:
    """Run multi-layer steered and unsteered generation for comparison.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: Input text.
        layer_steering: List of dicts with layer_idx, steering_vector, coefficient.
        max_tokens: Max tokens to generate.
        device: Inference device.
        config_name: Label for this configuration.
        progress_callback: Called with (step, total) across both runs.
        last_token_only: If True, only steer the last token position.
        unsteered_cache: If provided and prompt is a key, skip unsteered generation.

    Returns:
        MultiLayerSteeringResult with steered and unsteered outputs.
    """
    total = max_tokens * 2

    def steered_progress(step, _total):
        if progress_callback:
            progress_callback(step, total)

    def unsteered_progress(step, _total):
        if progress_callback:
            progress_callback(max_tokens + step, total)

    # Steered generation (all layers hooked simultaneously)
    steered = _generate_multi_layer(
        model, tokenizer, prompt, max_tokens, device,
        layer_steering=layer_steering,
        progress_callback=steered_progress,
        last_token_only=last_token_only,
    )

    # Unsteered baseline (use cache if available)
    if unsteered_cache is not None and prompt in unsteered_cache:
        unsteered = unsteered_cache[prompt]
    else:
        unsteered = _generate_multi_layer(
            model, tokenizer, prompt, max_tokens, device,
            layer_steering=None,
            progress_callback=unsteered_progress,
        )
        if unsteered_cache is not None:
            unsteered_cache[prompt] = unsteered

    config = MultiLayerSteeringConfig(
        layers=[
            {"layer_idx": ls["layer_idx"], "feature_idx": ls.get("feature_idx", -1),
             "coefficient": ls["coefficient"]}
            for ls in layer_steering
        ],
        max_tokens=max_tokens,
        config_name=config_name,
    )

    return MultiLayerSteeringResult(
        prompt=prompt,
        steered_text=steered["text"],
        unsteered_text=unsteered["text"],
        config=config,
        steered_tokens=steered["tokens"],
        unsteered_tokens=unsteered["tokens"],
        steered_top5=steered["top5"],
        unsteered_top5=unsteered["top5"],
    )


def generate_clamp_comparison(
    model,
    tokenizer,
    prompt: str,
    sae,
    layer_idx: int,
    feature_idx: int,
    clamp_value: float = 10.0,
    max_tokens: int = 50,
    device: str = "cpu",
    progress_callback=None,
) -> SteeringResult:
    """Run SAE clamp-steered and unsteered generation for comparison.

    Uses Anthropic's encode→clamp→decode→replace method instead of
    simple vector addition.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt: Input text.
        sae: Trained SparseAutoencoder (must be on device, eval mode).
        layer_idx: Layer to hook.
        feature_idx: SAE feature index to clamp.
        clamp_value: Fixed activation value for the clamped feature.
        max_tokens: Max tokens to generate.
        device: Inference device.
        progress_callback: Called with (step, total) across both runs.

    Returns:
        SteeringResult with steered (clamped) and unsteered outputs.
    """
    total = max_tokens * 2

    def steered_progress(step, _total):
        if progress_callback:
            progress_callback(step, total)

    def unsteered_progress(step, _total):
        if progress_callback:
            progress_callback(max_tokens + step, total)

    # Clamped generation
    steered = _generate_with_clamp(
        model, tokenizer, prompt, max_tokens, device,
        sae=sae,
        layer_idx=layer_idx,
        feature_idx=feature_idx,
        clamp_value=clamp_value,
        progress_callback=steered_progress,
    )

    # Unsteered baseline
    unsteered = _generate(
        model, tokenizer, prompt, max_tokens, device,
        progress_callback=unsteered_progress,
    )

    config = SteeringConfig(
        layer_idx=layer_idx,
        feature_idx=feature_idx,
        steering_coefficient=clamp_value,
        max_tokens=max_tokens,
    )

    return SteeringResult(
        prompt=prompt,
        steered_text=steered["text"],
        unsteered_text=unsteered["text"],
        config=config,
        feature_direction_norm=0.0,  # Not applicable for clamp method
        steered_tokens=steered["tokens"],
        unsteered_tokens=unsteered["tokens"],
        steered_top5=steered["top5"],
        unsteered_top5=unsteered["top5"],
    )
