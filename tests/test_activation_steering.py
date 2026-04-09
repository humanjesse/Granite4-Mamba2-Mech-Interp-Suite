"""Tests for activation steering engine."""

import torch
import pytest
from unittest.mock import MagicMock

from src.extraction.activation_steering import (
    SteeringConfig,
    SteeringResult,
    _generate,
    generate_comparison,
)


# ── Dataclass tests ──────────────────────────────────────────────────


def test_steering_config_fields():
    config = SteeringConfig(layer_idx=16, feature_idx=42, steering_coefficient=10.0, max_tokens=50)
    assert config.layer_idx == 16
    assert config.feature_idx == 42
    assert config.steering_coefficient == 10.0
    assert config.max_tokens == 50


def test_steering_result_fields():
    result = SteeringResult(
        prompt="test",
        steered_text="test about bees",
        unsteered_text="test about dogs",
        config=SteeringConfig(layer_idx=16, feature_idx=42),
        feature_direction_norm=1.0,
        steered_tokens=["about", "bees"],
        unsteered_tokens=["about", "dogs"],
        steered_top5=[],
        unsteered_top5=[],
    )
    assert result.prompt == "test"
    assert result.steered_text == "test about bees"
    assert result.unsteered_text == "test about dogs"
    assert result.feature_direction_norm == 1.0


def test_steering_result_defaults():
    result = SteeringResult(
        prompt="x", steered_text="x", unsteered_text="x",
        config=SteeringConfig(layer_idx=0, feature_idx=0),
        feature_direction_norm=0.5,
    )
    assert result.steered_tokens == []
    assert result.unsteered_tokens == []
    assert result.steered_top5 == []
    assert result.unsteered_top5 == []


# ── Steering vector math ────────────────────────────────────────────


def test_steering_vector_addition():
    """The core math: hidden + coefficient * vector."""
    hidden = torch.randn(1, 5, 32)
    steering_vector = torch.randn(32)
    coefficient = 10.0

    modified = hidden + coefficient * steering_vector
    diff = modified - hidden

    # Diff at every position should equal coefficient * vector
    for pos in range(5):
        assert torch.allclose(diff[0, pos], coefficient * steering_vector, atol=1e-5)


def test_steering_vector_zero_coefficient():
    """Zero coefficient should leave hidden unchanged."""
    hidden = torch.randn(1, 5, 32)
    steering_vector = torch.randn(32)
    coefficient = 0.0

    modified = hidden + coefficient * steering_vector
    assert torch.allclose(modified, hidden)


def test_steering_vector_broadcasts_to_all_positions():
    """Steering should apply to every sequence position."""
    hidden = torch.zeros(1, 10, 64)
    steering_vector = torch.ones(64)
    coefficient = 5.0

    modified = hidden + coefficient * steering_vector
    for pos in range(10):
        assert torch.allclose(modified[0, pos], torch.ones(64) * 5.0)


# ── Hook behavior ────────────────────────────────────────────────────


def test_hook_returns_modified_tuple():
    """When output is a tuple, hook should return (modified, *rest)."""
    hidden = torch.randn(1, 3, 32)
    extra = torch.randn(5)
    output = (hidden, extra)

    vec = torch.ones(32) * 0.1
    coefficient = 1.0

    # Simulate what the hook does
    if isinstance(output, tuple):
        h = output[0]
    else:
        h = output
    modified = h + coefficient * vec
    if isinstance(output, tuple):
        result = (modified,) + output[1:]
    else:
        result = modified

    assert isinstance(result, tuple)
    assert result[0].shape == hidden.shape
    assert torch.equal(result[1], extra)


def test_hook_returns_modified_tensor():
    """When output is a plain tensor, hook should return modified tensor."""
    hidden = torch.randn(1, 3, 32)
    vec = torch.ones(32) * 0.1
    coefficient = 1.0

    modified = hidden + coefficient * vec
    assert isinstance(modified, torch.Tensor)
    assert modified.shape == hidden.shape


# ── Generation with mock model ───────────────────────────────────────


def _make_mock_model_and_tokenizer(vocab_size=100, hidden_dim=32, num_layers=4):
    """Create a mock model that returns random logits and supports hooks."""
    model = MagicMock()

    # Set up layers with real hook support
    layers = []
    for _ in range(num_layers):
        layer = MagicMock()
        # Make register_forward_hook return a removable handle
        handle = MagicMock()
        handle.remove = MagicMock()
        layer.register_forward_hook = MagicMock(return_value=handle)
        layers.append(layer)
    model.model.layers = layers

    # Model parameters (for dtype detection) — return fresh iterator each call
    param = torch.nn.Parameter(torch.zeros(1))
    model.parameters = MagicMock(side_effect=lambda: iter([param]))

    # Model forward: return logits with high prob on token 42
    def model_forward(**kwargs):
        seq_len = kwargs["input_ids"].shape[1]
        logits = torch.randn(1, seq_len, vocab_size)
        # Make token 42 the argmax at the last position
        logits[0, -1, 42] = 100.0
        output = MagicMock()
        output.logits = logits
        return output

    model.__call__ = model_forward
    model.side_effect = model_forward

    # Tokenizer
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    tokenizer.decode = MagicMock(side_effect=lambda ids, **kw: "token" if isinstance(ids, int) else "hello token")
    tokenizer.convert_ids_to_tokens = MagicMock(return_value="tok")
    tokenizer.eos_token_id = 0

    return model, tokenizer


def test_generate_without_steering():
    model, tokenizer = _make_mock_model_and_tokenizer()
    result = _generate(model, tokenizer, "hello", max_tokens=3, device="cpu")
    assert "text" in result
    assert "tokens" in result
    assert "top5" in result
    assert len(result["tokens"]) <= 3
    assert len(result["top5"]) <= 3


def test_generate_with_steering_registers_hook():
    model, tokenizer = _make_mock_model_and_tokenizer()
    vec = torch.randn(32)
    result = _generate(
        model, tokenizer, "hello", max_tokens=2, device="cpu",
        steering_vector=vec, layer_idx=1, coefficient=10.0,
    )
    # Hook should have been registered on layer 1
    model.model.layers[1].register_forward_hook.assert_called()
    assert len(result["tokens"]) <= 2


def test_generate_hook_removed_after_step():
    model, tokenizer = _make_mock_model_and_tokenizer()
    handle = MagicMock()
    handle.remove = MagicMock()
    model.model.layers[0].register_forward_hook = MagicMock(return_value=handle)

    vec = torch.randn(32)
    _generate(
        model, tokenizer, "hello", max_tokens=1, device="cpu",
        steering_vector=vec, layer_idx=0, coefficient=5.0,
    )
    # Handle should be removed after the forward pass
    handle.remove.assert_called()


def test_generate_progress_callback():
    model, tokenizer = _make_mock_model_and_tokenizer()
    calls = []
    _generate(
        model, tokenizer, "hello", max_tokens=3, device="cpu",
        progress_callback=lambda s, t: calls.append((s, t)),
    )
    assert len(calls) > 0
    assert calls[0][1] == 3  # total is max_tokens


def test_generate_comparison_returns_result():
    model, tokenizer = _make_mock_model_and_tokenizer()
    vec = torch.randn(32)
    result = generate_comparison(
        model, tokenizer, "hello", vec,
        layer_idx=1, coefficient=10.0, max_tokens=2, device="cpu",
    )
    assert isinstance(result, SteeringResult)
    assert result.prompt == "hello"
    assert len(result.steered_tokens) <= 2
    assert len(result.unsteered_tokens) <= 2
    assert result.feature_direction_norm > 0
