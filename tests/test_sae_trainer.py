"""Tests for SAE training loop."""

import torch
import pytest

from src.sae.trainer import SAETrainingConfig, SAETrainingResult, train_sae, _get_builtin_texts
from src.sae.sparse_autoencoder import SparseAutoencoder


# ── Training result ──────────────────────────────────────────────────


def test_train_returns_result():
    activations = torch.randn(1000, 64)
    config = SAETrainingConfig(
        layer_idx=0, expansion_factor=4,
        l1_coefficient=1e-3, batch_size=128, num_epochs=1,
    )
    result = train_sae(activations, config)
    assert isinstance(result, SAETrainingResult)
    assert isinstance(result.sae, SparseAutoencoder)
    assert result.sae.input_dim == 64
    assert result.sae.latent_dim == 256


def test_train_loss_positive():
    activations = torch.randn(500, 32)
    config = SAETrainingConfig(
        layer_idx=0, expansion_factor=4, batch_size=128, num_epochs=1,
    )
    result = train_sae(activations, config)
    assert result.final_loss > 0
    assert result.final_reconstruction_loss > 0
    assert result.final_l1_loss >= 0


def test_train_loss_history_populated():
    activations = torch.randn(500, 32)
    config = SAETrainingConfig(
        layer_idx=0, expansion_factor=4, batch_size=128, num_epochs=1,
    )
    result = train_sae(activations, config)
    assert len(result.loss_history) > 0
    assert len(result.reconstruction_history) == len(result.loss_history)
    assert len(result.l1_history) == len(result.loss_history)


def test_train_loss_decreases():
    """Loss should generally decrease over training."""
    activations = torch.randn(2000, 32)
    config = SAETrainingConfig(
        layer_idx=0, expansion_factor=4,
        l1_coefficient=1e-4, batch_size=256, num_epochs=3,
    )
    result = train_sae(activations, config)
    # Compare average of first 3 vs last 3 losses
    early = sum(result.loss_history[:3]) / 3
    late = sum(result.loss_history[-3:]) / 3
    assert late < early


# ── Progress callback ────────────────────────────────────────────────


def test_train_progress_callback():
    activations = torch.randn(500, 32)
    config = SAETrainingConfig(
        layer_idx=0, expansion_factor=4, batch_size=128, num_epochs=1,
    )
    calls = []
    train_sae(activations, config, progress_callback=lambda s, t: calls.append((s, t)))
    assert len(calls) > 0
    # Last call should have step == total_steps
    assert calls[-1][0] == calls[-1][1]


# ── Config defaults ──────────────────────────────────────────────────


def test_config_defaults():
    config = SAETrainingConfig()
    assert config.layer_idx == 16
    assert config.expansion_factor == 8
    assert config.l1_coefficient == 1e-3
    assert config.learning_rate == 3e-4
    assert config.num_activations == 50_000
    assert config.batch_size == 256
    assert config.num_epochs == 1


# ── SAE properties after training ────────────────────────────────────


def test_trained_sae_decoder_normalized():
    """Decoder columns should be unit-normalized after training."""
    activations = torch.randn(500, 32)
    config = SAETrainingConfig(
        layer_idx=0, expansion_factor=4, batch_size=128, num_epochs=1,
    )
    result = train_sae(activations, config)
    norms = result.sae.decoder.weight.norm(dim=0)
    assert torch.allclose(norms, torch.ones(128), atol=1e-4)


def test_dead_features_count():
    """dead_features should be a non-negative integer."""
    activations = torch.randn(500, 32)
    config = SAETrainingConfig(
        layer_idx=0, expansion_factor=4, batch_size=128, num_epochs=1,
    )
    result = train_sae(activations, config)
    assert isinstance(result.dead_features, int)
    assert result.dead_features >= 0
    assert result.dead_features <= result.sae.latent_dim


def test_mean_active_features():
    """mean_active_features should be positive."""
    activations = torch.randn(500, 32)
    config = SAETrainingConfig(
        layer_idx=0, expansion_factor=4, batch_size=128, num_epochs=1,
    )
    result = train_sae(activations, config)
    assert result.mean_active_features > 0


# ── Built-in texts ───────────────────────────────────────────────────


def test_builtin_texts_not_empty():
    texts = _get_builtin_texts()
    assert len(texts) > 100


def test_builtin_texts_are_strings():
    texts = _get_builtin_texts()
    assert all(isinstance(t, str) and len(t) > 10 for t in texts)


# ── Edge cases ───────────────────────────────────────────────────────


def test_train_single_epoch_small_batch():
    activations = torch.randn(100, 16)
    config = SAETrainingConfig(
        layer_idx=0, expansion_factor=2, batch_size=32, num_epochs=1,
    )
    result = train_sae(activations, config)
    assert result.final_loss > 0


def test_train_multiple_epochs():
    activations = torch.randn(200, 16)
    config = SAETrainingConfig(
        layer_idx=0, expansion_factor=2, batch_size=64, num_epochs=3,
    )
    result = train_sae(activations, config)
    # Should have more steps than single epoch
    expected_steps_per_epoch = (200 + 63) // 64  # ceil division
    assert len(result.loss_history) == expected_steps_per_epoch * 3
