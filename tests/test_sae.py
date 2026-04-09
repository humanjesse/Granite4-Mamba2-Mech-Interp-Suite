"""Tests for sparse autoencoder model."""

import torch
import pytest

from src.sae.sparse_autoencoder import SparseAutoencoder


# ── Shape tests ──────────────────────────────────────────────────────


def test_forward_output_shapes():
    sae = SparseAutoencoder(input_dim=768, latent_dim=6144)
    x = torch.randn(32, 768)
    x_hat, z = sae(x)
    assert x_hat.shape == (32, 768)
    assert z.shape == (32, 6144)


def test_encode_shape():
    sae = SparseAutoencoder(input_dim=768, latent_dim=6144)
    z = sae.encode(torch.randn(16, 768))
    assert z.shape == (16, 6144)


def test_decode_shape():
    sae = SparseAutoencoder(input_dim=768, latent_dim=6144)
    x_hat = sae.decode(torch.randn(16, 6144))
    assert x_hat.shape == (16, 768)


def test_single_sample():
    sae = SparseAutoencoder(input_dim=64, latent_dim=256)
    x = torch.randn(1, 64)
    x_hat, z = sae(x)
    assert x_hat.shape == (1, 64)
    assert z.shape == (1, 256)


def test_get_feature_direction_shape():
    sae = SparseAutoencoder(input_dim=768, latent_dim=6144)
    direction = sae.get_feature_direction(42)
    assert direction.shape == (768,)


# ── Activation properties ───────────────────────────────────────────


def test_latent_activations_non_negative():
    """ReLU encoder should produce non-negative activations."""
    sae = SparseAutoencoder(input_dim=64, latent_dim=256)
    z = sae.encode(torch.randn(100, 64))
    assert (z >= 0).all()


def test_latent_activations_sparse():
    """Most latent activations should be zero (sparse)."""
    sae = SparseAutoencoder(input_dim=64, latent_dim=512)
    z = sae.encode(torch.randn(100, 64))
    sparsity = (z == 0).float().mean()
    # With random init and ReLU, expect at least ~40% zeros
    assert sparsity > 0.3


# ── Decoder normalization ───────────────────────────────────────────


def test_decoder_columns_unit_norm_at_init():
    sae = SparseAutoencoder(input_dim=64, latent_dim=256)
    norms = sae.decoder.weight.norm(dim=0)
    assert torch.allclose(norms, torch.ones(256), atol=1e-5)


def test_normalize_decoder_restores_unit_norm():
    sae = SparseAutoencoder(input_dim=64, latent_dim=256)
    sae.decoder.weight.data *= 5.0
    sae._normalize_decoder()
    norms = sae.decoder.weight.norm(dim=0)
    assert torch.allclose(norms, torch.ones(256), atol=1e-5)


# ── Feature direction ───────────────────────────────────────────────


def test_feature_direction_detached():
    sae = SparseAutoencoder(input_dim=64, latent_dim=256)
    direction = sae.get_feature_direction(0)
    assert not direction.requires_grad


def test_feature_direction_is_decoder_column():
    sae = SparseAutoencoder(input_dim=64, latent_dim=256)
    direction = sae.get_feature_direction(10)
    expected = sae.decoder.weight[:, 10].detach()
    assert torch.allclose(direction, expected)


def test_feature_direction_is_copy():
    """Modifying returned direction should not affect the SAE."""
    sae = SparseAutoencoder(input_dim=64, latent_dim=256)
    direction = sae.get_feature_direction(0)
    original = sae.decoder.weight[:, 0].clone()
    direction.fill_(999.0)
    assert torch.allclose(sae.decoder.weight[:, 0], original)


# ── Numerical stability ─────────────────────────────────────────────


def test_no_nan_inf_forward():
    sae = SparseAutoencoder(input_dim=768, latent_dim=6144)
    x = torch.randn(32, 768)
    x_hat, z = sae(x)
    assert not torch.isnan(x_hat).any()
    assert not torch.isinf(x_hat).any()
    assert not torch.isnan(z).any()
    assert not torch.isinf(z).any()


def test_no_nan_inf_large_input():
    sae = SparseAutoencoder(input_dim=64, latent_dim=256)
    x = torch.randn(32, 64) * 100
    x_hat, z = sae(x)
    assert not torch.isnan(x_hat).any()
    assert not torch.isinf(x_hat).any()


def test_zero_input():
    sae = SparseAutoencoder(input_dim=64, latent_dim=256)
    x = torch.zeros(8, 64)
    x_hat, z = sae(x)
    assert not torch.isnan(x_hat).any()
    assert not torch.isnan(z).any()


# ── Pre-bias ─────────────────────────────────────────────────────────


def test_pre_bias_initialized_zero():
    sae = SparseAutoencoder(input_dim=64, latent_dim=256)
    assert torch.allclose(sae.pre_bias, torch.zeros(64))


def test_pre_bias_affects_encoding():
    """Setting pre_bias should change encoder output."""
    sae = SparseAutoencoder(input_dim=64, latent_dim=256)
    x = torch.randn(8, 64)
    z_before = sae.encode(x).clone()
    sae.pre_bias.data = torch.ones(64) * 5.0
    z_after = sae.encode(x)
    assert not torch.allclose(z_before, z_after)


# ── Round-trip ───────────────────────────────────────────────────────


def test_reconstruction_not_identical():
    """With random init, reconstruction should differ from input (lossy)."""
    sae = SparseAutoencoder(input_dim=64, latent_dim=256)
    x = torch.randn(16, 64)
    x_hat, _ = sae(x)
    assert not torch.allclose(x, x_hat, atol=1e-3)
