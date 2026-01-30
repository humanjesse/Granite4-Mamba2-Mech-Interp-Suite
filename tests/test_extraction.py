"""Sanity checks for Mamba-2 hidden attention extraction math."""

import torch
import pytest
from src.extraction.mamba_hidden_attn import (
    compute_hidden_attention_mamba2,
    compute_hidden_attention_mamba2_memory_efficient,
)


# Test dimensions matching Granite 4.0 h-350m config
N_HEADS = 48
N_GROUPS = 1
STATE_SIZE = 128
SEQ_LEN = 16
BATCH = 1


def _make_test_inputs(seq_len=SEQ_LEN):
    """Create synthetic test inputs with realistic shapes."""
    dt = torch.randn(BATCH, seq_len, N_HEADS)
    B = torch.randn(BATCH, seq_len, N_GROUPS * STATE_SIZE)
    C = torch.randn(BATCH, seq_len, N_GROUPS * STATE_SIZE)
    A_log = torch.log(torch.arange(1, N_HEADS + 1).float())
    dt_bias = torch.ones(N_HEADS)
    return dt, B, C, A_log, dt_bias


def test_output_shape():
    """Hidden attention should be [batch, heads, seq, seq]."""
    dt, B, C, A_log, dt_bias = _make_test_inputs()
    result = compute_hidden_attention_mamba2(
        dt, B, C, A_log, dt_bias, N_HEADS, N_GROUPS
    )
    assert result.shape == (BATCH, N_HEADS, SEQ_LEN, SEQ_LEN)


def test_causal_mask():
    """Future positions (j > i) should have zero attention."""
    dt, B, C, A_log, dt_bias = _make_test_inputs()
    result = compute_hidden_attention_mamba2(
        dt, B, C, A_log, dt_bias, N_HEADS, N_GROUPS
    )
    # Upper triangle (excluding diagonal) should be zero
    for i in range(SEQ_LEN):
        for j in range(i + 1, SEQ_LEN):
            assert torch.allclose(
                result[0, :, i, j], torch.zeros(N_HEADS)
            ), f"Non-zero at future position [{i},{j}]"


def test_no_nan_inf():
    """Output should not contain NaN or Inf values."""
    dt, B, C, A_log, dt_bias = _make_test_inputs()
    result = compute_hidden_attention_mamba2(
        dt, B, C, A_log, dt_bias, N_HEADS, N_GROUPS
    )
    assert not torch.isnan(result).any(), "NaN values in output"
    assert not torch.isinf(result).any(), "Inf values in output"


def test_memory_efficient_matches():
    """Memory-efficient version should produce same results."""
    dt, B, C, A_log, dt_bias = _make_test_inputs()

    result_standard = compute_hidden_attention_mamba2(
        dt, B, C, A_log, dt_bias, N_HEADS, N_GROUPS
    )
    result_efficient = compute_hidden_attention_mamba2_memory_efficient(
        dt, B, C, A_log, dt_bias, N_HEADS, N_GROUPS, head_batch_size=8
    )

    assert torch.allclose(result_standard, result_efficient, atol=1e-5), (
        f"Max diff: {(result_standard - result_efficient).abs().max()}"
    )


def test_diagonal_nonzero():
    """Diagonal entries (self-attention) should generally be non-zero."""
    # Use positive dt values to ensure non-trivial output
    dt = torch.abs(torch.randn(BATCH, SEQ_LEN, N_HEADS)) + 0.5
    B = torch.randn(BATCH, SEQ_LEN, N_GROUPS * STATE_SIZE)
    C = torch.randn(BATCH, SEQ_LEN, N_GROUPS * STATE_SIZE)
    A_log = torch.log(torch.arange(1, N_HEADS + 1).float())
    dt_bias = torch.ones(N_HEADS)

    result = compute_hidden_attention_mamba2(
        dt, B, C, A_log, dt_bias, N_HEADS, N_GROUPS
    )

    diag = result[0, :, range(SEQ_LEN), range(SEQ_LEN)]  # [heads, seq]
    # At least some diagonal entries should be non-zero
    assert (diag.abs() > 1e-8).any(), "All diagonal entries are zero"


def test_row_sums_reasonable():
    """Row sums should be finite and non-trivial (not all identical)."""
    dt = torch.abs(torch.randn(BATCH, SEQ_LEN, N_HEADS)) + 0.1
    B = torch.randn(BATCH, SEQ_LEN, N_GROUPS * STATE_SIZE)
    C = torch.randn(BATCH, SEQ_LEN, N_GROUPS * STATE_SIZE)
    A_log = torch.log(torch.arange(1, N_HEADS + 1).float())
    dt_bias = torch.ones(N_HEADS)

    result = compute_hidden_attention_mamba2(
        dt, B, C, A_log, dt_bias, N_HEADS, N_GROUPS
    )

    row_sums = result[0].sum(dim=-1)  # [heads, seq]
    assert torch.isfinite(row_sums).all(), "Row sums contain non-finite values"
    # Row sums should vary (not all identical, which would suggest a bug)
    assert row_sums.std() > 1e-6, "All row sums are identical (suspicious)"


def test_longer_sequence():
    """Test with a longer sequence to check for issues with cumsum stability."""
    seq_len = 64
    dt, B, C, A_log, dt_bias = _make_test_inputs(seq_len=seq_len)
    result = compute_hidden_attention_mamba2(
        dt, B, C, A_log, dt_bias, N_HEADS, N_GROUPS
    )
    assert result.shape == (BATCH, N_HEADS, seq_len, seq_len)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()
