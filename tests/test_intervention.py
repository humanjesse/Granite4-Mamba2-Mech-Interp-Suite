"""Tests for causal intervention engine."""

import warnings
import torch
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from src.extraction.causal_intervention import (
    InterventionType,
    SweepResult,
    CausalInterventionEngine,
    MIN_RECOVERY_DENOMINATOR,
)


# ---- Test InterventionType enum ----

def test_intervention_type_values():
    """All intervention types should have string values."""
    assert InterventionType.ACTIVATION_PATCH.value == "activation_patch"
    assert InterventionType.ZERO_ABLATION.value == "zero_ablation"
    assert InterventionType.MEAN_ABLATION.value == "mean_ablation"
    assert InterventionType.NOISE.value == "noise"


def test_intervention_type_from_string():
    """Should be able to construct from string value."""
    assert InterventionType("activation_patch") == InterventionType.ACTIVATION_PATCH
    assert InterventionType("zero_ablation") == InterventionType.ZERO_ABLATION


# ---- Test recovery score computation ----

def test_recovery_score_full_recovery():
    """Patched == clean should give recovery of 1.0."""
    clean_logit_diff = 5.0
    corrupted_logit_diff = -3.0
    patched_logit_diff = 5.0  # fully recovered
    denominator = clean_logit_diff - corrupted_logit_diff

    recovery = (patched_logit_diff - corrupted_logit_diff) / denominator
    assert abs(recovery - 1.0) < 1e-6


def test_recovery_score_no_recovery():
    """Patched == corrupted should give recovery of 0.0."""
    clean_logit_diff = 5.0
    corrupted_logit_diff = -3.0
    patched_logit_diff = -3.0  # no recovery
    denominator = clean_logit_diff - corrupted_logit_diff

    recovery = (patched_logit_diff - corrupted_logit_diff) / denominator
    assert abs(recovery - 0.0) < 1e-6


def test_recovery_score_partial():
    """Partial recovery should be between 0 and 1."""
    clean_logit_diff = 5.0
    corrupted_logit_diff = -3.0
    patched_logit_diff = 1.0
    denominator = clean_logit_diff - corrupted_logit_diff

    recovery = (patched_logit_diff - corrupted_logit_diff) / denominator
    assert 0 < recovery < 1


def test_recovery_score_zero_denominator():
    """When clean == corrupted, recovery should be 0 (not crash)."""
    denominator = 0.0
    if abs(denominator) > MIN_RECOVERY_DENOMINATOR:
        recovery = 1.0 / denominator
    else:
        recovery = 0.0
    assert recovery == 0.0


# ---- Test SweepResult dataclass ----

def test_sweep_result_layer_sweep():
    """SweepResult with 1D recovery matrix (layer sweep)."""
    result = SweepResult(
        clean_tokens=["The", "capital"],
        corrupted_tokens=["The", "capital"],
        correct_token="Paris",
        incorrect_token="Warsaw",
        correct_token_id=1234,
        incorrect_token_id=5678,
        correct_token_display='"Paris" → token 1234 ("Paris")',
        incorrect_token_display='"Warsaw" → token 5678 ("Warsaw")',
        intervention_type=InterventionType.ACTIVATION_PATCH,
        recovery_matrix=torch.tensor([0.0, 0.5, 0.8, 0.1]),
        layer_indices=[0, 1, 2, 3],
        position_indices=[0, 1],
        clean_logit_diff=5.0,
        corrupted_logit_diff=-3.0,
        clean_prob_correct=0.9,
        corrupted_prob_correct=0.1,
    )
    assert result.recovery_matrix.dim() == 1
    assert result.recovery_matrix.shape[0] == 4
    assert result.correct_token == "Paris"


def test_sweep_result_full_sweep():
    """SweepResult with 2D recovery matrix (full sweep)."""
    result = SweepResult(
        clean_tokens=["The", "capital"],
        corrupted_tokens=["The", "capital"],
        correct_token="Paris",
        incorrect_token="Warsaw",
        correct_token_id=1234,
        incorrect_token_id=5678,
        correct_token_display='"Paris" → token 1234',
        incorrect_token_display='"Warsaw" → token 5678',
        intervention_type=InterventionType.ZERO_ABLATION,
        recovery_matrix=torch.randn(4, 2),
        layer_indices=[0, 1, 2, 3],
        position_indices=[0, 1],
        clean_logit_diff=5.0,
        corrupted_logit_diff=-3.0,
        clean_prob_correct=0.9,
        corrupted_prob_correct=0.1,
    )
    assert result.recovery_matrix.dim() == 2
    assert result.recovery_matrix.shape == (4, 2)


# ---- Test intervention hook logic ----

def test_zero_ablation_hook():
    """Zero ablation should set target positions to zero."""
    hidden = torch.randn(1, 5, 8)  # [batch, seq, hidden_dim]
    positions = [1, 3]

    # Simulate the hook logic from CausalInterventionEngine.run_with_intervention
    modified = hidden.clone()
    pos_idx = torch.tensor(positions)
    modified[:, pos_idx, :] = 0.0

    assert torch.allclose(modified[:, 1, :], torch.zeros(1, 8))
    assert torch.allclose(modified[:, 3, :], torch.zeros(1, 8))
    # Non-targeted positions should be unchanged
    assert torch.allclose(modified[:, 0, :], hidden[:, 0, :])
    assert torch.allclose(modified[:, 2, :], hidden[:, 2, :])
    assert torch.allclose(modified[:, 4, :], hidden[:, 4, :])


def test_mean_ablation_hook():
    """Mean ablation should replace target positions with the mean."""
    hidden = torch.randn(1, 5, 8)
    positions = [2]

    modified = hidden.clone()
    mean_val = hidden.mean(dim=1, keepdim=True)
    pos_idx = torch.tensor(positions)
    modified[:, pos_idx, :] = mean_val.expand(-1, len(positions), -1)

    assert torch.allclose(modified[:, 2, :], hidden.mean(dim=1))


def test_activation_patch_hook():
    """Activation patching should swap isolated layer contributions, not full residual."""
    corrupted_output = torch.randn(1, 5, 8)
    clean_contrib = torch.randn(1, 5, 8)
    corrupt_contrib = torch.randn(1, 5, 8)
    positions = [0, 4]

    # Simulate the delta-based hook logic
    modified = corrupted_output.clone()
    idx = torch.tensor(positions)
    modified[:, idx, :] = (
        corrupted_output[:, idx, :]
        - corrupt_contrib[:, idx, :]
        + clean_contrib[:, idx, :]
    )

    # Patched positions should have the delta applied
    expected = corrupted_output[:, idx, :] - corrupt_contrib[:, idx, :] + clean_contrib[:, idx, :]
    assert torch.allclose(modified[:, 0, :], expected[:, 0, :])
    assert torch.allclose(modified[:, 4, :], expected[:, 1, :])
    # Non-patched positions should retain corrupted values
    assert torch.allclose(modified[:, 1, :], corrupted_output[:, 1, :])
    assert torch.allclose(modified[:, 2, :], corrupted_output[:, 2, :])
    assert torch.allclose(modified[:, 3, :], corrupted_output[:, 3, :])


def test_noise_hook():
    """Noise injection should modify but not zero out activations."""
    hidden = torch.randn(1, 5, 8)
    positions = [1]

    modified = hidden.clone()
    pos_idx = torch.tensor(positions)
    noise = torch.randn_like(modified[:, pos_idx, :]) * 1.0
    modified[:, pos_idx, :] = modified[:, pos_idx, :] + noise

    # Position 1 should be different from original
    assert not torch.allclose(modified[:, 1, :], hidden[:, 1, :])
    # Other positions should be unchanged
    assert torch.allclose(modified[:, 0, :], hidden[:, 0, :])


# ---- Mock engine fixture ----

class _MockTokenizerOutput(dict):
    """Dict subclass that supports .to(device) like HuggingFace tokenizer output."""

    def to(self, device):
        return self


def _make_mock_model_output(seq_len=3, vocab_size=100):
    """Create a mock model output with logits."""
    logits = torch.randn(1, seq_len, vocab_size)
    output = MagicMock()
    output.logits = logits
    return output


@pytest.fixture
def mock_engine():
    """Create a CausalInterventionEngine with a mock model."""
    model = MagicMock()
    num_layers = 4
    layers = []
    for _ in range(num_layers):
        layer = MagicMock()
        layer._forward_hooks = {}

        def make_register(layer_ref):
            def register_forward_hook(fn, **kwargs):
                handle = MagicMock()
                handle.remove = MagicMock()
                return handle
            return register_forward_hook

        layer.register_forward_hook = make_register(layer)
        layers.append(layer)
    model.model.layers = layers

    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1234]
    tokenizer.decode.return_value = "Paris"
    tokenizer.convert_ids_to_tokens.return_value = ["The", "capital", "of"]
    tokenizer_output = _MockTokenizerOutput(input_ids=torch.tensor([[1, 2, 3]]))
    tokenizer.return_value = tokenizer_output

    arch_map = MagicMock()
    arch_map.num_layers = num_layers

    engine = CausalInterventionEngine(model, tokenizer, arch_map, device="cpu")
    return engine


# ---- Test resolve_token ----

def test_resolve_token_single_token(mock_engine):
    """Single-token input should resolve to that token ID."""
    mock_engine.tokenizer.encode.return_value = [42]
    mock_engine.tokenizer.decode.return_value = "Paris"
    token_id, display = mock_engine.resolve_token("Paris")
    assert token_id == 42
    assert "42" in display
    assert "Paris" in display


def test_resolve_token_multi_token_warns(mock_engine):
    """Multi-token input should trigger a warning and use the last token."""
    mock_engine.tokenizer.encode.return_value = [10, 20, 30]
    mock_engine.tokenizer.decode.return_value = "ris"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        token_id, display = mock_engine.resolve_token("Paris")
        assert token_id == 30
        assert len(w) == 1
        assert "3 tokens" in str(w[0].message)


def test_resolve_token_empty_raises(mock_engine):
    """Empty tokenization should raise ValueError."""
    mock_engine.tokenizer.encode.return_value = []
    with pytest.raises(ValueError, match="no tokens produced"):
        mock_engine.resolve_token("")


def test_resolve_token_display_format(mock_engine):
    """Display string should include token ID and decoded text."""
    mock_engine.tokenizer.encode.return_value = [99]
    mock_engine.tokenizer.decode.return_value = "hello"
    _, display = mock_engine.resolve_token("hello")
    assert "99" in display
    assert "hello" in display
    assert "→" in display


# ---- Test hook lifecycle ----

def test_run_and_cache_hook_cleanup(mock_engine):
    """All hooks (embedding + layers) should be removed after run_and_cache completes."""
    mock_output = _make_mock_model_output()
    mock_engine.model.return_value = mock_output

    embed_handle = MagicMock()
    embed_handle.remove = MagicMock()
    mock_engine.model.model.embed_tokens.register_forward_hook = MagicMock(return_value=embed_handle)

    layer_handles = []
    for layer in mock_engine.model.model.layers:
        handle = MagicMock()
        handle.remove = MagicMock()
        layer.register_forward_hook = MagicMock(return_value=handle)
        layer_handles.append(handle)

    mock_engine.run_and_cache("test prompt")

    embed_handle.remove.assert_called_once()
    for handle in layer_handles:
        handle.remove.assert_called_once()


def test_run_with_intervention_hook_cleanup(mock_engine):
    """Hook should be removed after run_with_intervention completes."""
    mock_output = _make_mock_model_output()
    mock_engine.model.return_value = mock_output

    handle = MagicMock()
    handle.remove = MagicMock()
    mock_engine.model.model.layers[0].register_forward_hook = MagicMock(return_value=handle)

    mock_engine.run_with_intervention(
        prompt="test",
        intervention_type=InterventionType.ZERO_ABLATION,
        target_layer=0,
        positions=[0],
    )

    handle.remove.assert_called_once()


def test_hook_cleanup_on_exception(mock_engine):
    """Hooks should be removed even if the forward pass raises."""
    mock_engine.model.side_effect = RuntimeError("forward pass failed")

    embed_handle = MagicMock()
    embed_handle.remove = MagicMock()
    mock_engine.model.model.embed_tokens.register_forward_hook = MagicMock(return_value=embed_handle)

    layer_handles = []
    for layer in mock_engine.model.model.layers:
        handle = MagicMock()
        handle.remove = MagicMock()
        layer.register_forward_hook = MagicMock(return_value=handle)
        layer_handles.append(handle)

    with pytest.raises(RuntimeError, match="forward pass failed"):
        mock_engine.run_and_cache("test prompt")

    embed_handle.remove.assert_called_once()
    for handle in layer_handles:
        handle.remove.assert_called_once()


# ---- Test caching ----

def test_cache_hit_returns_same_result(mock_engine):
    """Calling run_and_cache twice with the same prompt returns the same object."""
    mock_output = _make_mock_model_output()
    mock_engine.model.return_value = mock_output

    result1 = mock_engine.run_and_cache("test prompt")
    result2 = mock_engine.run_and_cache("test prompt")
    assert result1 is result2
    # Model should only be called once
    assert mock_engine.model.call_count == 1


def test_cache_miss_different_prompt(mock_engine):
    """Different prompts should produce different cache entries."""
    mock_output = _make_mock_model_output()
    mock_engine.model.return_value = mock_output

    result1 = mock_engine.run_and_cache("prompt A")
    result2 = mock_engine.run_and_cache("prompt B")
    assert result1 is not result2
    assert mock_engine.model.call_count == 2


def test_clear_cache(mock_engine):
    """clear_cache should empty the internal cache."""
    mock_output = _make_mock_model_output()
    mock_engine.model.return_value = mock_output

    mock_engine.run_and_cache("test prompt")
    assert len(mock_engine._prompt_cache) == 1
    mock_engine.clear_cache()
    assert len(mock_engine._prompt_cache) == 0


# ---- Test position handling ----

def test_out_of_bounds_positions_ignored():
    """Positions beyond seq_len should be filtered out."""
    hidden = torch.randn(1, 5, 8)
    positions = [1, 3, 10, 20]  # 10 and 20 are out of bounds

    valid_pos = [p for p in positions if p < hidden.shape[1]]
    assert valid_pos == [1, 3]


def test_empty_positions_returns_unmodified():
    """When all positions are out of bounds, output should be unmodified."""
    hidden = torch.randn(1, 5, 8)
    positions = [10, 20]  # all out of bounds

    valid_pos = [p for p in positions if p < hidden.shape[1]]
    assert valid_pos == []
    # The hook returns output unmodified when valid_pos is empty


def test_mismatched_seq_len_patches_intersection():
    """When clean and corrupted have different lengths, only shared positions are patched."""
    clean_hidden = torch.randn(1, 8, 16)   # 8 tokens
    corrupted_hidden = torch.randn(1, 10, 16)  # 10 tokens
    positions = [0, 1, 5, 9]

    valid_pos = [p for p in positions if p < corrupted_hidden.shape[1]]
    patch_pos = [p for p in valid_pos if p < clean_hidden.shape[1]]
    assert patch_pos == [0, 1, 5]  # 9 is beyond clean's seq_len


# ---- Test numerical edge cases ----

def test_bfloat16_logit_diff_precision():
    """Float32 cast should preserve small differences lost in bfloat16."""
    # Create values where bfloat16 subtraction loses precision
    a = torch.tensor(100.0, dtype=torch.bfloat16)
    b = torch.tensor(100.001, dtype=torch.bfloat16)

    # In bfloat16, the difference may be zero due to limited precision
    bf16_diff = (a - b).item()

    # After casting to float32, we get better precision
    f32_diff = (a.float() - b.float()).item()

    # The float32 path should give at least as good precision
    # (bfloat16 has ~3 decimal digits of precision, so small diffs vanish)
    assert abs(f32_diff) >= abs(bf16_diff)


def test_recovery_score_clamped_on_small_denominator():
    """When denominator is near the threshold, recovery should be 0."""
    denominator = MIN_RECOVERY_DENOMINATOR / 2  # below threshold
    if abs(denominator) > MIN_RECOVERY_DENOMINATOR:
        recovery = 1.0 / denominator
    else:
        recovery = 0.0
    assert recovery == 0.0


def test_recovery_no_nan_inf():
    """Recovery scores computed with the threshold should never be NaN or Inf."""
    test_cases = [
        (5.0, -3.0, 1.0),   # normal case
        (0.0, 0.0, 0.0),    # zero denominator
        (1e-7, 0.0, 0.5),   # tiny denominator
        (100.0, -100.0, 50.0),  # large values
    ]
    for clean_diff, corrupt_diff, patched_diff in test_cases:
        denom = clean_diff - corrupt_diff
        if abs(denom) > MIN_RECOVERY_DENOMINATOR:
            recovery = (patched_diff - corrupt_diff) / denom
        else:
            recovery = 0.0
        assert not (recovery != recovery), f"NaN for inputs {clean_diff}, {corrupt_diff}, {patched_diff}"  # NaN check
        assert abs(recovery) != float('inf'), f"Inf for inputs {clean_diff}, {corrupt_diff}, {patched_diff}"


# ---- Integration-style tests with mock model ----

def _setup_sweep_mocks(mock_engine, vocab_size=2000):
    """Set up mocks for sweep tests with consistent token IDs and vocab size."""
    mock_output = _make_mock_model_output(seq_len=3, vocab_size=vocab_size)
    mock_engine.model.return_value = mock_output
    # correct_token -> 10, incorrect_token -> 20 (within vocab_size)
    mock_engine.tokenizer.encode.side_effect = lambda text, **kw: {
        "Paris": [10], "Warsaw": [20],
    }.get(text.strip(), [10])
    mock_engine.tokenizer.decode.side_effect = lambda tid: {10: "Paris", 20: "Warsaw"}.get(tid, "?")


def _make_fake_contributions(num_layers, seq_len=3, hidden_dim=32):
    """Create fake per-layer contribution dicts for sweep tests."""
    return {i: torch.randn(1, seq_len, hidden_dim) for i in range(num_layers)}


def test_sweep_layers_calls_correct_number_of_forward_passes(mock_engine):
    """sweep_layers should call run_with_intervention once per layer."""
    _setup_sweep_mocks(mock_engine)
    num_layers = mock_engine.arch_map.num_layers

    with patch.object(mock_engine, 'run_with_intervention') as mock_rwi, \
         patch.object(mock_engine, '_compute_layer_output') as mock_clo:
        mock_rwi.return_value = torch.randn(1, 3, 2000)
        mock_clo.return_value = _make_fake_contributions(num_layers)
        result = mock_engine.sweep_layers(
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Poland is",
            correct_token="Paris",
            incorrect_token="Warsaw",
        )
        assert mock_rwi.call_count == num_layers


def test_sweep_layers_progress_callback(mock_engine):
    """Progress callback should be called num_layers times."""
    _setup_sweep_mocks(mock_engine)
    num_layers = mock_engine.arch_map.num_layers

    callback_calls = []

    with patch.object(mock_engine, 'run_with_intervention') as mock_rwi, \
         patch.object(mock_engine, '_compute_layer_output') as mock_clo:
        mock_rwi.return_value = torch.randn(1, 3, 2000)
        mock_clo.return_value = _make_fake_contributions(num_layers)
        mock_engine.sweep_layers(
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Poland is",
            correct_token="Paris",
            incorrect_token="Warsaw",
            progress_callback=lambda step, total: callback_calls.append((step, total)),
        )

    num_layers = mock_engine.arch_map.num_layers
    assert len(callback_calls) == num_layers
    assert callback_calls[-1] == (num_layers, num_layers)


def test_sweep_result_shapes(mock_engine):
    """Layer sweep should return a 1D recovery matrix of shape [num_layers]."""
    _setup_sweep_mocks(mock_engine)
    num_layers = mock_engine.arch_map.num_layers

    with patch.object(mock_engine, 'run_with_intervention') as mock_rwi, \
         patch.object(mock_engine, '_compute_layer_output') as mock_clo:
        mock_rwi.return_value = torch.randn(1, 3, 2000)
        mock_clo.return_value = _make_fake_contributions(num_layers)
        result = mock_engine.sweep_layers(
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Poland is",
            correct_token="Paris",
            incorrect_token="Warsaw",
        )

    assert result.recovery_matrix.dim() == 1
    assert result.recovery_matrix.shape[0] == mock_engine.arch_map.num_layers
    assert len(result.layer_indices) == mock_engine.arch_map.num_layers


# ---- sweep_positions_and_layers tests ----

def test_full_sweep_result_shape(mock_engine):
    """sweep_positions_and_layers should return a 2D [num_layers, seq_len] matrix."""
    _setup_sweep_mocks(mock_engine)
    num_layers = mock_engine.arch_map.num_layers

    with patch.object(mock_engine, 'run_with_intervention') as mock_rwi, \
         patch.object(mock_engine, '_compute_layer_output') as mock_clo:
        mock_rwi.return_value = torch.randn(1, 3, 2000)
        mock_clo.return_value = _make_fake_contributions(num_layers)
        result = mock_engine.sweep_positions_and_layers(
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Poland is",
            correct_token="Paris",
            incorrect_token="Warsaw",
        )

    num_layers = mock_engine.arch_map.num_layers
    seq_len = 3  # from convert_ids_to_tokens mock
    assert result.recovery_matrix.dim() == 2
    assert result.recovery_matrix.shape == (num_layers, seq_len)
    assert len(result.layer_indices) == num_layers
    assert len(result.position_indices) == seq_len


def test_full_sweep_forward_pass_count(mock_engine):
    """sweep_positions_and_layers should call run_with_intervention num_layers * seq_len times."""
    _setup_sweep_mocks(mock_engine)
    num_layers = mock_engine.arch_map.num_layers

    with patch.object(mock_engine, 'run_with_intervention') as mock_rwi, \
         patch.object(mock_engine, '_compute_layer_output') as mock_clo:
        mock_rwi.return_value = torch.randn(1, 3, 2000)
        mock_clo.return_value = _make_fake_contributions(num_layers)
        mock_engine.sweep_positions_and_layers(
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Poland is",
            correct_token="Paris",
            incorrect_token="Warsaw",
        )
        seq_len = 3
        assert mock_rwi.call_count == num_layers * seq_len


def test_full_sweep_progress_callback(mock_engine):
    """Progress callback should be called num_layers * seq_len times with correct totals."""
    _setup_sweep_mocks(mock_engine)
    num_layers = mock_engine.arch_map.num_layers
    callback_calls = []

    with patch.object(mock_engine, 'run_with_intervention') as mock_rwi, \
         patch.object(mock_engine, '_compute_layer_output') as mock_clo:
        mock_rwi.return_value = torch.randn(1, 3, 2000)
        mock_clo.return_value = _make_fake_contributions(num_layers)
        mock_engine.sweep_positions_and_layers(
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Poland is",
            correct_token="Paris",
            incorrect_token="Warsaw",
            progress_callback=lambda step, total: callback_calls.append((step, total)),
        )

    total = mock_engine.arch_map.num_layers * 3
    assert len(callback_calls) == total
    assert callback_calls[-1] == (total, total)
    # Steps should be monotonically increasing
    steps = [s for s, _ in callback_calls]
    assert steps == list(range(1, total + 1))


def test_full_sweep_layer_subset(mock_engine):
    """layer_subset should limit which layers are swept."""
    _setup_sweep_mocks(mock_engine)
    num_layers = mock_engine.arch_map.num_layers

    with patch.object(mock_engine, 'run_with_intervention') as mock_rwi, \
         patch.object(mock_engine, '_compute_layer_output') as mock_clo:
        mock_rwi.return_value = torch.randn(1, 3, 2000)
        mock_clo.return_value = _make_fake_contributions(num_layers)
        result = mock_engine.sweep_positions_and_layers(
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Poland is",
            correct_token="Paris",
            incorrect_token="Warsaw",
            layer_subset=[0, 2],
        )

    assert result.recovery_matrix.shape == (2, 3)  # 2 layers, 3 positions
    assert result.layer_indices == [0, 2]
    assert mock_rwi.call_count == 2 * 3


def test_full_sweep_each_call_patches_single_position(mock_engine):
    """Each run_with_intervention call should patch exactly one position."""
    _setup_sweep_mocks(mock_engine)
    num_layers = mock_engine.arch_map.num_layers

    with patch.object(mock_engine, 'run_with_intervention') as mock_rwi, \
         patch.object(mock_engine, '_compute_layer_output') as mock_clo:
        mock_rwi.return_value = torch.randn(1, 3, 2000)
        mock_clo.return_value = _make_fake_contributions(num_layers)
        mock_engine.sweep_positions_and_layers(
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Poland is",
            correct_token="Paris",
            incorrect_token="Warsaw",
        )

        for call in mock_rwi.call_args_list:
            positions = call.kwargs.get("positions") or call[1].get("positions")
            # positions is a keyword arg
            if positions is None:
                pytest.fail("positions should not be None in full sweep — expected [pos]")
            else:
                assert len(positions) == 1, f"Expected single position, got {positions}"


def test_full_sweep_no_nan_inf(mock_engine):
    """Full sweep recovery scores should not contain NaN or Inf."""
    _setup_sweep_mocks(mock_engine)
    num_layers = mock_engine.arch_map.num_layers

    with patch.object(mock_engine, 'run_with_intervention') as mock_rwi, \
         patch.object(mock_engine, '_compute_layer_output') as mock_clo:
        mock_rwi.return_value = torch.randn(1, 3, 2000)
        mock_clo.return_value = _make_fake_contributions(num_layers)
        result = mock_engine.sweep_positions_and_layers(
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Poland is",
            correct_token="Paris",
            incorrect_token="Warsaw",
        )

    assert not torch.isnan(result.recovery_matrix).any(), "Recovery matrix contains NaN"
    assert not torch.isinf(result.recovery_matrix).any(), "Recovery matrix contains Inf"


def test_full_sweep_populates_sweep_result_fields(mock_engine):
    """All SweepResult fields should be populated correctly."""
    _setup_sweep_mocks(mock_engine)

    with patch.object(mock_engine, 'run_with_intervention') as mock_rwi:
        mock_rwi.return_value = torch.randn(1, 3, 2000)
        result = mock_engine.sweep_positions_and_layers(
            clean_prompt="The capital of France is",
            corrupted_prompt="The capital of Poland is",
            correct_token="Paris",
            incorrect_token="Warsaw",
            intervention_type=InterventionType.ZERO_ABLATION,
        )

    assert result.correct_token == "Paris"
    assert result.incorrect_token == "Warsaw"
    assert result.correct_token_id == 10
    assert result.incorrect_token_id == 20
    assert result.intervention_type == InterventionType.ZERO_ABLATION
    assert isinstance(result.clean_logit_diff, float)
    assert isinstance(result.corrupted_logit_diff, float)
    assert isinstance(result.clean_prob_correct, float)
    assert isinstance(result.corrupted_prob_correct, float)
    assert len(result.clean_tokens) == 3
    assert len(result.corrupted_tokens) == 3


# ---- Test residual_stream key types ----

def test_residual_stream_has_only_integer_keys(mock_engine):
    """residual_stream should contain only integer keys; embedding is stored separately."""
    mock_output = _make_mock_model_output()
    mock_engine.model.return_value = mock_output

    result = mock_engine.run_and_cache("test prompt")

    for key in result["residual_stream"]:
        assert isinstance(key, int), f"Expected int key, got {type(key).__name__}: {key}"
    # "embedding" must exist as a top-level key, not inside residual_stream
    assert "embedding" in result
    assert "embedding" not in result["residual_stream"]
