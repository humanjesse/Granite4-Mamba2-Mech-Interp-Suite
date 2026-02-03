"""Tests for circuit discovery engine."""

import warnings
import torch
import pytest
from unittest.mock import MagicMock, patch, call

from src.extraction.circuit_discovery import (
    SweepGranularity,
    PathPatchResult,
    ComponentResult,
    CircuitNode,
    CircuitEdge,
    CircuitResult,
    CircuitDiscoveryEngine,
)
from src.extraction.intervention_base import MIN_RECOVERY_DENOMINATOR


# ---- Test dataclasses ----

def test_sweep_granularity_values():
    assert SweepGranularity.FAST.value == "fast"
    assert SweepGranularity.DETAILED.value == "detailed"


def test_path_patch_result():
    r = PathPatchResult(source_layer=0, target_layer=5, recovery_score=0.42,
                        source_type="mamba", target_type="attention")
    assert r.source_layer == 0
    assert r.target_layer == 5
    assert r.recovery_score == 0.42


def test_component_result():
    c = ComponentResult(layer_idx=10, component_idx=3, component_type="attention_head",
                        recovery_score=0.75)
    assert c.component_type == "attention_head"
    assert c.recovery_score == 0.75


def test_circuit_node_default_components():
    node = CircuitNode(layer_idx=5, layer_type="mamba", importance=0.3)
    assert node.components == []


def test_circuit_edge():
    edge = CircuitEdge(source_layer=2, target_layer=10, importance=0.6,
                       source_type="mamba", target_type="attention")
    assert edge.source_layer == 2
    assert edge.target_layer == 10


def test_circuit_result_construction():
    result = CircuitResult(
        clean_tokens=["The", "capital"],
        corrupted_tokens=["The", "capital"],
        correct_token="Paris",
        incorrect_token="Warsaw",
        correct_token_id=10,
        incorrect_token_id=20,
        correct_token_display='"Paris" → token 10',
        incorrect_token_display='"Warsaw" → token 20',
        path_matrix=torch.zeros(4, 4),
        layer_indices=[0, 1, 2, 3],
        layer_importance=torch.zeros(4),
        component_results=[],
        circuit_nodes=[],
        circuit_edges=[],
        clean_logit_diff=5.0,
        corrupted_logit_diff=-3.0,
        clean_prob_correct=0.9,
        corrupted_prob_correct=0.1,
        threshold=0.1,
        granularity=SweepGranularity.FAST,
    )
    assert result.path_matrix.shape == (4, 4)
    assert result.granularity == SweepGranularity.FAST


# ---- Test constructor validation ----

def test_constructor_rejects_none_model():
    with pytest.raises(ValueError, match="model cannot be None"):
        CircuitDiscoveryEngine(None, MagicMock(), MagicMock())


def test_constructor_rejects_none_tokenizer():
    with pytest.raises(ValueError, match="tokenizer cannot be None"):
        CircuitDiscoveryEngine(MagicMock(), None, MagicMock())


def test_constructor_rejects_none_arch_map():
    with pytest.raises(ValueError, match="arch_map cannot be None"):
        CircuitDiscoveryEngine(MagicMock(), MagicMock(), None)


# ---- Mock engine fixture ----

class _MockTokenizerOutput(dict):
    def to(self, device):
        return self


def _make_mock_model_output(seq_len=3, vocab_size=100):
    logits = torch.randn(1, seq_len, vocab_size)
    output = MagicMock()
    output.logits = logits
    return output


@pytest.fixture
def mock_engine():
    """Create a CircuitDiscoveryEngine with a mock model."""
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

        def make_pre_register(layer_ref):
            def register_forward_pre_hook(fn, **kwargs):
                handle = MagicMock()
                handle.remove = MagicMock()
                return handle
            return register_forward_pre_hook

        layer.register_forward_hook = make_register(layer)
        layer.register_forward_pre_hook = make_pre_register(layer)
        layers.append(layer)
    model.model.layers = layers
    model.model.embed_tokens = MagicMock()
    model.model.embed_tokens.register_forward_hook = lambda fn, **kw: MagicMock(remove=MagicMock())

    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1234]
    tokenizer.decode.return_value = "Paris"
    tokenizer.convert_ids_to_tokens.return_value = ["The", "capital", "of"]
    tokenizer_output = _MockTokenizerOutput(input_ids=torch.tensor([[1, 2, 3]]))
    tokenizer.return_value = tokenizer_output

    arch_map = MagicMock()
    arch_map.num_layers = num_layers
    arch_map.layer_type.side_effect = lambda idx: "mamba" if idx != 2 else "attention"
    arch_map.is_attention.side_effect = lambda idx: idx == 2
    arch_map.attention_indices = [2]
    arch_map.mamba_indices = [0, 1, 3]
    model.config.num_attention_heads = 4
    model.config.hidden_size = 32

    engine = CircuitDiscoveryEngine(model, tokenizer, arch_map, device="cpu")
    return engine


# ---- Test resolve_token (inherited from base) ----

def test_resolve_token_works_via_inheritance(mock_engine):
    mock_engine.tokenizer.encode.return_value = [42]
    mock_engine.tokenizer.decode.return_value = "Paris"
    token_id, display = mock_engine.resolve_token("Paris")
    assert token_id == 42
    assert "Paris" in display


# ---- Test _extract_circuit ----

def test_extract_circuit_empty_below_threshold(mock_engine):
    """All values below threshold should produce empty circuit."""
    path_matrix = torch.ones(4, 4) * 0.01  # all below 0.1 threshold
    layer_importance = torch.ones(4) * 0.01
    nodes, edges = mock_engine._extract_circuit(path_matrix, layer_importance, [], 0.1)
    assert nodes == []
    assert edges == []


def test_extract_circuit_finds_important_nodes(mock_engine):
    """Nodes with |importance| >= threshold should be included."""
    path_matrix = torch.zeros(4, 4)
    layer_importance = torch.tensor([0.0, 0.5, 0.0, 0.3])
    nodes, edges = mock_engine._extract_circuit(path_matrix, layer_importance, [], 0.1)
    assert len(nodes) == 2
    assert nodes[0].layer_idx == 1
    assert nodes[1].layer_idx == 3


def test_extract_circuit_finds_important_edges(mock_engine):
    """Paths with |recovery| >= threshold should become edges."""
    path_matrix = torch.zeros(4, 4)
    path_matrix[0, 3] = 0.5  # source=0, target=3
    layer_importance = torch.zeros(4)
    nodes, edges = mock_engine._extract_circuit(path_matrix, layer_importance, [], 0.1)
    assert len(edges) == 1
    assert edges[0].source_layer == 0
    assert edges[0].target_layer == 3
    # Both endpoints should be added as nodes even if layer_importance is 0
    assert len(nodes) == 2


def test_extract_circuit_includes_components(mock_engine):
    """Nodes should include matching component results."""
    path_matrix = torch.zeros(4, 4)
    layer_importance = torch.tensor([0.0, 0.0, 0.5, 0.0])  # layer 2 is important
    comp = ComponentResult(layer_idx=2, component_idx=1, component_type="attention_head",
                           recovery_score=0.3)
    nodes, edges = mock_engine._extract_circuit(path_matrix, layer_importance, [comp], 0.1)
    assert len(nodes) == 1
    assert len(nodes[0].components) == 1
    assert nodes[0].components[0].component_idx == 1


# ---- Test path_patch returns 0 for invalid pairs ----

def test_path_patch_source_ge_target_returns_zero(mock_engine):
    """path_patch should return 0.0 when source >= target."""
    score = mock_engine.path_patch(
        "clean", "corrupt", source_layer=3, target_layer=1,
        correct_id=10, incorrect_id=20,
        clean_contributions={}, corrupted_contributions={},
        corrupted_logit_diff=0.0, denominator=1.0,
    )
    assert score == 0.0


def test_path_patch_equal_layers_returns_zero(mock_engine):
    """path_patch should return 0.0 when source == target."""
    score = mock_engine.path_patch(
        "clean", "corrupt", source_layer=2, target_layer=2,
        correct_id=10, incorrect_id=20,
        clean_contributions={}, corrupted_contributions={},
        corrupted_logit_diff=0.0, denominator=1.0,
    )
    assert score == 0.0


# ---- Test patch_attention_head returns 0 for Mamba layers ----

def test_patch_attention_head_returns_zero_for_mamba(mock_engine):
    """Patching an attention head on a Mamba layer should return 0.0."""
    score = mock_engine.patch_attention_head(
        "clean", "corrupt", layer_idx=0, head_idx=0,
        correct_id=10, incorrect_id=20,
        corrupted_logit_diff=0.0, denominator=1.0,
    )
    assert score == 0.0


# ---- Test hook cleanup ----

def test_run_and_cache_hook_cleanup(mock_engine):
    """All hooks (embedding + layers) should be removed after run_and_cache."""
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


def test_hook_cleanup_on_exception(mock_engine):
    """Hooks should be removed even if forward pass raises."""
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


# ---- Test caching (inherited) ----

def test_cache_hit(mock_engine):
    mock_output = _make_mock_model_output()
    mock_engine.model.return_value = mock_output

    r1 = mock_engine.run_and_cache("test")
    r2 = mock_engine.run_and_cache("test")
    assert r1 is r2
    assert mock_engine.model.call_count == 1


def test_clear_cache(mock_engine):
    mock_output = _make_mock_model_output()
    mock_engine.model.return_value = mock_output

    mock_engine.run_and_cache("test")
    assert len(mock_engine._prompt_cache) == 1
    mock_engine.clear_cache()
    assert len(mock_engine._prompt_cache) == 0


# ---- Helper to set up sweep mocks ----

def _setup_sweep_mocks(engine, vocab_size=2000):
    """Set up mocks for sweep tests with consistent token IDs and vocab size."""
    mock_output = _make_mock_model_output(seq_len=3, vocab_size=vocab_size)
    engine.model.return_value = mock_output
    engine.tokenizer.encode.side_effect = lambda text, **kw: {
        "Paris": [10], "Warsaw": [20],
    }.get(text.strip(), [10])
    engine.tokenizer.decode.side_effect = lambda tid: {10: "Paris", 20: "Warsaw"}.get(tid, "?")


# ---- Test _run_with_layer_patch ----

def test_run_with_layer_patch_returns_logits(mock_engine):
    """_run_with_layer_patch should return logits tensor."""
    mock_output = _make_mock_model_output(seq_len=3, vocab_size=100)
    mock_engine.model.return_value = mock_output

    clean_contributions = {0: torch.randn(1, 3, 32)}
    corrupted_contributions = {0: torch.randn(1, 3, 32)}
    result = mock_engine._run_with_layer_patch(
        "test prompt", 0, clean_contributions, corrupted_contributions
    )
    assert result.shape[0] == 1  # batch
    assert result.shape[2] == 100  # vocab size


def test_run_with_layer_patch_hook_cleanup(mock_engine):
    """Hook should be removed after _run_with_layer_patch."""
    mock_output = _make_mock_model_output(seq_len=3, vocab_size=100)
    mock_engine.model.return_value = mock_output

    handle = MagicMock()
    handle.remove = MagicMock()
    mock_engine.model.model.layers[1].register_forward_hook = MagicMock(return_value=handle)

    clean_contributions = {1: torch.randn(1, 3, 32)}
    corrupted_contributions = {1: torch.randn(1, 3, 32)}
    mock_engine._run_with_layer_patch("test", 1, clean_contributions, corrupted_contributions)
    handle.remove.assert_called_once()


def test_run_with_layer_patch_hook_cleanup_on_exception(mock_engine):
    """Hook should be removed even if forward pass fails."""
    mock_engine.model.side_effect = RuntimeError("boom")

    handle = MagicMock()
    handle.remove = MagicMock()
    mock_engine.model.model.layers[0].register_forward_hook = MagicMock(return_value=handle)

    clean_contributions = {0: torch.randn(1, 3, 32)}
    corrupted_contributions = {0: torch.randn(1, 3, 32)}
    with pytest.raises(RuntimeError, match="boom"):
        mock_engine._run_with_layer_patch("test", 0, clean_contributions, corrupted_contributions)
    handle.remove.assert_called_once()


# ---- Test sweep_paths ----

def test_sweep_paths_returns_correct_shapes(mock_engine):
    """sweep_paths should return path_matrix[n,n], layer_importance[n], and metadata."""
    _setup_sweep_mocks(mock_engine)
    num_layers = mock_engine.arch_map.num_layers

    with patch.object(mock_engine, '_run_with_layer_patch') as mock_layer_patch, \
         patch.object(mock_engine, 'path_patch') as mock_path_patch, \
         patch.object(mock_engine, '_compute_layer_output') as mock_contrib:
        mock_layer_patch.return_value = torch.randn(1, 3, 2000)
        mock_path_patch.return_value = 0.5
        mock_contrib.return_value = {i: torch.randn(1, 3, 32) for i in range(num_layers)}

        path_matrix, layer_importance, metadata = mock_engine.sweep_paths(
            "The capital of France is", "The capital of Poland is",
            "Paris", "Warsaw",
        )

    assert path_matrix.shape == (num_layers, num_layers)
    assert layer_importance.shape == (num_layers,)
    assert "correct_id" in metadata
    assert "clean_tokens" in metadata


def test_sweep_paths_forward_pass_counts(mock_engine):
    """sweep_paths should call _run_with_layer_patch n times and path_patch n*(n-1)/2 times."""
    _setup_sweep_mocks(mock_engine)
    num_layers = mock_engine.arch_map.num_layers  # 4

    with patch.object(mock_engine, '_run_with_layer_patch') as mock_layer_patch, \
         patch.object(mock_engine, 'path_patch') as mock_path_patch, \
         patch.object(mock_engine, '_compute_layer_output') as mock_contrib:
        mock_layer_patch.return_value = torch.randn(1, 3, 2000)
        mock_path_patch.return_value = 0.0
        mock_contrib.return_value = {i: torch.randn(1, 3, 32) for i in range(num_layers)}

        mock_engine.sweep_paths(
            "The capital of France is", "The capital of Poland is",
            "Paris", "Warsaw",
        )

    # Layer sweep: n calls
    assert mock_layer_patch.call_count == num_layers
    # Path sweep: n*(n-1)/2 = 4*3/2 = 6 pairs
    expected_pairs = num_layers * (num_layers - 1) // 2
    assert mock_path_patch.call_count == expected_pairs


def test_sweep_paths_only_fills_upper_triangle(mock_engine):
    """Path matrix should only have values where source < target."""
    _setup_sweep_mocks(mock_engine)
    num_layers = mock_engine.arch_map.num_layers

    with patch.object(mock_engine, '_run_with_layer_patch') as mock_layer_patch, \
         patch.object(mock_engine, 'path_patch') as mock_path_patch, \
         patch.object(mock_engine, '_compute_layer_output') as mock_contrib:
        mock_layer_patch.return_value = torch.randn(1, 3, 2000)
        mock_path_patch.return_value = 0.42
        mock_contrib.return_value = {i: torch.randn(1, 3, 32) for i in range(num_layers)}

        path_matrix, _, _ = mock_engine.sweep_paths(
            "The capital of France is", "The capital of Poland is",
            "Paris", "Warsaw",
        )

    # Lower triangle and diagonal should be zero
    for i in range(num_layers):
        for j in range(i + 1):
            assert path_matrix[i, j].item() == 0.0, f"path_matrix[{i},{j}] should be 0"
    # Upper triangle should have values
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            assert path_matrix[i, j].item() != 0.0, f"path_matrix[{i},{j}] should be non-zero"


def test_sweep_paths_progress_callback(mock_engine):
    """Progress callback should be called for each path patch step."""
    _setup_sweep_mocks(mock_engine)
    num_layers = mock_engine.arch_map.num_layers
    callback_calls = []

    with patch.object(mock_engine, '_run_with_layer_patch') as mock_layer_patch, \
         patch.object(mock_engine, 'path_patch') as mock_path_patch, \
         patch.object(mock_engine, '_compute_layer_output') as mock_contrib:
        mock_layer_patch.return_value = torch.randn(1, 3, 2000)
        mock_path_patch.return_value = 0.0
        mock_contrib.return_value = {i: torch.randn(1, 3, 32) for i in range(num_layers)}

        mock_engine.sweep_paths(
            "The capital of France is", "The capital of Poland is",
            "Paris", "Warsaw",
            progress_callback=lambda step, total: callback_calls.append((step, total)),
        )

    # Should have been called for each path pair
    expected_pairs = num_layers * (num_layers - 1) // 2
    assert len(callback_calls) == expected_pairs
    # Total should be num_layers + pairs (layer sweep + path sweep)
    expected_total = num_layers + expected_pairs
    assert callback_calls[-1][1] == expected_total


# ---- Test sweep_components ----

def test_sweep_components_only_sweeps_transformer_layers(mock_engine):
    """sweep_components should only call patch_attention_head on Transformer layers."""
    with patch.object(mock_engine, '_capture_clean_attn_output', return_value=torch.randn(1, 3, 32)), \
         patch.object(mock_engine, 'patch_attention_head') as mock_pah:
        mock_pah.return_value = 0.3

        results = mock_engine.sweep_components(
            "clean", "corrupt",
            correct_id=10, incorrect_id=20,
            corrupted_logit_diff=0.0, denominator=1.0,
        )

    # Only layer 2 is attention in our mock (arch_map.attention_indices = [2])
    num_heads = mock_engine.model.config.num_attention_heads  # 4
    assert mock_pah.call_count == num_heads
    assert len(results) == num_heads
    # All results should be for layer 2
    for r in results:
        assert r.layer_idx == 2
        assert r.component_type == "attention_head"


def test_sweep_components_filters_by_important_layers(mock_engine):
    """When important_layers is provided, only those layers are swept."""
    with patch.object(mock_engine, '_capture_clean_attn_output', return_value=torch.randn(1, 3, 32)), \
         patch.object(mock_engine, 'patch_attention_head') as mock_pah:
        mock_pah.return_value = 0.0

        # Layer 2 is the only Transformer layer, but important_layers=[0,1]
        # (which are both Mamba layers), so no Transformer layer qualifies.
        results = mock_engine.sweep_components(
            "clean", "corrupt",
            correct_id=10, incorrect_id=20,
            corrupted_logit_diff=0.0, denominator=1.0,
            important_layers=[0, 1],  # neither is Transformer
        )

    assert mock_pah.call_count == 0
    assert results == []


def test_sweep_components_progress_callback(mock_engine):
    """Progress callback should fire for each head."""
    callback_calls = []

    with patch.object(mock_engine, '_capture_clean_attn_output', return_value=torch.randn(1, 3, 32)), \
         patch.object(mock_engine, 'patch_attention_head') as mock_pah:
        mock_pah.return_value = 0.1

        mock_engine.sweep_components(
            "clean", "corrupt",
            correct_id=10, incorrect_id=20,
            corrupted_logit_diff=0.0, denominator=1.0,
            progress_callback=lambda step, total: callback_calls.append((step, total)),
        )

    num_heads = mock_engine.model.config.num_attention_heads
    assert len(callback_calls) == num_heads
    assert callback_calls[-1] == (num_heads, num_heads)


def test_sweep_components_one_clean_pass_per_layer(mock_engine):
    """_capture_clean_attn_output should be called once per target layer, not per head."""
    with patch.object(mock_engine, '_capture_clean_attn_output', return_value=torch.randn(1, 3, 32)) as mock_cap, \
         patch.object(mock_engine, 'patch_attention_head') as mock_pah:
        mock_pah.return_value = 0.1

        mock_engine.sweep_components(
            "clean", "corrupt",
            correct_id=10, incorrect_id=20,
            corrupted_logit_diff=0.0, denominator=1.0,
        )

    # Only 1 Transformer layer (layer 2) in mock, so exactly 1 capture call
    assert mock_cap.call_count == 1
    # But patch_attention_head called once per head
    num_heads = mock_engine.model.config.num_attention_heads
    assert mock_pah.call_count == num_heads


# ---- Test find_circuit ----

def test_find_circuit_fast_mode(mock_engine):
    """find_circuit in FAST mode should return CircuitResult without component data."""
    _setup_sweep_mocks(mock_engine)
    num_layers = mock_engine.arch_map.num_layers

    fake_path_matrix = torch.zeros(num_layers, num_layers)
    fake_path_matrix[0, 2] = 0.5  # one important path
    fake_layer_importance = torch.tensor([0.3, 0.0, 0.6, 0.0])
    fake_metadata = {
        "correct_id": 10, "incorrect_id": 20,
        "correct_display": "Paris", "incorrect_display": "Warsaw",
        "clean_logit_diff": 5.0, "corrupted_logit_diff": -3.0,
        "denominator": 8.0,
        "clean_prob_correct": 0.9, "corrupted_prob_correct": 0.1,
        "clean_tokens": ["The", "capital", "of"],
        "corrupted_tokens": ["The", "capital", "of"],
        "clean_contributions": {}, "corrupted_contributions": {},
    }

    with patch.object(mock_engine, 'sweep_paths') as mock_sp:
        mock_sp.return_value = (fake_path_matrix, fake_layer_importance, fake_metadata)

        result = mock_engine.find_circuit(
            "The capital of France is", "The capital of Poland is",
            "Paris", "Warsaw",
            threshold=0.1,
            granularity=SweepGranularity.FAST,
        )

    assert isinstance(result, CircuitResult)
    assert result.granularity == SweepGranularity.FAST
    assert result.component_results == []  # FAST mode skips components
    assert len(result.circuit_nodes) > 0  # should have found nodes
    assert len(result.circuit_edges) > 0  # should have found edge 0->2
    assert result.threshold == 0.1


def test_find_circuit_detailed_mode_calls_sweep_components(mock_engine):
    """find_circuit in DETAILED mode should call sweep_components."""
    _setup_sweep_mocks(mock_engine)
    num_layers = mock_engine.arch_map.num_layers

    fake_path_matrix = torch.zeros(num_layers, num_layers)
    fake_layer_importance = torch.tensor([0.0, 0.0, 0.5, 0.0])  # layer 2 is important
    fake_metadata = {
        "correct_id": 10, "incorrect_id": 20,
        "correct_display": "Paris", "incorrect_display": "Warsaw",
        "clean_logit_diff": 5.0, "corrupted_logit_diff": -3.0,
        "denominator": 8.0,
        "clean_prob_correct": 0.9, "corrupted_prob_correct": 0.1,
        "clean_tokens": ["The", "capital", "of"],
        "corrupted_tokens": ["The", "capital", "of"],
        "clean_contributions": {}, "corrupted_contributions": {},
    }

    with patch.object(mock_engine, 'sweep_paths') as mock_sp, \
         patch.object(mock_engine, 'sweep_components') as mock_sc:
        mock_sp.return_value = (fake_path_matrix, fake_layer_importance, fake_metadata)
        mock_sc.return_value = [
            ComponentResult(layer_idx=2, component_idx=0,
                            component_type="attention_head", recovery_score=0.4),
        ]

        result = mock_engine.find_circuit(
            "The capital of France is", "The capital of Poland is",
            "Paris", "Warsaw",
            threshold=0.1,
            granularity=SweepGranularity.DETAILED,
        )

    mock_sc.assert_called_once()
    assert result.granularity == SweepGranularity.DETAILED
    assert len(result.component_results) == 1


def test_find_circuit_detailed_progress_monotonic(mock_engine):
    """In DETAILED mode with a callback, progress (step/total) should never decrease."""
    _setup_sweep_mocks(mock_engine)
    num_layers = mock_engine.arch_map.num_layers
    num_heads = mock_engine.model.config.num_attention_heads

    fake_path_matrix = torch.zeros(num_layers, num_layers)
    fake_layer_importance = torch.tensor([0.0, 0.0, 0.5, 0.0])  # layer 2 important
    fake_metadata = {
        "correct_id": 10, "incorrect_id": 20,
        "correct_display": "Paris", "incorrect_display": "Warsaw",
        "clean_logit_diff": 5.0, "corrupted_logit_diff": -3.0,
        "denominator": 8.0,
        "clean_prob_correct": 0.9, "corrupted_prob_correct": 0.1,
        "clean_tokens": ["The", "capital", "of"],
        "corrupted_tokens": ["The", "capital", "of"],
        "clean_contributions": {}, "corrupted_contributions": {},
    }

    callback_calls = []

    # We need sweep_paths and sweep_components to actually invoke the callback
    # so we can observe the wrapped values.
    def fake_sweep_paths(clean, corrupt, correct, incorrect, progress_callback=None):
        path_total = num_layers + num_layers * (num_layers - 1) // 2
        if progress_callback:
            for step in range(num_layers, path_total + 1):
                progress_callback(step, path_total)
        return (fake_path_matrix, fake_layer_importance, fake_metadata)

    def fake_sweep_components(clean, corrupt, correct_id, incorrect_id,
                              corrupted_logit_diff, denominator,
                              important_layers=None, progress_callback=None):
        total = num_heads  # 1 Transformer layer (layer 2) * num_heads
        if progress_callback:
            for step in range(1, total + 1):
                progress_callback(step, total)
        return [
            ComponentResult(layer_idx=2, component_idx=i,
                            component_type="attention_head", recovery_score=0.3)
            for i in range(num_heads)
        ]

    with patch.object(mock_engine, 'sweep_paths', side_effect=fake_sweep_paths), \
         patch.object(mock_engine, 'sweep_components', side_effect=fake_sweep_components):
        mock_engine.find_circuit(
            "The capital of France is", "The capital of Poland is",
            "Paris", "Warsaw",
            threshold=0.1,
            granularity=SweepGranularity.DETAILED,
            progress_callback=lambda step, total: callback_calls.append((step, total)),
        )

    assert len(callback_calls) > 0
    # Progress ratio should be monotonically non-decreasing
    ratios = [step / total for step, total in callback_calls]
    for i in range(1, len(ratios)):
        assert ratios[i] >= ratios[i - 1], (
            f"Progress went backwards at index {i}: {ratios[i-1]:.3f} -> {ratios[i]:.3f}"
        )
    # Final call should reach 100%
    assert ratios[-1] == pytest.approx(1.0)


def test_find_circuit_empty_tokens_raises(mock_engine):
    """find_circuit should raise ValueError when prompts produce no tokens."""
    mock_output = _make_mock_model_output(seq_len=3, vocab_size=100)
    mock_engine.model.return_value = mock_output
    mock_engine.tokenizer.convert_ids_to_tokens.return_value = []  # empty tokens

    with pytest.raises(ValueError, match="no tokens"):
        mock_engine.find_circuit(
            "clean", "corrupt", "Paris", "Warsaw",
        )


# ---- Test _compute_layer_output ----

def test_compute_layer_output_returns_per_layer_contributions(mock_engine):
    """_compute_layer_output should return a dict with one entry per layer."""
    num_layers = mock_engine.arch_map.num_layers

    fake_embedding = torch.randn(1, 3, 32)
    fake_residuals = {i: torch.randn(1, 3, 32) for i in range(num_layers)}
    fake_cache = {
        "residual_stream": fake_residuals,
        "embedding": fake_embedding,
        "logits": torch.randn(1, 3, 100),
        "tokens": ["The", "capital", "of"],
    }

    with patch.object(mock_engine, 'run_and_cache', return_value=fake_cache):
        contributions = mock_engine._compute_layer_output("test prompt")

    assert len(contributions) == num_layers
    for i in range(num_layers):
        assert i in contributions
        assert contributions[i].shape == (1, 3, 32)


def test_compute_layer_output_layer0_uses_embedding(mock_engine):
    """Layer 0's contribution should be residual[0] - embedding_output."""
    num_layers = mock_engine.arch_map.num_layers

    embedding = torch.ones(1, 3, 32) * 2.0
    residual_0 = torch.ones(1, 3, 32) * 5.0
    fake_residuals = {0: residual_0}
    for i in range(1, num_layers):
        fake_residuals[i] = torch.randn(1, 3, 32)

    fake_cache = {
        "residual_stream": fake_residuals,
        "embedding": embedding,
        "logits": torch.randn(1, 3, 100),
        "tokens": ["The", "capital", "of"],
    }

    with patch.object(mock_engine, 'run_and_cache', return_value=fake_cache):
        contributions = mock_engine._compute_layer_output("test prompt")

    # Layer 0 contribution = residual[0] - embedding = 5.0 - 2.0 = 3.0
    expected = residual_0 - embedding
    assert torch.allclose(contributions[0], expected)


# ---- Test patch_attention_head with missing self_attn ----

def test_patch_attention_head_returns_zero_when_no_self_attn(mock_engine):
    """Should return 0.0 if the layer doesn't have a self_attn attribute."""
    # Layer 2 is attention per arch_map, but we remove self_attn
    mock_engine.arch_map.is_attention.side_effect = lambda idx: True  # pretend it's attention
    del mock_engine.model.model.layers[0].self_attn  # remove the attribute

    score = mock_engine.patch_attention_head(
        "clean", "corrupt", layer_idx=0, head_idx=0,
        correct_id=10, incorrect_id=20,
        corrupted_logit_diff=0.0, denominator=1.0,
    )
    assert score == 0.0
