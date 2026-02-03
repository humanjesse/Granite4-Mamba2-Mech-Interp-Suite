"""Tests for circuit discovery visualization."""

import torch
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import MagicMock

from src.extraction.circuit_discovery import (
    SweepGranularity,
    ComponentResult,
    CircuitNode,
    CircuitEdge,
    CircuitResult,
)
from src.visualization.circuit_viz import (
    create_circuit_dashboard,
    format_circuit_info,
    create_path_matrix_plotly,
    create_layer_importance_plotly,
    create_circuit_diagram_plotly,
    create_component_importance_plotly,
    create_circuit_summary_markdown,
)
import plotly.graph_objects as go


def _make_circuit_result(num_layers=4, with_components=False, with_circuit=True):
    """Create a synthetic CircuitResult for testing."""
    path_matrix = torch.randn(num_layers, num_layers) * 0.3
    # Zero lower triangle (only upper is meaningful)
    path_matrix = torch.triu(path_matrix, diagonal=1)

    layer_importance = torch.randn(num_layers) * 0.5

    component_results = []
    if with_components:
        for head in range(4):
            component_results.append(ComponentResult(
                layer_idx=2, component_idx=head,
                component_type="attention_head",
                recovery_score=float(torch.randn(1).item() * 0.3),
            ))

    nodes = []
    edges = []
    if with_circuit:
        nodes = [
            CircuitNode(layer_idx=0, layer_type="mamba", importance=0.4),
            CircuitNode(layer_idx=2, layer_type="attention", importance=0.6,
                        components=component_results[:2] if with_components else []),
        ]
        edges = [
            CircuitEdge(source_layer=0, target_layer=2, importance=0.35,
                        source_type="mamba", target_type="attention"),
        ]

    arch_map = MagicMock()
    arch_map.num_layers = num_layers
    arch_map.layer_type.side_effect = lambda idx: "mamba" if idx != 2 else "attention"
    arch_map.attention_indices = [2]

    result = CircuitResult(
        clean_tokens=["The", "capital", "of", "France"],
        corrupted_tokens=["The", "capital", "of", "Poland"],
        correct_token="Paris",
        incorrect_token="Warsaw",
        correct_token_id=10,
        incorrect_token_id=20,
        correct_token_display='"Paris" → token 10 ("Paris")',
        incorrect_token_display='"Warsaw" → token 20 ("Warsaw")',
        path_matrix=path_matrix,
        layer_indices=list(range(num_layers)),
        layer_importance=layer_importance,
        component_results=component_results,
        circuit_nodes=nodes,
        circuit_edges=edges,
        clean_logit_diff=5.0,
        corrupted_logit_diff=-3.0,
        clean_prob_correct=0.9,
        corrupted_prob_correct=0.1,
        threshold=0.1,
        granularity=SweepGranularity.DETAILED if with_components else SweepGranularity.FAST,
    )
    return result, arch_map


# ---- Test create_circuit_dashboard ----

def test_dashboard_returns_figure():
    """Dashboard should return a matplotlib Figure."""
    result, arch_map = _make_circuit_result()
    fig = create_circuit_dashboard(result, arch_map)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_dashboard_with_components():
    """DETAILED mode dashboard should work with component data."""
    result, arch_map = _make_circuit_result(with_components=True)
    fig = create_circuit_dashboard(result, arch_map)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_dashboard_empty_circuit():
    """Dashboard should handle empty circuit (no nodes/edges)."""
    result, arch_map = _make_circuit_result(with_circuit=False)
    fig = create_circuit_dashboard(result, arch_map)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_dashboard_none_result():
    """Dashboard should handle None input gracefully."""
    arch_map = MagicMock()
    fig = create_circuit_dashboard(None, arch_map)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---- Test format_circuit_info ----

def test_format_info_returns_string():
    result, _ = _make_circuit_result()
    info = format_circuit_info(result)
    assert isinstance(info, str)
    assert "Paris" in info
    assert "Warsaw" in info


def test_format_info_includes_stats():
    result, _ = _make_circuit_result()
    info = format_circuit_info(result)
    assert "Circuit:" in info
    assert "nodes" in info
    assert "edges" in info


def test_format_info_none_result():
    info = format_circuit_info(None)
    assert "No circuit" in info


def test_format_info_empty_circuit():
    result, _ = _make_circuit_result(with_circuit=False)
    info = format_circuit_info(result)
    assert "0 nodes" in info
    assert "0 edges" in info


def test_format_info_includes_logit_diffs():
    result, _ = _make_circuit_result()
    info = format_circuit_info(result)
    assert "logit diff" in info.lower()
    assert "5.000" in info or "+5.000" in info


# ---- Plotly visualization tests ----

def test_path_matrix_plotly_returns_figure():
    result, arch_map = _make_circuit_result()
    fig = create_path_matrix_plotly(result, arch_map)
    assert isinstance(fig, go.Figure)


def test_layer_importance_plotly_returns_figure():
    result, arch_map = _make_circuit_result()
    fig = create_layer_importance_plotly(result, arch_map)
    assert isinstance(fig, go.Figure)


def test_circuit_diagram_plotly_returns_figure():
    result, arch_map = _make_circuit_result()
    fig = create_circuit_diagram_plotly(result, arch_map)
    assert isinstance(fig, go.Figure)


def test_circuit_diagram_plotly_filters_top_n():
    """With many nodes, diagram should filter to top-N."""
    num_layers = 32
    result, arch_map = _make_circuit_result(num_layers=num_layers, with_circuit=False)
    # Manually add 32 nodes
    nodes = []
    for i in range(num_layers):
        lt = arch_map.layer_type(i)
        nodes.append(CircuitNode(layer_idx=i, layer_type=lt,
                                 importance=float(torch.randn(1).item())))
    edges = [CircuitEdge(source_layer=0, target_layer=5, importance=0.3,
                         source_type="mamba", target_type="attention")]
    result.circuit_nodes = nodes
    result.circuit_edges = edges

    fig = create_circuit_diagram_plotly(result, arch_map, max_nodes=10)
    assert isinstance(fig, go.Figure)
    # Should have filtered note in title
    assert "top 10" in fig.layout.title.text.lower()


def test_component_importance_plotly_returns_figure():
    result, arch_map = _make_circuit_result(with_components=True)
    fig = create_component_importance_plotly(result, arch_map)
    assert isinstance(fig, go.Figure)
    # Should have heatmap data
    assert len(fig.data) > 0


def test_component_importance_plotly_empty():
    """No component data should return a valid figure with message."""
    result, arch_map = _make_circuit_result(with_components=False)
    fig = create_component_importance_plotly(result, arch_map)
    assert isinstance(fig, go.Figure)


def test_circuit_summary_markdown_returns_string():
    result, arch_map = _make_circuit_result(with_components=True)
    md = create_circuit_summary_markdown(result, arch_map)
    assert isinstance(md, str)
    assert "Circuit Discovery" in md
    assert "Paris" in md


def test_plotly_functions_handle_none():
    """All Plotly functions should handle None input gracefully."""
    arch_map = MagicMock()
    assert isinstance(create_path_matrix_plotly(None, arch_map), go.Figure)
    assert isinstance(create_layer_importance_plotly(None, arch_map), go.Figure)
    assert isinstance(create_circuit_diagram_plotly(None, arch_map), go.Figure)
    assert isinstance(create_component_importance_plotly(None, arch_map), go.Figure)
    assert isinstance(create_circuit_summary_markdown(None, arch_map), str)
