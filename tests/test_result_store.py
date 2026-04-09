"""Tests for the result storage module."""

import json
import time

import pytest
import torch

from src.storage.result_store import ResultStore
from src.extraction.causal_intervention import SweepResult, InterventionType
from src.extraction.circuit_discovery import (
    CircuitResult,
    SweepGranularity,
    ComponentResult,
    CircuitNode,
    CircuitEdge,
)


@pytest.fixture
def store(tmp_path):
    return ResultStore(base_dir=tmp_path / "results")


@pytest.fixture
def sample_sweep():
    return SweepResult(
        clean_tokens=["The", " capital", " of", " France", " is"],
        corrupted_tokens=["The", " capital", " of", " Poland", " is"],
        correct_token="Paris",
        incorrect_token="Warsaw",
        correct_token_id=1234,
        incorrect_token_id=5678,
        correct_token_display='"Paris" -> token 1234',
        incorrect_token_display='"Warsaw" -> token 5678',
        intervention_type=InterventionType.ACTIVATION_PATCH,
        recovery_matrix=torch.tensor([0.0, 0.1, 0.5, 0.8]),
        layer_indices=[0, 1, 2, 3],
        position_indices=[0, 1, 2, 3, 4],
        clean_logit_diff=5.0,
        corrupted_logit_diff=-3.0,
        clean_prob_correct=0.9,
        corrupted_prob_correct=0.1,
    )


@pytest.fixture
def sample_circuit():
    return CircuitResult(
        clean_tokens=["The", " capital"],
        corrupted_tokens=["The", " capital"],
        correct_token="Paris",
        incorrect_token="Warsaw",
        correct_token_id=10,
        incorrect_token_id=20,
        correct_token_display='"Paris"',
        incorrect_token_display='"Warsaw"',
        path_matrix=torch.randn(4, 4),
        layer_indices=[0, 1, 2, 3],
        layer_importance=torch.tensor([0.1, 0.5, 0.8, 0.2]),
        component_results=[
            ComponentResult(
                layer_idx=1, component_idx=0,
                component_type="attention_head", recovery_score=0.3,
            ),
        ],
        circuit_nodes=[
            CircuitNode(
                layer_idx=2, layer_type="mamba", importance=0.8,
                components=[
                    ComponentResult(
                        layer_idx=2, component_idx=1,
                        component_type="attention_head", recovery_score=0.4,
                    ),
                ],
            ),
        ],
        circuit_edges=[
            CircuitEdge(
                source_layer=1, target_layer=2, importance=0.6,
                source_type="attention", target_type="mamba",
            ),
        ],
        clean_logit_diff=5.0,
        corrupted_logit_diff=-3.0,
        clean_prob_correct=0.9,
        corrupted_prob_correct=0.1,
        threshold=0.1,
        granularity=SweepGranularity.FAST,
    )


# ── Round-trip tests ────────────────────────────────────────────────────


def test_save_load_intervention(store, sample_sweep):
    save_dir = store.save_intervention(sample_sweep, notes="test run")
    assert (save_dir / "metadata.json").exists()
    assert (save_dir / "tensors.pt").exists()

    loaded, meta = store.load_intervention(save_dir)
    assert meta["notes"] == "test run"
    assert loaded.correct_token == "Paris"
    assert loaded.intervention_type == InterventionType.ACTIVATION_PATCH
    assert torch.allclose(loaded.recovery_matrix, sample_sweep.recovery_matrix)
    assert loaded.clean_tokens == sample_sweep.clean_tokens
    assert loaded.clean_logit_diff == sample_sweep.clean_logit_diff


def test_save_load_circuit(store, sample_circuit):
    save_dir = store.save_circuit(sample_circuit, notes="circuit test")
    loaded, meta = store.load_circuit(save_dir)

    assert meta["notes"] == "circuit test"
    assert loaded.correct_token == "Paris"
    assert loaded.granularity == SweepGranularity.FAST
    assert loaded.threshold == 0.1
    assert torch.allclose(loaded.path_matrix, sample_circuit.path_matrix)
    assert torch.allclose(loaded.layer_importance, sample_circuit.layer_importance)
    assert len(loaded.component_results) == 1
    assert loaded.component_results[0].recovery_score == 0.3
    assert len(loaded.circuit_nodes) == 1
    assert len(loaded.circuit_nodes[0].components) == 1
    assert loaded.circuit_nodes[0].components[0].recovery_score == 0.4
    assert len(loaded.circuit_edges) == 1
    assert loaded.circuit_edges[0].source_type == "attention"


def test_save_load_2d_recovery_matrix(store):
    result = SweepResult(
        clean_tokens=["a", "b"],
        corrupted_tokens=["a", "b"],
        correct_token="x",
        incorrect_token="y",
        correct_token_id=1,
        incorrect_token_id=2,
        correct_token_display="x",
        incorrect_token_display="y",
        intervention_type=InterventionType.ZERO_ABLATION,
        recovery_matrix=torch.randn(4, 3),
        layer_indices=[0, 1, 2, 3],
        position_indices=[0, 1, 2],
        clean_logit_diff=1.0,
        corrupted_logit_diff=-1.0,
        clean_prob_correct=0.8,
        corrupted_prob_correct=0.2,
    )
    save_dir = store.save_intervention(result)
    loaded, meta = store.load_intervention(save_dir)
    assert meta["sweep_mode"] == "full"
    assert loaded.recovery_matrix.shape == (4, 3)
    assert torch.allclose(loaded.recovery_matrix, result.recovery_matrix)


# ── List experiments ────────────────────────────────────────────────────


def test_list_empty(store):
    assert store.list_experiments() == []


def test_list_after_save(store, sample_sweep, sample_circuit):
    store.save_intervention(sample_sweep, notes="first")
    store.save_circuit(sample_circuit, notes="second")

    all_exps = store.list_experiments()
    assert len(all_exps) == 2

    assert len(store.list_experiments(tool_type="intervention")) == 1
    assert len(store.list_experiments(tool_type="circuit")) == 1


def test_list_sorted_newest_first(store, sample_sweep):
    store.save_intervention(sample_sweep, notes="older")
    time.sleep(1.1)
    store.save_intervention(sample_sweep, notes="newer")

    exps = store.list_experiments()
    assert exps[0]["notes"] == "newer"
    assert exps[1]["notes"] == "older"


# ── Edge cases ──────────────────────────────────────────────────────────


def test_notes_multiline(store, sample_sweep):
    notes = "Testing layer 15\nMulti-line notes"
    save_dir = store.save_intervention(sample_sweep, notes=notes)
    _, meta = store.load_intervention(save_dir)
    assert meta["notes"] == notes


def test_folder_name_sanitization(store):
    result = SweepResult(
        clean_tokens=["a"],
        corrupted_tokens=["a"],
        correct_token="cafe/resume",
        incorrect_token="naive\\path",
        correct_token_id=1,
        incorrect_token_id=2,
        correct_token_display="x",
        incorrect_token_display="y",
        intervention_type=InterventionType.ACTIVATION_PATCH,
        recovery_matrix=torch.tensor([0.5]),
        layer_indices=[0],
        position_indices=[0],
        clean_logit_diff=1.0,
        corrupted_logit_diff=-1.0,
        clean_prob_correct=0.8,
        corrupted_prob_correct=0.2,
    )
    save_dir = store.save_intervention(result)
    assert save_dir.exists()
    assert "/" not in save_dir.name
    assert "\\" not in save_dir.name


def test_all_intervention_types_roundtrip(store):
    for int_type in InterventionType:
        result = SweepResult(
            clean_tokens=["a"],
            corrupted_tokens=["a"],
            correct_token="x",
            incorrect_token="y",
            correct_token_id=1,
            incorrect_token_id=2,
            correct_token_display="x",
            incorrect_token_display="y",
            intervention_type=int_type,
            recovery_matrix=torch.tensor([0.5]),
            layer_indices=[0],
            position_indices=[0],
            clean_logit_diff=1.0,
            corrupted_logit_diff=-1.0,
            clean_prob_correct=0.8,
            corrupted_prob_correct=0.2,
        )
        save_dir = store.save_intervention(result)
        loaded, _ = store.load_intervention(save_dir)
        assert loaded.intervention_type == int_type


def test_metadata_json_readable(store, sample_sweep):
    save_dir = store.save_intervention(sample_sweep, notes="readable")
    with open(save_dir / "metadata.json") as f:
        data = json.load(f)
    assert data["correct_token"] == "Paris"
    assert data["notes"] == "readable"
    assert "saved_at" in data
    assert data["tool_type"] == "intervention"


def test_short_label_format(store, sample_sweep):
    store.save_intervention(sample_sweep, notes="my note")
    exps = store.list_experiments()
    label = exps[0]["short_label"]
    assert "[intervention]" in label
    assert "Paris" in label
    assert "Warsaw" in label
    assert "my note" in label
