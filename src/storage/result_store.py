"""Save and load experiment results (intervention + circuit) to disk.

Storage format per experiment:
    results/{tool_type}/{timestamp}_{description}/
        metadata.json   -- all scalar/string/list fields, plus user notes
        tensors.pt      -- all torch.Tensor fields bundled into one file
"""

import json
import re
from datetime import datetime
from pathlib import Path

import torch

from src.extraction.causal_intervention import SweepResult, InterventionType
from src.extraction.circuit_discovery import (
    CircuitResult,
    SweepGranularity,
    ComponentResult,
    CircuitNode,
    CircuitEdge,
)

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def _sanitize(text: str, max_len: int = 20) -> str:
    """Remove non-word characters and truncate for filesystem-safe names."""
    return re.sub(r"[^\w]", "", text)[:max_len]


class ResultStore:
    """Filesystem-based storage for experiment results."""

    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_RESULTS_DIR

    # ── Save ────────────────────────────────────────────────────────────

    def save_intervention(self, result: SweepResult, notes: str = "") -> Path:
        """Save a SweepResult to disk. Returns the saved directory path."""
        metadata, tensors = self._sweep_to_dict(result)
        metadata["tool_type"] = "intervention"
        metadata["saved_at"] = datetime.now().isoformat()
        metadata["notes"] = notes
        metadata["sweep_mode"] = "full" if result.recovery_matrix.dim() == 2 else "layer"

        folder = self._make_folder_name(
            result.correct_token, result.incorrect_token, result.intervention_type.value
        )
        save_dir = self.base_dir / "intervention" / folder
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        torch.save(tensors, save_dir / "tensors.pt")

        return save_dir

    def save_circuit(self, result: CircuitResult, notes: str = "") -> Path:
        """Save a CircuitResult to disk. Returns the saved directory path."""
        metadata, tensors = self._circuit_to_dict(result)
        metadata["tool_type"] = "circuit"
        metadata["saved_at"] = datetime.now().isoformat()
        metadata["notes"] = notes

        folder = self._make_folder_name(
            result.correct_token, result.incorrect_token, result.granularity.value
        )
        save_dir = self.base_dir / "circuit" / folder
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        torch.save(tensors, save_dir / "tensors.pt")

        return save_dir

    # ── Load ────────────────────────────────────────────────────────────

    def load_intervention(self, experiment_dir: Path | str) -> tuple[SweepResult, dict]:
        """Load a SweepResult from disk. Returns (result, metadata_dict)."""
        experiment_dir = Path(experiment_dir)
        with open(experiment_dir / "metadata.json") as f:
            metadata = json.load(f)
        tensors = torch.load(experiment_dir / "tensors.pt", weights_only=True)
        return self._dict_to_sweep(metadata, tensors), metadata

    def load_circuit(self, experiment_dir: Path | str) -> tuple[CircuitResult, dict]:
        """Load a CircuitResult from disk. Returns (result, metadata_dict)."""
        experiment_dir = Path(experiment_dir)
        with open(experiment_dir / "metadata.json") as f:
            metadata = json.load(f)
        tensors = torch.load(experiment_dir / "tensors.pt", weights_only=True)
        return self._dict_to_circuit(metadata, tensors), metadata

    # ── List ────────────────────────────────────────────────────────────

    def list_experiments(self, tool_type: str | None = None) -> list[dict]:
        """Scan results directory and return metadata summaries, newest first."""
        experiments = []
        tool_types = [tool_type] if tool_type else ["intervention", "circuit"]

        for tt in tool_types:
            type_dir = self.base_dir / tt
            if not type_dir.exists():
                continue
            for entry in type_dir.iterdir():
                if not entry.is_dir():
                    continue
                meta_path = entry / "metadata.json"
                if not meta_path.exists():
                    continue
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    experiments.append({
                        "path": str(entry),
                        "tool_type": tt,
                        "folder_name": entry.name,
                        "timestamp": meta.get("saved_at", ""),
                        "notes": meta.get("notes", ""),
                        "correct_token": meta.get("correct_token", ""),
                        "incorrect_token": meta.get("incorrect_token", ""),
                        "short_label": self._make_short_label(meta),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue

        experiments.sort(key=lambda e: e["timestamp"], reverse=True)
        return experiments

    # ── Private helpers ─────────────────────────────────────────────────

    def _make_folder_name(self, correct: str, incorrect: str, type_hint: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{timestamp}_{_sanitize(correct)}_vs_{_sanitize(incorrect)}_{_sanitize(type_hint, 30)}"

    def _make_short_label(self, meta: dict) -> str:
        tool = meta.get("tool_type", "?")
        correct = meta.get("correct_token", "?")
        incorrect = meta.get("incorrect_token", "?")
        timestamp = meta.get("saved_at", "")[:19]
        notes = meta.get("notes", "")
        label = f"[{tool}] {correct} vs {incorrect} ({timestamp})"
        if notes:
            label += f" -- {notes[:40]}"
        return label

    # ── Serialization: SweepResult ──────────────────────────────────────

    def _sweep_to_dict(self, result: SweepResult) -> tuple[dict, dict]:
        metadata = {
            "clean_tokens": result.clean_tokens,
            "corrupted_tokens": result.corrupted_tokens,
            "correct_token": result.correct_token,
            "incorrect_token": result.incorrect_token,
            "correct_token_id": result.correct_token_id,
            "incorrect_token_id": result.incorrect_token_id,
            "correct_token_display": result.correct_token_display,
            "incorrect_token_display": result.incorrect_token_display,
            "intervention_type": result.intervention_type.value,
            "layer_indices": result.layer_indices,
            "position_indices": result.position_indices,
            "clean_logit_diff": result.clean_logit_diff,
            "corrupted_logit_diff": result.corrupted_logit_diff,
            "clean_prob_correct": result.clean_prob_correct,
            "corrupted_prob_correct": result.corrupted_prob_correct,
            "recovery_matrix_shape": list(result.recovery_matrix.shape),
        }
        tensors = {"recovery_matrix": result.recovery_matrix}
        return metadata, tensors

    def _dict_to_sweep(self, metadata: dict, tensors: dict) -> SweepResult:
        return SweepResult(
            clean_tokens=metadata["clean_tokens"],
            corrupted_tokens=metadata["corrupted_tokens"],
            correct_token=metadata["correct_token"],
            incorrect_token=metadata["incorrect_token"],
            correct_token_id=metadata["correct_token_id"],
            incorrect_token_id=metadata["incorrect_token_id"],
            correct_token_display=metadata["correct_token_display"],
            incorrect_token_display=metadata["incorrect_token_display"],
            intervention_type=InterventionType(metadata["intervention_type"]),
            recovery_matrix=tensors["recovery_matrix"],
            layer_indices=metadata["layer_indices"],
            position_indices=metadata["position_indices"],
            clean_logit_diff=metadata["clean_logit_diff"],
            corrupted_logit_diff=metadata["corrupted_logit_diff"],
            clean_prob_correct=metadata["clean_prob_correct"],
            corrupted_prob_correct=metadata["corrupted_prob_correct"],
        )

    # ── Serialization: CircuitResult ────────────────────────────────────

    def _circuit_to_dict(self, result: CircuitResult) -> tuple[dict, dict]:
        metadata = {
            "clean_tokens": result.clean_tokens,
            "corrupted_tokens": result.corrupted_tokens,
            "correct_token": result.correct_token,
            "incorrect_token": result.incorrect_token,
            "correct_token_id": result.correct_token_id,
            "incorrect_token_id": result.incorrect_token_id,
            "correct_token_display": result.correct_token_display,
            "incorrect_token_display": result.incorrect_token_display,
            "layer_indices": result.layer_indices,
            "threshold": result.threshold,
            "granularity": result.granularity.value,
            "clean_logit_diff": result.clean_logit_diff,
            "corrupted_logit_diff": result.corrupted_logit_diff,
            "clean_prob_correct": result.clean_prob_correct,
            "corrupted_prob_correct": result.corrupted_prob_correct,
            "component_results": [
                {
                    "layer_idx": c.layer_idx,
                    "component_idx": c.component_idx,
                    "component_type": c.component_type,
                    "recovery_score": c.recovery_score,
                }
                for c in result.component_results
            ],
            "circuit_nodes": [
                {
                    "layer_idx": n.layer_idx,
                    "layer_type": n.layer_type,
                    "importance": n.importance,
                    "components": [
                        {
                            "layer_idx": c.layer_idx,
                            "component_idx": c.component_idx,
                            "component_type": c.component_type,
                            "recovery_score": c.recovery_score,
                        }
                        for c in n.components
                    ],
                }
                for n in result.circuit_nodes
            ],
            "circuit_edges": [
                {
                    "source_layer": e.source_layer,
                    "target_layer": e.target_layer,
                    "importance": e.importance,
                    "source_type": e.source_type,
                    "target_type": e.target_type,
                }
                for e in result.circuit_edges
            ],
            "path_matrix_shape": list(result.path_matrix.shape),
            "layer_importance_shape": list(result.layer_importance.shape),
        }
        tensors = {
            "path_matrix": result.path_matrix,
            "layer_importance": result.layer_importance,
        }
        return metadata, tensors

    def _dict_to_circuit(self, metadata: dict, tensors: dict) -> CircuitResult:
        component_results = [
            ComponentResult(**c) for c in metadata["component_results"]
        ]
        circuit_nodes = [
            CircuitNode(
                layer_idx=n["layer_idx"],
                layer_type=n["layer_type"],
                importance=n["importance"],
                components=[ComponentResult(**c) for c in n.get("components", [])],
            )
            for n in metadata["circuit_nodes"]
        ]
        circuit_edges = [CircuitEdge(**e) for e in metadata["circuit_edges"]]

        return CircuitResult(
            clean_tokens=metadata["clean_tokens"],
            corrupted_tokens=metadata["corrupted_tokens"],
            correct_token=metadata["correct_token"],
            incorrect_token=metadata["incorrect_token"],
            correct_token_id=metadata["correct_token_id"],
            incorrect_token_id=metadata["incorrect_token_id"],
            correct_token_display=metadata["correct_token_display"],
            incorrect_token_display=metadata["incorrect_token_display"],
            path_matrix=tensors["path_matrix"],
            layer_indices=metadata["layer_indices"],
            layer_importance=tensors["layer_importance"],
            component_results=component_results,
            circuit_nodes=circuit_nodes,
            circuit_edges=circuit_edges,
            clean_logit_diff=metadata["clean_logit_diff"],
            corrupted_logit_diff=metadata["corrupted_logit_diff"],
            clean_prob_correct=metadata["clean_prob_correct"],
            corrupted_prob_correct=metadata["corrupted_prob_correct"],
            threshold=metadata["threshold"],
            granularity=SweepGranularity(metadata["granularity"]),
        )
