"""Causal intervention engine for activation patching experiments.

Activation patching works by:
1. Running a clean prompt (correct answer) and caching residual stream activations
2. Running a corrupted prompt (wrong answer) with intervention hooks that replace
   specific layer/position activations with clean-run values
3. Measuring how much the correct answer is recovered (fractional recovery)

If patching layer L restores the correct answer, that layer is causally responsible
for the model's knowledge of that fact.
"""

import torch
from dataclasses import dataclass
from enum import Enum

from .intervention_base import InterventionBase, MAX_SEQ_LEN, MIN_RECOVERY_DENOMINATOR


class InterventionType(Enum):
    """Types of causal interventions supported."""
    ACTIVATION_PATCH = "activation_patch"   # Replace corrupted with clean activations
    ZERO_ABLATION = "zero_ablation"         # Set activations to zero
    MEAN_ABLATION = "mean_ablation"         # Replace with mean across positions
    NOISE = "noise"                         # Add Gaussian noise


@dataclass
class SweepResult:
    """Result of sweeping interventions across layers (and optionally positions)."""
    clean_tokens: list[str]
    corrupted_tokens: list[str]
    correct_token: str
    incorrect_token: str
    correct_token_id: int
    incorrect_token_id: int
    correct_token_display: str          # How the token was resolved (for UI)
    incorrect_token_display: str
    intervention_type: InterventionType
    # Core data — shape depends on sweep mode:
    #   Layer sweep: [num_layers] (1D)
    #   Full sweep:  [num_layers, seq_len] (2D)
    recovery_matrix: torch.Tensor
    layer_indices: list[int]
    position_indices: list[int]
    # Baselines
    clean_logit_diff: float
    corrupted_logit_diff: float
    clean_prob_correct: float
    corrupted_prob_correct: float


class CausalInterventionEngine(InterventionBase):
    """Run causal intervention experiments on the Granite hybrid model.

    This engine patches activations from a clean run into a corrupted run
    to determine which layers and positions are causally responsible for
    a model's prediction.
    """

    def run_with_intervention(
        self,
        prompt: str,
        intervention_type: InterventionType,
        target_layer: int,
        positions: list[int] | None,
        clean_cache: dict | None = None,
        clean_contributions: dict | None = None,
        corrupted_contributions: dict | None = None,
        noise_std: float = 1.0,
    ) -> torch.Tensor:
        """Run a forward pass with an intervention hook on a specific layer.

        The hook modifies the layer's output (residual stream) at the
        specified positions according to the intervention type. PyTorch
        replaces the module output when a forward hook returns a value.

        Returns patched logits [batch, seq, vocab_size].
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        ).to(self.device)
        seq_len = inputs["input_ids"].shape[1]

        if positions is None:
            positions = list(range(seq_len))

        def make_intervention_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output

                modified = hidden.clone()
                # Build index tensor for the positions we want to patch
                valid_pos = [p for p in positions if p < modified.shape[1]]
                if not valid_pos:
                    return output
                pos_idx = torch.tensor(valid_pos, device=modified.device)

                if intervention_type == InterventionType.ACTIVATION_PATCH:
                    clean_c = clean_contributions[layer_idx].to(modified.device)
                    corrupt_c = corrupted_contributions[layer_idx].to(modified.device)
                    # Only patch positions that exist in both contribution tensors
                    patch_pos = [p for p in valid_pos
                                 if p < clean_c.shape[1] and p < corrupt_c.shape[1]]
                    if patch_pos:
                        idx = torch.tensor(patch_pos, device=modified.device)
                        modified[:, idx, :] = (
                            modified[:, idx, :]
                            - corrupt_c[:, idx, :]
                            + clean_c[:, idx, :]
                        )

                elif intervention_type == InterventionType.ZERO_ABLATION:
                    modified[:, pos_idx, :] = 0.0

                elif intervention_type == InterventionType.MEAN_ABLATION:
                    # Replaces target positions with the mean activation across all
                    # sequence positions. This is a sequence-level mean, not a
                    # dataset-level mean (which would require pre-computed statistics).
                    mean_val = hidden.mean(dim=1, keepdim=True)
                    modified[:, pos_idx, :] = mean_val.expand(
                        -1, len(valid_pos), -1
                    )

                elif intervention_type == InterventionType.NOISE:
                    noise = torch.randn_like(modified[:, pos_idx, :]) * noise_std
                    modified[:, pos_idx, :] = modified[:, pos_idx, :] + noise

                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                return modified

            return hook_fn

        # Register intervention hook on target layer only
        layer = self.model.model.layers[target_layer]
        hook = layer.register_forward_hook(make_intervention_hook(target_layer))

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
        finally:
            hook.remove()

        return outputs.logits.detach().cpu()

    def sweep_layers(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        correct_token: str,
        incorrect_token: str,
        intervention_type: InterventionType = InterventionType.ACTIVATION_PATCH,
        positions: list[int] | None = None,
        progress_callback=None,
    ) -> SweepResult:
        """Sweep intervention across all layers.

        By default patches all positions at each layer (answering "which layers
        matter?"). Pass specific positions to test a single token position
        across all layers (e.g., "at which layer does the France token matter?").

        Requires num_layers forward passes (32 for Granite).
        """
        # Resolve tokens
        correct_id, correct_display = self.resolve_token(correct_token)
        incorrect_id, incorrect_display = self.resolve_token(incorrect_token)

        # Phase 1: Cache clean and corrupted activations
        clean_cache = self.run_and_cache(clean_prompt)
        corrupted_cache = self.run_and_cache(corrupted_prompt)

        # Compute baselines
        (
            clean_logit_diff,
            corrupted_logit_diff,
            denominator,
            clean_prob_correct,
            corrupted_prob_correct,
        ) = self._compute_baselines(clean_cache, corrupted_cache, correct_id, incorrect_id)

        # Compute per-layer isolated contributions for activation patching
        clean_contributions = None
        corrupted_contributions = None
        if intervention_type == InterventionType.ACTIVATION_PATCH:
            clean_contributions = self._compute_layer_output(clean_prompt)
            corrupted_contributions = self._compute_layer_output(corrupted_prompt)

        # Phase 2: Sweep across layers
        num_layers = self.arch_map.num_layers
        recovery_scores = torch.zeros(num_layers)

        for layer_idx in range(num_layers):
            patched_logits = self.run_with_intervention(
                prompt=corrupted_prompt,
                intervention_type=intervention_type,
                target_layer=layer_idx,
                positions=positions,
                clean_contributions=clean_contributions,
                corrupted_contributions=corrupted_contributions,
            )
            patched_diff = (
                patched_logits[0, -1, correct_id].float()
                - patched_logits[0, -1, incorrect_id].float()
            ).item()

            if abs(denominator) > MIN_RECOVERY_DENOMINATOR:
                recovery_scores[layer_idx] = (patched_diff - corrupted_logit_diff) / denominator
            else:
                recovery_scores[layer_idx] = 0.0

            if progress_callback:
                progress_callback(layer_idx + 1, num_layers)

        return SweepResult(
            clean_tokens=clean_cache["tokens"],
            corrupted_tokens=corrupted_cache["tokens"],
            correct_token=correct_token,
            incorrect_token=incorrect_token,
            correct_token_id=correct_id,
            incorrect_token_id=incorrect_id,
            correct_token_display=correct_display,
            incorrect_token_display=incorrect_display,
            intervention_type=intervention_type,
            recovery_matrix=recovery_scores,
            layer_indices=list(range(num_layers)),
            position_indices=positions or list(range(len(corrupted_cache["tokens"]))),
            clean_logit_diff=clean_logit_diff,
            corrupted_logit_diff=corrupted_logit_diff,
            clean_prob_correct=clean_prob_correct,
            corrupted_prob_correct=corrupted_prob_correct,
        )

    def sweep_positions_and_layers(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        correct_token: str,
        incorrect_token: str,
        intervention_type: InterventionType = InterventionType.ACTIVATION_PATCH,
        layer_subset: list[int] | None = None,
        progress_callback=None,
    ) -> SweepResult:
        """Full position × layer sweep — patches ONE position at a time per layer.

        Produces the full [num_layers, seq_len] heatmap but requires
        num_layers × seq_len forward passes.
        """
        correct_id, correct_display = self.resolve_token(correct_token)
        incorrect_id, incorrect_display = self.resolve_token(incorrect_token)

        clean_cache = self.run_and_cache(clean_prompt)
        corrupted_cache = self.run_and_cache(corrupted_prompt)

        (
            clean_logit_diff,
            corrupted_logit_diff,
            denominator,
            clean_prob_correct,
            corrupted_prob_correct,
        ) = self._compute_baselines(clean_cache, corrupted_cache, correct_id, incorrect_id)

        # Compute per-layer isolated contributions for activation patching
        clean_contributions = None
        corrupted_contributions = None
        if intervention_type == InterventionType.ACTIVATION_PATCH:
            clean_contributions = self._compute_layer_output(clean_prompt)
            corrupted_contributions = self._compute_layer_output(corrupted_prompt)

        layers = layer_subset or list(range(self.arch_map.num_layers))
        seq_len = len(corrupted_cache["tokens"])
        recovery_matrix = torch.zeros(len(layers), seq_len)

        total_steps = len(layers) * seq_len
        step = 0

        for i, layer_idx in enumerate(layers):
            for pos in range(seq_len):
                patched_logits = self.run_with_intervention(
                    prompt=corrupted_prompt,
                    intervention_type=intervention_type,
                    target_layer=layer_idx,
                    positions=[pos],
                    clean_contributions=clean_contributions,
                    corrupted_contributions=corrupted_contributions,
                )
                patched_diff = (
                    patched_logits[0, -1, correct_id].float()
                    - patched_logits[0, -1, incorrect_id].float()
                ).item()

                if abs(denominator) > MIN_RECOVERY_DENOMINATOR:
                    recovery_matrix[i, pos] = (patched_diff - corrupted_logit_diff) / denominator
                else:
                    recovery_matrix[i, pos] = 0.0

                step += 1
                if progress_callback:
                    progress_callback(step, total_steps)

        return SweepResult(
            clean_tokens=clean_cache["tokens"],
            corrupted_tokens=corrupted_cache["tokens"],
            correct_token=correct_token,
            incorrect_token=incorrect_token,
            correct_token_id=correct_id,
            incorrect_token_id=incorrect_id,
            correct_token_display=correct_display,
            incorrect_token_display=incorrect_display,
            intervention_type=intervention_type,
            recovery_matrix=recovery_matrix,
            layer_indices=layers,
            position_indices=list(range(seq_len)),
            clean_logit_diff=clean_logit_diff,
            corrupted_logit_diff=corrupted_logit_diff,
            clean_prob_correct=clean_prob_correct,
            corrupted_prob_correct=corrupted_prob_correct,
        )
