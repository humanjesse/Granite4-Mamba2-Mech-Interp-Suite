"""Circuit discovery via path patching for hybrid Mamba-2/Transformer models.

Path patching extends activation patching to trace *connections* between layers.
Instead of replacing an entire layer's output, path patching replaces the
contribution of a source layer *only as it flows into a specific target layer*,
leaving the source layer's contribution to all other downstream layers untouched.

In a residual stream architecture, every layer's output is added to the stream.
To patch the path from source layer S to target layer T:

    At layer T's input:
    residual = corrupted_residual
    residual = residual - corrupted_layer_S_output + clean_layer_S_output

If recovery is high, the S->T connection specifically carries critical information.

This module implements:
  Phase 1: Path patching engine (layer-to-layer connection importance)
  Phase 2: Component-level patching (individual attention heads in Transformer layers)
  Phase 3: Automatic circuit extraction (find minimal subgraph above threshold)

The hybrid Mamba-2/Transformer architecture makes this particularly interesting:
Mamba layers route information through compressed recurrent states (implicit routing),
while Transformer layers use explicit attention-based routing. Circuit discovery
reveals how these two mechanisms collaborate.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from enum import Enum

from ..architecture_map import ArchitectureMap

MAX_SEQ_LEN = 256


class SweepGranularity(Enum):
    """Controls how detailed the circuit discovery sweep is."""
    FAST = "fast"              # Layer-level paths only (O(n^2) forward passes)
    DETAILED = "detailed"      # Layer paths + component-level for important connections


@dataclass
class PathPatchResult:
    """Result of path patching between a source and target layer pair."""
    source_layer: int
    target_layer: int
    recovery_score: float
    source_type: str    # "mamba" or "attention"
    target_type: str


@dataclass
class ComponentResult:
    """Result of patching a single component (attention head) within a layer."""
    layer_idx: int
    component_idx: int     # head index for Transformer layers
    component_type: str    # "attention_head"
    recovery_score: float


@dataclass
class CircuitNode:
    """A node in the discovered circuit."""
    layer_idx: int
    layer_type: str          # "mamba" or "attention"
    importance: float        # recovery score from layer-level patching
    components: list[ComponentResult] = field(default_factory=list)


@dataclass
class CircuitEdge:
    """An edge (path) in the discovered circuit."""
    source_layer: int
    target_layer: int
    importance: float        # path patching recovery score
    source_type: str
    target_type: str


@dataclass
class CircuitResult:
    """Complete circuit discovery result."""
    clean_tokens: list[str]
    corrupted_tokens: list[str]
    correct_token: str
    incorrect_token: str
    correct_token_id: int
    incorrect_token_id: int
    correct_token_display: str
    incorrect_token_display: str
    # Path patching matrix: [num_layers, num_layers]
    # Entry [i, j] = recovery when patching path from layer i to layer j
    path_matrix: torch.Tensor
    layer_indices: list[int]
    # Layer-level importance (from standard activation patching, 1D)
    layer_importance: torch.Tensor
    # Component-level results (for Transformer layers)
    component_results: list[ComponentResult]
    # Extracted circuit (nodes + edges above threshold)
    circuit_nodes: list[CircuitNode]
    circuit_edges: list[CircuitEdge]
    # Baselines
    clean_logit_diff: float
    corrupted_logit_diff: float
    clean_prob_correct: float
    corrupted_prob_correct: float
    # Settings
    threshold: float
    granularity: SweepGranularity


class CircuitDiscoveryEngine:
    """Discover circuits in the Granite hybrid model via path patching.

    Builds on the CausalInterventionEngine pattern but adds path-specific
    patching and automatic circuit extraction.
    """

    def __init__(self, model, tokenizer, arch_map: ArchitectureMap, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.arch_map = arch_map
        self.device = device
        self._prompt_cache: dict[str, dict] = {}

    def resolve_token(self, text: str) -> tuple[int, str]:
        """Resolve a text string to a token ID (last token from tokenization)."""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Could not tokenize '{text}' — no tokens produced")
        if len(token_ids) > 1:
            import warnings
            warnings.warn(
                f"'{text}' tokenizes to {len(token_ids)} tokens; using last token only."
            )
        token_id = token_ids[-1]
        decoded = self.tokenizer.decode(token_id)
        display = f'"{text}" → token {token_id} ("{decoded}")'
        return token_id, display

    def run_and_cache(self, prompt: str) -> dict:
        """Run a forward pass and cache both residual stream and per-layer outputs.

        Returns dict with:
            residual_stream: {layer_idx: Tensor[batch, seq, hidden_dim]} — output of each layer
            logits: Tensor[batch, seq, vocab_size]
            tokens: list[str]
        """
        cache_key = prompt.strip()
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        ).to(self.device)
        input_ids = inputs["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        storage = {}
        hooks = []
        for layer_idx in range(self.arch_map.num_layers):
            layer = self.model.model.layers[layer_idx]

            def hook_fn(module, input, output, _idx=layer_idx):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                storage[_idx] = hidden.detach().clone().cpu()

            h = layer.register_forward_hook(hook_fn)
            hooks.append(h)

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        result = {
            "residual_stream": storage,
            "logits": outputs.logits.detach().cpu(),
            "tokens": tokens,
        }
        self._prompt_cache[cache_key] = result
        return result

    def _compute_layer_output(self, prompt: str) -> dict:
        """Compute the *isolated contribution* of each layer to the residual stream.

        In a residual stream architecture, layer L's output is added to the stream:
            residual_after_L = residual_before_L + layer_L_contribution

        We capture the full residual after each layer in run_and_cache. The isolated
        contribution is: layer_L_output = residual_after_L - residual_before_L.

        For layer 0, residual_before is the embedding output. We capture it here.
        """
        cache = self.run_and_cache(prompt)
        residual = cache["residual_stream"]

        # Get the embedding output (input to layer 0)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        ).to(self.device)

        embedding_output = None

        def embed_hook(module, input, output):
            nonlocal embedding_output
            embedding_output = output.detach().clone().cpu()

        # Hook the embedding layer (model.model.embed_tokens)
        h = self.model.model.embed_tokens.register_forward_hook(embed_hook)
        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            h.remove()

        # Compute per-layer isolated contributions
        layer_contributions = {}
        num_layers = self.arch_map.num_layers
        for layer_idx in range(num_layers):
            if layer_idx == 0:
                prev_residual = embedding_output
            else:
                prev_residual = residual[layer_idx - 1]
            layer_contributions[layer_idx] = residual[layer_idx] - prev_residual

        return layer_contributions

    def clear_cache(self):
        """Clear the internal prompt cache."""
        self._prompt_cache.clear()

    def _get_logit_diff(self, logits: torch.Tensor, correct_id: int, incorrect_id: int) -> float:
        """Compute logit difference at last position."""
        return (
            logits[0, -1, correct_id].float()
            - logits[0, -1, incorrect_id].float()
        ).item()

    def _compute_baselines(
        self,
        clean_cache: dict,
        corrupted_cache: dict,
        correct_id: int,
        incorrect_id: int,
    ) -> tuple[float, float, float, float, float]:
        """Compute baseline logit diffs, probabilities, and denominator."""
        clean_logit_diff = self._get_logit_diff(clean_cache["logits"], correct_id, incorrect_id)
        corrupted_logit_diff = self._get_logit_diff(corrupted_cache["logits"], correct_id, incorrect_id)
        denominator = clean_logit_diff - corrupted_logit_diff

        clean_probs = torch.softmax(clean_cache["logits"][0, -1].float(), dim=-1)
        corrupted_probs = torch.softmax(corrupted_cache["logits"][0, -1].float(), dim=-1)

        return (
            clean_logit_diff,
            corrupted_logit_diff,
            denominator,
            clean_probs[correct_id].item(),
            corrupted_probs[correct_id].item(),
        )

    def path_patch(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        source_layer: int,
        target_layer: int,
        correct_id: int,
        incorrect_id: int,
        clean_contributions: dict,
        corrupted_contributions: dict,
        corrupted_logit_diff: float,
        denominator: float,
    ) -> float:
        """Patch the path from source_layer to target_layer only.

        Runs the corrupted prompt but hooks target_layer's input to replace
        source_layer's contribution with the clean version:
            residual_at_target = corrupted_residual
                - corrupted_source_contribution
                + clean_source_contribution

        Only meaningful when source_layer < target_layer (information flows forward).

        Returns fractional recovery score.
        """
        if source_layer >= target_layer:
            return 0.0

        clean_contrib = clean_contributions[source_layer]
        corrupted_contrib = corrupted_contributions[source_layer]

        inputs = self.tokenizer(
            corrupted_prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        ).to(self.device)

        # We hook the target layer to modify its *input* (the residual stream).
        # In HuggingFace, forward hooks get (module, input, output).
        # We use a pre-hook (forward_pre_hook) to modify the input to the target layer.
        def path_patch_hook(module, args):
            # args is a tuple; first element is the hidden_states tensor
            if isinstance(args, tuple) and len(args) > 0:
                hidden = args[0]
                if isinstance(hidden, torch.Tensor):
                    # Swap source layer's contribution
                    device = hidden.device
                    # Ensure shapes match (handle different sequence lengths)
                    seq_len = hidden.shape[1]
                    clean_c = clean_contrib.to(device)
                    corrupt_c = corrupted_contrib.to(device)
                    min_seq = min(seq_len, clean_c.shape[1], corrupt_c.shape[1])
                    modified = hidden.clone()
                    modified[:, :min_seq, :] = (
                        hidden[:, :min_seq, :]
                        - corrupt_c[:, :min_seq, :]
                        + clean_c[:, :min_seq, :]
                    )
                    return (modified,) + args[1:]
            return args

        layer = self.model.model.layers[target_layer]
        hook = layer.register_forward_pre_hook(path_patch_hook)

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
        finally:
            hook.remove()

        patched_logits = outputs.logits.detach().cpu()
        patched_diff = self._get_logit_diff(patched_logits, correct_id, incorrect_id)

        if abs(denominator) > 1e-6:
            return (patched_diff - corrupted_logit_diff) / denominator
        return 0.0

    def sweep_paths(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        correct_token: str,
        incorrect_token: str,
        progress_callback=None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Sweep all source->target layer pairs to build the path matrix.

        Returns (path_matrix, layer_importance, metadata).

        path_matrix[i, j] = recovery when patching path from layer i to layer j.
        layer_importance[i] = recovery when patching layer i entirely (standard activation patching).

        This requires O(n^2) forward passes for path patching + O(n) for layer sweeps
        (but only for source < target, so ~n*(n-1)/2 path patches).
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

        # Compute per-layer contributions for both prompts
        clean_contributions = self._compute_layer_output(clean_prompt)
        corrupted_contributions = self._compute_layer_output(corrupted_prompt)

        num_layers = self.arch_map.num_layers

        # Phase 1: Standard layer sweep (reuse logic from CausalInterventionEngine)
        layer_importance = torch.zeros(num_layers)
        for layer_idx in range(num_layers):
            patched_logits = self._run_with_layer_patch(
                corrupted_prompt, layer_idx, clean_cache
            )
            patched_diff = self._get_logit_diff(patched_logits, correct_id, incorrect_id)
            if abs(denominator) > 1e-6:
                layer_importance[layer_idx] = (patched_diff - corrupted_logit_diff) / denominator

        # Phase 2: Path patching for all source->target pairs
        path_matrix = torch.zeros(num_layers, num_layers)

        # Count total steps for progress
        total_pairs = sum(1 for s in range(num_layers) for t in range(s + 1, num_layers))
        total_steps = num_layers + total_pairs  # layer sweep + path sweep
        current_step = num_layers  # layer sweep already done

        for source in range(num_layers):
            for target in range(source + 1, num_layers):
                score = self.path_patch(
                    clean_prompt, corrupted_prompt,
                    source, target,
                    correct_id, incorrect_id,
                    clean_contributions, corrupted_contributions,
                    corrupted_logit_diff, denominator,
                )
                path_matrix[source, target] = score
                current_step += 1
                if progress_callback:
                    progress_callback(current_step, total_steps)

        metadata = {
            "correct_id": correct_id,
            "incorrect_id": incorrect_id,
            "correct_display": correct_display,
            "incorrect_display": incorrect_display,
            "clean_logit_diff": clean_logit_diff,
            "corrupted_logit_diff": corrupted_logit_diff,
            "denominator": denominator,
            "clean_prob_correct": clean_prob_correct,
            "corrupted_prob_correct": corrupted_prob_correct,
            "clean_tokens": clean_cache["tokens"],
            "corrupted_tokens": corrupted_cache["tokens"],
            "clean_contributions": clean_contributions,
            "corrupted_contributions": corrupted_contributions,
        }
        return path_matrix, layer_importance, metadata

    def _run_with_layer_patch(
        self,
        prompt: str,
        target_layer: int,
        clean_cache: dict,
    ) -> torch.Tensor:
        """Run forward pass with full layer output replaced by clean activations."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        ).to(self.device)

        def hook_fn(module, input, output, _idx=target_layer):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            clean_hidden = clean_cache["residual_stream"][_idx].to(hidden.device)
            min_seq = min(hidden.shape[1], clean_hidden.shape[1])
            modified = hidden.clone()
            modified[:, :min_seq, :] = clean_hidden[:, :min_seq, :]

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        layer = self.model.model.layers[target_layer]
        hook = layer.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
        finally:
            hook.remove()

        return outputs.logits.detach().cpu()

    def patch_attention_head(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        layer_idx: int,
        head_idx: int,
        correct_id: int,
        incorrect_id: int,
        clean_cache: dict,
        corrupted_logit_diff: float,
        denominator: float,
    ) -> float:
        """Patch a single attention head's output within a Transformer layer.

        Hooks into the attention output projection and replaces only the
        specified head's contribution with the clean-run value.

        Only applicable to Transformer (attention) layers.
        """
        if not self.arch_map.is_attention(layer_idx):
            return 0.0

        inputs = self.tokenizer(
            corrupted_prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        ).to(self.device)

        # We need to capture the clean attention output for this layer.
        # Run clean prompt and capture the self_attn output.
        clean_attn_output = None
        corrupted_attn_output = None

        def capture_clean_hook(module, input, output):
            nonlocal clean_attn_output
            if isinstance(output, tuple):
                clean_attn_output = output[0].detach().clone()
            else:
                clean_attn_output = output.detach().clone()

        def capture_corrupt_hook(module, input, output):
            nonlocal corrupted_attn_output
            if isinstance(output, tuple):
                corrupted_attn_output = output[0].detach().clone()
            else:
                corrupted_attn_output = output.detach().clone()

        # Get clean attention output
        clean_inputs = self.tokenizer(
            clean_prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        ).to(self.device)

        attn_module = self.model.model.layers[layer_idx].self_attn
        h = attn_module.register_forward_hook(capture_clean_hook)
        try:
            with torch.no_grad():
                self.model(**clean_inputs)
        finally:
            h.remove()

        if clean_attn_output is None:
            return 0.0

        # Now run corrupted with head-specific patching
        num_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.hidden_size // num_heads

        def head_patch_hook(module, input, output):
            if isinstance(output, tuple):
                attn_out = output[0]
            else:
                attn_out = output

            # attn_out shape: [batch, seq, hidden_size]
            # Reshape to [batch, seq, num_heads, head_dim]
            batch, seq, hidden = attn_out.shape
            modified = attn_out.clone()
            modified_heads = modified.view(batch, seq, num_heads, head_dim)
            clean_heads = clean_attn_output.to(attn_out.device)

            min_seq = min(seq, clean_heads.shape[1])
            clean_reshaped = clean_heads[:, :min_seq, :].view(
                clean_heads.shape[0], min_seq, num_heads, head_dim
            )

            # Replace only the target head
            modified_heads[:, :min_seq, head_idx, :] = clean_reshaped[:, :min_seq, head_idx, :]
            modified = modified_heads.view(batch, seq, hidden)

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        h = attn_module.register_forward_hook(head_patch_hook)
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
        finally:
            h.remove()

        patched_logits = outputs.logits.detach().cpu()
        patched_diff = self._get_logit_diff(patched_logits, correct_id, incorrect_id)

        if abs(denominator) > 1e-6:
            return (patched_diff - corrupted_logit_diff) / denominator
        return 0.0

    def sweep_components(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        correct_id: int,
        incorrect_id: int,
        clean_cache: dict,
        corrupted_logit_diff: float,
        denominator: float,
        important_layers: list[int] | None = None,
        progress_callback=None,
    ) -> list[ComponentResult]:
        """Sweep individual attention heads in Transformer layers.

        Only patches heads in Transformer layers (Mamba layers don't have
        discrete attention heads). If important_layers is provided, only
        sweeps Transformer layers in that list.
        """
        results = []
        num_heads = self.model.config.num_attention_heads

        # Only sweep Transformer layers
        target_layers = [
            idx for idx in self.arch_map.attention_indices
            if important_layers is None or idx in important_layers
        ]

        total_steps = len(target_layers) * num_heads
        step = 0

        for layer_idx in target_layers:
            for head_idx in range(num_heads):
                score = self.patch_attention_head(
                    clean_prompt, corrupted_prompt,
                    layer_idx, head_idx,
                    correct_id, incorrect_id,
                    clean_cache,
                    corrupted_logit_diff, denominator,
                )
                results.append(ComponentResult(
                    layer_idx=layer_idx,
                    component_idx=head_idx,
                    component_type="attention_head",
                    recovery_score=score,
                ))
                step += 1
                if progress_callback:
                    progress_callback(step, total_steps)

        return results

    def find_circuit(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        correct_token: str,
        incorrect_token: str,
        threshold: float = 0.1,
        granularity: SweepGranularity = SweepGranularity.FAST,
        progress_callback=None,
    ) -> CircuitResult:
        """Discover the minimal circuit for a clean/corrupt prompt pair.

        Steps:
        1. Layer sweep → identify important layers (recovery > threshold)
        2. Path sweep between all layers → identify important connections
        3. (If DETAILED) Component sweep within important Transformer layers
        4. Extract circuit: nodes + edges above threshold

        Returns a CircuitResult with the full path matrix, layer importance,
        component results, and the extracted circuit graph.
        """
        # Phase 1+2: Sweep paths (includes layer sweep internally)
        path_matrix, layer_importance, metadata = self.sweep_paths(
            clean_prompt, corrupted_prompt,
            correct_token, incorrect_token,
            progress_callback=progress_callback,
        )

        # Phase 3: Component-level sweep (DETAILED mode only)
        component_results = []
        if granularity == SweepGranularity.DETAILED:
            important_layers = [
                i for i in range(self.arch_map.num_layers)
                if abs(layer_importance[i].item()) >= threshold
            ]
            component_results = self.sweep_components(
                clean_prompt, corrupted_prompt,
                metadata["correct_id"], metadata["incorrect_id"],
                self.run_and_cache(clean_prompt),
                metadata["corrupted_logit_diff"], metadata["denominator"],
                important_layers=important_layers,
                progress_callback=progress_callback,
            )

        # Phase 4: Extract circuit graph
        nodes, edges = self._extract_circuit(
            path_matrix, layer_importance, component_results, threshold
        )

        return CircuitResult(
            clean_tokens=metadata["clean_tokens"],
            corrupted_tokens=metadata["corrupted_tokens"],
            correct_token=correct_token,
            incorrect_token=incorrect_token,
            correct_token_id=metadata["correct_id"],
            incorrect_token_id=metadata["incorrect_id"],
            correct_token_display=metadata["correct_display"],
            incorrect_token_display=metadata["incorrect_display"],
            path_matrix=path_matrix,
            layer_indices=list(range(self.arch_map.num_layers)),
            layer_importance=layer_importance,
            component_results=component_results,
            circuit_nodes=nodes,
            circuit_edges=edges,
            clean_logit_diff=metadata["clean_logit_diff"],
            corrupted_logit_diff=metadata["corrupted_logit_diff"],
            clean_prob_correct=metadata["clean_prob_correct"],
            corrupted_prob_correct=metadata["corrupted_prob_correct"],
            threshold=threshold,
            granularity=granularity,
        )

    def _extract_circuit(
        self,
        path_matrix: torch.Tensor,
        layer_importance: torch.Tensor,
        component_results: list[ComponentResult],
        threshold: float,
    ) -> tuple[list[CircuitNode], list[CircuitEdge]]:
        """Extract the minimal circuit above threshold.

        Nodes: layers with |importance| >= threshold
        Edges: paths with |recovery| >= threshold between any two layers
        """
        num_layers = self.arch_map.num_layers

        # Identify important nodes
        nodes = []
        important_layer_set = set()
        for idx in range(num_layers):
            imp = layer_importance[idx].item()
            if abs(imp) >= threshold:
                components = [
                    c for c in component_results
                    if c.layer_idx == idx and abs(c.recovery_score) >= threshold
                ]
                nodes.append(CircuitNode(
                    layer_idx=idx,
                    layer_type=self.arch_map.layer_type(idx),
                    importance=imp,
                    components=components,
                ))
                important_layer_set.add(idx)

        # Identify important edges
        edges = []
        for s in range(num_layers):
            for t in range(s + 1, num_layers):
                score = path_matrix[s, t].item()
                if abs(score) >= threshold:
                    edges.append(CircuitEdge(
                        source_layer=s,
                        target_layer=t,
                        importance=score,
                        source_type=self.arch_map.layer_type(s),
                        target_type=self.arch_map.layer_type(t),
                    ))
                    # Ensure both endpoints are in the node set
                    for layer_idx in (s, t):
                        if layer_idx not in important_layer_set:
                            nodes.append(CircuitNode(
                                layer_idx=layer_idx,
                                layer_type=self.arch_map.layer_type(layer_idx),
                                importance=layer_importance[layer_idx].item(),
                                components=[
                                    c for c in component_results
                                    if c.layer_idx == layer_idx
                                ],
                            ))
                            important_layer_set.add(layer_idx)

        # Sort nodes by layer index
        nodes.sort(key=lambda n: n.layer_idx)
        edges.sort(key=lambda e: (e.source_layer, e.target_layer))

        return nodes, edges
