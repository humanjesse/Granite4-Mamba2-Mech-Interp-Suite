"""Shared base class for causal intervention and circuit discovery engines.

Both engines need to resolve tokens, cache forward-pass results, and compute
baseline logit diffs. This base class eliminates that duplication.
"""

import warnings

import torch

from ..architecture_map import ArchitectureMap

MAX_SEQ_LEN = 256
# Minimum denominator (clean_logit_diff - corrupted_logit_diff) to compute
# a meaningful recovery score.  Below this the clean and corrupted runs are
# too similar for activation patching to be informative.
MIN_RECOVERY_DENOMINATOR = 1e-6


class InterventionBase:
    """Common functionality for engines that do activation patching."""

    def __init__(self, model, tokenizer, arch_map: ArchitectureMap, device: str = "cpu"):
        if model is None:
            raise ValueError("model cannot be None")
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None")
        if arch_map is None:
            raise ValueError("arch_map cannot be None")
        self.model = model
        self.tokenizer = tokenizer
        self.arch_map = arch_map
        self.device = device
        # Internal cache: prompt string -> {residual_stream, logits, tokens}
        self._prompt_cache: dict[str, dict] = {}

    def resolve_token(self, text: str) -> tuple[int, str]:
        """Resolve a text string to a token ID.

        Takes the last token from the tokenization of the input text.
        Returns (token_id, display_string) where display_string shows
        the exact resolution for UI feedback.

        This handles the space-prefix issue: "Paris" and " Paris" may
        tokenize differently. The display string makes this visible.
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Could not tokenize '{text}' — no tokens produced")
        if len(token_ids) > 1:
            warnings.warn(
                f"'{text}' tokenizes to {len(token_ids)} tokens; using last token only."
            )
        token_id = token_ids[-1]
        decoded = self.tokenizer.decode(token_id)
        display = f'"{text}" → token {token_id} ("{decoded}")'
        return token_id, display

    def run_and_cache(self, prompt: str) -> dict:
        """Run a forward pass and cache residual stream activations at every layer.

        Uses an internal cache so repeated calls with the same prompt
        (e.g., switching from layer sweep to full sweep) reuse results.

        Returns dict with:
            residual_stream: {layer_idx: Tensor[batch, seq, hidden_dim]}
            embedding: Tensor[batch, seq, hidden_dim]  (embedding layer output)
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

        # Register hooks (read-only): embedding + residual stream at every layer
        storage = {}
        hooks = []
        embedding_output = None

        def embed_hook_fn(module, input, output):
            nonlocal embedding_output
            embedding_output = output.detach().clone().cpu()

        hooks.append(self.model.model.embed_tokens.register_forward_hook(embed_hook_fn))

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
            "embedding": embedding_output,
            "logits": outputs.logits.detach().cpu(),
            "tokens": tokens,
        }
        self._prompt_cache[cache_key] = result
        return result

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
        """Compute baseline logit diffs, probabilities, and denominator.

        Returns (clean_logit_diff, corrupted_logit_diff, denominator,
                 clean_prob_correct, corrupted_prob_correct).
        """
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

    def _compute_layer_output(self, prompt: str) -> dict:
        """Compute the *isolated contribution* of each layer to the residual stream.

        In a residual stream architecture, layer L's output is added to the stream:
            residual_after_L = residual_before_L + layer_L_contribution

        We capture the full residual after each layer in run_and_cache. The isolated
        contribution is: layer_L_output = residual_after_L - residual_before_L.

        For layer 0, residual_before is the embedding output (cached by run_and_cache).
        """
        cache = self.run_and_cache(prompt)
        residual = cache["residual_stream"]
        embedding_output = cache["embedding"]

        layer_contributions = {}
        num_layers = self.arch_map.num_layers
        for layer_idx in range(num_layers):
            prev = embedding_output if layer_idx == 0 else residual[layer_idx - 1]
            layer_contributions[layer_idx] = residual[layer_idx] - prev

        return layer_contributions
