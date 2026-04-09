"""Conv1d output steering: steer Mamba layers by modifying conv1d output.

In Mamba's pipeline, conv1d is a depthwise causal convolution (kernel=4)
that mixes local context BEFORE the SSM processes it. Its output (1792 dims)
contains the hidden states, B, and C that drive the SSM.

We hook conv1d and modify its output before SiLU activation and B/C split.
Output is in channels-first format: [batch, 1792, seq_len+padding].
"""

import torch


class Conv1dSteeringEngine:
    """Extract conv1d outputs and steer generation by modifying them."""

    def __init__(self, model, tokenizer, arch_map, device):
        self.model = model
        self.tokenizer = tokenizer
        self.arch_map = arch_map
        self.device = device
        # conv_dim = intermediate_size + 2 * n_groups * d_state
        intermediate_size = model.config.mamba_expand * model.config.hidden_size
        n_groups = model.config.mamba_n_groups
        d_state = model.config.mamba_d_state
        self.conv_dim = intermediate_size + 2 * n_groups * d_state

    def extract_conv1d_values(self, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
        """Extract average conv1d output values at specified Mamba layers.

        Args:
            prompt: Input text.
            layers: Layer indices to extract from (must be Mamba layers).

        Returns:
            dict mapping layer_idx -> Tensor[conv_dim] (averaged across seq positions).
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        seq_len = input_ids.shape[1]

        storage = {}
        hooks = []

        for layer_idx in layers:
            if not self.arch_map.is_mamba(layer_idx):
                continue
            mamba_layer = self.model.model.layers[layer_idx].mamba

            def hook_fn(module, input, output, _idx=layer_idx, _seq_len=seq_len):
                # output is channels-first: [batch, conv_dim, seq_len+padding]
                # Slice to actual seq_len, transpose to [batch, seq_len, conv_dim]
                trimmed = output[:, :, :_seq_len].transpose(1, 2).detach()
                # Average across batch and seq positions -> [conv_dim]
                storage[_idx] = trimmed[0].mean(dim=0)

            h = mamba_layer.conv1d.register_forward_hook(hook_fn)
            hooks.append(h)

        with torch.no_grad():
            self.model(input_ids=input_ids)

        for h in hooks:
            h.remove()

        return storage

    def compute_contrastive_vectors(
        self,
        bee_prompts: list[str],
        control_prompts: list[str],
        layers: list[int],
    ) -> dict[int, torch.Tensor]:
        """Compute contrastive conv1d vectors (mean_bee - mean_control).

        Returns:
            dict mapping layer_idx -> Tensor[conv_dim] (unnormalized contrastive vector).
        """
        bee_vals = {layer: [] for layer in layers}
        for prompt in bee_prompts:
            extracted = self.extract_conv1d_values(prompt, layers)
            for layer_idx, val in extracted.items():
                bee_vals[layer_idx].append(val)

        control_vals = {layer: [] for layer in layers}
        for prompt in control_prompts:
            extracted = self.extract_conv1d_values(prompt, layers)
            for layer_idx, val in extracted.items():
                control_vals[layer_idx].append(val)

        vectors = {}
        for layer_idx in layers:
            if not bee_vals[layer_idx] or not control_vals[layer_idx]:
                continue
            mean_bee = torch.stack(bee_vals[layer_idx]).mean(dim=0)
            mean_control = torch.stack(control_vals[layer_idx]).mean(dim=0)
            vectors[layer_idx] = mean_bee - mean_control

        return vectors

    def _generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Unsteered generation (greedy decoding)."""
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        all_ids = list(input_ids)
        for step in range(max_new_tokens):
            ids_tensor = torch.tensor([all_ids], device=self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=ids_tensor)
            logits = outputs.logits[0, -1, :].float()
            next_id = logits.argmax().item()
            all_ids.append(next_id)
            if next_id == self.tokenizer.eos_token_id:
                break
        return self.tokenizer.decode(all_ids, skip_special_tokens=False)

    def generate_with_conv1d_steering(
        self,
        prompt: str,
        layer_idx: int,
        vector: torch.Tensor,
        coefficient: float,
        max_new_tokens: int = 100,
    ) -> str:
        """Generate text with conv1d output steering at a single layer.

        Registers the hook ONCE before the generation loop for efficiency.
        The vector is added to conv1d output in channels-first format.
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        all_ids = list(input_ids)
        model_dtype = next(self.model.parameters()).dtype
        # Reshape vector to [1, conv_dim, 1] for channels-first broadcast
        vec = vector.to(self.device, dtype=model_dtype).unsqueeze(0).unsqueeze(-1)

        mamba_layer = self.model.model.layers[layer_idx].mamba

        def conv1d_hook(module, input, output, _vec=vec, _coeff=coefficient):
            # output: [batch, conv_dim, seq_len+padding]
            return output + _coeff * _vec

        hook = mamba_layer.conv1d.register_forward_hook(conv1d_hook)

        try:
            for step in range(max_new_tokens):
                ids_tensor = torch.tensor([all_ids], device=self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids=ids_tensor)
                logits = outputs.logits[0, -1, :].float()
                next_id = logits.argmax().item()
                all_ids.append(next_id)
                if next_id == self.tokenizer.eos_token_id:
                    break
        finally:
            hook.remove()

        return self.tokenizer.decode(all_ids, skip_special_tokens=False)
