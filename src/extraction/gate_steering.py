"""Gate (z) steering engine: steer Mamba layers by modifying the output gate.

In Mamba's pipeline, the gate (z) is a multiplicative mask that controls which
dimensions of the SSM output pass through to out_proj and back to the residual
stream. It's the first chunk of in_proj's output (1536 dims), applied after
SiLU activation as: output = SiLU(gate) * ssm_output.

We hook in_proj and modify the first intermediate_size dimensions.
"""

import torch


class GateSteeringEngine:
    """Extract gate values and steer generation by modifying the gate."""

    def __init__(self, model, tokenizer, arch_map, device):
        self.model = model
        self.tokenizer = tokenizer
        self.arch_map = arch_map
        self.device = device
        self.intermediate_size = int(model.config.mamba_expand * model.config.hidden_size)  # 1536

    def extract_gate_values(self, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
        """Extract average gate values at specified Mamba layers.

        Returns:
            dict mapping layer_idx -> Tensor[intermediate_size] (averaged across seq positions).
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(self.device)

        storage = {}
        hooks = []

        for layer_idx in layers:
            if not self.arch_map.is_mamba(layer_idx):
                continue
            mamba_layer = self.model.model.layers[layer_idx].mamba

            def hook_fn(module, input, output, _idx=layer_idx, _sz=self.intermediate_size):
                # First intermediate_size dims of in_proj output are the gate
                gate_raw = output[:, :, :_sz].detach()
                # Average across batch and sequence positions -> [intermediate_size]
                storage[_idx] = gate_raw[0].mean(dim=0)

            h = mamba_layer.in_proj.register_forward_hook(hook_fn)
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
        """Compute contrastive gate vectors (mean_bee - mean_control).

        Returns:
            dict mapping layer_idx -> Tensor[intermediate_size] (unnormalized).
        """
        bee_vals = {layer: [] for layer in layers}
        for prompt in bee_prompts:
            extracted = self.extract_gate_values(prompt, layers)
            for layer_idx, val in extracted.items():
                bee_vals[layer_idx].append(val)

        control_vals = {layer: [] for layer in layers}
        for prompt in control_prompts:
            extracted = self.extract_gate_values(prompt, layers)
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

    def generate_with_gate_steering(
        self,
        prompt: str,
        layer_idx: int,
        vector: torch.Tensor,
        coefficient: float,
        max_new_tokens: int = 100,
    ) -> str:
        """Generate text with gate steering at a single layer.

        Registers the hook ONCE before the generation loop for efficiency.
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        all_ids = list(input_ids)
        model_dtype = next(self.model.parameters()).dtype
        vec = vector.to(self.device, dtype=model_dtype)
        sz = self.intermediate_size

        mamba_layer = self.model.model.layers[layer_idx].mamba

        def gate_hook(module, input, output, _vec=vec, _coeff=coefficient, _sz=sz):
            modified = output.clone()
            modified[:, :, :_sz] = modified[:, :, :_sz] + _coeff * _vec
            return modified

        hook = mamba_layer.in_proj.register_forward_hook(gate_hook)

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
