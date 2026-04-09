"""dt steering engine: steer Mamba layers by modifying the dt (timestep) parameter.

In the Mamba SSM update h[t] = h[t-1] * exp(dt*A) + dt * B * x[t],
dt controls how much each token updates the hidden state — it's Mamba's
analog of attention strength.

We hook in_proj (the linear projection before softplus) and modify the
last num_heads dimensions, which are the raw dt values.
"""

import torch


class DtSteeringEngine:
    """Extract dt values and steer generation by modifying dt."""

    def __init__(self, model, tokenizer, arch_map, device):
        self.model = model
        self.tokenizer = tokenizer
        self.arch_map = arch_map
        self.device = device
        self.num_heads = model.config.mamba_n_heads  # 48

    def extract_dt_values(self, prompt: str, layers: list[int]) -> dict[int, torch.Tensor]:
        """Extract average dt values at specified Mamba layers.

        Args:
            prompt: Input text.
            layers: Layer indices to extract from (must be Mamba layers).

        Returns:
            dict mapping layer_idx -> Tensor[num_heads] (averaged across seq positions).
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(self.device)

        storage = {}
        hooks = []

        for layer_idx in layers:
            if not self.arch_map.is_mamba(layer_idx):
                continue
            mamba_layer = self.model.model.layers[layer_idx].mamba

            def hook_fn(module, input, output, _idx=layer_idx):
                # Last num_heads dims of in_proj output are raw dt
                dt_raw = output[:, :, -self.num_heads:].detach()
                # Average across batch and sequence positions -> [num_heads]
                storage[_idx] = dt_raw[0].mean(dim=0)

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
        """Compute contrastive dt vectors (mean_bee - mean_control).

        Args:
            bee_prompts: Prompts containing the target concept.
            control_prompts: Neutral/control prompts.
            layers: Layer indices to compute vectors for.

        Returns:
            dict mapping layer_idx -> Tensor[num_heads] (unnormalized contrastive vector).
        """
        # Collect bee dt values
        bee_dts = {layer: [] for layer in layers}
        for prompt in bee_prompts:
            dt_vals = self.extract_dt_values(prompt, layers)
            for layer_idx, dt in dt_vals.items():
                bee_dts[layer_idx].append(dt)

        # Collect control dt values
        control_dts = {layer: [] for layer in layers}
        for prompt in control_prompts:
            dt_vals = self.extract_dt_values(prompt, layers)
            for layer_idx, dt in dt_vals.items():
                control_dts[layer_idx].append(dt)

        # Compute contrastive vectors (unnormalized to preserve dt scale)
        vectors = {}
        for layer_idx in layers:
            if not bee_dts[layer_idx] or not control_dts[layer_idx]:
                continue
            mean_bee = torch.stack(bee_dts[layer_idx]).mean(dim=0)
            mean_control = torch.stack(control_dts[layer_idx]).mean(dim=0)
            vectors[layer_idx] = mean_bee - mean_control

        return vectors

    def _generate(self, prompt: str, max_new_tokens: int = 200) -> str:
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

    def generate_with_dt_steering(
        self,
        prompt: str,
        layer_idx: int,
        vector: torch.Tensor,
        coefficient: float,
        max_new_tokens: int = 200,
    ) -> str:
        """Generate text with dt steering at a single layer.

        Registers the hook ONCE before the generation loop for efficiency.
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        all_ids = list(input_ids)
        model_dtype = next(self.model.parameters()).dtype
        vec = vector.to(self.device, dtype=model_dtype)
        num_heads = self.num_heads

        mamba_layer = self.model.model.layers[layer_idx].mamba

        def dt_hook(module, input, output, _vec=vec, _coeff=coefficient, _nh=num_heads):
            modified = output.clone()
            modified[:, :, -_nh:] = modified[:, :, -_nh:] + _coeff * _vec
            return modified

        hook = mamba_layer.in_proj.register_forward_hook(dt_hook)

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
