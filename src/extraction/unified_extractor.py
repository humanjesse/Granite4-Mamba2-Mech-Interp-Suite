"""Orchestrator: hooks + forward pass + extraction for all layer types."""

import torch
from ..architecture_map import ArchitectureMap
from .mamba_hidden_attn import (
    compute_hidden_attention_mamba2,
    compute_hidden_attention_mamba2_memory_efficient,
)
from .transformer_attn import extract_transformer_attention_via_hooks
from .neuron_extractor import register_neuron_hooks

MAX_SEQ_LEN = 256


class GraniteAttentionExtractor:
    """Extract attention patterns from all layers of a Granite hybrid model."""

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.arch_map = ArchitectureMap(model.config)

        # Cache config values
        self.n_heads = model.config.mamba_n_heads
        self.n_groups = model.config.mamba_n_groups
        self.ssm_state_size = model.config.mamba_d_state
        self.intermediate_size = int(model.config.mamba_expand * model.config.hidden_size)
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

    def extract(self, prompt: str, max_length: int = MAX_SEQ_LEN) -> dict:
        """Run extraction on a prompt.

        Returns dict with:
            tokens: list of token strings
            mamba_attention: dict[layer_idx -> Tensor[batch, heads, seq, seq]]
            transformer_attention: dict[layer_idx -> Tensor[batch, heads, seq, seq]]
            layer_map: ArchitectureMap
            logit_lens: dict with prediction evolution across layers
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        seq_len = input_ids.shape[1]

        # Set up hooks for Mamba layers
        mamba_hooks = []
        mamba_raw_data = {}

        for layer_idx in self.arch_map.mamba_indices:
            mamba_layer = self.model.model.layers[layer_idx].mamba
            hooks, storage = self._register_mamba_hooks(mamba_layer, layer_idx)
            mamba_hooks.extend(hooks)
            mamba_raw_data[layer_idx] = storage

        # Set up hooks for Transformer attention layers
        attn_hooks, attn_storage = extract_transformer_attention_via_hooks(
            self.model, self.arch_map.attention_indices
        )

        # Set up hooks for residual stream (logit lens)
        residual_hooks, residual_storage = self._register_residual_hooks()

        # Set up hooks for MLP neuron activations
        neuron_hooks, neuron_storage = register_neuron_hooks(self.model, self.arch_map)

        # Forward pass (output_attentions=True doesn't work for this model,
        # so we rely on hooks for both Mamba and Transformer attention)
        with torch.no_grad():
            outputs = self.model(**inputs)
            output_logits = outputs.logits.detach().cpu()

        # Remove hooks
        for h in mamba_hooks + attn_hooks + residual_hooks + neuron_hooks:
            h.remove()

        # Compute hidden attention for each Mamba layer
        mamba_attention = {}
        for layer_idx in self.arch_map.mamba_indices:
            raw = mamba_raw_data[layer_idx]
            if "dt" not in raw or "B" not in raw or "C" not in raw:
                continue

            mamba_layer = self.model.model.layers[layer_idx].mamba
            A_log = mamba_layer.A_log.detach()
            dt_bias = mamba_layer.dt_bias.detach()

            if seq_len > 128:
                hidden_attn = compute_hidden_attention_mamba2_memory_efficient(
                    dt=raw["dt"],
                    B=raw["B"],
                    C=raw["C"],
                    A_log=A_log,
                    dt_bias=dt_bias,
                    n_heads=self.n_heads,
                    n_groups=self.n_groups,
                )
            else:
                hidden_attn = compute_hidden_attention_mamba2(
                    dt=raw["dt"],
                    B=raw["B"],
                    C=raw["C"],
                    A_log=A_log,
                    dt_bias=dt_bias,
                    n_heads=self.n_heads,
                    n_groups=self.n_groups,
                )

            mamba_attention[layer_idx] = hidden_attn.cpu()

        # Collect transformer attention from hooks
        transformer_attention = {k: v.cpu() for k, v in attn_storage.items()}

        # Compute logit lens
        logit_lens = self._compute_logit_lens(residual_storage, tokens)

        return {
            "tokens": tokens,
            "mamba_attention": mamba_attention,
            "transformer_attention": transformer_attention,
            "layer_map": self.arch_map,
            "logit_lens": logit_lens,
            "neuron_activations": {k: v.cpu() for k, v in neuron_storage.items()},
            "residual_stream": {k: v.cpu() for k, v in residual_storage.items()},
            "logits": output_logits,
        }

    def _register_mamba_hooks(self, mamba_layer, layer_idx):
        """Register hooks to capture dt, B, C from Mamba layer's torch_forward.

        We hook in_proj to get the raw dt, and conv1d to get B/C after convolution.
        """
        storage = {}
        hooks = []

        # Hook in_proj to capture raw dt (last num_heads dims of projection)
        def in_proj_hook(module, input, output, _idx=layer_idx):
            # projected_states split: [gate, hidden_states_B_C, dt]
            # sizes: [intermediate_size, conv_dim, num_heads]
            dt_raw = output[:, :, -self.n_heads:].detach()
            storage["dt"] = dt_raw

        h1 = mamba_layer.in_proj.register_forward_hook(in_proj_hook)
        hooks.append(h1)

        # Hook conv1d to capture B and C after convolution + activation
        # The conv1d output contains [hidden_states, B, C] concatenated
        def conv1d_hook(module, input, output, _idx=layer_idx):
            # output is after conv1d but before slicing to seq_len
            # Shape: [batch, conv_dim, seq_len + padding]
            # The calling code does: conv1d(...).transpose(1,2)[:, :seq_len]
            # then applies act() and splits into hidden_states, B, C
            # We need to capture the post-activation split.
            # Since we can't hook between act and split, we store raw conv output
            # and the caller will apply act + split.
            storage["conv1d_raw"] = output.detach()

        h2 = mamba_layer.conv1d.register_forward_hook(conv1d_hook)
        hooks.append(h2)

        # Hook the entire mamba layer to capture post-processed B and C
        # by intercepting torch_forward's internal state.
        # Mamba layer receives hidden_states as a kwarg, not positional arg.
        def mamba_forward_hook(module, args, kwargs, output, _idx=layer_idx):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and len(args) > 0:
                hidden_states = args[0]
            if hidden_states is None:
                return

            batch, seq_len, _ = hidden_states.shape

            with torch.no_grad():
                projected = module.in_proj(hidden_states)
                gate, hidden_states_BC, dt_raw = projected.split(
                    [self.intermediate_size, self.conv_dim, self.n_heads], dim=-1
                )

                # Run conv1d (same as torch_forward)
                hidden_states_BC = module.act(
                    module.conv1d(hidden_states_BC.transpose(1, 2))[..., :seq_len].transpose(1, 2)
                )

                groups_state_size = module.n_groups * module.ssm_state_size
                _, B, C = torch.split(
                    hidden_states_BC,
                    [self.intermediate_size, groups_state_size, groups_state_size],
                    dim=-1,
                )

                storage["B"] = B.detach()
                storage["C"] = C.detach()

        h3 = mamba_layer.register_forward_hook(mamba_forward_hook, with_kwargs=True)
        hooks.append(h3)

        return hooks, storage

    def _register_residual_hooks(self):
        """Register hooks to capture residual stream (hidden states) at each layer output."""
        storage = {}
        hooks = []

        for layer_idx in range(self.arch_map.num_layers):
            layer = self.model.model.layers[layer_idx]

            def hook_fn(module, input, output, _idx=layer_idx):
                # Layer output is typically a tuple; first element is hidden states
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                storage[_idx] = hidden.detach()

            h = layer.register_forward_hook(hook_fn)
            hooks.append(h)

        return hooks, storage

    def _compute_logit_lens(self, residual_states: dict, tokens: list, k: int = 10) -> dict:
        """Project each layer's residual stream through lm_head to get predictions.

        Returns dict with logit lens data for the last token position.
        """
        lm_head = self.model.lm_head
        last_pos = -1
        layers_data = []

        with torch.no_grad():
            for layer_idx in range(self.arch_map.num_layers):
                if layer_idx not in residual_states:
                    continue

                hidden = residual_states[layer_idx][:, last_pos, :]  # [batch, hidden_dim]

                # Cast to float32 for numerical stability
                logits = hidden.float() @ lm_head.weight.T.float()
                if lm_head.bias is not None:
                    logits = logits + lm_head.bias.float()

                probs = torch.softmax(logits, dim=-1)  # [batch, vocab_size]
                top_probs, top_ids = probs[0].topk(k)

                top_tokens = []
                for tid in top_ids:
                    decoded = self.tokenizer.decode(tid.item()).strip()
                    if not decoded:
                        decoded = repr(self.tokenizer.convert_ids_to_tokens(tid.item()))
                    top_tokens.append(decoded)

                layers_data.append({
                    "layer": layer_idx,
                    "tokens": top_tokens,
                    "probs": top_probs.cpu().tolist(),
                })

        return {
            "position": len(tokens) - 1,
            "position_token": tokens[-1] if tokens else "",
            "layers": layers_data,
        }
