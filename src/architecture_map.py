"""Map layer indices to types (mamba/attention) for Granite hybrid models."""


class ArchitectureMap:
    """Maps layer indices to their types and provides lookup utilities."""

    def __init__(self, config):
        self.layer_types = list(config.layer_types)
        self.num_layers = config.num_hidden_layers

        self.mamba_indices = [
            i for i, t in enumerate(self.layer_types) if t == "mamba"
        ]
        self.attention_indices = [
            i for i, t in enumerate(self.layer_types) if t == "attention"
        ]

    def is_mamba(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "mamba"

    def is_attention(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "attention"

    def layer_type(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    def nearest_attention(self, layer_idx: int) -> int | None:
        """Find nearest attention layer to given index."""
        if not self.attention_indices:
            return None
        return min(self.attention_indices, key=lambda x: abs(x - layer_idx))

    def nearest_mamba(self, layer_idx: int) -> int | None:
        """Find nearest mamba layer to given index."""
        if not self.mamba_indices:
            return None
        return min(self.mamba_indices, key=lambda x: abs(x - layer_idx))

    def summary(self) -> str:
        n_mamba = len(self.mamba_indices)
        n_attn = len(self.attention_indices)
        return (
            f"{self.num_layers} layers: {n_mamba} Mamba-2, {n_attn} Transformer attention\n"
            f"Mamba layers: {self.mamba_indices}\n"
            f"Attention layers: {self.attention_indices}"
        )
