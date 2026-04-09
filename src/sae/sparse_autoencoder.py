"""Sparse Autoencoder for extracting monosemantic features from residual streams.

Implements the vanilla SAE architecture from Anthropic's "Towards Monosemanticity"
paper. The decoder columns are unit-normalized and serve as feature directions
(steering vectors) in residual stream space.
"""

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Pre-encoder bias: learned mean subtraction (centers the data)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))

        # Encoder: Linear + ReLU (applied in encode())
        self.encoder = nn.Linear(input_dim, latent_dim)

        # Decoder: Linear, no bias (pre_bias is added back after decoding)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

        # Initialize and normalize decoder columns to unit norm
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.encoder.weight)
            nn.init.kaiming_uniform_(self.decoder.weight)
            self._normalize_decoder()

    def _normalize_decoder(self):
        """Unit-normalize each decoder column (feature direction)."""
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse latent activations.

        Args:
            x: [batch, input_dim] activation vectors.

        Returns:
            [batch, latent_dim] non-negative sparse activations.
        """
        return torch.relu(self.encoder(x - self.pre_bias))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent activations back to input space.

        Args:
            z: [batch, latent_dim] sparse activations.

        Returns:
            [batch, input_dim] reconstructed activation vectors.
        """
        return self.decoder(z) + self.pre_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            x: [batch, input_dim] activation vectors.

        Returns:
            (reconstruction, latent_activations) tuple.
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def get_feature_direction(self, feature_idx: int) -> torch.Tensor:
        """Get the decoder weight vector for a specific feature.

        This is the direction in residual stream space that the feature
        represents — used as the steering vector for activation steering.

        Args:
            feature_idx: Index into the latent dimension.

        Returns:
            [input_dim] tensor (detached, no grad).
        """
        return self.decoder.weight[:, feature_idx].detach().clone()
