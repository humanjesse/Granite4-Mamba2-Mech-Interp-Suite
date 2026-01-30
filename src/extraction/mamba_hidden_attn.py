"""Core: extract hidden attention matrices from Mamba-2 layers.

Implements the formula from Ali et al. (2024) "Hidden Attention of Mamba Models":
    α̃[i,j] = exp(Σ_{k=j+1}^{i} dt[k] * a) × (C[i] · B[j]) × dt[j]

Mamba-2 simplification: A is scalar × identity, so the matrix exponential
collapses to a scalar exponential, making extraction much cheaper than Mamba-1.

The key parameters are captured after conv1d processing in the slow PyTorch path:
- dt (Δ): step size / dwell time after softplus + dt_bias — [batch, seq, heads]
- B: input projection — [batch, seq, n_groups, state_size]
- C: output projection — [batch, seq, n_groups, state_size]
- A_log: log of state transition scalar (static parameter) — [heads]
"""

import torch
import torch.nn.functional as F


def compute_hidden_attention_mamba2(
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    n_heads: int,
    n_groups: int,
    time_step_limit: tuple[float, float] = (0.0, float("inf")),
) -> torch.Tensor:
    """Compute hidden attention matrix for one Mamba-2 layer.

    Args:
        dt: Raw dt from in_proj, before softplus — [batch, seq, heads]
        B: Input projection after conv1d — [batch, seq, n_groups * state_size]
        C: Output projection after conv1d — [batch, seq, n_groups * state_size]
        A_log: Log of state transition scalar — [heads]
        dt_bias: Bias added to dt before softplus — [heads]
        n_heads: Number of SSM heads
        n_groups: Number of groups for B/C
        time_step_limit: Clamp limits for dt

    Returns:
        Hidden attention matrix — [batch, heads, seq, seq], lower-triangular (causal)
    """
    # Use float32 for numerical stability
    dt = dt.float()
    B = B.float()
    C = C.float()
    A_log = A_log.float()
    dt_bias = dt_bias.float()

    batch, seq_len, _ = dt.shape
    state_size = B.shape[-1] // n_groups

    # Apply softplus and clamp to dt (matching model's torch_forward)
    dt = F.softplus(dt + dt_bias)
    dt = torch.clamp(dt, time_step_limit[0], time_step_limit[1])
    # dt: [batch, seq, heads]

    # A is negative (decay): A = -exp(A_log)
    A = -torch.exp(A_log)  # [heads]

    # dtA[t] = dt[t] * A for each position and head
    # [batch, seq, heads]
    dtA = dt * A.unsqueeze(0).unsqueeze(0)

    # Cumulative sum for exponential decay
    # cumsum_dtA[t] = Σ_{k=0}^{t} dtA[k]
    cumsum_dtA = torch.cumsum(dtA, dim=1)  # [batch, seq, heads]

    # Decay matrix: decay[i,j] = exp(cumsum[i] - cumsum[j])
    # This gives exp(Σ_{k=j+1}^{i} dtA[k]) when i >= j
    # [batch, heads, seq, seq]
    cumsum_dtA_t = cumsum_dtA.permute(0, 2, 1)  # [batch, heads, seq]
    diff = cumsum_dtA_t.unsqueeze(-1) - cumsum_dtA_t.unsqueeze(-2)

    # Apply causal mask BEFORE exp to avoid inf in upper triangle
    # For i >= j (causal), diff <= 0 (decay), so exp is in [0, 1]
    # For i < j (future), set to -inf so exp gives 0
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=dt.device, dtype=torch.bool)
    )
    diff = diff.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -float("inf"))

    # Clamp to prevent overflow for very long sequences
    diff = torch.clamp(diff, min=-30.0, max=0.0)
    decay = torch.exp(diff)  # [batch, heads, seq, seq]

    # Reshape B and C for dot product computation
    # B: [batch, seq, n_groups, state_size]
    B = B.reshape(batch, seq_len, n_groups, state_size)
    C = C.reshape(batch, seq_len, n_groups, state_size)

    # Expand groups to heads: repeat each group for heads_per_group heads
    heads_per_group = n_heads // n_groups
    # B: [batch, seq, heads, state_size]
    B = B.repeat_interleave(heads_per_group, dim=2)
    C = C.repeat_interleave(heads_per_group, dim=2)

    # Compute C[i] · B[j] dot products
    # C: [batch, heads, seq_i, state_size]
    # B: [batch, heads, seq_j, state_size]
    C_perm = C.permute(0, 2, 1, 3)  # [batch, heads, seq, state_size]
    B_perm = B.permute(0, 2, 1, 3)  # [batch, heads, seq, state_size]

    # CB[i,j] = C[i] · B[j] = sum over state_size
    CB = torch.matmul(C_perm, B_perm.transpose(-1, -2))  # [batch, heads, seq, seq]

    # dt[j] factor: broadcast dt for the j-th position
    # [batch, heads, 1, seq]
    dt_j = dt.permute(0, 2, 1).unsqueeze(-2)  # [batch, heads, 1, seq]

    # Final hidden attention: decay × CB × dt[j]
    hidden_attn = decay * CB * dt_j

    return hidden_attn


def compute_hidden_attention_mamba2_memory_efficient(
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    n_heads: int,
    n_groups: int,
    time_step_limit: tuple[float, float] = (0.0, float("inf")),
    head_batch_size: int = 8,
) -> torch.Tensor:
    """Memory-efficient version that processes heads in batches.

    Same signature as compute_hidden_attention_mamba2 but with reduced peak memory.
    """
    dt = dt.float()
    B = B.float()
    C = C.float()
    A_log = A_log.float()
    dt_bias = dt_bias.float()

    batch, seq_len, _ = dt.shape
    state_size = B.shape[-1] // n_groups

    dt = F.softplus(dt + dt_bias)
    dt = torch.clamp(dt, time_step_limit[0], time_step_limit[1])

    A = -torch.exp(A_log)

    B = B.reshape(batch, seq_len, n_groups, state_size)
    C = C.reshape(batch, seq_len, n_groups, state_size)
    heads_per_group = n_heads // n_groups
    B = B.repeat_interleave(heads_per_group, dim=2)
    C = C.repeat_interleave(heads_per_group, dim=2)

    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=dt.device, dtype=torch.float32)
    )

    result = torch.zeros(batch, n_heads, seq_len, seq_len, device=dt.device)

    for h_start in range(0, n_heads, head_batch_size):
        h_end = min(h_start + head_batch_size, n_heads)
        h_slice = slice(h_start, h_end)

        dt_h = dt[:, :, h_slice]  # [batch, seq, h_batch]
        dtA_h = dt_h * A[h_slice].unsqueeze(0).unsqueeze(0)
        cumsum_h = torch.cumsum(dtA_h, dim=1).permute(0, 2, 1)

        diff_h = cumsum_h.unsqueeze(-1) - cumsum_h.unsqueeze(-2)
        diff_h = diff_h.masked_fill(~causal_mask.bool().unsqueeze(0).unsqueeze(0), -float("inf"))
        diff_h = torch.clamp(diff_h, min=-30.0, max=0.0)
        decay_h = torch.exp(diff_h)

        C_h = C[:, :, h_slice, :].permute(0, 2, 1, 3)
        B_h = B[:, :, h_slice, :].permute(0, 2, 1, 3)
        CB_h = torch.matmul(C_h, B_h.transpose(-1, -2))

        dt_j_h = dt_h.permute(0, 2, 1).unsqueeze(-2)

        result[:, h_slice] = decay_h * CB_h * dt_j_h

    return result
