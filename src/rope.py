"""Rotary Position Embeddings (RoPE) for transformer attention.

RoPE encodes position information by rotating Q and K vectors in pairs of dimensions.
This is the position encoding used by Qwen and many modern LLMs.

References:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - Qwen2: https://github.com/QwenLM/Qwen2
"""

import torch
import torch.nn as nn
from typing import Tuple


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Precompute complex exponentials for RoPE rotation.

    Args:
        dim: Embedding dimension (must be even)
        max_seq_len: Maximum sequence length to support
        theta: Base for computing rotation frequencies (default 10000)
        device: Device to place tensor on

    Returns:
        Complex tensor of shape [max_seq_len, dim//2] containing rotation coefficients

    Raises:
        ValueError: If dim is not even

    Examples:
        >>> freqs = precompute_freqs_cis(64, 128, theta=10000.0)
        >>> freqs.shape
        torch.Size([128, 32])
        >>> freqs.dtype
        torch.complex64
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")

    # Compute frequencies for each dimension pair
    # freqs[i] = 1 / (theta^(2i/dim)) for i in [0, dim/2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Create position indices [0, 1, ..., max_seq_len-1]
    t = torch.arange(max_seq_len, device=device)

    # Outer product: [max_seq_len, dim//2]
    # Each position gets rotated by its own set of frequencies
    freqs = torch.outer(t, freqs)

    # Convert to complex exponentials e^(i*theta)
    # This gives us the rotation in complex plane
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to input tensor.

    Args:
        x: Input tensor of shape [..., seq_len, n_heads, head_dim]
        freqs_cis: Precomputed rotation coefficients [seq_len, head_dim//2]

    Returns:
        Tensor with same shape as input, with rotary embeddings applied

    Examples:
        >>> x = torch.randn(2, 16, 4, 64)  # [batch, seq_len, heads, head_dim]
        >>> freqs = precompute_freqs_cis(64, 16)
        >>> out = apply_rotary_emb(x, freqs)
        >>> out.shape
        torch.Size([2, 16, 4, 64])
    """
    # Reshape input to complex pairs: [..., seq_len, n_heads, head_dim//2, 2]
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)

    # Convert pairs to complex numbers
    x_complex = torch.view_as_complex(x_reshaped)

    # Broadcast freqs_cis to match x_complex shape
    # freqs_cis: [seq_len, head_dim//2] -> [..., seq_len, 1, head_dim//2]
    freqs_cis = freqs_cis.view(*([1] * (x_complex.ndim - 3)), freqs_cis.shape[0], 1, freqs_cis.shape[1])

    # Complex multiplication performs rotation
    x_rotated = x_complex * freqs_cis

    # Convert back to real representation
    x_out = torch.view_as_real(x_rotated).flatten(-2)

    return x_out.type_as(x)


class RoPE(nn.Module):
    """
    Rotary Position Embedding module.

    This module precomputes and applies rotary position embeddings to query and key tensors
    in transformer attention. RoPE provides relative position information without requiring
    learned position embeddings.

    Args:
        dim: Dimension per attention head (head_dim)
        max_seq_len: Maximum sequence length to support
        theta: Base for computing rotation frequencies (default 10000.0)

    Examples:
        >>> rope = RoPE(dim=64, max_seq_len=512)
        >>> q = torch.randn(2, 128, 8, 64)  # [batch, seq, heads, head_dim]
        >>> k = torch.randn(2, 128, 8, 64)
        >>> q_rot, k_rot = rope(q, k)
        >>> q_rot.shape, k_rot.shape
        (torch.Size([2, 128, 8, 64]), torch.Size([2, 128, 8, 64]))
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Register as buffer so it moves with the model to different devices
        freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor [..., seq_len, n_heads, head_dim]
            k: Key tensor [..., seq_len, n_heads, head_dim]
            seq_len: Actual sequence length (uses q.shape[-3] if None)

        Returns:
            Tuple of (rotated_q, rotated_k)

        Raises:
            ValueError: If sequence length exceeds max_seq_len
        """
        if seq_len is None:
            seq_len = q.shape[-3]

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}. "
                f"Create RoPE with larger max_seq_len."
            )

        # Get frequencies for this sequence length
        freqs_cis = self.freqs_cis[:seq_len]

        # Apply to both q and k
        q_rotated = apply_rotary_emb(q, freqs_cis)
        k_rotated = apply_rotary_emb(k, freqs_cis)

        return q_rotated, k_rotated

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, theta={self.theta}"
