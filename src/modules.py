"""Transformer modules for contextualizer encoder and MLM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint

from src.rope import RoPE


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with RoPE (Rotary Position Embeddings).

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        max_seq_len: Maximum sequence length for RoPE
        dropout: Dropout probability
        rope_theta: RoPE theta parameter

    Examples:
        >>> attn = MultiHeadAttention(d_model=512, n_heads=8)
        >>> x = torch.randn(2, 16, 512)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([2, 16, 512])
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # RoPE for position encoding
        self.rope = RoPE(dim=self.head_dim, max_seq_len=max_seq_len, theta=rope_theta)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [seq_len, seq_len] or [batch, seq_len, seq_len]
            padding_mask: Padding mask [batch_size, seq_len] (True = padding)

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: [batch, seq_len, n_heads, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k, seq_len=seq_len)

        # Transpose for attention: [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores: [batch, n_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply masks if provided
        if mask is not None:
            # Expand mask to [batch, n_heads, seq_len, seq_len] if needed
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))

        if padding_mask is not None:
            # padding_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [batch, n_heads, seq_len, head_dim]

        # Reshape back: [batch, seq_len, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        out = self.o_proj(out)

        return out


class MLP(nn.Module):
    """
    Feed-forward MLP with GELU activation.

    Args:
        d_model: Model dimension
        d_ff: Hidden dimension of feed-forward layer
        dropout: Dropout probability

    Examples:
        >>> mlp = MLP(d_model=512, d_ff=2048)
        >>> x = torch.randn(2, 16, 512)
        >>> out = mlp(x)
        >>> out.shape
        torch.Size([2, 16, 512])
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MLP.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-norm architecture.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        rope_theta: RoPE theta parameter

    Examples:
        >>> block = TransformerBlock(d_model=512, n_heads=8, d_ff=2048)
        >>> x = torch.randn(2, 16, 512)
        >>> out = block(x)
        >>> out.shape
        torch.Size([2, 16, 512])
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope_theta=rope_theta,
        )
        self.mlp = MLP(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of transformer block.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask
            padding_mask: Padding mask

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Pre-norm architecture with residual connections
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            x = x + checkpoint(self._attn_block, x, mask, padding_mask, use_reentrant=False)
            x = x + checkpoint(self._mlp_block, x, use_reentrant=False)
        else:
            x = x + self._attn_block(x, mask, padding_mask)
            x = x + self._mlp_block(x)

        return x

    def _attn_block(self, x, mask, padding_mask):
        """Attention block for checkpointing."""
        return self.attn(self.norm1(x), mask=mask, padding_mask=padding_mask)

    def _mlp_block(self, x):
        """MLP block for checkpointing."""
        return self.mlp(self.norm2(x))


class ContextualizerEncoder(nn.Module):
    """
    Stack of transformer blocks for contextual encoding.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        rope_theta: RoPE theta parameter
        gradient_checkpointing: Whether to use gradient checkpointing

    Examples:
        >>> encoder = ContextualizerEncoder(
        ...     vocab_size=10000, d_model=512, n_heads=8, n_layers=4, d_ff=2048
        ... )
        >>> input_ids = torch.randint(0, 10000, (2, 16))
        >>> out = encoder(input_ids)
        >>> out.shape
        torch.Size([2, 16, 512])
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout,
                rope_theta=rope_theta,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            for layer in self.layers:
                layer.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of encoder.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] (True = attend, False = ignore)

        Returns:
            Hidden states [batch_size, seq_len, d_model]
        """
        # Embed tokens
        x = self.embed(input_ids)

        # Convert attention mask to padding mask (invert logic)
        padding_mask = None
        if attention_mask is not None:
            padding_mask = ~attention_mask  # True = padding

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)

        # Final normalization
        x = self.norm(x)

        return x


class MLMHead(nn.Module):
    """
    MLM (Masked Language Modeling) prediction head.

    Args:
        d_model: Model dimension
        vocab_size: Size of vocabulary

    Examples:
        >>> head = MLMHead(d_model=512, vocab_size=10000)
        >>> x = torch.randn(2, 16, 512)
        >>> logits = head(x)
        >>> logits.shape
        torch.Size([2, 16, 10000])
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MLM head.

        Args:
            x: Hidden states [batch_size, seq_len, d_model]

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        x = self.norm(x)
        logits = self.proj(x)
        return logits


class ContextualizerForMLM(nn.Module):
    """
    Complete model for MLM pretraining (Encoder + MLM Head).

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        rope_theta: RoPE theta parameter
        gradient_checkpointing: Whether to use gradient checkpointing

    Examples:
        >>> model = ContextualizerForMLM(
        ...     vocab_size=10000, d_model=512, n_heads=8, n_layers=4, d_ff=2048
        ... )
        >>> input_ids = torch.randint(0, 10000, (2, 16))
        >>> logits = model(input_ids)
        >>> logits.shape
        torch.Size([2, 16, 10000])
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.encoder = ContextualizerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope_theta=rope_theta,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.mlm_head = MLMHead(d_model=d_model, vocab_size=vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of MLM model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for MLM loss [batch_size, seq_len] (use -100 to ignore)

        Returns:
            Tuple of (logits, loss) where loss is None if labels not provided
        """
        # Encode
        hidden_states = self.encoder(input_ids, attention_mask=attention_mask)

        # Get MLM logits
        logits = self.mlm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,  # Standard MLM convention
            )
            return logits, loss

        return logits, loss
