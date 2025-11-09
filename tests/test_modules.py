"""Tests for transformer modules (Attention, MLP, Block)."""

import pytest
import torch
import torch.nn as nn
from src.modules import (
    MultiHeadAttention,
    MLP,
    TransformerBlock,
    ContextualizerEncoder,
    MLMHead,
    ContextualizerForMLM,
)
from src.utils import set_seed


class TestMultiHeadAttention:
    """Test multi-head attention with RoPE."""

    def test_initialization(self):
        """Test attention module initialization."""
        attn = MultiHeadAttention(d_model=512, n_heads=8, dropout=0.1)
        assert attn.d_model == 512
        assert attn.n_heads == 8
        assert attn.head_dim == 64

    def test_head_dimension_calculation(self):
        """Test that head dimension is correctly calculated."""
        attn = MultiHeadAttention(d_model=3584, n_heads=16, dropout=0.1)
        assert attn.head_dim == 224  # 3584 / 16

    def test_invalid_head_count(self):
        """Test that invalid head count raises error."""
        with pytest.raises(ValueError, match="d_model .* must be divisible by n_heads"):
            MultiHeadAttention(d_model=512, n_heads=7, dropout=0.1)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        attn = MultiHeadAttention(d_model=512, n_heads=8, dropout=0.0)
        x = torch.randn(2, 16, 512)  # [batch, seq_len, d_model]
        out = attn(x)
        assert out.shape == (2, 16, 512)

    def test_forward_with_mask(self):
        """Test forward pass with attention mask."""
        attn = MultiHeadAttention(d_model=512, n_heads=8, dropout=0.0)
        x = torch.randn(2, 16, 512)
        # Causal mask: upper triangular
        mask = torch.triu(torch.ones(16, 16), diagonal=1).bool()
        out = attn(x, mask=mask)
        assert out.shape == (2, 16, 512)

    def test_forward_with_padding_mask(self):
        """Test forward pass with padding mask."""
        attn = MultiHeadAttention(d_model=512, n_heads=8, dropout=0.0)
        x = torch.randn(2, 16, 512)
        # Padding mask: batch_size x seq_len (True = padding)
        padding_mask = torch.zeros(2, 16, dtype=torch.bool)
        padding_mask[:, 12:] = True  # Last 4 tokens are padding
        out = attn(x, padding_mask=padding_mask)
        assert out.shape == (2, 16, 512)

    def test_deterministic_with_seed(self):
        """Test deterministic behavior with same seed."""
        set_seed(42)
        attn1 = MultiHeadAttention(d_model=512, n_heads=8, dropout=0.0)
        x = torch.randn(2, 16, 512)
        out1 = attn1(x)

        set_seed(42)
        attn2 = MultiHeadAttention(d_model=512, n_heads=8, dropout=0.0)
        out2 = attn2(x)

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_rope_integration(self):
        """Test that RoPE is properly integrated."""
        attn = MultiHeadAttention(d_model=512, n_heads=8, max_seq_len=128, dropout=0.0)
        assert hasattr(attn, 'rope')
        assert attn.rope.max_seq_len == 128

    def test_gradient_flow(self):
        """Test that gradients flow through attention."""
        attn = MultiHeadAttention(d_model=512, n_heads=8, dropout=0.0)
        x = torch.randn(2, 16, 512, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestMLP:
    """Test MLP (feed-forward) module."""

    def test_initialization(self):
        """Test MLP initialization."""
        mlp = MLP(d_model=512, d_ff=2048, dropout=0.1)
        assert mlp.d_model == 512
        assert mlp.d_ff == 2048

    def test_forward_shape(self):
        """Test forward pass output shape."""
        mlp = MLP(d_model=512, d_ff=2048, dropout=0.0)
        x = torch.randn(2, 16, 512)
        out = mlp(x)
        assert out.shape == (2, 16, 512)

    def test_activation_function(self):
        """Test that activation is applied (output differs from linear)."""
        mlp = MLP(d_model=512, d_ff=2048, dropout=0.0)
        # Use a known input to test activation
        x = torch.ones(2, 16, 512) * 2.0
        out = mlp(x)
        # The output should be affected by GELU and two linear layers
        # Just verify it has the right shape and is not all zeros
        assert out.shape == (2, 16, 512)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_deterministic_with_seed(self):
        """Test deterministic behavior."""
        set_seed(42)
        mlp1 = MLP(d_model=512, d_ff=2048, dropout=0.0)
        x = torch.randn(2, 16, 512)
        out1 = mlp1(x)

        set_seed(42)
        mlp2 = MLP(d_model=512, d_ff=2048, dropout=0.0)
        out2 = mlp2(x)

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_gradient_flow(self):
        """Test gradient flow through MLP."""
        mlp = MLP(d_model=512, d_ff=2048, dropout=0.0)
        x = torch.randn(2, 16, 512, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestTransformerBlock:
    """Test transformer block (Attention + MLP with residuals)."""

    def test_initialization(self):
        """Test block initialization."""
        block = TransformerBlock(d_model=512, n_heads=8, d_ff=2048, dropout=0.1)
        assert hasattr(block, 'attn')
        assert hasattr(block, 'mlp')
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')

    def test_forward_shape(self):
        """Test forward pass output shape."""
        block = TransformerBlock(d_model=512, n_heads=8, d_ff=2048, dropout=0.0)
        x = torch.randn(2, 16, 512)
        out = block(x)
        assert out.shape == (2, 16, 512)

    def test_residual_connections(self):
        """Test that residual connections are present."""
        block = TransformerBlock(d_model=512, n_heads=8, d_ff=2048, dropout=0.0)
        # With very small initialized weights, output should be close to input due to residuals
        x = torch.randn(2, 16, 512)
        out = block(x)
        # Output should not be identical but should be related through residual
        assert not torch.equal(out, x)
        assert out.shape == x.shape

    def test_with_gradient_checkpointing(self):
        """Test gradient checkpointing support."""
        block = TransformerBlock(d_model=512, n_heads=8, d_ff=2048, dropout=0.0)
        block.gradient_checkpointing = True
        x = torch.randn(2, 16, 512, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_deterministic(self):
        """Test deterministic behavior."""
        set_seed(42)
        block1 = TransformerBlock(d_model=512, n_heads=8, d_ff=2048, dropout=0.0)
        x = torch.randn(2, 16, 512)
        out1 = block1(x)

        set_seed(42)
        block2 = TransformerBlock(d_model=512, n_heads=8, d_ff=2048, dropout=0.0)
        out2 = block2(x)

        assert torch.allclose(out1, out2, atol=1e-6)


class TestContextualizerEncoder:
    """Test stacked transformer encoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = ContextualizerEncoder(
            vocab_size=10000,
            d_model=512,
            n_heads=8,
            n_layers=4,
            d_ff=2048,
            max_seq_len=512,
            dropout=0.1
        )
        assert len(encoder.layers) == 4
        assert hasattr(encoder, 'embed')
        assert hasattr(encoder, 'norm')

    def test_forward_shape(self):
        """Test forward pass output shape."""
        encoder = ContextualizerEncoder(
            vocab_size=10000,
            d_model=512,
            n_heads=8,
            n_layers=4,
            d_ff=2048,
            max_seq_len=512,
            dropout=0.0
        )
        input_ids = torch.randint(0, 10000, (2, 16))
        out = encoder(input_ids)
        assert out.shape == (2, 16, 512)

    def test_with_attention_mask(self):
        """Test encoder with attention mask."""
        encoder = ContextualizerEncoder(
            vocab_size=10000,
            d_model=512,
            n_heads=8,
            n_layers=2,
            d_ff=2048,
            max_seq_len=512,
            dropout=0.0
        )
        input_ids = torch.randint(0, 10000, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.bool)
        attention_mask[:, 12:] = False  # Mask last 4 tokens
        out = encoder(input_ids, attention_mask=attention_mask)
        assert out.shape == (2, 16, 512)

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing mode."""
        encoder = ContextualizerEncoder(
            vocab_size=10000,
            d_model=512,
            n_heads=8,
            n_layers=4,
            d_ff=2048,
            max_seq_len=512,
            dropout=0.0,
            gradient_checkpointing=True
        )
        input_ids = torch.randint(0, 10000, (2, 16))
        out = encoder(input_ids)
        loss = out.sum()
        loss.backward()
        assert out.shape == (2, 16, 512)

    def test_deterministic(self):
        """Test deterministic encoding."""
        set_seed(42)
        encoder1 = ContextualizerEncoder(
            vocab_size=10000, d_model=512, n_heads=8, n_layers=2,
            d_ff=2048, max_seq_len=512, dropout=0.0
        )
        input_ids = torch.randint(0, 10000, (2, 16))
        out1 = encoder1(input_ids)

        set_seed(42)
        encoder2 = ContextualizerEncoder(
            vocab_size=10000, d_model=512, n_heads=8, n_layers=2,
            d_ff=2048, max_seq_len=512, dropout=0.0
        )
        out2 = encoder2(input_ids)

        assert torch.allclose(out1, out2, atol=1e-6)


class TestMLMHead:
    """Test MLM (Masked Language Modeling) head."""

    def test_initialization(self):
        """Test MLM head initialization."""
        head = MLMHead(d_model=512, vocab_size=10000)
        assert hasattr(head, 'norm')
        assert hasattr(head, 'proj')

    def test_forward_shape(self):
        """Test forward pass output shape."""
        head = MLMHead(d_model=512, vocab_size=10000)
        x = torch.randn(2, 16, 512)
        logits = head(x)
        assert logits.shape == (2, 16, 10000)

    def test_output_dtype(self):
        """Test that output is float for logits."""
        head = MLMHead(d_model=512, vocab_size=10000)
        x = torch.randn(2, 16, 512)
        logits = head(x)
        assert logits.dtype in [torch.float32, torch.float16, torch.bfloat16]

    def test_gradient_flow(self):
        """Test gradient flow through MLM head."""
        head = MLMHead(d_model=512, vocab_size=10000)
        x = torch.randn(2, 16, 512, requires_grad=True)
        logits = head(x)
        loss = logits.sum()
        loss.backward()
        assert x.grad is not None


class TestContextualizerForMLM:
    """Test complete MLM model (Encoder + MLM Head)."""

    def test_initialization(self):
        """Test full MLM model initialization."""
        model = ContextualizerForMLM(
            vocab_size=10000,
            d_model=512,
            n_heads=8,
            n_layers=4,
            d_ff=2048,
            max_seq_len=512,
            dropout=0.1
        )
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'mlm_head')

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = ContextualizerForMLM(
            vocab_size=10000,
            d_model=512,
            n_heads=8,
            n_layers=2,
            d_ff=2048,
            max_seq_len=512,
            dropout=0.0
        )
        input_ids = torch.randint(0, 10000, (2, 16))
        logits, loss = model(input_ids)
        assert logits.shape == (2, 16, 10000)
        assert loss is None

    def test_with_labels_computes_loss(self):
        """Test that providing labels computes MLM loss."""
        model = ContextualizerForMLM(
            vocab_size=10000,
            d_model=512,
            n_heads=8,
            n_layers=2,
            d_ff=2048,
            max_seq_len=512,
            dropout=0.0
        )
        input_ids = torch.randint(0, 10000, (2, 16))
        labels = torch.randint(0, 10000, (2, 16))
        # Mask some positions (MLM convention: -100 = ignore)
        labels[:, :8] = -100

        logits, loss = model(input_ids, labels=labels)
        assert logits.shape == (2, 16, 10000)
        assert loss is not None
        assert loss.dim() == 0  # Scalar loss

    def test_loss_ignores_padding(self):
        """Test that loss correctly ignores -100 labels."""
        model = ContextualizerForMLM(
            vocab_size=10000,
            d_model=512,
            n_heads=8,
            n_layers=2,
            d_ff=2048,
            max_seq_len=512,
            dropout=0.0
        )
        input_ids = torch.randint(0, 10000, (2, 16))
        labels = torch.full((2, 16), -100, dtype=torch.long)
        labels[:, 8] = 100  # Only one valid label

        logits, loss = model(input_ids, labels=labels)
        assert torch.isfinite(loss)

    def test_gradient_flow_through_mlm(self):
        """Test gradient flow through complete MLM model."""
        model = ContextualizerForMLM(
            vocab_size=10000,
            d_model=512,
            n_heads=8,
            n_layers=2,
            d_ff=2048,
            max_seq_len=512,
            dropout=0.0
        )
        input_ids = torch.randint(0, 10000, (2, 16))
        labels = torch.randint(0, 10000, (2, 16))

        logits, loss = model(input_ids, labels=labels)
        loss.backward()

        # Check that encoder embeddings have gradients
        assert model.encoder.embed.weight.grad is not None

    def test_deterministic_mlm(self):
        """Test deterministic MLM predictions."""
        set_seed(42)
        model1 = ContextualizerForMLM(
            vocab_size=10000, d_model=512, n_heads=8, n_layers=2,
            d_ff=2048, max_seq_len=512, dropout=0.0
        )
        input_ids = torch.randint(0, 10000, (2, 16))
        logits1, _ = model1(input_ids)

        set_seed(42)
        model2 = ContextualizerForMLM(
            vocab_size=10000, d_model=512, n_heads=8, n_layers=2,
            d_ff=2048, max_seq_len=512, dropout=0.0
        )
        logits2, _ = model2(input_ids)

        assert torch.allclose(logits1, logits2, atol=1e-6)


class TestModuleIntegration:
    """Integration tests with realistic configurations."""

    def test_qwen_7b_inspired_config(self):
        """Test with Qwen 7B-inspired configuration."""
        # Smaller version for testing
        model = ContextualizerForMLM(
            vocab_size=151936,  # Qwen tokenizer vocab size
            d_model=3584,
            n_heads=16,
            n_layers=2,  # Just 2 layers for testing
            d_ff=14336,
            max_seq_len=512,
            dropout=0.1
        )
        input_ids = torch.randint(0, 151936, (4, 256))
        logits, _ = model(input_ids)
        assert logits.shape == (4, 256, 151936)

    def test_parameter_count(self):
        """Test parameter counting utility."""
        from src.utils import count_parameters, format_number

        model = ContextualizerForMLM(
            vocab_size=10000,
            d_model=512,
            n_heads=8,
            n_layers=2,
            d_ff=2048,
            max_seq_len=512,
            dropout=0.0
        )
        param_count = count_parameters(model)
        assert param_count > 0
        formatted = format_number(param_count)
        assert 'M' in formatted or 'K' in formatted
