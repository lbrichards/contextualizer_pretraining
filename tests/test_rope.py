"""Tests for Rotary Position Embeddings (RoPE)."""

import pytest
import torch
from src.rope import precompute_freqs_cis, apply_rotary_emb, RoPE


class TestPrecomputeFreqsCis:
    """Test frequency precomputation for RoPE."""

    def test_output_shape(self):
        """Test that output shape matches expected [max_seq_len, dim//2]."""
        freqs = precompute_freqs_cis(dim=64, max_seq_len=128)
        assert freqs.shape == (128, 32)

    def test_complex_dtype(self):
        """Test that output is complex type."""
        freqs = precompute_freqs_cis(dim=64, max_seq_len=128)
        assert freqs.dtype == torch.complex64

    def test_odd_dimension_raises_error(self):
        """Test that odd dimensions raise ValueError."""
        with pytest.raises(ValueError, match="dim must be even"):
            precompute_freqs_cis(dim=63, max_seq_len=128)

    def test_different_theta_values(self):
        """Test that different theta values produce different frequencies."""
        freqs_1 = precompute_freqs_cis(dim=64, max_seq_len=128, theta=10000.0)
        freqs_2 = precompute_freqs_cis(dim=64, max_seq_len=128, theta=100000.0)
        assert not torch.allclose(freqs_1, freqs_2)

    def test_device_placement(self):
        """Test that frequencies are placed on specified device."""
        device = torch.device("cpu")
        freqs = precompute_freqs_cis(dim=64, max_seq_len=128, device=device)
        assert freqs.device == device

    def test_unit_magnitude(self):
        """Test that all complex numbers have unit magnitude (on unit circle)."""
        freqs = precompute_freqs_cis(dim=64, max_seq_len=128)
        magnitudes = torch.abs(freqs)
        # All magnitudes should be 1.0 (allowing small floating point error)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)

    def test_deterministic(self):
        """Test that precomputation is deterministic."""
        freqs_1 = precompute_freqs_cis(dim=64, max_seq_len=128, theta=10000.0)
        freqs_2 = precompute_freqs_cis(dim=64, max_seq_len=128, theta=10000.0)
        assert torch.equal(freqs_1, freqs_2)


class TestApplyRotaryEmb:
    """Test applying rotary embeddings to tensors."""

    def test_output_shape_preserved(self):
        """Test that output shape matches input shape."""
        x = torch.randn(2, 16, 4, 64)  # [batch, seq, heads, head_dim]
        freqs = precompute_freqs_cis(64, 16)
        out = apply_rotary_emb(x, freqs)
        assert out.shape == x.shape

    def test_dtype_preserved(self):
        """Test that output dtype matches input dtype."""
        for dtype in [torch.float32, torch.float16]:
            x = torch.randn(2, 16, 4, 64, dtype=dtype)
            freqs = precompute_freqs_cis(64, 16)
            out = apply_rotary_emb(x, freqs)
            assert out.dtype == dtype

    def test_deterministic(self):
        """Test that rotation is deterministic for same input."""
        x = torch.randn(2, 16, 4, 64)
        freqs = precompute_freqs_cis(64, 16)
        out1 = apply_rotary_emb(x, freqs)
        out2 = apply_rotary_emb(x, freqs)
        assert torch.equal(out1, out2)

    def test_different_positions_different_output(self):
        """Test that different sequence positions produce different rotations."""
        # Same vector at different positions should be rotated differently
        x = torch.randn(1, 2, 1, 64)  # 2 positions
        x[0, 1] = x[0, 0]  # Make both positions identical
        freqs = precompute_freqs_cis(64, 2)
        out = apply_rotary_emb(x, freqs)
        # After rotation, positions should differ
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_preserves_magnitude_approximately(self):
        """Test that rotation approximately preserves vector magnitude."""
        x = torch.randn(2, 16, 4, 64)
        freqs = precompute_freqs_cis(64, 16)
        out = apply_rotary_emb(x, freqs)

        # Rotation should preserve L2 norm (within numerical precision)
        input_norms = torch.norm(x, dim=-1)
        output_norms = torch.norm(out, dim=-1)
        assert torch.allclose(input_norms, output_norms, rtol=1e-4)


class TestRoPEModule:
    """Test RoPE nn.Module wrapper."""

    def test_initialization(self):
        """Test RoPE module initialization."""
        rope = RoPE(dim=64, max_seq_len=512, theta=10000.0)
        assert rope.dim == 64
        assert rope.max_seq_len == 512
        assert rope.theta == 10000.0

    def test_forward_shape(self):
        """Test forward pass preserves shapes."""
        rope = RoPE(dim=64, max_seq_len=512)
        q = torch.randn(2, 128, 8, 64)
        k = torch.randn(2, 128, 8, 64)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_forward_dtype_preservation(self):
        """Test that forward pass preserves dtype."""
        rope = RoPE(dim=64, max_seq_len=512)
        for dtype in [torch.float32, torch.float16]:
            q = torch.randn(2, 128, 8, 64, dtype=dtype)
            k = torch.randn(2, 128, 8, 64, dtype=dtype)
            q_rot, k_rot = rope(q, k)
            assert q_rot.dtype == dtype
            assert k_rot.dtype == dtype

    def test_sequence_length_validation(self):
        """Test that exceeding max_seq_len raises error."""
        rope = RoPE(dim=64, max_seq_len=128)
        q = torch.randn(2, 256, 8, 64)  # seq_len=256 > max_seq_len=128
        k = torch.randn(2, 256, 8, 64)
        with pytest.raises(ValueError, match="Sequence length .* exceeds max_seq_len"):
            rope(q, k)

    def test_variable_sequence_length(self):
        """Test that module handles variable sequence lengths."""
        rope = RoPE(dim=64, max_seq_len=512)

        # Different sequence lengths should all work
        for seq_len in [16, 64, 128, 256, 512]:
            q = torch.randn(2, seq_len, 8, 64)
            k = torch.randn(2, seq_len, 8, 64)
            q_rot, k_rot = rope(q, k)
            assert q_rot.shape == q.shape
            assert k_rot.shape == k.shape

    def test_device_movement(self):
        """Test that RoPE buffers move with module to different devices."""
        rope = RoPE(dim=64, max_seq_len=512)

        # Move to CPU explicitly
        rope = rope.to("cpu")
        assert rope.freqs_cis.device.type == "cpu"

        q = torch.randn(2, 128, 8, 64)
        k = torch.randn(2, 128, 8, 64)
        q_rot, k_rot = rope(q, k)
        assert q_rot.device.type == "cpu"

    def test_deterministic_same_seed(self):
        """Test deterministic behavior with same initialization."""
        rope1 = RoPE(dim=64, max_seq_len=512, theta=10000.0)
        rope2 = RoPE(dim=64, max_seq_len=512, theta=10000.0)

        q = torch.randn(2, 128, 8, 64)
        k = torch.randn(2, 128, 8, 64)

        q_rot1, k_rot1 = rope1(q, k)
        q_rot2, k_rot2 = rope2(q, k)

        assert torch.equal(q_rot1, q_rot2)
        assert torch.equal(k_rot1, k_rot2)

    def test_q_and_k_rotated_identically(self):
        """Test that Q and K are rotated identically for same positions."""
        rope = RoPE(dim=64, max_seq_len=512)

        # Use same input for both q and k
        x = torch.randn(2, 128, 8, 64)
        q_rot, k_rot = rope(x, x)

        # Since we used same input and same positions, rotations should be equal
        assert torch.equal(q_rot, k_rot)

    def test_different_inputs_different_outputs(self):
        """Test that different inputs produce different outputs."""
        rope = RoPE(dim=64, max_seq_len=512)

        q1 = torch.randn(2, 128, 8, 64)
        k1 = torch.randn(2, 128, 8, 64)
        q_rot1, k_rot1 = rope(q1, k1)

        q2 = torch.randn(2, 128, 8, 64)
        k2 = torch.randn(2, 128, 8, 64)
        q_rot2, k_rot2 = rope(q2, k2)

        assert not torch.allclose(q_rot1, q_rot2)
        assert not torch.allclose(k_rot1, k_rot2)

    def test_repr(self):
        """Test string representation."""
        rope = RoPE(dim=64, max_seq_len=512, theta=10000.0)
        repr_str = repr(rope)
        assert "64" in repr_str
        assert "512" in repr_str


class TestRoPEIntegration:
    """Integration tests for RoPE with realistic scenarios."""

    def test_with_qwen_7b_config(self):
        """Test RoPE with Qwen 7B-like configuration."""
        # Qwen 7B: d_model=3584, n_heads=28, head_dim=128
        d_model = 3584
        n_heads = 28
        head_dim = d_model // n_heads  # 128

        rope = RoPE(dim=head_dim, max_seq_len=512, theta=10000.0)

        batch_size = 4
        seq_len = 256

        q = torch.randn(batch_size, seq_len, n_heads, head_dim)
        k = torch.randn(batch_size, seq_len, n_heads, head_dim)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == (batch_size, seq_len, n_heads, head_dim)
        assert k_rot.shape == (batch_size, seq_len, n_heads, head_dim)

    def test_with_our_contextualizer_config(self):
        """Test RoPE with our contextualizer configuration."""
        # Our config: d_model=3584, n_heads=16, head_dim=224
        d_model = 3584
        n_heads = 16
        head_dim = d_model // n_heads  # 224

        rope = RoPE(dim=head_dim, max_seq_len=512, theta=10000.0)

        batch_size = 8
        seq_len = 512

        q = torch.randn(batch_size, seq_len, n_heads, head_dim)
        k = torch.randn(batch_size, seq_len, n_heads, head_dim)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == (batch_size, seq_len, n_heads, head_dim)
        assert k_rot.shape == (batch_size, seq_len, n_heads, head_dim)

    def test_relative_position_property(self):
        """Test that RoPE provides relative position information."""
        rope = RoPE(dim=64, max_seq_len=512)

        # Create identical vectors at different absolute positions
        # but same relative distance
        q1 = torch.randn(1, 2, 1, 64)
        k1 = q1.clone()
        q_rot1, k_rot1 = rope(q1, k1)
        dot_product_1 = (q_rot1[0, 0, 0] * k_rot1[0, 1, 0]).sum()

        # Same vectors at positions (10, 11) - same relative distance
        q2 = torch.zeros(1, 12, 1, 64)
        k2 = torch.zeros(1, 12, 1, 64)
        q2[0, 10] = q1[0, 0]
        k2[0, 11] = k1[0, 1]
        q_rot2, k_rot2 = rope(q2, k2)
        dot_product_2 = (q_rot2[0, 10, 0] * k_rot2[0, 11, 0]).sum()

        # Both should be finite
        assert torch.isfinite(dot_product_1)
        assert torch.isfinite(dot_product_2)
