"""Tests for Qwen constants and validation."""

import pytest
from src.constants import (
    QWEN_PRESETS,
    get_valid_hidden_dims,
    validate_hidden_dim,
    get_qwen_preset,
    infer_qwen_size,
    validate_model_config,
)


class TestQwenPresets:
    """Test Qwen preset constants."""

    def test_presets_exist(self):
        """Test that all expected Qwen sizes are defined."""
        expected_sizes = ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]
        assert set(QWEN_PRESETS.keys()) == set(expected_sizes)

    def test_preset_structure(self):
        """Test that each preset has required keys."""
        required_keys = {"hidden", "layers", "heads", "ff"}
        for size, preset in QWEN_PRESETS.items():
            assert set(preset.keys()) == required_keys, f"Size {size} missing keys"

    def test_qwen_7b_preset(self):
        """Test specific values for Qwen 7B (our target model)."""
        preset = QWEN_PRESETS["7B"]
        assert preset["hidden"] == 3584
        assert preset["layers"] == 28
        assert preset["heads"] == 28
        assert preset["ff"] == 18944

    def test_hidden_dims_unique(self):
        """Test that hidden dimensions are unique across presets."""
        hidden_dims = [p["hidden"] for p in QWEN_PRESETS.values()]
        # Note: 5120 appears in both 14B and 32B
        assert len(set(hidden_dims)) == 6  # 896, 1536, 2048, 3584, 5120, 8192


class TestValidHiddenDims:
    """Test helper functions for hidden dimensions."""

    def test_get_valid_hidden_dims(self):
        """Test getting sorted list of valid dimensions."""
        dims = get_valid_hidden_dims()
        assert isinstance(dims, list)
        assert dims == sorted(dims)
        assert 3584 in dims  # Qwen 7B
        assert 2048 in dims  # Qwen 3B

    def test_valid_dims_count(self):
        """Test that we have correct number of unique dimensions."""
        dims = get_valid_hidden_dims()
        assert len(dims) == 6


class TestValidateHiddenDim:
    """Test hidden dimension validation."""

    def test_validate_valid_dimension(self):
        """Test validation passes for valid Qwen dimensions."""
        validate_hidden_dim(3584)  # Qwen 7B
        validate_hidden_dim(2048)  # Qwen 3B
        validate_hidden_dim(896)   # Qwen 0.5B

    def test_validate_invalid_dimension_strict(self):
        """Test validation fails for non-Qwen dimensions in strict mode."""
        with pytest.raises(ValueError, match="hidden size 768 not in Qwen presets"):
            validate_hidden_dim(768, strict=True)

    def test_validate_invalid_dimension_non_strict(self, capsys):
        """Test validation warns for non-Qwen dimensions in non-strict mode."""
        validate_hidden_dim(768, strict=False)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "768" in captured.out


class TestGetQwenPreset:
    """Test getting presets by model size."""

    def test_get_valid_preset(self):
        """Test retrieving valid preset."""
        preset = get_qwen_preset("7B")
        assert preset["hidden"] == 3584
        assert preset["layers"] == 28

    def test_get_all_presets(self):
        """Test retrieving all presets."""
        for size in ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]:
            preset = get_qwen_preset(size)
            assert "hidden" in preset
            assert "layers" in preset

    def test_get_invalid_preset(self):
        """Test error on invalid model size."""
        with pytest.raises(KeyError, match="Model size '10B' not found"):
            get_qwen_preset("10B")

    def test_preset_copy(self):
        """Test that returned preset is a copy (not reference)."""
        preset1 = get_qwen_preset("7B")
        preset2 = get_qwen_preset("7B")
        preset1["hidden"] = 9999
        assert preset2["hidden"] == 3584  # Original unchanged


class TestInferQwenSize:
    """Test inferring model size from hidden dimension."""

    def test_infer_7b(self):
        """Test inferring Qwen 7B from dimension."""
        assert infer_qwen_size(3584) == "7B"

    def test_infer_3b(self):
        """Test inferring Qwen 3B from dimension."""
        assert infer_qwen_size(2048) == "3B"

    def test_infer_all_sizes(self):
        """Test inferring all model sizes."""
        expected = {
            896: "0.5B",
            1536: "1.5B",
            2048: "3B",
            3584: "7B",
        }
        for hidden, size in expected.items():
            assert infer_qwen_size(hidden) == size

    def test_infer_ambiguous_dimension(self):
        """Test that 5120 (used by both 14B and 32B) returns one of them."""
        result = infer_qwen_size(5120)
        assert result in ["14B", "32B"]

    def test_infer_invalid_dimension(self):
        """Test error on invalid dimension."""
        with pytest.raises(ValueError, match="Hidden dim 768 doesn't match"):
            infer_qwen_size(768)


class TestValidateModelConfig:
    """Test model configuration validation."""

    def test_validate_valid_config(self):
        """Test validation passes for valid configurations."""
        validate_model_config(3584, 16)  # 3584 / 16 = 224
        validate_model_config(2048, 8)   # 2048 / 8 = 256

    def test_validate_indivisible_heads(self):
        """Test error when d_model not divisible by n_heads."""
        with pytest.raises(ValueError, match="d_model .* must be divisible by n_heads"):
            validate_model_config(3584, 15)

    def test_validate_invalid_preset_strict(self):
        """Test error on non-Qwen dimension in strict mode."""
        with pytest.raises(ValueError, match="hidden size 768 not in Qwen presets"):
            validate_model_config(768, 8, strict=True)

    def test_validate_invalid_preset_non_strict(self, capsys):
        """Test warning on non-Qwen dimension in non-strict mode."""
        validate_model_config(768, 8, strict=False)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_validate_qwen_7b_config(self):
        """Test our target Qwen 7B configuration."""
        # Our config uses d_model=3584, n_heads=16
        validate_model_config(3584, 16, strict=True)
