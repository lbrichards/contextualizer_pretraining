"""Tests for utility functions."""

import pytest
import torch
import tempfile
import os
from pathlib import Path

from src.utils import (
    get_device,
    get_dtype,
    set_seed,
    load_config,
    save_config,
    count_parameters,
    format_number,
    get_data_dir,
    get_project_paths,
    ensure_dirs,
)


class TestDeviceDetection:
    """Test device detection for cross-platform compatibility."""

    def test_get_device_auto(self):
        """Test auto device detection returns valid device."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]

    def test_get_device_cpu(self):
        """Test explicit CPU device."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_get_device_explicit_cuda(self):
        """Test explicit CUDA specification (may not be available)."""
        device = get_device("cuda")
        assert device.type == "cuda"

    def test_get_device_explicit_mps(self):
        """Test explicit MPS specification (may not be available)."""
        device = get_device("mps")
        assert device.type == "mps"


class TestDtypeConversion:
    """Test dtype string to torch.dtype conversion."""

    def test_bf16_conversion(self):
        """Test bfloat16 conversion."""
        assert get_dtype("bf16") == torch.bfloat16
        assert get_dtype("bfloat16") == torch.bfloat16

    def test_fp16_conversion(self):
        """Test float16 conversion."""
        assert get_dtype("fp16") == torch.float16
        assert get_dtype("float16") == torch.float16

    def test_fp32_conversion(self):
        """Test float32 conversion."""
        assert get_dtype("fp32") == torch.float32
        assert get_dtype("float32") == torch.float32

    def test_invalid_dtype(self):
        """Test invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dtype"):
            get_dtype("invalid")


class TestSeeding:
    """Test deterministic seeding."""

    def test_set_seed_deterministic(self):
        """Test that seeding produces deterministic results."""
        set_seed(42)
        tensor1 = torch.rand(5)

        set_seed(42)
        tensor2 = torch.rand(5)

        assert torch.allclose(tensor1, tensor2)

    def test_set_seed_different_seeds(self):
        """Test different seeds produce different results."""
        set_seed(42)
        tensor1 = torch.rand(5)

        set_seed(123)
        tensor2 = torch.rand(5)

        assert not torch.allclose(tensor1, tensor2)


class TestConfigManagement:
    """Test configuration loading and saving."""

    def test_load_config(self):
        """Test loading YAML configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("seed: 42\ndevice: auto\nmodel:\n  d_model: 3584\n")
            tmp_path = f.name

        try:
            config = load_config(tmp_path)
            assert config['seed'] == 42
            assert config['device'] == 'auto'
            assert config['model']['d_model'] == 3584
        finally:
            os.unlink(tmp_path)

    def test_save_and_load_config(self):
        """Test round-trip config save and load."""
        config = {
            'seed': 42,
            'device': 'auto',
            'model': {'d_model': 3584, 'n_layers': 2}
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'test_config.yaml')
            save_config(config, config_path)

            loaded_config = load_config(config_path)
            assert loaded_config == config


class TestParameterCounting:
    """Test model parameter counting."""

    def test_count_parameters_linear(self):
        """Test counting parameters in a linear layer."""
        model = torch.nn.Linear(10, 5)
        # 10 * 5 weights + 5 biases = 55
        assert count_parameters(model) == 55

    def test_count_parameters_no_bias(self):
        """Test counting parameters without bias."""
        model = torch.nn.Linear(10, 5, bias=False)
        assert count_parameters(model) == 50

    def test_count_parameters_frozen(self):
        """Test that frozen parameters are not counted."""
        model = torch.nn.Linear(10, 5)
        for param in model.parameters():
            param.requires_grad = False
        assert count_parameters(model) == 0

    def test_count_parameters_sequential(self):
        """Test counting parameters in sequential model."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),  # 200 + 20
            torch.nn.Linear(20, 5),   # 100 + 5
        )
        assert count_parameters(model) == 325


class TestNumberFormatting:
    """Test human-readable number formatting."""

    def test_format_small_numbers(self):
        """Test formatting numbers under 1000."""
        assert format_number(500) == "500"
        assert format_number(999) == "999"

    def test_format_thousands(self):
        """Test formatting thousands with K suffix."""
        assert format_number(1000) == "1.0K"
        assert format_number(1500) == "1.5K"
        assert format_number(999_999) == "1000.0K"

    def test_format_millions(self):
        """Test formatting millions with M suffix."""
        assert format_number(1_000_000) == "1.0M"
        assert format_number(2_500_000) == "2.5M"
        assert format_number(999_999_999) == "1000.0M"

    def test_format_billions(self):
        """Test formatting billions with B suffix."""
        assert format_number(1_000_000_000) == "1.0B"
        assert format_number(7_000_000_000) == "7.0B"


class TestPathUtilities:
    """Test path management utilities."""

    def test_get_data_dir_default(self):
        """Test getting data directory with default."""
        import os
        # Temporarily unset env var to test default
        old_val = os.environ.pop("DATA_SHARDS_DIR", None)
        try:
            from pathlib import Path
            data_dir = get_data_dir()
            assert isinstance(data_dir, Path)
            assert str(data_dir) == "data/shards"
        finally:
            if old_val:
                os.environ["DATA_SHARDS_DIR"] = old_val

    def test_get_data_dir_from_env(self):
        """Test getting data directory from environment variable."""
        import os
        old_val = os.environ.get("DATA_SHARDS_DIR")
        try:
            os.environ["DATA_SHARDS_DIR"] = "/custom/path"
            # Reload module to pick up new env var
            import importlib
            import src.utils
            importlib.reload(src.utils)
            from src.utils import get_data_dir as gdd
            data_dir = gdd()
            assert str(data_dir) == "/custom/path"
        finally:
            if old_val:
                os.environ["DATA_SHARDS_DIR"] = old_val
            else:
                os.environ.pop("DATA_SHARDS_DIR", None)
            # Reload again to reset
            importlib.reload(src.utils)

    def test_get_project_paths(self):
        """Test getting all project paths."""
        from pathlib import Path
        paths = get_project_paths()
        assert isinstance(paths, dict)
        assert "data" in paths
        assert "checkpoints" in paths
        assert "logs" in paths
        assert "exports" in paths
        assert "encoded" in paths
        for path in paths.values():
            assert isinstance(path, Path)

    def test_ensure_dirs_creates_directories(self):
        """Test that ensure_dirs creates directories."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            test_paths = {
                "test1": Path(tmpdir) / "dir1",
                "test2": Path(tmpdir) / "dir2" / "subdir",
            }

            # Directories should not exist yet
            assert not test_paths["test1"].exists()
            assert not test_paths["test2"].exists()

            # Create directories
            ensure_dirs(test_paths)

            # Now they should exist
            assert test_paths["test1"].exists()
            assert test_paths["test2"].exists()
            assert test_paths["test1"].is_dir()
            assert test_paths["test2"].is_dir()

    def test_ensure_dirs_idempotent(self):
        """Test that ensure_dirs can be called multiple times safely."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            test_paths = {"test": Path(tmpdir) / "dir1"}

            # Call twice
            ensure_dirs(test_paths)
            ensure_dirs(test_paths)

            # Should still work
            assert test_paths["test"].exists()
