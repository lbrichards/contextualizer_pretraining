"""Utility functions for device detection, seeding, and config management."""

import os
import random
import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_device(device_spec: str = "auto") -> torch.device:
    """
    Get PyTorch device supporting CUDA (deployment) and MPS (Apple Silicon dev).

    Args:
        device_spec: One of "auto", "cuda", "mps", "cpu"

    Returns:
        torch.device object

    Examples:
        >>> device = get_device("auto")  # doctest: +SKIP
        >>> isinstance(device, torch.device)  # doctest: +SKIP
        True
    """
    if device_spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_spec)


def get_dtype(dtype_spec: str) -> torch.dtype:
    """
    Convert string dtype specification to torch.dtype.

    Args:
        dtype_spec: One of "bf16", "fp16", "fp32"

    Returns:
        torch.dtype object

    Examples:
        >>> get_dtype("bf16")
        torch.bfloat16
        >>> get_dtype("fp32")
        torch.float32
    """
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if dtype_spec not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_spec}. Choose from {list(dtype_map.keys())}")
    return dtype_map[dtype_spec]


def set_seed(seed: int):
    """
    Set random seeds for reproducibility across random, numpy, and torch.

    Args:
        seed: Random seed value

    Examples:
        >>> set_seed(42)
        >>> random.randint(0, 100)  # doctest: +SKIP
        81
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing configuration

    Examples:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        ...     _ = f.write("seed: 42\\ndevice: auto")
        ...     tmp_path = f.name
        >>> config = load_config(tmp_path)
        >>> config['seed']
        42
        >>> import os; os.unlink(tmp_path)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], path: str):
    """
    Save configuration dictionary to YAML file.

    Args:
        config: Configuration dictionary
        path: Output path for YAML file
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters

    Examples:
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 5)
        >>> count_parameters(model)
        55
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    """
    Format large numbers with K/M/B suffixes.

    Args:
        num: Number to format

    Returns:
        Formatted string

    Examples:
        >>> format_number(1500)
        '1.5K'
        >>> format_number(2_500_000)
        '2.5M'
        >>> format_number(1_000_000_000)
        '1.0B'
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def get_data_dir() -> Path:
    """
    Get data directory from environment variable or default.

    Returns:
        Path to data shards directory

    Examples:
        >>> data_dir = get_data_dir()  # doctest: +SKIP
        >>> isinstance(data_dir, Path)  # doctest: +SKIP
        True
    """
    data_dir = os.getenv("DATA_SHARDS_DIR", "data/shards")
    return Path(data_dir)


def get_project_paths() -> Dict[str, Path]:
    """
    Get all project paths with environment variable overrides.

    Returns:
        Dictionary mapping path names to Path objects

    Examples:
        >>> paths = get_project_paths()
        >>> 'data' in paths
        True
        >>> 'checkpoints' in paths
        True
    """
    return {
        "data": Path(os.getenv("DATA_SHARDS_DIR", "data/shards")),
        "checkpoints": Path(os.getenv("CHECKPOINTS_DIR", "checkpoints")),
        "logs": Path(os.getenv("LOGS_DIR", "logs")),
        "exports": Path(os.getenv("EXPORTS_DIR", "exports")),
        "encoded": Path(os.getenv("ENCODED_DIR", "encoded")),
    }


def ensure_dirs(paths: Optional[Dict[str, Path]] = None):
    """
    Ensure all project directories exist.

    Args:
        paths: Optional dictionary of paths to create. If None, uses get_project_paths()

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     test_paths = {"test": Path(tmpdir) / "test_dir"}
        ...     ensure_dirs(test_paths)
        ...     test_paths["test"].exists()
        True
    """
    if paths is None:
        paths = get_project_paths()

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
