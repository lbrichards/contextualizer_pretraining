"""Constants and validation for Qwen model presets."""

from typing import Dict, List

# Qwen 2.5 model dimension presets
# Source: https://huggingface.co/Qwen
QWEN_PRESETS = {
    "0.5B": {"hidden": 896, "layers": 24, "heads": 14, "ff": 4864},
    "1.5B": {"hidden": 1536, "layers": 28, "heads": 12, "ff": 8960},
    "3B": {"hidden": 2048, "layers": 36, "heads": 16, "ff": 11008},
    "7B": {"hidden": 3584, "layers": 28, "heads": 28, "ff": 18944},
    "14B": {"hidden": 5120, "layers": 48, "heads": 40, "ff": 13824},
    "32B": {"hidden": 5120, "layers": 64, "heads": 40, "ff": 27648},
    "72B": {"hidden": 8192, "layers": 80, "heads": 64, "ff": 29568},
}


def get_valid_hidden_dims() -> List[int]:
    """
    Get list of valid hidden dimensions from Qwen presets.

    Returns:
        Sorted list of valid hidden dimensions

    Examples:
        >>> dims = get_valid_hidden_dims()
        >>> 3584 in dims  # Qwen 7B
        True
        >>> 768 in dims  # Not a Qwen preset
        False
    """
    return sorted({preset["hidden"] for preset in QWEN_PRESETS.values()})


def validate_hidden_dim(target_hidden: int, strict: bool = True):
    """
    Validate that hidden dimension matches a Qwen preset.

    Args:
        target_hidden: Target hidden dimension to validate
        strict: If True, raise error on mismatch; if False, only warn

    Raises:
        ValueError: If target_hidden not in Qwen presets and strict=True

    Examples:
        >>> validate_hidden_dim(3584)  # Qwen 7B - valid
        >>> validate_hidden_dim(768)  # doctest: +SKIP
        Traceback (most recent call last):
        ...
        ValueError: hidden size 768 not in Qwen presets [896, 1536, 2048, 3584, 5120, 8192]
    """
    valid_dims = get_valid_hidden_dims()
    if target_hidden not in valid_dims:
        msg = f"hidden size {target_hidden} not in Qwen presets {valid_dims}"
        if strict:
            raise ValueError(msg)
        else:
            print(f"WARNING: {msg}")


def get_qwen_preset(model_size: str) -> Dict[str, int]:
    """
    Get Qwen model preset by size name.

    Args:
        model_size: Model size string (e.g., "7B", "3B")

    Returns:
        Dictionary with hidden, layers, heads, ff keys

    Raises:
        KeyError: If model_size not found in presets

    Examples:
        >>> preset = get_qwen_preset("7B")
        >>> preset['hidden']
        3584
        >>> preset['layers']
        28
    """
    if model_size not in QWEN_PRESETS:
        available = list(QWEN_PRESETS.keys())
        raise KeyError(f"Model size '{model_size}' not found. Available: {available}")
    return QWEN_PRESETS[model_size].copy()


def infer_qwen_size(hidden_dim: int) -> str:
    """
    Infer Qwen model size from hidden dimension.

    Args:
        hidden_dim: Hidden dimension size

    Returns:
        Model size string (e.g., "7B")

    Raises:
        ValueError: If hidden_dim doesn't match any preset

    Examples:
        >>> infer_qwen_size(3584)
        '7B'
        >>> infer_qwen_size(2048)
        '3B'
    """
    for size, specs in QWEN_PRESETS.items():
        if specs["hidden"] == hidden_dim:
            return size

    valid_dims = get_valid_hidden_dims()
    raise ValueError(f"Hidden dim {hidden_dim} doesn't match any Qwen preset. Valid: {valid_dims}")


def validate_model_config(d_model: int, n_heads: int, strict: bool = True):
    """
    Validate that model configuration is compatible.

    Args:
        d_model: Model hidden dimension
        n_heads: Number of attention heads
        strict: Whether to enforce Qwen presets

    Raises:
        ValueError: If configuration is invalid

    Examples:
        >>> validate_model_config(3584, 16)  # Valid
        >>> validate_model_config(3584, 15)  # doctest: +SKIP
        Traceback (most recent call last):
        ...
        ValueError: d_model (3584) must be divisible by n_heads (15)
    """
    # Check divisibility (required for multi-head attention)
    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

    # Validate against Qwen presets
    validate_hidden_dim(d_model, strict=strict)
