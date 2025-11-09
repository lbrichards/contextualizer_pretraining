"""Export trained contextualizer encoder for downstream use."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

from src.modules import ContextualizerEncoder, ContextualizerForMLM
from src.train_mlm import load_checkpoint, TrainingConfig
from src.utils import load_config


class ExportedEncoder(nn.Module):
    """
    Exported encoder for inference with pooling.

    This wraps the ContextualizerEncoder and adds pooling to produce
    a single vector representation per sequence.

    Args:
        encoder: The contextualizer encoder
        pool_mode: Pooling mode ("cls" or "mean")

    Examples:
        >>> from src.modules import ContextualizerEncoder
        >>> encoder = ContextualizerEncoder(
        ...     vocab_size=1000, d_model=128, n_heads=4, n_layers=2, d_ff=512
        ... )
        >>> exported = ExportedEncoder(encoder, pool_mode="cls")
        >>> input_ids = torch.randint(0, 1000, (2, 32))
        >>> h_prime = exported(input_ids)
        >>> h_prime.shape
        torch.Size([2, 128])
    """

    def __init__(self, encoder: ContextualizerEncoder, pool_mode: str = "cls"):
        super().__init__()
        if pool_mode not in ["cls", "mean"]:
            raise ValueError(f"pool_mode must be 'cls' or 'mean', got {pool_mode}")

        self.encoder = encoder
        self.pool_mode = pool_mode

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode input and pool to single vector.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Pooled representation [batch_size, d_model]
        """
        # Get hidden states from encoder
        hidden_states = self.encoder(input_ids, attention_mask=attention_mask)

        # Pool to single vector
        if self.pool_mode == "cls":
            # Use first token (CLS-style)
            pooled = hidden_states[:, 0, :]
        else:  # mean
            # Average over sequence, respecting attention mask
            if attention_mask is not None:
                # Convert to float and add dimension for broadcasting
                mask = attention_mask.float().unsqueeze(-1)
                # Sum and divide by number of non-masked tokens
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                # Simple mean over sequence
                pooled = hidden_states.mean(dim=1)

        return pooled


def create_export_manifest(
    model_config: Dict[str, Any],
    pool_mode: str,
    seed: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Create manifest file for exported encoder.

    Args:
        model_config: Model configuration dictionary
        pool_mode: Pooling mode used
        seed: Random seed used during training
        **kwargs: Additional metadata

    Returns:
        Manifest dictionary
    """
    manifest = {
        "model_config": model_config,
        "export_config": {
            "pool_mode": pool_mode,
            "timestamp": datetime.now().isoformat(),
        },
    }

    if seed is not None:
        manifest["export_config"]["seed"] = seed

    # Add any additional metadata
    manifest["export_config"].update(kwargs)

    return manifest


def export_encoder(
    checkpoint_path: Path,
    output_path: Path,
    pool_mode: str = "cls",
) -> Path:
    """
    Export encoder from training checkpoint.

    This function:
    1. Loads the full MLM model from checkpoint
    2. Extracts the encoder (discarding MLM head)
    3. Wraps it with pooling logic
    4. Saves to safetensors format
    5. Creates a manifest file with metadata

    Args:
        checkpoint_path: Path to training checkpoint
        output_path: Path to save exported encoder
        pool_mode: Pooling mode ("cls" or "mean")

    Returns:
        Path to exported encoder file

    Examples:
        >>> # After training
        >>> export_encoder(
        ...     checkpoint_path=Path("checkpoints/best_model.pt"),
        ...     output_path=Path("exports/encoder.safetensors"),
        ...     pool_mode="cls",
        ... )  # doctest: +SKIP
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get config from checkpoint
    config = TrainingConfig(**checkpoint["config"])

    print("Reconstructing model...")
    # Reconstruct full MLM model
    model = ContextualizerForMLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=0.0,  # No dropout for inference
        rope_theta=config.rope_theta,
        gradient_checkpointing=False,
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Exporting encoder with pool_mode={pool_mode}...")
    # Extract encoder and wrap with pooling
    exported = ExportedEncoder(encoder=model.encoder, pool_mode=pool_mode)

    # Set to eval mode
    exported.eval()

    # Save using safetensors
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert state dict to format safetensors expects
    state_dict = {k: v.clone().cpu() for k, v in exported.state_dict().items()}
    save_file(state_dict, output_path)

    print(f"Saved encoder to {output_path}")

    # Create and save manifest
    manifest = create_export_manifest(
        model_config={
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "d_ff": config.d_ff,
            "max_seq_len": config.max_seq_len,
            "rope_theta": config.rope_theta,
        },
        pool_mode=pool_mode,
        seed=config.seed,
        step=checkpoint.get("step", None),
        loss=checkpoint.get("loss", None),
    )

    manifest_path = output_path.with_suffix(".json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest to {manifest_path}")

    return output_path


def load_exported_encoder(
    export_path: Path,
    device: str = "cpu",
) -> ExportedEncoder:
    """
    Load exported encoder for inference.

    Args:
        export_path: Path to exported encoder (.safetensors)
        device: Device to load model on

    Returns:
        ExportedEncoder ready for inference

    Examples:
        >>> encoder = load_exported_encoder(Path("exports/encoder.safetensors"))  # doctest: +SKIP
        >>> input_ids = torch.randint(0, 1000, (2, 32))  # doctest: +SKIP
        >>> h_prime = encoder(input_ids)  # doctest: +SKIP
    """
    # Load manifest
    manifest_path = export_path.with_suffix(".json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    model_config = manifest["model_config"]
    pool_mode = manifest["export_config"]["pool_mode"]

    print(f"Loading encoder from {export_path}...")
    print(f"Config: d_model={model_config['d_model']}, n_layers={model_config['n_layers']}")
    print(f"Pool mode: {pool_mode}")

    # Reconstruct encoder
    encoder = ContextualizerEncoder(
        vocab_size=model_config["vocab_size"],
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        d_ff=model_config["d_ff"],
        max_seq_len=model_config["max_seq_len"],
        dropout=0.0,
        rope_theta=model_config["rope_theta"],
        gradient_checkpointing=False,
    )

    # Wrap with pooling
    exported = ExportedEncoder(encoder=encoder, pool_mode=pool_mode)

    # Load weights
    state_dict = load_file(export_path, device=device)
    exported.load_state_dict(state_dict)

    # Set to eval mode
    exported.eval()

    print("Encoder loaded successfully!")

    return exported


def main():
    """Main entry point for export script."""
    parser = argparse.ArgumentParser(description="Export trained contextualizer encoder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output path for exported encoder")
    parser.add_argument(
        "--pool",
        type=str,
        default="cls",
        choices=["cls", "mean"],
        help="Pooling mode (default: cls)",
    )

    args = parser.parse_args()

    export_encoder(
        checkpoint_path=Path(args.checkpoint),
        output_path=Path(args.output),
        pool_mode=args.pool,
    )


if __name__ == "__main__":
    main()
