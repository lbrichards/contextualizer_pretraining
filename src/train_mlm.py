"""Training loop for MLM contextualizer pretraining."""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from src.modules import ContextualizerForMLM
from src.data import create_mlm_dataloaders
from src.utils import (
    set_seed,
    get_device,
    get_dtype,
    count_parameters,
    format_number,
    get_project_paths,
    ensure_dirs,
    load_config,
    save_config,
)
from src.constants import validate_model_config


@dataclass
class TrainingConfig:
    """Configuration for MLM training."""

    # Model architecture
    vocab_size: int = 151936  # Qwen tokenizer size
    d_model: int = 3584
    n_heads: int = 16
    n_layers: int = 2
    d_ff: int = 14336
    max_seq_len: int = 512
    dropout: float = 0.1
    rope_theta: float = 10000.0

    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    gradient_clip: float = 1.0
    warmup_steps: int = 500
    max_steps: int = 10000

    # Data
    mask_prob: float = 0.15
    val_split: float = 0.1

    # Logging and checkpointing
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "outputs"

    # System
    seed: int = 42
    device: str = "auto"
    dtype: str = "bf16"
    gradient_checkpointing: bool = False
    use_amp: bool = True

    # Scheduler
    scheduler_type: str = "cosine"  # "cosine" or "linear"

    # Validation
    strict_validation: bool = True  # Enforce Qwen presets

    def __post_init__(self):
        """Validate configuration after initialization."""
        validate_model_config(self.d_model, self.n_heads, strict=self.strict_validation)
        if isinstance(self.betas, list):
            self.betas = tuple(self.betas)


def create_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.95),
) -> AdamW:
    """
    Create AdamW optimizer with weight decay.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters

    Returns:
        AdamW optimizer
    """
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Don't apply weight decay to bias and layer norm parameters
        if "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, betas=betas)
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    num_warmup_steps: int,
):
    """
    Create learning rate scheduler with warmup.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ("cosine" or "linear")
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps

    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    config: TrainingConfig,
    save_path: Path,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    loss: Optional[float] = None,
    **kwargs,
):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optional optimizer state to save
        step: Current training step
        config: Training configuration
        save_path: Path to save checkpoint
        scheduler: Optional scheduler state
        loss: Optional loss value
        **kwargs: Additional items to save
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if loss is not None:
        checkpoint["loss"] = loss

    checkpoint.update(kwargs)

    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


class Trainer:
    """
    Trainer for MLM pretraining.

    Args:
        config: Training configuration
        data_dir: Directory containing training data
        resume_from: Optional checkpoint path to resume from
    """

    def __init__(
        self,
        config: TrainingConfig,
        data_dir: str,
        resume_from: Optional[str] = None,
    ):
        self.config = config
        self.data_dir = data_dir

        # Set seed for reproducibility
        set_seed(config.seed)

        # Setup device and dtype
        self.device = get_device(config.device)
        self.dtype = get_dtype(config.dtype)

        print(f"Using device: {self.device}, dtype: {self.dtype}")

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        save_config(asdict(config), self.output_dir / "config.yaml")

        # Initialize tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

        # Update vocab size if tokenizer differs
        if len(self.tokenizer) != config.vocab_size:
            print(f"Updating vocab_size from {config.vocab_size} to {len(self.tokenizer)}")
            config.vocab_size = len(self.tokenizer)

        # Create dataloaders
        print(f"Loading data from {data_dir}...")
        self.train_loader, self.val_loader = create_mlm_dataloaders(
            data_dir=data_dir,
            tokenizer=self.tokenizer,
            batch_size=config.batch_size,
            max_length=config.max_seq_len,
            mask_prob=config.mask_prob,
            val_split=config.val_split,
            seed=config.seed,
        )

        print(f"Training batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Validation batches: {len(self.val_loader)}")

        # Initialize model
        print("Initializing model...")
        self.model = ContextualizerForMLM(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            rope_theta=config.rope_theta,
            gradient_checkpointing=config.gradient_checkpointing,
        )

        self.model = self.model.to(self.device)

        # Print model info
        num_params = count_parameters(self.model)
        print(f"Model parameters: {format_number(num_params)}")

        # Create optimizer
        self.optimizer = create_optimizer(
            self.model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

        # Create scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=config.scheduler_type,
            num_training_steps=config.max_steps,
            num_warmup_steps=config.warmup_steps,
        )

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Resume from checkpoint if specified
        if resume_from:
            print(f"Resuming from checkpoint: {resume_from}")
            checkpoint = load_checkpoint(
                Path(resume_from), self.model, self.optimizer, self.scheduler
            )
            self.global_step = checkpoint["step"]
            if "best_val_loss" in checkpoint:
                self.best_val_loss = checkpoint["best_val_loss"]

        # Setup AMP scaler if using mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if config.use_amp and self.device.type == "cuda" else None

    def training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Perform single training step.

        Args:
            batch: Batch of data

        Returns:
            Loss value
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass with mixed precision if enabled
        if self.scaler:
            with torch.amp.autocast("cuda", dtype=self.dtype):
                logits, loss = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
        else:
            logits, loss = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate on validation set.

        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            logits, loss = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else float("inf")

    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config.max_steps} steps...")
        print(f"Logging every {self.config.log_interval} steps")
        print(f"Evaluating every {self.config.eval_interval} steps")
        print(f"Saving every {self.config.save_interval} steps\n")

        running_loss = 0.0
        num_loss_steps = 0

        # Training loop
        epoch = 0
        while self.global_step < self.config.max_steps:
            epoch += 1

            for batch in self.train_loader:
                loss = self.training_step(batch)
                running_loss += loss
                num_loss_steps += 1
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = running_loss / num_loss_steps
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"Step {self.global_step}/{self.config.max_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )
                    running_loss = 0.0
                    num_loss_steps = 0

                # Evaluation
                if self.global_step % self.config.eval_interval == 0:
                    val_loss = self.evaluate()
                    print(f"Validation loss: {val_loss:.4f}")

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_path = self.output_dir / "best_model.pt"
                        save_checkpoint(
                            self.model,
                            self.optimizer,
                            self.global_step,
                            self.config,
                            save_path,
                            scheduler=self.scheduler,
                            loss=val_loss,
                            best_val_loss=self.best_val_loss,
                        )
                        print(f"Saved best model to {save_path}")

                # Checkpointing
                if self.global_step % self.config.save_interval == 0:
                    save_path = self.output_dir / f"checkpoint_step_{self.global_step}.pt"
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.global_step,
                        self.config,
                        save_path,
                        scheduler=self.scheduler,
                        best_val_loss=self.best_val_loss,
                    )
                    print(f"Saved checkpoint to {save_path}")

                # Check if we've reached max steps
                if self.global_step >= self.config.max_steps:
                    break

        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Save final checkpoint
        save_path = self.output_dir / "final_model.pt"
        save_checkpoint(
            self.model,
            self.optimizer,
            self.global_step,
            self.config,
            save_path,
            scheduler=self.scheduler,
            best_val_loss=self.best_val_loss,
        )
        print(f"Saved final model to {save_path}")


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="Train MLM contextualizer")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--data", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config:
        config_dict = load_config(args.config)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()

    # Override output directory
    config.output_dir = args.output

    # Create trainer and run
    trainer = Trainer(config=config, data_dir=args.data, resume_from=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
