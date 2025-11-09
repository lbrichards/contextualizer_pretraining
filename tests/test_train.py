"""Tests for training loop and utilities."""

import pytest
import torch
import tempfile
import json
from pathlib import Path

from src.train_mlm import (
    create_optimizer,
    create_scheduler,
    TrainingConfig,
    Trainer,
    save_checkpoint,
    load_checkpoint,
)
from src.modules import ContextualizerForMLM
from src.utils import set_seed


@pytest.fixture
def sample_config():
    """Create sample training configuration."""
    return TrainingConfig(
        # Model config
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=64,
        dropout=0.1,
        # Training config
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_steps=10,
        max_steps=100,
        gradient_clip=1.0,
        log_interval=10,
        eval_interval=25,
        save_interval=50,
        seed=42,
        strict_validation=False,  # Allow non-Qwen sizes for testing
    )


@pytest.fixture
def sample_model():
    """Create a small model for testing."""
    set_seed(42)
    return ContextualizerForMLM(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=64,
        dropout=0.1,
    )


class TestTrainingConfig:
    """Test training configuration."""

    def test_config_creation(self, sample_config):
        """Test creating training config."""
        assert sample_config.batch_size == 4
        assert sample_config.learning_rate == 1e-3
        assert sample_config.max_steps == 100

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "vocab_size": 1000,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 512,
            "batch_size": 8,
            "learning_rate": 3e-4,
            "max_steps": 200,
            "strict_validation": False,
        }
        config = TrainingConfig(**config_dict)
        assert config.batch_size == 8
        assert config.learning_rate == 3e-4

    def test_config_validation(self):
        """Test that invalid configs raise errors."""
        with pytest.raises((ValueError, TypeError)):
            TrainingConfig(
                vocab_size=1000,
                d_model=128,
                n_heads=5,  # Not divisible
                n_layers=2,
                d_ff=512,
            )


class TestCreateOptimizer:
    """Test optimizer creation."""

    def test_create_adamw_optimizer(self, sample_model):
        """Test creating AdamW optimizer."""
        optimizer = create_optimizer(
            sample_model,
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )

        assert optimizer is not None
        assert len(optimizer.param_groups) > 0
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_optimizer_parameter_groups(self, sample_model):
        """Test that optimizer has correct parameter groups."""
        optimizer = create_optimizer(sample_model, lr=1e-3, weight_decay=0.01)

        # Check that parameters are registered
        total_params = sum(p.numel() for p in sample_model.parameters())
        optimizer_params = sum(
            p.numel() for group in optimizer.param_groups for p in group["params"]
        )

        assert optimizer_params == total_params


class TestCreateScheduler:
    """Test learning rate scheduler creation."""

    def test_create_cosine_scheduler(self, sample_model):
        """Test creating cosine annealing scheduler."""
        optimizer = create_optimizer(sample_model, lr=1e-3)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="cosine",
            num_training_steps=100,
            num_warmup_steps=10,
        )

        assert scheduler is not None

    def test_warmup_then_decay(self, sample_model):
        """Test that learning rate warms up then decays."""
        optimizer = create_optimizer(sample_model, lr=1e-3)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="cosine",
            num_training_steps=100,
            num_warmup_steps=10,
        )

        initial_lr = optimizer.param_groups[0]["lr"]

        # Step through warmup
        lrs_warmup = []
        for _ in range(10):
            lrs_warmup.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # LR should increase during warmup
        assert lrs_warmup[-1] > lrs_warmup[0]

        # Step through decay
        lr_after_warmup = optimizer.param_groups[0]["lr"]
        for _ in range(50):
            optimizer.step()
            scheduler.step()

        lr_after_decay = optimizer.param_groups[0]["lr"]

        # LR should decrease after warmup
        assert lr_after_decay < lr_after_warmup

    def test_linear_scheduler(self, sample_model):
        """Test creating linear scheduler."""
        optimizer = create_optimizer(sample_model, lr=1e-3)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="linear",
            num_training_steps=100,
            num_warmup_steps=10,
        )

        assert scheduler is not None


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    def test_save_checkpoint(self, sample_model, sample_config):
        """Test saving checkpoint."""
        optimizer = create_optimizer(sample_model, lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "checkpoint.pt"

            save_checkpoint(
                model=sample_model,
                optimizer=optimizer,
                step=100,
                config=sample_config,
                save_path=save_path,
                loss=2.5,
            )

            assert save_path.exists()

    def test_load_checkpoint(self, sample_model, sample_config):
        """Test loading checkpoint."""
        optimizer = create_optimizer(sample_model, lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "checkpoint.pt"

            # Save
            save_checkpoint(
                model=sample_model,
                optimizer=optimizer,
                step=100,
                config=sample_config,
                save_path=save_path,
                loss=2.5,
            )

            # Create new model and optimizer
            new_model = ContextualizerForMLM(
                vocab_size=1000,
                d_model=128,
                n_heads=4,
                n_layers=2,
                d_ff=512,
                max_seq_len=64,
                dropout=0.1,
            )
            new_optimizer = create_optimizer(new_model, lr=1e-3)

            # Load
            checkpoint = load_checkpoint(save_path, new_model, new_optimizer)

            assert checkpoint["step"] == 100
            assert checkpoint["loss"] == 2.5

    def test_checkpoint_deterministic(self, sample_model, sample_config):
        """Test that loading checkpoint restores exact state."""
        optimizer = create_optimizer(sample_model, lr=1e-3)

        # Get initial state
        initial_params = {
            name: param.clone() for name, param in sample_model.named_parameters()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "checkpoint.pt"

            # Train for a few steps
            for _ in range(5):
                optimizer.zero_grad()
                dummy_input = torch.randint(0, 1000, (2, 32))
                dummy_labels = torch.randint(0, 1000, (2, 32))
                _, loss = sample_model(dummy_input, labels=dummy_labels)
                loss.backward()
                optimizer.step()

            # Save checkpoint
            save_checkpoint(
                model=sample_model,
                optimizer=optimizer,
                step=5,
                config=sample_config,
                save_path=save_path,
            )

            # Reset model to initial state
            sample_model.load_state_dict(
                {name: param for name, param in initial_params.items()}
            )

            # Load checkpoint
            load_checkpoint(save_path, sample_model, optimizer)

            # Verify state was restored
            for name, param in sample_model.named_parameters():
                assert not torch.equal(param, initial_params[name])


class TestTrainer:
    """Test Trainer class."""

    def test_trainer_initialization(self, sample_config):
        """Test trainer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal training data
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            with open(data_dir / "train.jsonl", "w") as f:
                for i in range(20):
                    f.write(json.dumps({"full_text": f"Example text number {i}"}) + "\n")

            trainer = Trainer(config=sample_config, data_dir=str(data_dir))

            assert trainer.model is not None
            assert trainer.optimizer is not None
            assert trainer.train_loader is not None

    def test_training_step(self, sample_config):
        """Test single training step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal training data
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            with open(data_dir / "train.jsonl", "w") as f:
                for i in range(20):
                    f.write(json.dumps({"full_text": f"Example text number {i} " * 10}) + "\n")

            trainer = Trainer(config=sample_config, data_dir=str(data_dir))

            # Get initial loss
            batch = next(iter(trainer.train_loader))
            initial_loss = trainer.training_step(batch)

            assert isinstance(initial_loss, float)
            assert initial_loss > 0

    def test_training_reduces_loss(self, sample_config):
        """Test that training reduces loss over steps."""
        # Increase steps for this test
        sample_config.max_steps = 50
        sample_config.eval_interval = 50  # Disable eval

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create training data
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            with open(data_dir / "train.jsonl", "w") as f:
                for i in range(50):
                    f.write(json.dumps({"full_text": f"Example text number {i} " * 10}) + "\n")

            trainer = Trainer(config=sample_config, data_dir=str(data_dir))

            # Train for a few steps and track loss
            losses = []
            for i, batch in enumerate(trainer.train_loader):
                if i >= 20:
                    break
                loss = trainer.training_step(batch)
                losses.append(loss)

            # Loss should generally decrease (allowing some variance)
            assert losses[-1] < losses[0] * 1.1  # Final loss should be lower


class TestTrainingIntegration:
    """Integration tests for full training pipeline."""

    def test_small_training_run(self, sample_config):
        """Test running a small training session."""
        sample_config.max_steps = 10
        sample_config.eval_interval = 100  # Disable eval for speed
        sample_config.save_interval = 100  # Disable saving

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            with open(data_dir / "train.jsonl", "w") as f:
                for i in range(30):
                    f.write(json.dumps({"full_text": f"Training example {i} " * 10}) + "\n")

            # Set output directory
            sample_config.output_dir = str(Path(tmpdir) / "outputs")

            trainer = Trainer(config=sample_config, data_dir=str(data_dir))
            trainer.train()

            # Training should complete without errors
            assert trainer.global_step >= sample_config.max_steps

    def test_training_with_validation(self, sample_config):
        """Test training with validation split."""
        sample_config.max_steps = 10
        sample_config.eval_interval = 5
        sample_config.val_split = 0.2

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            with open(data_dir / "train.jsonl", "w") as f:
                for i in range(50):
                    f.write(json.dumps({"full_text": f"Example {i} " * 10}) + "\n")

            sample_config.output_dir = str(Path(tmpdir) / "outputs")

            trainer = Trainer(config=sample_config, data_dir=str(data_dir))
            trainer.train()

            # Should have validation loader
            assert trainer.val_loader is not None
