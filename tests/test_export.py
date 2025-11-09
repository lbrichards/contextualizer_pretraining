"""Tests for encoder export utilities."""

import pytest
import torch
import tempfile
import json
from pathlib import Path

from src.export_encoder import (
    ExportedEncoder,
    export_encoder,
    load_exported_encoder,
    create_export_manifest,
)
from src.modules import ContextualizerForMLM
from src.train_mlm import TrainingConfig, save_checkpoint
from src.utils import set_seed


@pytest.fixture
def trained_model():
    """Create a small trained model for testing."""
    set_seed(42)
    model = ContextualizerForMLM(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=64,
        dropout=0.0,
    )
    return model


class TestExportedEncoder:
    """Test ExportedEncoder wrapper class."""

    def test_initialization(self, trained_model):
        """Test creating ExportedEncoder from trained model."""
        exported = ExportedEncoder(
            encoder=trained_model.encoder,
            pool_mode="cls",
        )
        assert exported.encoder is not None
        assert exported.pool_mode == "cls"

    def test_forward_cls_pooling(self, trained_model):
        """Test forward pass with CLS pooling."""
        exported = ExportedEncoder(
            encoder=trained_model.encoder,
            pool_mode="cls",
        )

        input_ids = torch.randint(0, 1000, (2, 32))
        h_prime = exported(input_ids)

        # Should output [batch_size, d_model]
        assert h_prime.shape == (2, 128)

    def test_forward_mean_pooling(self, trained_model):
        """Test forward pass with mean pooling."""
        exported = ExportedEncoder(
            encoder=trained_model.encoder,
            pool_mode="mean",
        )

        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)
        h_prime = exported(input_ids, attention_mask=attention_mask)

        assert h_prime.shape == (2, 128)

    def test_mean_pooling_respects_mask(self, trained_model):
        """Test that mean pooling respects attention mask."""
        exported = ExportedEncoder(
            encoder=trained_model.encoder,
            pool_mode="mean",
        )

        input_ids = torch.randint(0, 1000, (2, 32))

        # All tokens attended
        mask_all = torch.ones(2, 32)
        h_all = exported(input_ids, attention_mask=mask_all)

        # Only half tokens attended
        mask_half = torch.ones(2, 32)
        mask_half[:, 16:] = 0
        h_half = exported(input_ids, attention_mask=mask_half)

        # Results should differ
        assert not torch.allclose(h_all, h_half)

    def test_invalid_pool_mode(self, trained_model):
        """Test that invalid pool mode raises error."""
        with pytest.raises(ValueError, match="pool_mode must be"):
            ExportedEncoder(
                encoder=trained_model.encoder,
                pool_mode="invalid",
            )

    def test_deterministic_encoding(self, trained_model):
        """Test deterministic encoding."""
        exported = ExportedEncoder(
            encoder=trained_model.encoder,
            pool_mode="cls",
        )

        input_ids = torch.randint(0, 1000, (2, 32))

        # Encode twice
        h1 = exported(input_ids)
        h2 = exported(input_ids)

        assert torch.equal(h1, h2)


class TestExportEncoder:
    """Test export_encoder function."""

    def test_export_from_checkpoint(self, trained_model):
        """Test exporting encoder from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save a checkpoint
            config = TrainingConfig(
                vocab_size=1000,
                d_model=128,
                n_heads=4,
                n_layers=2,
                d_ff=512,
                strict_validation=False,
            )

            ckpt_path = Path(tmpdir) / "checkpoint.pt"
            save_checkpoint(
                model=trained_model,
                optimizer=None,
                step=100,
                config=config,
                save_path=ckpt_path,
            )

            # Export
            export_path = Path(tmpdir) / "exported.safetensors"
            export_encoder(
                checkpoint_path=ckpt_path,
                output_path=export_path,
                pool_mode="cls",
            )

            assert export_path.exists()

    def test_export_creates_manifest(self, trained_model):
        """Test that export creates manifest file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                vocab_size=1000,
                d_model=128,
                n_heads=4,
                n_layers=2,
                d_ff=512,
                strict_validation=False,
            )

            ckpt_path = Path(tmpdir) / "checkpoint.pt"
            save_checkpoint(
                model=trained_model,
                optimizer=None,
                step=100,
                config=config,
                save_path=ckpt_path,
            )

            export_path = Path(tmpdir) / "exported.safetensors"
            export_encoder(
                checkpoint_path=ckpt_path,
                output_path=export_path,
                pool_mode="cls",
            )

            # Check manifest
            manifest_path = export_path.with_suffix(".json")
            assert manifest_path.exists()

            with open(manifest_path) as f:
                manifest = json.load(f)

            assert "model_config" in manifest
            assert "export_config" in manifest
            assert "pool_mode" in manifest["export_config"]

    def test_export_deterministic(self, trained_model):
        """Test that export is deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                vocab_size=1000,
                d_model=128,
                n_heads=4,
                n_layers=2,
                d_ff=512,
                strict_validation=False,
            )

            ckpt_path = Path(tmpdir) / "checkpoint.pt"
            save_checkpoint(
                model=trained_model,
                optimizer=None,
                step=100,
                config=config,
                save_path=ckpt_path,
            )

            # Export twice
            export_path1 = Path(tmpdir) / "exported1.safetensors"
            export_encoder(ckpt_path, export_path1, pool_mode="cls")

            export_path2 = Path(tmpdir) / "exported2.safetensors"
            export_encoder(ckpt_path, export_path2, pool_mode="cls")

            # Files should be identical
            with open(export_path1, "rb") as f1, open(export_path2, "rb") as f2:
                assert f1.read() == f2.read()


class TestLoadExportedEncoder:
    """Test loading exported encoder."""

    def test_load_exported_encoder(self, trained_model):
        """Test loading exported encoder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                vocab_size=1000,
                d_model=128,
                n_heads=4,
                n_layers=2,
                d_ff=512,
                strict_validation=False,
            )

            # Save and export
            ckpt_path = Path(tmpdir) / "checkpoint.pt"
            save_checkpoint(
                model=trained_model,
                optimizer=None,
                step=100,
                config=config,
                save_path=ckpt_path,
            )

            export_path = Path(tmpdir) / "exported.safetensors"
            export_encoder(ckpt_path, export_path, pool_mode="cls")

            # Load
            loaded_encoder = load_exported_encoder(export_path)

            assert loaded_encoder is not None
            assert loaded_encoder.pool_mode == "cls"

    def test_loaded_encoder_produces_same_output(self, trained_model):
        """Test that loaded encoder produces same output as original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                vocab_size=1000,
                d_model=128,
                n_heads=4,
                n_layers=2,
                d_ff=512,
                strict_validation=False,
            )

            # Original encoder
            original_exported = ExportedEncoder(
                encoder=trained_model.encoder,
                pool_mode="cls",
            )

            input_ids = torch.randint(0, 1000, (2, 32))
            original_output = original_exported(input_ids)

            # Save, export, and load
            ckpt_path = Path(tmpdir) / "checkpoint.pt"
            save_checkpoint(
                model=trained_model,
                optimizer=None,
                step=100,
                config=config,
                save_path=ckpt_path,
            )

            export_path = Path(tmpdir) / "exported.safetensors"
            export_encoder(ckpt_path, export_path, pool_mode="cls")

            loaded_encoder = load_exported_encoder(export_path)
            loaded_output = loaded_encoder(input_ids)

            # Outputs should match
            assert torch.allclose(original_output, loaded_output, atol=1e-5)


class TestExportManifest:
    """Test manifest creation."""

    def test_create_manifest(self):
        """Test creating export manifest."""
        model_config = {
            "vocab_size": 1000,
            "d_model": 128,
            "n_heads": 4,
        }

        manifest = create_export_manifest(
            model_config=model_config,
            pool_mode="cls",
            seed=42,
        )

        assert "model_config" in manifest
        assert "export_config" in manifest
        assert manifest["export_config"]["pool_mode"] == "cls"
        assert manifest["export_config"]["seed"] == 42

    def test_manifest_has_timestamp(self):
        """Test that manifest includes timestamp."""
        manifest = create_export_manifest(
            model_config={"d_model": 128},
            pool_mode="cls",
        )

        assert "export_config" in manifest
        assert "timestamp" in manifest["export_config"]


class TestExportIntegration:
    """Integration tests for export pipeline."""

    def test_full_export_pipeline(self, trained_model):
        """Test complete export pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                vocab_size=1000,
                d_model=128,
                n_heads=4,
                n_layers=2,
                d_ff=512,
                strict_validation=False,
            )

            # 1. Save checkpoint
            ckpt_path = Path(tmpdir) / "checkpoint.pt"
            save_checkpoint(
                model=trained_model,
                optimizer=None,
                step=100,
                config=config,
                save_path=ckpt_path,
            )

            # 2. Export encoder
            export_path = Path(tmpdir) / "encoder.safetensors"
            export_encoder(ckpt_path, export_path, pool_mode="mean")

            # 3. Load exported encoder
            encoder = load_exported_encoder(export_path)

            # 4. Use for encoding
            input_ids = torch.randint(0, 1000, (4, 32))
            attention_mask = torch.ones(4, 32)
            h_prime = encoder(input_ids, attention_mask=attention_mask)

            # Verify output shape
            assert h_prime.shape == (4, 128)
            assert h_prime.dtype in [torch.float32, torch.float16, torch.bfloat16]
