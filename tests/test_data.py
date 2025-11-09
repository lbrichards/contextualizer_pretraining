"""Tests for data loading and MLM masking."""

import pytest
import torch
import tempfile
import json
from pathlib import Path

from src.data import (
    load_jsonl_shard,
    MLMDataset,
    create_mlm_dataloaders,
    apply_mlm_masking,
)
from src.utils import set_seed


@pytest.fixture
def sample_jsonl_file():
    """Create a temporary JSONL file for testing."""
    data = [
        {"sentence_id": "test_001", "full_text": "This is a test sentence for MLM training."},
        {"sentence_id": "test_002", "full_text": "Another example with different content."},
        {"sentence_id": "test_003", "full_text": "Short text."},
        {
            "sentence_id": "test_004",
            "full_text": "A longer sentence that contains more words and should be useful for testing.",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        tmp_path = f.name

    yield tmp_path

    # Cleanup
    Path(tmp_path).unlink()


@pytest.fixture
def sample_data_dir():
    """Create a temporary directory with multiple JSONL shards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 3 shard files
        for i in range(3):
            data = [
                {
                    "sentence_id": f"shard{i}_example{j}",
                    "full_text": f"This is example {j} from shard {i}.",
                }
                for j in range(5)
            ]
            shard_path = Path(tmpdir) / f"shard_{i:03d}.jsonl"
            with open(shard_path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")

        yield tmpdir


class TestLoadJsonlShard:
    """Test loading JSONL shard files."""

    def test_load_single_shard(self, sample_jsonl_file):
        """Test loading a single JSONL file."""
        texts = load_jsonl_shard(sample_jsonl_file, text_field="full_text")
        assert len(texts) == 4
        assert texts[0] == "This is a test sentence for MLM training."
        assert texts[2] == "Short text."

    def test_load_with_limit(self, sample_jsonl_file):
        """Test loading with max_examples limit."""
        texts = load_jsonl_shard(sample_jsonl_file, text_field="full_text", max_examples=2)
        assert len(texts) == 2

    def test_custom_text_field(self, sample_jsonl_file):
        """Test loading with custom text field."""
        ids = load_jsonl_shard(sample_jsonl_file, text_field="sentence_id")
        assert len(ids) == 4
        assert ids[0] == "test_001"

    def test_empty_file(self):
        """Test handling of empty JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            tmp_path = f.name

        try:
            texts = load_jsonl_shard(tmp_path, text_field="full_text")
            assert len(texts) == 0
        finally:
            Path(tmp_path).unlink()


class TestMLMMasking:
    """Test MLM masking logic."""

    def test_apply_mlm_masking_shape(self):
        """Test that masking preserves tensor shape."""
        input_ids = torch.randint(0, 1000, (4, 16))
        mask_token_id = 999

        masked_ids, labels = apply_mlm_masking(
            input_ids, mask_token_id=mask_token_id, mask_prob=0.15
        )

        assert masked_ids.shape == input_ids.shape
        assert labels.shape == input_ids.shape

    def test_mlm_masking_creates_labels(self):
        """Test that labels are properly created (mostly -100)."""
        input_ids = torch.randint(0, 1000, (4, 16))
        mask_token_id = 999

        _, labels = apply_mlm_masking(input_ids, mask_token_id=mask_token_id, mask_prob=0.15)

        # Most positions should be -100 (ignored)
        num_masked = (labels != -100).sum().item()
        total = labels.numel()

        # Should be roughly 15% masked
        assert 0 < num_masked < total
        assert num_masked / total < 0.3  # Some variance allowed

    def test_mlm_masking_deterministic(self):
        """Test deterministic masking with seed."""
        input_ids = torch.randint(0, 1000, (4, 16))
        mask_token_id = 999

        set_seed(42)
        masked_ids1, labels1 = apply_mlm_masking(
            input_ids, mask_token_id=mask_token_id, mask_prob=0.15
        )

        set_seed(42)
        masked_ids2, labels2 = apply_mlm_masking(
            input_ids, mask_token_id=mask_token_id, mask_prob=0.15
        )

        assert torch.equal(masked_ids1, masked_ids2)
        assert torch.equal(labels1, labels2)

    def test_mlm_masking_strategies(self):
        """Test that masking includes mask/random/keep strategies."""
        input_ids = torch.ones(100, 50, dtype=torch.long) * 500
        mask_token_id = 999

        set_seed(42)
        masked_ids, labels = apply_mlm_masking(
            input_ids,
            mask_token_id=mask_token_id,
            mask_prob=0.15,
            replace_probs={"mask": 0.8, "random": 0.1, "keep": 0.1},
        )

        # Find masked positions
        masked_positions = labels != -100

        # At masked positions, check different strategies were applied
        masked_tokens = masked_ids[masked_positions]

        # Should have some mask tokens
        num_mask_tokens = (masked_tokens == mask_token_id).sum().item()
        assert num_mask_tokens > 0

        # Should have some original tokens (keep strategy)
        num_kept = (masked_tokens == 500).sum().item()
        assert num_kept > 0

    def test_zero_mask_prob(self):
        """Test that zero mask probability doesn't mask anything."""
        input_ids = torch.randint(0, 1000, (4, 16))
        mask_token_id = 999

        masked_ids, labels = apply_mlm_masking(
            input_ids, mask_token_id=mask_token_id, mask_prob=0.0
        )

        # No masking should occur
        assert torch.equal(masked_ids, input_ids)
        assert torch.all(labels == -100)


class TestMLMDataset:
    """Test MLM dataset."""

    def test_dataset_initialization(self, sample_data_dir):
        """Test dataset initialization."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        dataset = MLMDataset(
            data_dir=sample_data_dir,
            tokenizer=tokenizer,
            max_length=128,
            mask_prob=0.15,
        )

        assert len(dataset) > 0
        assert dataset.max_length == 128

    def test_dataset_getitem(self, sample_data_dir):
        """Test getting items from dataset."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        dataset = MLMDataset(
            data_dir=sample_data_dir,
            tokenizer=tokenizer,
            max_length=128,
            mask_prob=0.15,
        )

        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

        assert item["input_ids"].shape[0] == 128
        assert item["attention_mask"].shape[0] == 128
        assert item["labels"].shape[0] == 128

    def test_dataset_length(self, sample_data_dir):
        """Test dataset length calculation."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        dataset = MLMDataset(
            data_dir=sample_data_dir,
            tokenizer=tokenizer,
            max_length=128,
            mask_prob=0.15,
        )

        # We created 3 shards with 5 examples each = 15 total
        assert len(dataset) == 15

    def test_dataset_deterministic(self, sample_data_dir):
        """Test deterministic dataset with seed."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

        set_seed(42)
        dataset1 = MLMDataset(
            data_dir=sample_data_dir,
            tokenizer=tokenizer,
            max_length=128,
            mask_prob=0.15,
            seed=42,
        )
        item1 = dataset1[0]

        set_seed(42)
        dataset2 = MLMDataset(
            data_dir=sample_data_dir,
            tokenizer=tokenizer,
            max_length=128,
            mask_prob=0.15,
            seed=42,
        )
        item2 = dataset2[0]

        # Masking should be deterministic
        assert torch.equal(item1["input_ids"], item2["input_ids"])
        assert torch.equal(item1["labels"], item2["labels"])


class TestCreateMLMDataloaders:
    """Test dataloader creation."""

    def test_create_train_val_split(self, sample_data_dir):
        """Test creating train/val dataloaders."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

        train_loader, val_loader = create_mlm_dataloaders(
            data_dir=sample_data_dir,
            tokenizer=tokenizer,
            batch_size=2,
            max_length=128,
            mask_prob=0.15,
            val_split=0.2,
            seed=42,
        )

        assert train_loader is not None
        assert val_loader is not None

        # Check we can iterate
        train_batch = next(iter(train_loader))
        assert "input_ids" in train_batch
        assert train_batch["input_ids"].shape[0] == 2  # batch size

    def test_dataloader_batch_size(self, sample_data_dir):
        """Test that dataloader respects batch size."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

        train_loader, _ = create_mlm_dataloaders(
            data_dir=sample_data_dir,
            tokenizer=tokenizer,
            batch_size=4,
            max_length=128,
            mask_prob=0.15,
            val_split=0.2,
            seed=42,
        )

        batch = next(iter(train_loader))
        assert batch["input_ids"].shape[0] == 4

    def test_no_val_split(self, sample_data_dir):
        """Test creating dataloader without validation split."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

        train_loader, val_loader = create_mlm_dataloaders(
            data_dir=sample_data_dir,
            tokenizer=tokenizer,
            batch_size=2,
            max_length=128,
            mask_prob=0.15,
            val_split=0.0,
            seed=42,
        )

        assert train_loader is not None
        assert val_loader is None


class TestDataIntegration:
    """Integration tests with real data format."""

    def test_with_lima_format(self, sample_jsonl_file):
        """Test loading data in LIMA format."""
        texts = load_jsonl_shard(sample_jsonl_file, text_field="full_text")
        assert len(texts) > 0
        assert all(isinstance(text, str) for text in texts)

    def test_end_to_end_pipeline(self, sample_data_dir):
        """Test complete pipeline from data loading to batching."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

        # Create dataloaders
        train_loader, val_loader = create_mlm_dataloaders(
            data_dir=sample_data_dir,
            tokenizer=tokenizer,
            batch_size=4,
            max_length=128,
            mask_prob=0.15,
            val_split=0.2,
            seed=42,
        )

        # Get a batch from each
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        # Verify shapes
        assert train_batch["input_ids"].shape == (4, 128)
        assert val_batch["input_ids"].shape[0] <= 4
        assert val_batch["input_ids"].shape[1] == 128

        # Verify labels have masked positions
        assert (train_batch["labels"] != -100).any()
