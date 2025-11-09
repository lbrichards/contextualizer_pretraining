"""Data loading and MLM masking for contextualizer training."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import PreTrainedTokenizer
import numpy as np

from src.utils import set_seed


def load_jsonl_shard(
    filepath: str,
    text_field: str = "full_text",
    max_examples: Optional[int] = None,
) -> List[str]:
    """
    Load texts from a JSONL (JSON Lines) shard file.

    Args:
        filepath: Path to JSONL file
        text_field: Field name containing the text
        max_examples: Maximum number of examples to load (None = all)

    Returns:
        List of text strings

    Examples:
        >>> import tempfile, json
        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        ...     _ = f.write(json.dumps({"full_text": "Hello world"}) + "\\n")
        ...     _ = f.write(json.dumps({"full_text": "Second line"}) + "\\n")
        ...     tmp = f.name
        >>> texts = load_jsonl_shard(tmp, text_field="full_text")
        >>> len(texts)
        2
        >>> texts[0]
        'Hello world'
        >>> import os; os.unlink(tmp)
    """
    texts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                if text_field in data:
                    texts.append(data[text_field])
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line {i+1} in {filepath}")
                continue

    return texts


def apply_mlm_masking(
    input_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int = 151936,  # Qwen vocab size
    mask_prob: float = 0.15,
    replace_probs: Optional[Dict[str, float]] = None,
    special_token_ids: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply MLM masking to input token IDs.

    Masking strategy (Devlin et al., 2019):
    - 15% of tokens are selected for masking
    - Of those selected:
        - 80% replaced with [MASK]
        - 10% replaced with random token
        - 10% kept unchanged

    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        mask_token_id: ID of the mask token
        vocab_size: Size of vocabulary for random replacement
        mask_prob: Probability of masking a token (default 0.15)
        replace_probs: Dict with 'mask', 'random', 'keep' probabilities
        special_token_ids: List of special token IDs to never mask

    Returns:
        Tuple of (masked_input_ids, labels) where labels are -100 for unmasked positions

    Examples:
        >>> input_ids = torch.randint(0, 1000, (2, 10))
        >>> masked, labels = apply_mlm_masking(input_ids, mask_token_id=999, vocab_size=1000)
        >>> masked.shape == input_ids.shape
        True
        >>> labels.shape == input_ids.shape
        True
    """
    if replace_probs is None:
        replace_probs = {"mask": 0.8, "random": 0.1, "keep": 0.1}

    if special_token_ids is None:
        special_token_ids = []

    # Clone to avoid modifying original
    input_ids = input_ids.clone()
    labels = input_ids.clone()

    # Create masking probability matrix
    probability_matrix = torch.full(input_ids.shape, mask_prob)

    # Don't mask special tokens
    for special_id in special_token_ids:
        special_tokens_mask = input_ids == special_id
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Randomly select positions to mask
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Set labels to -100 for unmasked positions (ignored in loss)
    labels[~masked_indices] = -100

    # Of the masked positions, determine replacement strategy
    num_masked = masked_indices.sum().item()
    if num_masked == 0:
        return input_ids, labels

    # Get indices of masked positions
    masked_positions = masked_indices.nonzero(as_tuple=True)

    # Split into mask/random/keep according to probabilities
    num_mask = int(num_masked * replace_probs["mask"])
    num_random = int(num_masked * replace_probs["random"])
    # num_keep is the remainder

    # Shuffle masked positions
    perm = torch.randperm(num_masked)

    # Apply mask token
    mask_positions = perm[:num_mask]
    for idx in mask_positions:
        batch_idx = masked_positions[0][idx]
        seq_idx = masked_positions[1][idx]
        input_ids[batch_idx, seq_idx] = mask_token_id

    # Apply random token
    random_positions = perm[num_mask : num_mask + num_random]
    for idx in random_positions:
        batch_idx = masked_positions[0][idx]
        seq_idx = masked_positions[1][idx]
        input_ids[batch_idx, seq_idx] = torch.randint(0, vocab_size, (1,)).item()

    # Keep original tokens for remaining positions (no change needed)

    return input_ids, labels


class MLMDataset(Dataset):
    """
    Dataset for Masked Language Modeling.

    Args:
        data_dir: Directory containing JSONL shard files
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        mask_prob: Probability of masking tokens
        text_field: Field name in JSONL containing text
        file_pattern: Glob pattern for shard files
        replace_probs: Masking strategy probabilities
        seed: Random seed for deterministic masking

    Examples:
        >>> import tempfile
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")  # doctest: +SKIP
        >>> # Create test data
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     import json
        ...     with open(f"{tmpdir}/test.jsonl", "w") as f:
        ...         f.write(json.dumps({"full_text": "Hello world"}) + "\\n")
        ...     dataset = MLMDataset(tmpdir, tokenizer, max_length=32)
        ...     len(dataset) > 0
        True
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        mask_prob: float = 0.15,
        text_field: str = "full_text",
        file_pattern: str = "*.json*",  # Matches .json and .jsonl
        replace_probs: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.text_field = text_field
        self.replace_probs = replace_probs or {"mask": 0.8, "random": 0.1, "keep": 0.1}
        self.seed = seed

        # Find all shard files
        self.shard_files = sorted(self.data_dir.glob(file_pattern))
        if not self.shard_files:
            raise ValueError(f"No files matching {file_pattern} found in {data_dir}")

        # Load all texts from all shards
        self.texts = []
        for shard_file in self.shard_files:
            shard_texts = load_jsonl_shard(shard_file, text_field=text_field)
            self.texts.extend(shard_texts)

        if not self.texts:
            raise ValueError(f"No texts found in {data_dir}")

        # Get special token IDs to avoid masking
        self.special_token_ids = list(tokenizer.all_special_ids)

        # Store mask token ID and vocab size
        self.mask_token_id = tokenizer.mask_token_id
        if self.mask_token_id is None:
            # Qwen and some tokenizers don't have mask token
            # Add it if it doesn't exist
            if tokenizer.mask_token is None:
                special_tokens = {"mask_token": "<mask>"}
                tokenizer.add_special_tokens(special_tokens)
                self.mask_token_id = tokenizer.mask_token_id
            else:
                self.mask_token_id = tokenizer.mask_token_id

        self.vocab_size = len(tokenizer)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example with MLM masking applied.

        Args:
            idx: Index of example

        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        # Set seed for deterministic masking if provided
        if self.seed is not None:
            set_seed(self.seed + idx)

        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Get tensors and squeeze batch dimension
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Apply MLM masking
        masked_input_ids, labels = apply_mlm_masking(
            input_ids.unsqueeze(0),  # Add batch dim
            mask_token_id=self.mask_token_id,
            vocab_size=self.vocab_size,
            mask_prob=self.mask_prob,
            replace_probs=self.replace_probs,
            special_token_ids=self.special_token_ids,
        )

        # Remove batch dimension
        masked_input_ids = masked_input_ids.squeeze(0)
        labels = labels.squeeze(0)

        # Don't compute loss on padding tokens
        labels[attention_mask == 0] = -100

        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_mlm_dataloaders(
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    mask_prob: float = 0.15,
    val_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    replace_probs: Optional[Dict[str, float]] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders for MLM.

    Args:
        data_dir: Directory containing JSONL files
        tokenizer: Hugging Face tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        mask_prob: Probability of masking
        val_split: Fraction of data for validation (0.0 = no val split)
        num_workers: Number of dataloader workers
        seed: Random seed
        replace_probs: Masking strategy probabilities

    Returns:
        Tuple of (train_loader, val_loader). val_loader is None if val_split=0

    Examples:
        >>> from transformers import AutoTokenizer
        >>> import tempfile, json
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")  # doctest: +SKIP
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     with open(f"{tmpdir}/test.jsonl", "w") as f:
        ...         for i in range(10):
        ...             f.write(json.dumps({"full_text": f"Example {i}"}) + "\\n")
        ...     train_loader, val_loader = create_mlm_dataloaders(
        ...         tmpdir, tokenizer, batch_size=2, val_split=0.2, seed=42
        ...     )
        ...     train_loader is not None
        True
    """
    # Create full dataset
    dataset = MLMDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        mask_prob=mask_prob,
        replace_probs=replace_probs,
        seed=seed,
    )

    # Split into train/val if requested
    if val_split > 0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        # Use deterministic split
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        return train_loader, val_loader
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        return train_loader, None
