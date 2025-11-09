# Contextualizer Pretraining Harness

MLM (Masked Language Modeling) pretraining system for Qwen-based contextualizer encoders with comprehensive testing.

## Overview

This project trains a lightweight contextualizer encoder using masked language modeling, then exports it for downstream use in Z-model distillation pipelines. The encoder learns to produce contextual embeddings (H') that can replace traditional embedding layers.

## Features

- ✅ **157 tests with 92% coverage**
- ✅ **Cross-platform**: CUDA (deployment) + MPS (Apple Silicon dev) + CPU
- ✅ **TDD**: Test-driven development throughout
- ✅ **Deterministic**: Reproducible training and encoding
- ✅ **Production-ready**: Checkpointing, resumption, mixed precision
- ✅ **Qwen-compatible**: Dimension presets for all Qwen model sizes

## Quick Start

### Installation

```bash
# Using Poetry (recommended)
poetry install

# Or with pip
pip install -e .
```

### Setup

1. Copy environment template:
```bash
cp .env.example .env
```

2. Edit `.env` and set your data path:
```bash
DATA_SHARDS_DIR=/path/to/your/data/shards
```

### Training

```bash
# Train contextualizer
poetry run python -m src.train_mlm \
    --config configs/mlm_qwen7b_3584.yaml \
    --data $DATA_SHARDS_DIR \
    --output outputs/my_training
```

### Export

```bash
# Export trained encoder
poetry run python -m src.export_encoder \
    --checkpoint outputs/my_training/best_model.pt \
    --output exports/contextualizer_v1.safetensors \
    --pool cls
```

### Use Exported Encoder

```python
from src.export_encoder import load_exported_encoder
import torch

# Load encoder
encoder = load_exported_encoder("exports/contextualizer_v1.safetensors")

# Encode text
input_ids = torch.randint(0, 151936, (4, 512))
h_prime = encoder(input_ids)  # [4, 3584]
```

## Architecture

**Default Configuration (Qwen 7B compatible):**
- `d_model`: 3584 (Qwen 7B hidden size)
- `n_heads`: 16
- `n_layers`: 2
- `d_ff`: 14336 (4× d_model)
- `max_seq_len`: 512
- RoPE position embeddings (θ=10000)

## Data Format

JSONL files with `full_text` field:
```jsonl
{"sentence_id": "example_001", "full_text": "Your text here..."}
{"sentence_id": "example_002", "full_text": "Another example..."}
```

## Training Features

- **MLM Masking**: 80% mask / 10% random / 10% keep
- **Optimizations**:
  - AdamW with weight decay
  - Cosine/Linear LR schedules with warmup
  - Gradient clipping
  - Mixed precision (AMP)
  - Gradient checkpointing
- **Checkpointing**: Automatic best model tracking
- **Resumption**: Continue from any checkpoint

## Configuration

Edit `configs/mlm_qwen7b_3584.yaml`:

```yaml
# Model
d_model: 3584
n_heads: 16
n_layers: 2

# Training
batch_size: 8
learning_rate: 3e-4
max_steps: 10000
warmup_steps: 500

# Data
mask_prob: 0.15
val_split: 0.1
```

## Testing

```bash
# Run all tests
poetry run pytest

# With coverage
poetry run pytest --cov=src --cov-report=html

# Specific test file
poetry run pytest tests/test_modules.py -v
```

## Project Structure

```
.
├── configs/
│   └── mlm_qwen7b_3584.yaml      # Training config
├── src/
│   ├── constants.py               # Qwen presets
│   ├── rope.py                    # RoPE embeddings
│   ├── modules.py                 # Transformer components
│   ├── data.py                    # Data loading + MLM masking
│   ├── train_mlm.py               # Training loop
│   ├── export_encoder.py          # Export utilities
│   └── utils.py                   # Helper functions
├── tests/                         # 157 tests
├── scripts/
│   └── quickstart.sh              # End-to-end demo
└── .env                           # Local configuration (gitignored)
```

## Development

### Code Quality

```bash
# Format code
poetry run black src tests

# Lint
poetry run ruff check src tests
```

### Adding Tests

All new features should include tests. See `tests/` for examples:
- `test_modules.py` - Model architecture tests
- `test_data.py` - Data pipeline tests
- `test_train.py` - Training loop tests
- `test_export.py` - Export utilities tests

## Supported Qwen Models

| Model | d_model | n_layers | n_heads |
|-------|---------|----------|---------|
| 0.5B  | 896     | 24       | 14      |
| 1.5B  | 1536    | 28       | 12      |
| 3B    | 2048    | 36       | 16      |
| **7B** | **3584** | **28**  | **28**  |
| 14B   | 5120    | 48       | 40      |
| 32B   | 5120    | 64       | 40      |
| 72B   | 8192    | 80       | 64      |

## Citation

If you use this code, please cite:

```bibtex
@software{contextualizer_harness_2024,
  title = {Contextualizer Pretraining Harness},
  author = {Richards, Larry},
  year = {2024},
  url = {https://github.com/lbrichards/contextualizer_pretraining}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Add tests for new features
2. Maintain >90% coverage
3. Follow existing code style
4. Update documentation

## Acknowledgments

Built with:
- PyTorch 2.2+
- Transformers 4.44+
- Poetry for dependency management
- Comprehensive test suite with pytest
