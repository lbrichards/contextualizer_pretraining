# Contextualizer Pretraining Harness

Self-training contextualizer with MLM (Masked Language Modeling) for Qwen models.

## Overview

This project implements a lightweight contextualizer encoder that can be trained with a masked-token objective and then frozen for use in downstream Z-model pipelines.

## Installation

Using Poetry (recommended):

```bash
poetry install
```

## Quick Start

Coming soon...

## Development

Run tests:

```bash
poetry run pytest
```

Format code:

```bash
poetry run black src tests
poetry run ruff check src tests
```

## License

MIT
