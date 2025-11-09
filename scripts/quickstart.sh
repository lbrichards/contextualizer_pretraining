#!/usr/bin/env bash
set -euo pipefail

# Contextualizer Pretraining Quickstart Script
# This script demonstrates the complete pipeline

CFG="configs/mlm_qwen7b_3584.yaml"

# Get data directory from environment or default
DATA_DIR="${DATA_SHARDS_DIR:-data/shards}"

echo "==================================="
echo "Contextualizer Pretraining Pipeline"
echo "==================================="
echo ""
echo "Data directory: $DATA_DIR"
echo "Config: $CFG"
echo ""

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Please set DATA_SHARDS_DIR environment variable or create data/shards/"
    exit 1
fi

# 1) Train contextualizer (MLM)
echo "Step 1: Training contextualizer with MLM..."
python -m src.train_mlm \
    --config "$CFG" \
    --data "$DATA_DIR" \
    --output "outputs/train_$(date +%Y%m%d_%H%M%S)"

# 2) Export frozen encoder (strip MLM head)
echo ""
echo "Step 2: Exporting trained encoder..."
python -m src.export_encoder \
    --checkpoint "outputs/*/best_model.pt" \
    --output "exports/contextualizer_v1.safetensors" \
    --pool cls

# 3) Encode corpus into H' shards for Z-model
echo ""
echo "Step 3: Encoding corpus to H' embeddings..."
echo "Note: This step requires implementation of encode_hprime.py"
echo "      which will generate embeddings for downstream use"

echo ""
echo "==================================="
echo "Pipeline complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "  1. Check exports/contextualizer_v1.safetensors"
echo "  2. Use exported encoder for downstream tasks"
echo "  3. Train Z-head on encoded embeddings + teacher logits"
