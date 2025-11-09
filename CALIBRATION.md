# Training Calibration Results

## Test Environment
- **Machine**: Apple Silicon (MPS)
- **Data**: LIMA dataset (786 examples, 8 shards)
- **Model**: Contextualizer (1.4B parameters)
  - d_model: 3584
  - n_heads: 16
  - n_layers: 2
  - d_ff: 14336

## Performance Metrics

### Data Loading
- Dataset creation: 0.91s
- Train batches: 177
- Val batches: 20
- Iteration speed: 6.9ms/batch
- Throughput: 580 examples/sec

### Training (20 steps, batch_size=4)
- **Average step time**: 1,751ms (1.75s)
- **Throughput**: 1,169 tokens/sec
- **Tokens per batch**: 2,048 (4 × 512)
- **Initial loss**: 12.11
- **Final loss**: 9.00 (loss decreased in 20 steps!)

### Full Training Estimates

For 10,000 steps (typical full training):
- **Estimated time on MPS**: ~4.9 hours
- **Estimated time on A100**: ~0.5-1.0 hours (10x faster)

For 50,000 steps (extended training):
- **Estimated time on MPS**: ~24 hours
- **Estimated time on A100**: ~2.5-5 hours

## Recommendations

### For Development (MPS)
- Use batch_size=4 for stability
- Run shorter experiments (1K-2K steps)
- Quick iteration and debugging

### For Production (CUDA)
- Increase batch_size to 16-32
- Use full training runs (10K-50K steps)
- Expected 10x speedup on modern GPUs

## Next Steps

1. **Transfer to GPU server** for production training
2. **Adjust batch_size** based on GPU memory
3. **Monitor convergence** with validation loss
4. **Export best checkpoint** when complete
