#!/usr/bin/env python
"""Calibration script to measure training throughput on local machine."""

import time
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer

from src.train_mlm import TrainingConfig, Trainer
from src.data import create_mlm_dataloaders
from src.utils import set_seed, get_device, count_parameters, format_number


def calibrate_data_loading(data_dir: str, batch_size: int = 8, num_batches: int = 10):
    """Measure data loading and preprocessing speed."""
    print("\n" + "="*60)
    print("DATA LOADING CALIBRATION")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

    print(f"\nLoading data from: {data_dir}")
    start = time.time()

    train_loader, val_loader = create_mlm_dataloaders(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=512,
        mask_prob=0.15,
        val_split=0.1,
        seed=42,
    )

    load_time = time.time() - start
    print(f"Dataset creation time: {load_time:.2f}s")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader) if val_loader else 0}")

    # Measure batch iteration speed
    print(f"\nMeasuring iteration speed (first {num_batches} batches)...")
    start = time.time()

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

    iter_time = time.time() - start
    time_per_batch = iter_time / num_batches

    print(f"Total time: {iter_time:.2f}s")
    print(f"Time per batch: {time_per_batch*1000:.1f}ms")
    print(f"Throughput: {batch_size * num_batches / iter_time:.1f} examples/sec")

    return train_loader, val_loader, tokenizer


def calibrate_training(
    data_dir: str,
    num_steps: int = 50,
    batch_size: int = 8,
    device: str = "auto",
):
    """Measure training speed and memory usage."""
    print("\n" + "="*60)
    print("TRAINING CALIBRATION")
    print("="*60)

    # Create minimal config for calibration
    config = TrainingConfig(
        # Model (small version for calibration)
        vocab_size=151936,
        d_model=3584,
        n_heads=16,
        n_layers=2,
        d_ff=14336,
        max_seq_len=512,
        dropout=0.1,
        # Training
        batch_size=batch_size,
        learning_rate=3e-4,
        max_steps=num_steps,
        warmup_steps=10,
        log_interval=10,
        eval_interval=num_steps + 1,  # Skip eval during calibration
        save_interval=num_steps + 1,  # Skip saving
        # System
        device=device,
        dtype="bf16",
        use_amp=True,
        gradient_checkpointing=True,
        seed=42,
    )

    print(f"\nDevice: {get_device(device)}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: 512")
    print(f"Calibration steps: {num_steps}")

    # Create trainer
    print("\nInitializing model...")
    trainer = Trainer(config=config, data_dir=data_dir)

    # Print memory usage if CUDA
    if torch.cuda.is_available():
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        print("Running on MPS (Apple Silicon)")

    # Run calibration
    print(f"\nRunning {num_steps} training steps...")
    start = time.time()

    step_times = []
    losses = []

    for i, batch in enumerate(trainer.train_loader):
        if i >= num_steps:
            break

        step_start = time.time()
        loss = trainer.training_step(batch)
        step_time = time.time() - step_start

        step_times.append(step_time)
        losses.append(loss)

        if (i + 1) % 10 == 0:
            avg_time = sum(step_times[-10:]) / len(step_times[-10:])
            avg_loss = sum(losses[-10:]) / len(losses[-10:])
            print(f"Step {i+1}/{num_steps} | Loss: {avg_loss:.4f} | "
                  f"Time: {avg_time*1000:.1f}ms/step")

    total_time = time.time() - start

    # Calculate statistics
    avg_step_time = sum(step_times) / len(step_times)
    tokens_per_batch = batch_size * 512
    tokens_per_sec = tokens_per_batch / avg_step_time

    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Average step time: {avg_step_time*1000:.1f}ms")
    print(f"Tokens per batch: {format_number(tokens_per_batch)}")
    print(f"Throughput: {format_number(int(tokens_per_sec))} tokens/sec")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory: {peak_memory:.2f} GB")

    # Estimate full training time
    full_steps = config.max_steps if hasattr(config, 'max_steps') else 10000
    estimated_hours = (full_steps * avg_step_time) / 3600

    print(f"\nEstimated time for {full_steps} steps: {estimated_hours:.2f} hours")

    return {
        "avg_step_time_ms": avg_step_time * 1000,
        "tokens_per_sec": int(tokens_per_sec),
        "total_time": total_time,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate training on local machine")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Data directory (uses DATA_SHARDS_DIR env var if not specified)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of calibration steps (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)",
    )

    args = parser.parse_args()

    # Get data directory
    if args.data:
        data_dir = args.data
    else:
        import os
        data_dir = os.getenv("DATA_SHARDS_DIR")
        if not data_dir:
            print("Error: Please specify --data or set DATA_SHARDS_DIR environment variable")
            return

    # Check data exists
    if not Path(data_dir).exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    print("="*60)
    print("CONTEXTUALIZER TRAINING CALIBRATION")
    print("="*60)
    print(f"\nData directory: {data_dir}")
    print(f"Calibration steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")

    # Run calibrations
    calibrate_data_loading(data_dir, batch_size=args.batch_size)
    results = calibrate_training(
        data_dir=data_dir,
        num_steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Save results
    import json
    results_path = Path("calibration_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
