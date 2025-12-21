#!/usr/bin/env python3
"""
Benchmark training speed: train_ratio.py vs dinov3_train.py

Compares:
1. models_ratio.py DirectDINO (original)
2. dinov3_models.py DINOv3Direct (new)

Usage:
    python -m src.benchmark_training --device-type mps
"""
import argparse
import gc
import os
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .dataset import BiomassDataset, get_train_transforms, prepare_dataframe
from .device import DeviceType, get_device, get_device_type, empty_cache


def create_real_dataloader(
    base_path: str,
    batch_size: int = 8,
    img_size: int = 576,
    max_samples: int = 100,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = False,  # Only for CUDA
    persistent_workers: bool = False,  # Can cause issues on MPS
) -> DataLoader:
    """Create dataloader with real images."""
    train_csv = os.path.join(base_path, "train.csv")
    image_dir = os.path.join(base_path, "train")
    
    df = prepare_dataframe(train_csv)
    
    # Limit samples for faster benchmarking
    if max_samples and len(df) > max_samples:
        df = df.head(max_samples)
    
    transform = get_train_transforms(img_size, aug_prob=0.5)
    
    dataset = BiomassDataset(df, image_dir, transform, is_train=True)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=True,
    )
    
    return loader


def benchmark_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    device_type: DeviceType,
    n_epochs: int = 3,
    warmup_epochs: int = 1,
) -> Tuple[float, float]:
    """
    Benchmark a model.
    
    Returns:
        avg_epoch_time: Average time per epoch (excluding warmup)
        avg_batch_time: Average time per batch
    """
    model.to(device)
    model.train()
    
    # Freeze backbone for fair comparison (head-only training)
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False
    
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    loss_fn = nn.MSELoss()
    
    use_channels_last = device_type == DeviceType.MPS
    
    epoch_times = []
    batch_times = []
    
    for epoch in range(n_epochs):
        epoch_start = time.perf_counter()
        
        for batch in loader:
            batch_start = time.perf_counter()
            
            # BiomassDataset returns (left, right, targets, ...)
            x_left, x_right, targets = batch[0], batch[1], batch[2]
            
            if use_channels_last:
                x_left = x_left.to(device, memory_format=torch.channels_last)
                x_right = x_right.to(device, memory_format=torch.channels_last)
            else:
                x_left = x_left.to(device)
                x_right = x_right.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            green, dead, clover, gdm, total = model(x_left, x_right)
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
            loss = loss_fn(preds, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Sync for accurate timing
            if device_type == DeviceType.MPS:
                torch.mps.synchronize()
            elif device_type == DeviceType.CUDA:
                torch.cuda.synchronize()
            
            batch_time = time.perf_counter() - batch_start
            
            # Only record after warmup
            if epoch >= warmup_epochs:
                batch_times.append(batch_time)
        
        epoch_time = time.perf_counter() - epoch_start
        
        if epoch >= warmup_epochs:
            epoch_times.append(epoch_time)
        
        print(f"    Epoch {epoch + 1}: {epoch_time:.2f}s")
    
    avg_epoch = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0
    
    return avg_epoch, avg_batch


def main():
    parser = argparse.ArgumentParser(description="Benchmark training speed")
    parser.add_argument("--device-type", type=str, default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=8, help="Increase for faster training if you have memory")
    parser.add_argument("--img-size", type=int, default=576)
    parser.add_argument("--n-samples", type=int, default=100, help="Number of real images to use")
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (try 8-12)")
    parser.add_argument("--prefetch", type=int, default=2, help="Batches to prefetch per worker")
    args = parser.parse_args()
    
    device_type = DeviceType(args.device_type) if args.device_type else get_device_type()
    device = get_device(device_type)
    
    print("=" * 60)
    print("Training Benchmark (Real Images)")
    print("=" * 60)
    print(f"Device: {device} ({device_type.value})")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Grid: {args.grid}x{args.grid}")
    print(f"Samples: {args.n_samples}")
    print(f"Workers: {args.num_workers} (prefetch={args.prefetch})")
    print(f"Epochs: {args.n_epochs} (1 warmup)")
    print("=" * 60)
    
    # Create real data loader
    base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    print(f"\nLoading real images from: {base_path}")
    is_cuda = device_type == DeviceType.CUDA
    loader = create_real_dataloader(
        base_path, 
        args.batch_size, 
        args.img_size, 
        max_samples=args.n_samples,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
        pin_memory=is_cuda,  # Only for CUDA
        persistent_workers=is_cuda,  # Can cause issues on MPS
    )
    
    results = {}
    
    # Benchmark 1: models_ratio.py DirectDINO (original)
    print("\n" + "-" * 60)
    print("1. models_ratio.py - DirectDINO (original)")
    print("-" * 60)
    
    try:
        from .models_ratio import build_ratio_model
        
        model1 = build_ratio_model(
            backbone_name="vit_base_patch16_dinov3",
            grid=(args.grid, args.grid),
            pretrained=True,
            model_type="direct",
        )
        
        avg_epoch1, avg_batch1 = benchmark_model(
            model1, loader, device, device_type, args.n_epochs
        )
        
        results["models_ratio"] = {
            "avg_epoch": avg_epoch1,
            "avg_batch": avg_batch1,
        }
        
        del model1
        empty_cache(device_type)
        gc.collect()
        
    except Exception as e:
        print(f"  Error: {e}")
        results["models_ratio"] = None
    
    # Benchmark 2: dinov3_models.py DINOv3Direct (new)
    print("\n" + "-" * 60)
    print("2. dinov3_models.py - DINOv3Direct (new)")
    print("-" * 60)
    
    try:
        from .dinov3_models import DINOv3Direct
        
        model2 = DINOv3Direct(
            grid=(args.grid, args.grid),
            pretrained=True,
        )
        
        avg_epoch2, avg_batch2 = benchmark_model(
            model2, loader, device, device_type, args.n_epochs
        )
        
        results["dinov3_models"] = {
            "avg_epoch": avg_epoch2,
            "avg_batch": avg_batch2,
        }
        
        del model2
        empty_cache(device_type)
        gc.collect()
        
    except Exception as e:
        print(f"  Error: {e}")
        results["dinov3_models"] = None
    
    # Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    print(f"\n{'Model':<25} {'Epoch (s)':<12} {'Batch (ms)':<12} {'Batches/s':<12}")
    print("-" * 60)
    
    for name, data in results.items():
        if data:
            batches_per_sec = 1.0 / data["avg_batch"] if data["avg_batch"] > 0 else 0
            print(f"{name:<25} {data['avg_epoch']:<12.2f} {data['avg_batch']*1000:<12.1f} {batches_per_sec:<12.2f}")
        else:
            print(f"{name:<25} {'FAILED':<12}")
    
    # Comparison
    if results.get("models_ratio") and results.get("dinov3_models"):
        r1 = results["models_ratio"]
        r2 = results["dinov3_models"]
        
        speedup = r1["avg_epoch"] / r2["avg_epoch"] if r2["avg_epoch"] > 0 else 0
        
        print("\n" + "-" * 60)
        if speedup > 1:
            print(f"dinov3_models is {speedup:.2f}x FASTER than models_ratio")
        elif speedup < 1:
            print(f"dinov3_models is {1/speedup:.2f}x SLOWER than models_ratio")
        else:
            print("Both are approximately the same speed")


if __name__ == "__main__":
    main()

