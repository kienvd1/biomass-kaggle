#!/usr/bin/env python3
"""
MPS Memory Profiler and Batch Size Finder for Apple Silicon.

Usage:
    python -m src.profile_mps --backbone vit_base_patch14_reg4_dinov2.lvd142m
"""
import argparse
import gc
import time
from typing import Tuple

import torch
import psutil

from .models import build_model


def get_memory_info() -> dict:
    """Get current memory usage."""
    # System memory
    vm = psutil.virtual_memory()
    
    info = {
        "system_total_gb": vm.total / (1024**3),
        "system_available_gb": vm.available / (1024**3),
        "system_used_gb": vm.used / (1024**3),
        "system_percent": vm.percent,
    }
    
    # MPS memory (if available)
    if torch.backends.mps.is_available():
        # MPS uses unified memory, so we track via system memory
        # But we can check MPS allocator stats
        if hasattr(torch.mps, "current_allocated_memory"):
            info["mps_allocated_gb"] = torch.mps.current_allocated_memory() / (1024**3)
        if hasattr(torch.mps, "driver_allocated_memory"):
            info["mps_driver_gb"] = torch.mps.driver_allocated_memory() / (1024**3)
    
    return info


def find_max_batch_size(
    backbone: str,
    grid: Tuple[int, int] = (2, 2),
    img_size: int = 518,
    start_batch: int = 4,
    max_batch: int = 128,
    num_targets: int = 5,
) -> int:
    """Find maximum batch size that fits in memory."""
    
    device = torch.device("mps")
    print(f"\n{'='*60}")
    print(f"Finding max batch size for: {backbone}")
    print(f"Grid: {grid}, Image size: {img_size}")
    print(f"{'='*60}")
    
    # Initial memory
    gc.collect()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    
    mem_before = get_memory_info()
    print(f"\nInitial memory: {mem_before['system_used_gb']:.1f} GB used / {mem_before['system_total_gb']:.1f} GB total")
    
    # Build model
    print(f"\nBuilding model...")
    model = build_model(
        backbone_name=backbone,
        model_type="tiled_film",
        grid=grid,
        pretrained=True,
        dropout=0.3,
        hidden_ratio=0.25,
    ).to(device)
    model.train()
    
    mem_after_model = get_memory_info()
    model_mem = mem_after_model['system_used_gb'] - mem_before['system_used_gb']
    print(f"Model loaded: +{model_mem:.2f} GB")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")
    
    # Binary search for max batch size
    best_batch = start_batch
    batch_size = start_batch
    
    print(f"\nTesting batch sizes...")
    
    while batch_size <= max_batch:
        gc.collect()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        
        try:
            # Create dummy inputs
            x_left = torch.randn(batch_size, 3, img_size, img_size, device=device)
            x_right = torch.randn(batch_size, 3, img_size, img_size, device=device)
            targets = torch.randn(batch_size, num_targets, device=device)
            
            # Forward pass
            with torch.autocast(device_type="mps", dtype=torch.float16):
                green, dead, clover, gdm, total = model(x_left, x_right)
                preds = torch.cat([green, dead, clover, gdm, total], dim=1)
                loss = torch.nn.functional.mse_loss(preds, targets)
            
            # Backward pass
            loss.backward()
            
            # Sync MPS
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            
            mem_info = get_memory_info()
            print(f"  Batch {batch_size:3d}: ✓ ({mem_info['system_used_gb']:.1f} GB used, {mem_info['system_percent']:.0f}%)")
            
            best_batch = batch_size
            
            # Clear for next iteration
            del x_left, x_right, targets, preds, loss, green, dead, clover, gdm, total
            model.zero_grad(set_to_none=True)
            
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "mps" in str(e).lower():
                print(f"  Batch {batch_size:3d}: ✗ OOM")
                break
            else:
                raise e
    
    # Cleanup
    del model
    gc.collect()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    
    # Recommend slightly lower for safety
    safe_batch = max(best_batch // 2, start_batch) if best_batch > start_batch else best_batch
    
    print(f"\n{'='*60}")
    print(f"Results for {backbone}:")
    print(f"  Max tested batch size: {best_batch}")
    print(f"  Recommended batch size: {safe_batch} (with headroom)")
    print(f"{'='*60}")
    
    return safe_batch


def benchmark_training_speed(
    backbone: str,
    batch_size: int,
    grid: Tuple[int, int] = (2, 2),
    img_size: int = 518,
    num_iterations: int = 10,
) -> float:
    """Benchmark training speed (iterations per second)."""
    
    device = torch.device("mps")
    
    print(f"\nBenchmarking {backbone} with batch_size={batch_size}...")
    
    # Build model
    model = build_model(
        backbone_name=backbone,
        model_type="tiled_film",
        grid=grid,
        pretrained=True,
        dropout=0.3,
        hidden_ratio=0.25,
    ).to(device)
    model.train()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Warmup
    for _ in range(3):
        x_left = torch.randn(batch_size, 3, img_size, img_size, device=device)
        x_right = torch.randn(batch_size, 3, img_size, img_size, device=device)
        targets = torch.randn(batch_size, 5, device=device)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="mps", dtype=torch.float16):
            green, dead, clover, gdm, total = model(x_left, x_right)
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
            loss = torch.nn.functional.mse_loss(preds, targets)
        loss.backward()
        optimizer.step()
    
    if hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    
    # Benchmark
    start_time = time.perf_counter()
    
    for _ in range(num_iterations):
        x_left = torch.randn(batch_size, 3, img_size, img_size, device=device)
        x_right = torch.randn(batch_size, 3, img_size, img_size, device=device)
        targets = torch.randn(batch_size, 5, device=device)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="mps", dtype=torch.float16):
            green, dead, clover, gdm, total = model(x_left, x_right)
            preds = torch.cat([green, dead, clover, gdm, total], dim=1)
            loss = torch.nn.functional.mse_loss(preds, targets)
        loss.backward()
        optimizer.step()
    
    if hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    
    elapsed = time.perf_counter() - start_time
    iter_per_sec = num_iterations / elapsed
    samples_per_sec = iter_per_sec * batch_size
    
    print(f"  {iter_per_sec:.2f} iter/s, {samples_per_sec:.1f} samples/s")
    
    # Cleanup
    del model, optimizer
    gc.collect()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    
    return samples_per_sec


def main():
    parser = argparse.ArgumentParser(description="MPS Memory Profiler")
    parser.add_argument(
        "--backbone", type=str, 
        default="vit_base_patch14_reg4_dinov2.lvd142m",
        help="Backbone model name"
    )
    parser.add_argument("--all", action="store_true", help="Profile all DINOv2 backbones")
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark")
    
    args = parser.parse_args()
    
    if not torch.backends.mps.is_available():
        print("MPS not available! This script is for Apple Silicon Macs.")
        return
    
    print(f"\n{'='*60}")
    print("MPS Memory Profiler for Apple Silicon")
    print(f"{'='*60}")
    
    mem = get_memory_info()
    print(f"System Memory: {mem['system_total_gb']:.1f} GB total")
    print(f"Available: {mem['system_available_gb']:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    
    backbones = [
        "vit_base_patch14_reg4_dinov2.lvd142m",
        "vit_small_patch14_reg4_dinov2.lvd142m",
    ] if args.all else [args.backbone]
    
    results = {}
    
    for backbone in backbones:
        batch_size = find_max_batch_size(backbone)
        results[backbone] = {"batch_size": batch_size}
        
        if args.benchmark:
            speed = benchmark_training_speed(backbone, batch_size)
            results[backbone]["samples_per_sec"] = speed
    
    # Summary
    print(f"\n{'='*60}")
    print("RECOMMENDED SETTINGS")
    print(f"{'='*60}")
    
    for backbone, info in results.items():
        name = "base" if "base" in backbone else "small"
        print(f"\n{name.upper()} model ({backbone}):")
        print(f"  --batch-size {info['batch_size']}")
        if "samples_per_sec" in info:
            print(f"  Throughput: {info['samples_per_sec']:.1f} samples/sec")
    
    print(f"\nGeneral MPS optimizations:")
    print(f"  --num-workers 4        # Fewer workers for unified memory")
    print(f"  --compile              # Enable torch.compile (PyTorch 2.0+)")
    print(f"  --amp-dtype float16    # Use float16 instead of bfloat16 for MPS")


if __name__ == "__main__":
    main()


