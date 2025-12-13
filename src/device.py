"""Device utilities for CUDA, MPS (Apple Silicon), and CPU support."""
from enum import Enum
from typing import Tuple

import torch
import torch.nn as nn


class DeviceType(str, Enum):
    """Supported device types."""
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


def get_device_type() -> DeviceType:
    """Auto-detect the best available device type."""
    if torch.cuda.is_available():
        return DeviceType.CUDA
    elif torch.backends.mps.is_available():
        return DeviceType.MPS
    return DeviceType.CPU


def get_device(device_type: DeviceType | None = None, local_rank: int = 0) -> torch.device:
    """
    Get torch device based on device type.
    
    Args:
        device_type: Specific device type, or None for auto-detect
        local_rank: GPU index for CUDA (ignored for MPS/CPU)
    
    Returns:
        torch.device
    """
    if device_type is None:
        device_type = get_device_type()
    
    if device_type == DeviceType.CUDA:
        return torch.device(f"cuda:{local_rank}")
    elif device_type == DeviceType.MPS:
        return torch.device("mps")
    return torch.device("cpu")


def get_amp_settings(device_type: DeviceType, amp_dtype: str = "bfloat16") -> Tuple[bool, str, torch.dtype]:
    """
    Get AMP settings based on device type.
    
    Args:
        device_type: Device type
        amp_dtype: Preferred AMP dtype ("bfloat16" or "float16")
    
    Returns:
        Tuple of (use_amp, autocast_device_type, dtype)
    """
    if device_type == DeviceType.CUDA:
        dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        return True, "cuda", dtype
    elif device_type == DeviceType.MPS:
        # MPS supports float16 autocast (PyTorch 2.0+)
        # bfloat16 is NOT supported on MPS - always use float16
        return True, "mps", torch.float16
    # CPU - no AMP benefit
    return False, "cpu", torch.float32


def get_optimal_mps_settings() -> dict:
    """Get optimal settings for MPS (Apple Silicon) training."""
    import psutil
    
    total_mem_gb = psutil.virtual_memory().total / (1024**3)
    
    # Scale settings based on unified memory
    if total_mem_gb >= 64:
        return {"batch_size": 32, "num_workers": 4, "prefetch_factor": 4}
    elif total_mem_gb >= 32:
        return {"batch_size": 16, "num_workers": 4, "prefetch_factor": 2}
    elif total_mem_gb >= 16:
        return {"batch_size": 8, "num_workers": 2, "prefetch_factor": 2}
    else:
        return {"batch_size": 4, "num_workers": 2, "prefetch_factor": 2}


def supports_distributed(device_type: DeviceType) -> bool:
    """Check if device type supports distributed training (DDP)."""
    return device_type == DeviceType.CUDA


def supports_fused_optimizer(device_type: DeviceType) -> bool:
    """Check if device supports fused optimizer (CUDA only)."""
    return device_type == DeviceType.CUDA


def get_cudnn_settings(device_type: DeviceType) -> Tuple[bool, bool]:
    """
    Get CUDNN settings (deterministic, benchmark) based on device.
    
    Returns:
        (deterministic, benchmark) tuple
    """
    if device_type == DeviceType.CUDA:
        return False, True  # Non-deterministic for speed, benchmark enabled
    return True, False


def empty_cache(device_type: DeviceType) -> None:
    """Clear device cache based on device type."""
    if device_type == DeviceType.CUDA:
        torch.cuda.empty_cache()
    elif device_type == DeviceType.MPS:
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def synchronize(device_type: DeviceType) -> None:
    """Synchronize device."""
    if device_type == DeviceType.CUDA:
        torch.cuda.synchronize()
    elif device_type == DeviceType.MPS:
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def set_device_seed(seed: int, device_type: DeviceType) -> None:
    """Set device-specific random seed."""
    if device_type == DeviceType.CUDA:
        torch.cuda.manual_seed_all(seed)
    elif device_type == DeviceType.MPS:
        # MPS uses torch.manual_seed for seeding
        torch.mps.manual_seed(seed) if hasattr(torch.mps, "manual_seed") else None


def get_device_count(device_type: DeviceType) -> int:
    """Get number of available devices."""
    if device_type == DeviceType.CUDA:
        return torch.cuda.device_count()
    # MPS and CPU are single device
    return 1


def pin_memory_available(device_type: DeviceType) -> bool:
    """Check if pin_memory is beneficial for this device."""
    return device_type == DeviceType.CUDA


def to_device(tensor: torch.Tensor, device: torch.device, non_blocking: bool = True) -> torch.Tensor:
    """Move tensor to device with appropriate settings."""
    return tensor.to(device, non_blocking=non_blocking)


def get_dataloader_kwargs(device_type: DeviceType, num_workers: int = 4, prefetch_factor: int = 2) -> dict:
    """
    Get optimal DataLoader kwargs for device type.
    
    Args:
        device_type: Device type
        num_workers: Number of workers
        prefetch_factor: Prefetch factor
    
    Returns:
        dict of DataLoader kwargs
    """
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory_available(device_type),
    }
    
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True
    
    return kwargs


def model_to_device(
    model: nn.Module,
    device: torch.device,
    device_type: DeviceType,
    compile_model: bool = False,
    compile_mode: str = "default",
) -> nn.Module:
    """
    Move model to device with optional compilation.
    
    Args:
        model: Model to move
        device: Target device
        device_type: Device type
        compile_model: Whether to compile with torch.compile
        compile_mode: Compilation mode
    
    Returns:
        Model on device
    """
    model = model.to(device)
    
    if compile_model and device_type in (DeviceType.CUDA, DeviceType.MPS):
        # torch.compile works on CUDA and MPS (PyTorch 2.0+)
        model = torch.compile(model, mode=compile_mode, dynamic=False, fullgraph=False)
    
    return model


def print_device_info(device_type: DeviceType, device: torch.device) -> None:
    """Print device information."""
    print(f"Device type: {device_type.value}")
    print(f"Device: {device}")
    
    if device_type == DeviceType.CUDA:
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    elif device_type == DeviceType.MPS:
        print("MPS (Apple Silicon) backend active")
    else:
        print("Running on CPU")

