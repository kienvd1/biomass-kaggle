#!/usr/bin/env python3
"""GPU monitoring utility for optimizing training/inference workloads."""

import subprocess
import time
import argparse
from datetime import datetime
from pathlib import Path


def get_gpu_stats() -> list[dict]:
    """Query GPU stats using nvidia-smi."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 9:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "mem_used_mb": float(parts[2]),
                    "mem_total_mb": float(parts[3]),
                    "mem_free_mb": float(parts[4]),
                    "gpu_util": float(parts[5]) if parts[5] != "[N/A]" else 0,
                    "mem_util": float(parts[6]) if parts[6] != "[N/A]" else 0,
                    "temp_c": float(parts[7]) if parts[7] != "[N/A]" else 0,
                    "power_w": float(parts[8]) if parts[8] != "[N/A]" else 0,
                })
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error querying GPU: {e}")
        return []


def format_bar(value: float, max_val: float = 100, width: int = 20) -> str:
    """Create a simple ASCII progress bar."""
    filled = int((value / max_val) * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def print_stats(gpus: list[dict], clear: bool = True) -> None:
    """Print GPU stats in a formatted way."""
    if clear:
        print("\033[2J\033[H", end="")  # Clear screen
    
    print(f"{'═' * 70}")
    print(f"  GPU Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 70}")
    
    for gpu in gpus:
        mem_pct = (gpu["mem_used_mb"] / gpu["mem_total_mb"]) * 100
        print(f"\n  GPU {gpu['index']}: {gpu['name']}")
        print(f"  {'─' * 66}")
        print(f"  Memory:  {format_bar(mem_pct)} {gpu['mem_used_mb']:,.0f} / {gpu['mem_total_mb']:,.0f} MB ({mem_pct:.1f}%)")
        print(f"  GPU Util:{format_bar(gpu['gpu_util'])} {gpu['gpu_util']:.0f}%")
        print(f"  Mem BW:  {format_bar(gpu['mem_util'])} {gpu['mem_util']:.0f}%")
        print(f"  Temp: {gpu['temp_c']:.0f}°C  |  Power: {gpu['power_w']:.0f}W")
    
    print(f"\n{'═' * 70}")
    print("  Press Ctrl+C to stop")


def log_stats(gpus: list[dict], log_file: Path) -> None:
    """Append stats to CSV log file."""
    timestamp = datetime.now().isoformat()
    
    # Write header if file doesn't exist
    if not log_file.exists():
        with open(log_file, "w") as f:
            f.write("timestamp,gpu_index,mem_used_mb,mem_total_mb,gpu_util,mem_util,temp_c,power_w\n")
    
    with open(log_file, "a") as f:
        for gpu in gpus:
            f.write(f"{timestamp},{gpu['index']},{gpu['mem_used_mb']},{gpu['mem_total_mb']},"
                    f"{gpu['gpu_util']},{gpu['mem_util']},{gpu['temp_c']},{gpu['power_w']}\n")


def monitor(interval: float = 1.0, log_path: str | None = None, duration: float | None = None) -> None:
    """Main monitoring loop."""
    log_file = Path(log_path) if log_path else None
    start_time = time.time()
    
    try:
        while True:
            gpus = get_gpu_stats()
            if not gpus:
                print("No GPUs found or nvidia-smi not available")
                break
            
            print_stats(gpus)
            
            if log_file:
                log_stats(gpus, log_file)
                print(f"  Logging to: {log_file}")
            
            if duration and (time.time() - start_time) >= duration:
                print("\n  Duration reached, stopping...")
                break
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n  Monitoring stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor GPU usage")
    parser.add_argument("-i", "--interval", type=float, default=1.0, help="Update interval in seconds (default: 1.0)")
    parser.add_argument("-l", "--log", type=str, help="Log stats to CSV file")
    parser.add_argument("-d", "--duration", type=float, help="Stop after N seconds")
    parser.add_argument("--once", action="store_true", help="Print stats once and exit")
    
    args = parser.parse_args()
    
    if args.once:
        gpus = get_gpu_stats()
        print_stats(gpus, clear=False)
    else:
        monitor(interval=args.interval, log_path=args.log, duration=args.duration)


if __name__ == "__main__":
    main()


