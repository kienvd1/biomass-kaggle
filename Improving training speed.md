# Improving Training Speed on MPS (Apple Silicon)

This document outlines optimizations to improve training speed for the CSIRO Biomass model on Apple Silicon (MPS backend).

## Summary

| Optimization | Impact | Complexity |
|-------------|--------|------------|
| Batched tile processing | 20-40% | Medium |
| Channels-last memory format | 5-15% | Low |
| TurboJPEG image loading | 10-20% | Low |
| Image caching | 15-25% | Low |
| Increased validation batch size | 5-10% | Low |
| ~~torch.compile~~ | ❌ Not supported on MPS | - |

---

## 1. Batched Tile Processing (High Impact)

**Problem**: Current `_tiles_backbone` in `models.py` processes tiles sequentially through the backbone, which is inefficient.

**Solution**: Batch all tiles together and process in a single forward pass.

### Changes to `src/models.py`

Replace the `_tiles_backbone` method in `TwoStreamDINOTiledFiLM` class (around line 216):

```python
def _tiles_backbone(self, x: torch.Tensor) -> torch.Tensor:
    """Extract features from image tiles - batched for speed."""
    B, C, H, W = x.shape
    r, c = self.grid
    rows = _make_edges(H, r)
    cols = _make_edges(W, c)

    # Collect all tiles
    tiles = []
    for rs, re in rows:
        for cs, ce in cols:
            xt = x[:, :, rs:re, cs:ce]
            if xt.shape[-2:] != (self.input_res, self.input_res):
                xt = F.interpolate(
                    xt, size=(self.input_res, self.input_res),
                    mode="bilinear", align_corners=False,
                )
            tiles.append(xt)

    # Stack and process all tiles in one forward pass
    num_tiles = len(tiles)
    tiles = torch.cat(tiles, dim=0)  # (B * num_tiles, C, H, W)
    feats = self.backbone(tiles)     # (B * num_tiles, D)
    feats = feats.view(num_tiles, B, -1).permute(1, 0, 2)  # (B, num_tiles, D)
    return feats
```

Also update the same method in `TwoStreamDINOTiled` class (around line 169).

---

## 2. Channels-Last Memory Format (Medium Impact)

**Problem**: Default memory format (NCHW contiguous) is not optimal for MPS.

**Solution**: Use `channels_last` memory format for better memory access patterns.

### Changes to `src/trainer.py`

After model creation in `train_fold` method (around line 547):

```python
model = model.to(self.device)

# Add channels-last for MPS
if self.device_type == DeviceType.MPS:
    model = model.to(memory_format=torch.channels_last)
```

In the training loop `train_one_epoch` method (around line 372):

```python
# Convert inputs to channels-last format
if self.device_type == DeviceType.MPS:
    x_left = x_left.to(self.device, non_blocking=True, memory_format=torch.channels_last)
    x_right = x_right.to(self.device, non_blocking=True, memory_format=torch.channels_last)
else:
    x_left = x_left.to(self.device, non_blocking=True)
    x_right = x_right.to(self.device, non_blocking=True)
```

Same change in `validate` method (around line 432).

---

## 3. TurboJPEG Image Loading (Medium Impact)

**Problem**: OpenCV's `cv2.imread` is slower than specialized JPEG decoders.

**Solution**: Use TurboJPEG for faster JPEG decoding.

### Installation

```bash
# macOS
brew install libjpeg-turbo
pip install PyTurboJPEG
```

### Changes to `src/dataset.py`

Add at the top of the file:

```python
# Fast JPEG loading
try:
    from turbojpeg import TurboJPEG
    _jpeg = TurboJPEG()
    _USE_TURBOJPEG = True
except ImportError:
    _USE_TURBOJPEG = False
```

Replace the `_load_image` method in `BiomassDataset` class:

```python
def _load_image(self, idx: int) -> np.ndarray:
    """Load image from disk or cache."""
    if idx in self._cache:
        return self._cache[idx]

    filename = os.path.basename(self.paths[idx])
    full_path = os.path.join(self.image_dir, filename)

    img = None
    if _USE_TURBOJPEG and full_path.lower().endswith(('.jpg', '.jpeg')):
        try:
            with open(full_path, 'rb') as f:
                img = _jpeg.decode(f.read())  # Returns RGB directly
        except Exception:
            img = None

    if img is None:
        img = cv2.imread(full_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)

    if self.cache_images:
        self._cache[idx] = img
    return img
```

---

## 4. Image Caching (High Impact for I/O Bound Training)

**Problem**: Repeated disk I/O for loading images slows down training.

**Solution**: Cache images in RAM if sufficient memory is available (32GB+ recommended).

### Usage

```bash
# Enable via command line (if implemented)
python -m src.train --device-type mps --cache-images

# Or set in config
cfg.cache_images = True
```

### Memory Estimation

- ~2000 images × ~6MB each ≈ 12GB RAM for caching
- Recommended for systems with 32GB+ unified memory

---

## 5. Increase Validation Batch Size (Low Impact)

**Problem**: Validation doesn't need gradients, so can use larger batches.

**Solution**: Increase validation batch size from 2x to 4x.

### Changes to `src/trainer.py`

In `train_fold` method (around line 530):

```python
valid_loader = DataLoader(
    valid_ds,
    batch_size=self.cfg.batch_size * 4,  # Increased from 2x to 4x
    shuffle=False,
    **loader_kwargs,
)
```

---

## 6. Disable torch.compile for MPS

**Problem**: `torch.compile` with TorchInductor does NOT support MPS backend. It only supports NVIDIA/AMD GPUs (via Triton) and Intel CPUs.

**Solution**: Remove MPS from compile targets.

### Changes to `src/device.py`

Update `model_to_device` function (around line 198):

```python
def model_to_device(
    model: nn.Module,
    device: torch.device,
    device_type: DeviceType,
    compile_model: bool = False,
    compile_mode: str = "default",
) -> nn.Module:
    model = model.to(device)

    # torch.compile only works on CUDA (not MPS)
    if compile_model and device_type == DeviceType.CUDA:
        model = torch.compile(model, mode=compile_mode, dynamic=False, fullgraph=False)

    return model
```

---

## 7. Multiprocessing Start Method (Stability)

**Problem**: Default multiprocessing can cause issues on macOS.

**Solution**: Use 'spawn' start method.

### Changes to `src/train.py`

Add near the top of the file after imports:

```python
import torch.multiprocessing as mp

# Set multiprocessing start method for macOS stability
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
```

---

## Recommended Training Command

```bash
python -m src.train \
    --device-type mps \
    --batch-size 8 \
    --num-workers 4 \
    --grad-ckpt \
    --no-compile
```

### Flags Explanation

| Flag | Purpose |
|------|---------|
| `--device-type mps` | Use Apple Silicon GPU |
| `--batch-size 8` | Per-device batch size (adjust based on memory) |
| `--num-workers 4` | Fewer workers for unified memory architecture |
| `--grad-ckpt` | Enable gradient checkpointing to save memory |
| `--no-compile` | Disable torch.compile (not supported on MPS) |

---

## Finding Optimal Batch Size

Use the included profiler to find the maximum batch size for your system:

```bash
python -m src.profile_mps --backbone vit_base_patch14_reg4_dinov2.lvd142m --benchmark
```

---

## References

- [MPS Backend — PyTorch Documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Apple Developer - Accelerated PyTorch on Mac](https://developer.apple.com/metal/pytorch/)
- [PyTorch Lightning - MPS Training](https://lightning.ai/docs/pytorch/stable/accelerators/mps_basic.html)
- [torch.compiler — PyTorch Documentation](https://docs.pytorch.org/docs/stable/torch.compiler.html)
