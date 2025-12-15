# Train Ratio Model - Arguments Reference

## Model Architecture

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-type` | `softmax` | `softmax`: SoftmaxRatioDINO (predicts Total + softmax ratios)<br>`hierarchical`: HierarchicalRatioDINO (predicts Total → GDM/Total → Green/GDM) |
| `--backbone` | `vit_base_patch14_reg4_dinov2.lvd142m` | DINOv2 backbone model |
| `--grid` | `2 2` | Tile grid size (e.g., `2 2` = 4 tiles per image) |
| `--dropout` | `0.2` | Dropout rate in regression heads |
| `--hidden-ratio` | `0.5` | Hidden layer size ratio (relative to backbone dim) |
| `--no-film` | `False` | Disable FiLM cross-conditioning between L/R streams |
| `--no-attention-pool` | `False` | Disable attention pooling (use mean pooling instead) |
| `--grad-ckpt` | `False` | Enable gradient checkpointing (saves memory) |
| `--ratio-temperature` | `1.0` | Temperature for softmax ratios (only for softmax model) |

## Training Strategy

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `50` | Total training epochs |
| `--batch-size` | `8` | Batch size per device |
| `--patience` | `10` | Early stopping patience (epochs without improvement) |
| `--freeze-backbone` | `False` | **Head-only mode**: Keep backbone frozen throughout |
| `--two-stage` | `False` | **2-stage training**: Freeze backbone first, then finetune |
| `--freeze-epochs` | `5` | Number of epochs with frozen backbone (stage 1) |

### Training Modes

```bash
# Mode 1: Head-only (fastest, good baseline)
--freeze-backbone --head-lr-stage1 3e-4

# Mode 2: Two-stage (recommended for best results)
--two-stage --freeze-epochs 5 --head-lr-stage1 1e-3 --lr 2e-4 --backbone-lr 1e-5

# Mode 3: Full finetune from start (risk of overfitting)
--lr 2e-4 --backbone-lr 1e-5
```

## Learning Rates

| Argument | Default | Description |
|----------|---------|-------------|
| `--head-lr-stage1` | `1e-3` | Head learning rate for stage 1 (frozen backbone) |
| `--lr` | `2e-4` | Head learning rate for stage 2 (unfrozen) |
| `--backbone-lr` | `1e-5` | Backbone learning rate for stage 2 |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--warmup-epochs` | `2` | Linear warmup epochs |

## Gradient Control

| Argument | Default | Description |
|----------|---------|-------------|
| `--grad-accum` | `1` | Gradient accumulation steps (effective BS = batch_size × grad_accum) |
| `--grad-clip` | `0.5` | Gradient clipping max norm |

## Loss Function

| Argument | Default | Description |
|----------|---------|-------------|
| `--target-weights` | `0.1 0.1 0.1 0.2 0.5` | Loss weights for [Green, Dead, Clover, GDM, Total] |
| `--no-huber-for-dead` | `False` | Use MSE instead of Huber for Dead target |
| `--huber-delta` | `5.0` | Huber loss delta for Dead |
| `--ratio-loss-weight` | `0.0` | Weight for auxiliary KL divergence on ratios |

## Data Augmentation

| Argument | Default | Description |
|----------|---------|-------------|
| `--img-size` | `518` | Input image size |
| `--aug-prob` | `0.5` | Base augmentation probability |
| `--stereo-correct-aug` | `False` | Apply photometric aug independently per L/R view |
| `--stereo-swap-prob` | `0.0` | Probability of swapping L/R images |
| `--mixup-prob` | `0.0` | MixUp augmentation probability |
| `--mixup-alpha` | `0.4` | MixUp beta distribution alpha |
| `--cutmix-prob` | `0.0` | CutMix augmentation probability |
| `--cutmix-alpha` | `1.0` | CutMix beta distribution alpha |
| `--use-log-target` | `False` | Apply log1p transform to targets |

## Cross-Validation

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-folds` | `5` | Number of CV folds |
| `--train-folds` | `0 1 2 3 4` | Which folds to train |
| `--cv-strategy` | `group_date` | CV split strategy |
| `--fold-csv` | `None` | Path to pre-defined fold CSV |

## System

| Argument | Default | Description |
|----------|---------|-------------|
| `--device-type` | auto | `cuda`, `mps`, or `cpu` |
| `--multi-gpu` | `False` | Enable DataParallel for multi-GPU |
| `--num-workers` | `4` | DataLoader workers |
| `--cache-images` | `False` | Cache images in RAM |
| `--prefetch-factor` | `4` | DataLoader prefetch factor |
| `--amp-dtype` | `float16` | AMP dtype: `float16` or `bfloat16` |
| `--seed` | `42` | Random seed |

## Output

| Argument | Default | Description |
|----------|---------|-------------|
| `--base-path` | `./data` | Data directory |
| `--output-dir` | auto | Output directory (auto-generated with timestamp) |
| `--compute-oof` | `False` | Compute OOF predictions after each fold |

---

## Example Commands

### Simple Baseline (Head-only)
```bash
python -m src.train_ratio \
    --base-path ./data \
    --fold-csv ./data/trainfold.csv \
    --model-type hierarchical \
    --freeze-backbone \
    --head-lr-stage1 1e-4 \
    --epochs 30 \
    --dropout 0.3 \
    --patience 15 \
    --compute-oof \
    --device-type mps
```

### Strong Baseline (2-stage + augmentation)
```bash
python -m src.train_ratio \
    --base-path ./data \
    --fold-csv ./data/trainfold.csv \
    --model-type hierarchical \
    --two-stage \
    --freeze-epochs 5 \
    --head-lr-stage1 1e-3 \
    --lr 2e-4 \
    --backbone-lr 1e-5 \
    --epochs 30 \
    --dropout 0.3 \
    --grad-accum 2 \
    --stereo-correct-aug \
    --stereo-swap-prob 0.3 \
    --compute-oof \
    --device-type mps
```

### Quick 1-fold Test
```bash
python -m src.train_ratio \
    --train-folds 0 \
    --epochs 10 \
    --freeze-backbone \
    --head-lr-stage1 3e-4 \
    --device-type mps
```

---

## Model Comparison

| Model | Predicts | Derives | Constraint |
|-------|----------|---------|------------|
| **softmax** | Total, softmax(G,D,C ratios) | GDM=G+C | G+D+C=Total (exact) |
| **hierarchical** | Total, GDM/Total, Green/GDM | Dead=T-GDM, Clover=GDM-G | All ≥0 (sigmoid bounded) |

**Recommendation**: Use `hierarchical` when Dead/Clover are hard to predict directly.
