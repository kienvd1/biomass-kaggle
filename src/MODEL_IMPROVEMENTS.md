# CSIRO Biomass — CV + Performance Improvement Notes (image-only)

This repo trains an **image-only** model (stereo pair split into left/right) to predict 5 targets:
`Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`, `GDM_g`, `Dry_Total_g`.

This document captures **practical improvements** you can apply *right now*, given you **do not have labeled test data** and the **test CSV has no metadata**.

---

## 1) What’s in the data (important for CV + modeling)

### 1.1 Train vs test schema

- **Train** (`data/train.csv`): long format with columns
  `sample_id`, `image_path`, `Sampling_Date`, `State`, `Species`, `Pre_GSHH_NDVI`, `Height_Ave_cm`, `target_name`, `target`.
- **Test** (`data/test.csv`): **only**
  `sample_id`, `image_path`, `target_name`.

✅ Conclusion: **final model must not require metadata** (State/Species/NDVI/Height/Date) at inference time.

### 1.2 Wide-form training rows

`src/dataset.py:prepare_dataframe()` pivots the long CSV into **one row per image_path** with 5 numeric targets.

### 1.3 Target distribution notes (from a scan of `data/train.csv`)

- There are **357 unique images** (and 1785 long rows = 357 × 5 targets).
- `Dry_Clover_g` is **heavily zero-inflated** (~38% zeros).
- `Dry_Dead_g` has meaningful zero mass as well (~11% zeros).
- `GDM_g` and `Dry_Total_g` are always > 0 in the current scan.

Implication: handling **heavy tails** and **zero inflation** can improve score.

---

## 2) “Proper” cross validation when you have no test labels

### 2.1 Non-negotiable rules

- **Split at the image level** (one row per `image_path` in the wide table).
- Any evaluation must use **fold-pure OOF**:
  - fold-*k* model predicts only fold-*k* validation rows.
  - never score a model on rows it was trained on.

### 2.2 Recommended CV view(s)

Your current default is `cv_strategy="group_month"` in `src/dataset.py:create_folds()`.

**Problem**: only ~10 months exist → group-by-month can be *too coarse* (few groups), leading to unstable or pessimistic estimates.

Use two complementary CV views:

- **Primary (for model selection)**: `cv_strategy="group_date"`
  - `StratifiedGroupKFold` grouped by `Sampling_Date`, stratified by bins of `Dry_Total_g`.
  - More groups than month → more stable signal for hyperparameters/architecture selection.

- **Robustness check (distribution shift)**: `cv_strategy="group_month"`
  - Keep as a “stress test” to ensure seasonal robustness.
  - Do not tune solely to this, or you risk underfitting your main objective.

### 2.3 Avoiding CV overfitting

If you iterate a lot, you will overfit the CV split itself.

Prefer one of:

- **Nested CV** (best, expensive):
  - Outer folds for unbiased estimation.
  - Inner folds for tuning (Optuna / hand tuning).

- **Pseudo-holdout fold** (cheap):
  - Reserve 1 fold as “final validation” and never tune on it.
  - Tune on remaining folds only, then do one final check on the holdout fold.

---

## 3) Performance improvements that do NOT use test metadata

### 3.1 Make OOF evaluation match inference (high ROI)

Inference already uses TTA (see `notebooks/inference_v1.ipynb`).
OOF evaluation in `src/evaluate_oof.py` currently uses a single deterministic view.

**Recommendation**
- Add optional **TTA in OOF**, so you select models/weights using the same policy you’ll submit.
- Ensure your “local testing” in notebooks is fold-pure; avoid scoring with an ensemble that includes models trained on the same sample.

### 3.2 Train in log space for regression stability (high ROI)

Targets have heavy tails; log scaling usually helps.

**Recommendation**
- Train on: `log1p(y)` for all 5 targets.
- At inference: apply `expm1` to outputs.

Notes:
- You already enforce non-negativity in the model via `Softplus`.
- Log-space training often improves stability, especially with outliers.

### 3.3 Zero-inflation modeling for Clover / Dead (medium–high ROI)

Given the amount of zeros, especially for `Dry_Clover_g`, consider:

- **Two-part head** per sparse target:
  - classifier: `p(nonzero)` (BCE)
  - regressor: `amount` (e.g., on `log1p(y)`)
  - prediction: `p * amount`

Alternative: Tweedie / hurdle-like loss, but the two-part approach is straightforward.

### 3.4 Better tile aggregation (medium ROI)

Current tiled variants mean-pool the tile features.

**Recommendation**
- Replace mean pooling with **learned attention pooling** over tiles.
  - lets the model focus on informative regions (often better than uniform averaging).

### 3.5 Increase tile grid diversity (medium ROI)

Try `--grid 3 3` (and keep `2 2`):
- Often improves local detail capture.
- Also creates ensemble diversity across checkpoints.

### 3.6 Ensemble weights: fit them from OOF (high ROI)

You currently hand-set ensemble weights (e.g. `W_A/W_B`) in notebook inference.

**Recommendation**
- Compute OOF predictions for each backbone/setting.
- Fit non-negative weights to maximize weighted R² on OOF.
- Lock weights and use them for test inference.

---

## 4) Concrete “do next” commands (using existing scripts)

### 4.1 Train with a stronger CV split (primary)

```bash
python -m src.train \
  --cv-strategy group_date \
  --backbone vit_base_patch14_reg4_dinov2.lvd142m \
  --output-dir /root/workspace/outputs/base_group_date
```

### 4.2 Train a diverse companion model (for ensemble)

```bash
python -m src.train \
  --cv-strategy group_date \
  --backbone vit_small_patch14_reg4_dinov2.lvd142m \
  --output-dir /root/workspace/outputs/small_group_date
```

### 4.3 Evaluate fold-pure OOF for each model dir

```bash
python -m src.evaluate_oof --model-dir /root/workspace/outputs/base_group_date
python -m src.evaluate_oof --model-dir /root/workspace/outputs/small_group_date
```

### 4.4 Run inference / submission (ensemble)

Use your notebook (`notebooks/inference_v1.ipynb`) or `src/inference.py` (if you prefer script-based).
Prefer ensemble weights learned from OOF.

---

## 5) Optional (uses metadata only for training): teacher → student distillation

If you want to exploit train-time metadata **without requiring it at test**:

- Train a **teacher** that uses metadata + image.
- Train an **image-only student** to match the teacher’s predictions (distillation).

This can transfer useful structure while keeping the final model valid for test inference.


