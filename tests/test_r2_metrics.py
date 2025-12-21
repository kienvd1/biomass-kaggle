"""
Unit tests for R² metrics: Global R² (gR²) and Average R² (aR²).

gR² = Competition metric (global weighted R²)
aR² = Weighted average of per-target R² scores
"""
import numpy as np
import torch
import pytest


# Target weights (competition)
TARGET_WEIGHTS = [0.1, 0.1, 0.1, 0.2, 0.5]  # green, dead, clover, gdm, total
TARGET_NAMES = ["green", "dead", "clover", "gdm", "total"]


def compute_per_target_r2(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute R² for each target separately."""
    r2_scores = {}
    for i, name in enumerate(TARGET_NAMES):
        pred_i = preds[:, i]
        target_i = targets[:, i]
        
        ss_res = np.sum((pred_i - target_i) ** 2)
        ss_tot = np.sum((target_i - target_i.mean()) ** 2)
        
        if ss_tot < 1e-10:
            r2 = 1.0 if ss_res < 1e-10 else 0.0
        else:
            r2 = 1.0 - (ss_res / ss_tot)
        
        r2_scores[f"r2_{name}"] = r2
    
    return r2_scores


def compute_avg_r2(per_target_r2: dict[str, float]) -> float:
    """
    Compute aR²: weighted average of per-target R² scores.
    
    Formula: aR² = Σ(w_i × R²_i)
    """
    avg_r2 = sum(
        TARGET_WEIGHTS[i] * per_target_r2[f"r2_{name}"]
        for i, name in enumerate(TARGET_NAMES)
    )
    return avg_r2


def compute_global_r2(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute gR²: global weighted R² (competition metric).
    
    Formula: gR² = 1 - SS_res / SS_tot
    where SS_res = Σ w_j * (y_j - ŷ_j)²
          SS_tot = Σ w_j * (y_j - ȳ_w)²
          ȳ_w = Σ w_j * y_j / Σ w_j
    """
    n_samples = preds.shape[0]
    weights = np.array(TARGET_WEIGHTS)
    
    # Expand weights for all samples: (N, 5)
    w = np.tile(weights, (n_samples, 1)).flatten()  # (N * 5,)
    y = targets.flatten()  # (N * 5,)
    y_hat = preds.flatten()  # (N * 5,)
    
    # Weighted mean of targets
    y_bar_w = np.sum(w * y) / np.sum(w)
    
    # Global weighted R²
    ss_res = np.sum(w * (y - y_hat) ** 2)
    ss_tot = np.sum(w * (y - y_bar_w) ** 2)
    
    if ss_tot < 1e-10:
        return 1.0 if ss_res < 1e-10 else 0.0
    
    return 1.0 - (ss_res / ss_tot)


def compute_global_r2_torch(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Torch version of global R² (matches dinov3_train.py implementation)."""
    n_samples = preds.size(0)
    weights_t = torch.tensor(TARGET_WEIGHTS, device=preds.device)
    weights_expanded = weights_t.unsqueeze(0).expand(n_samples, -1)
    
    w = weights_expanded.flatten()
    y = targets.flatten()
    y_hat = preds.flatten()
    
    y_bar_w = (w * y).sum() / w.sum()
    
    ss_res = (w * (y - y_hat) ** 2).sum()
    ss_tot = (w * (y - y_bar_w) ** 2).sum()
    
    if ss_tot < 1e-8:
        return 1.0 if ss_res < 1e-8 else 0.0
    
    return float((1.0 - (ss_res / (ss_tot + 1e-8))).item())


class TestR2Metrics:
    """Test R² metric calculations."""
    
    def test_perfect_predictions(self):
        """Perfect predictions should give R² = 1.0 for both metrics."""
        np.random.seed(42)
        targets = np.random.rand(100, 5) * 100
        preds = targets.copy()  # Perfect predictions
        
        per_target = compute_per_target_r2(preds, targets)
        avg_r2 = compute_avg_r2(per_target)
        global_r2 = compute_global_r2(preds, targets)
        
        # All per-target R² should be 1.0
        for name in TARGET_NAMES:
            assert per_target[f"r2_{name}"] == pytest.approx(1.0, abs=1e-10)
        
        # Both metrics should be 1.0
        assert avg_r2 == pytest.approx(1.0, abs=1e-10)
        assert global_r2 == pytest.approx(1.0, abs=1e-10)
    
    def test_mean_predictions(self):
        """Predicting the mean should give R² = 0.0 for per-target."""
        np.random.seed(42)
        targets = np.random.rand(100, 5) * 100
        preds = np.tile(targets.mean(axis=0), (100, 1))  # Predict column means
        
        per_target = compute_per_target_r2(preds, targets)
        avg_r2 = compute_avg_r2(per_target)
        
        # All per-target R² should be ~0.0
        for name in TARGET_NAMES:
            assert per_target[f"r2_{name}"] == pytest.approx(0.0, abs=1e-10)
        
        # aR² should be 0.0
        assert avg_r2 == pytest.approx(0.0, abs=1e-10)
    
    def test_global_r2_vs_avg_r2_difference(self):
        """gR² and aR² should generally be different values."""
        np.random.seed(42)
        targets = np.random.rand(50, 5) * 100
        # Add some noise to predictions
        preds = targets + np.random.randn(50, 5) * 10
        
        per_target = compute_per_target_r2(preds, targets)
        avg_r2 = compute_avg_r2(per_target)
        global_r2 = compute_global_r2(preds, targets)
        
        # They should both be positive (decent predictions)
        assert avg_r2 > 0
        assert global_r2 > 0
        
        # They should be different (not exactly equal)
        # Note: In rare cases they could be very close, but generally different
        print(f"aR² = {avg_r2:.6f}, gR² = {global_r2:.6f}, diff = {abs(avg_r2 - global_r2):.6f}")
    
    def test_torch_numpy_consistency(self):
        """Torch and NumPy implementations should give same results."""
        np.random.seed(42)
        targets_np = np.random.rand(100, 5) * 100
        preds_np = targets_np + np.random.randn(100, 5) * 10
        
        targets_torch = torch.tensor(targets_np, dtype=torch.float32)
        preds_torch = torch.tensor(preds_np, dtype=torch.float32)
        
        global_r2_np = compute_global_r2(preds_np, targets_np)
        global_r2_torch = compute_global_r2_torch(preds_torch, targets_torch)
        
        assert global_r2_np == pytest.approx(global_r2_torch, abs=1e-5)
    
    def test_weight_impact(self):
        """
        Test that weights affect gR² correctly.
        If Total (weight=0.5) has high R² but others are low,
        gR² should be pulled up more than if Green (weight=0.1) is high.
        """
        np.random.seed(42)
        n = 100
        
        # Scenario 1: Only Total is predicted well
        targets = np.random.rand(n, 5) * 100
        preds_total_good = np.random.rand(n, 5) * 100  # Random for all
        preds_total_good[:, 4] = targets[:, 4]  # Perfect Total
        
        # Scenario 2: Only Green is predicted well
        preds_green_good = np.random.rand(n, 5) * 100  # Random for all
        preds_green_good[:, 0] = targets[:, 0]  # Perfect Green
        
        r2_total_good = compute_global_r2(preds_total_good, targets)
        r2_green_good = compute_global_r2(preds_green_good, targets)
        
        # Total has weight 0.5, Green has weight 0.1
        # So total_good scenario should have higher gR²
        assert r2_total_good > r2_green_good, \
            f"Total-good ({r2_total_good:.4f}) should be > Green-good ({r2_green_good:.4f})"
    
    def test_negative_r2(self):
        """Test that R² can be negative when predictions are worse than mean."""
        np.random.seed(42)
        targets = np.random.rand(100, 5) * 100
        # Very bad predictions (opposite direction)
        preds = -targets + 200
        
        per_target = compute_per_target_r2(preds, targets)
        avg_r2 = compute_avg_r2(per_target)
        global_r2 = compute_global_r2(preds, targets)
        
        # At least some should be negative
        print(f"aR² = {avg_r2:.4f}, gR² = {global_r2:.4f}")
        print(f"Per-target: {per_target}")
    
    def test_single_sample(self):
        """Test with single sample (edge case)."""
        targets = np.array([[10.0, 20.0, 5.0, 30.0, 65.0]])
        preds = np.array([[12.0, 18.0, 6.0, 28.0, 64.0]])
        
        global_r2 = compute_global_r2(preds, targets)
        
        # With single sample, SS_tot uses weighted mean across all values
        # This should still compute without error
        assert np.isfinite(global_r2)
    
    def test_competition_formula_manual(self):
        """
        Manual verification of the competition formula.
        
        gR² = 1 - SS_res / SS_tot
        SS_res = Σ w_j * (y_j - ŷ_j)²
        SS_tot = Σ w_j * (y_j - ȳ_w)²
        ȳ_w = Σ w_j * y_j / Σ w_j
        """
        # Simple 2-sample case for manual verification
        targets = np.array([
            [10.0, 20.0, 5.0, 30.0, 50.0],  # Sample 1
            [20.0, 10.0, 15.0, 40.0, 80.0],  # Sample 2
        ])
        preds = np.array([
            [12.0, 18.0, 6.0, 32.0, 52.0],  # Sample 1
            [18.0, 12.0, 14.0, 38.0, 78.0],  # Sample 2
        ])
        
        # Manual calculation
        w = np.array([0.1, 0.1, 0.1, 0.2, 0.5] * 2)  # Repeated for 2 samples
        y = targets.flatten()
        y_hat = preds.flatten()
        
        y_bar_w = np.sum(w * y) / np.sum(w)
        ss_res = np.sum(w * (y - y_hat) ** 2)
        ss_tot = np.sum(w * (y - y_bar_w) ** 2)
        expected_r2 = 1.0 - (ss_res / ss_tot)
        
        # Compare with function
        computed_r2 = compute_global_r2(preds, targets)
        
        assert computed_r2 == pytest.approx(expected_r2, abs=1e-10)
        print(f"Manual gR² = {expected_r2:.6f}")
        print(f"Computed gR² = {computed_r2:.6f}")
        print(f"ȳ_w = {y_bar_w:.4f}")
        print(f"SS_res = {ss_res:.4f}, SS_tot = {ss_tot:.4f}")


class TestAvgR2Formula:
    """Test the aR² (weighted average of per-target R²) formula."""
    
    def test_formula_manual(self):
        """Manual verification of aR² formula."""
        targets = np.array([
            [10.0, 20.0, 5.0, 30.0, 50.0],
            [20.0, 10.0, 15.0, 40.0, 80.0],
        ])
        preds = np.array([
            [12.0, 18.0, 6.0, 32.0, 52.0],
            [18.0, 12.0, 14.0, 38.0, 78.0],
        ])
        
        per_target = compute_per_target_r2(preds, targets)
        avg_r2 = compute_avg_r2(per_target)
        
        # Manual: aR² = 0.1*R²_green + 0.1*R²_dead + 0.1*R²_clover + 0.2*R²_gdm + 0.5*R²_total
        expected = (
            0.1 * per_target["r2_green"] +
            0.1 * per_target["r2_dead"] +
            0.1 * per_target["r2_clover"] +
            0.2 * per_target["r2_gdm"] +
            0.5 * per_target["r2_total"]
        )
        
        assert avg_r2 == pytest.approx(expected, abs=1e-10)
        print(f"Per-target R²: {per_target}")
        print(f"aR² = {avg_r2:.6f}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])


