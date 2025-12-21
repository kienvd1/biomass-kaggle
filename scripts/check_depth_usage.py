#!/usr/bin/env python3
"""Check if depth features are being used by the trained model."""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dinov3_models import DINOv3Direct


def analyze_depth_weights(checkpoint_path: str) -> None:
    """Analyze how the model uses depth features."""
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {checkpoint_path}")
    print('='*60)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt if isinstance(ckpt, dict) and 'state_dict' not in ckpt else ckpt.get('state_dict', ckpt)
    
    # Check if depth module exists
    depth_keys = [k for k in state_dict.keys() if 'depth' in k.lower()]
    
    if not depth_keys:
        print("\n❌ No depth module found in checkpoint!")
        print("   Model was trained WITHOUT --use-depth flag")
        return
    
    print(f"\n✅ Depth module found! Keys: {len(depth_keys)}")
    
    # Analyze depth projection layer
    proj_weight = None
    proj_bias = None
    
    for k in depth_keys:
        if 'proj.0.weight' in k:
            proj_weight = state_dict[k]
        if 'proj.0.bias' in k:
            proj_bias = state_dict[k]
        print(f"  - {k}: {state_dict[k].shape}")
    
    if proj_weight is not None:
        print(f"\n📊 Depth Projection Analysis:")
        print(f"   Input dim: {proj_weight.shape[1]} (22 stats: 10 left + 10 right + 2 stereo)")
        print(f"   Output dim: {proj_weight.shape[0]}")
        
        # Which input stats have highest weights?
        input_importance = proj_weight.abs().mean(dim=0)  # Average abs weight per input
        
        stat_names = [
            "L_mean", "L_std", "L_max", "L_min", "L_p10", "L_p90", 
            "L_range", "L_volume", "L_gradient", "L_high_ratio",
            "R_mean", "R_std", "R_max", "R_min", "R_p10", "R_p90",
            "R_range", "R_volume", "R_gradient", "R_high_ratio",
            "LR_diff", "LR_corr"
        ]
        
        print(f"\n   📈 Feature Importance (by weight magnitude):")
        sorted_idx = input_importance.argsort(descending=True)
        for i, idx in enumerate(sorted_idx[:10]):
            print(f"      {i+1}. {stat_names[idx]:15s}: {input_importance[idx]:.4f}")
        
        # Check if weights are non-trivial (not near initialization)
        weight_std = proj_weight.std().item()
        weight_mean = proj_weight.abs().mean().item()
        print(f"\n   Weight stats: mean_abs={weight_mean:.4f}, std={weight_std:.4f}")
        
        if weight_std < 0.01:
            print("   ⚠️  Weights seem close to initialization - may not be learning from depth")
        else:
            print("   ✅ Weights have diverged from initialization - depth is being used!")
    
    # Check head weights to see depth contribution
    print(f"\n📊 Head Weight Analysis (depth contribution):")
    
    for target in ['green', 'dead', 'clover', 'gdm', 'total']:
        head_key = f'heads.{target}_head.0.weight'
        if head_key in state_dict:
            head_weight = state_dict[head_key]
            # Depth features are at the end of the concatenated features
            # Need to figure out where depth starts
            total_input = head_weight.shape[1]
            print(f"   {target:8s} head input dim: {total_input}")


def compare_with_without_depth(
    model_with: str | None = None, 
    model_without: str | None = None
) -> None:
    """Compare predictions with and without depth features."""
    
    if model_with is None or model_without is None:
        print("\n⚠️  Provide both --with and --without checkpoints for comparison")
        return
    
    print(f"\n{'='*60}")
    print("Comparing predictions WITH vs WITHOUT depth")
    print('='*60)
    
    # This would require loading both models and running inference
    # Placeholder for now
    print("TODO: Implement prediction comparison")


def gradient_analysis(checkpoint_path: str) -> None:
    """Check if gradients flow through depth features."""
    
    print(f"\n{'='*60}")
    print("Gradient Flow Analysis")
    print('='*60)
    
    # Load model
    model = DINOv3Direct(use_depth=True, depth_model_size='small')
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt if isinstance(ckpt, dict) and 'state_dict' not in ckpt else ckpt.get('state_dict', ckpt)
    
    # Filter state dict for model keys only
    model_keys = set(model.state_dict().keys())
    filtered_state = {k: v for k, v in state_dict.items() if k in model_keys}
    
    try:
        model.load_state_dict(filtered_state, strict=False)
    except Exception as e:
        print(f"⚠️  Could not load full state dict: {e}")
        return
    
    model.eval()
    
    # Create dummy input
    dummy_left = torch.randn(1, 3, 518, 518)
    dummy_right = torch.randn(1, 3, 518, 518)
    
    # Enable gradients for depth proj layer
    if hasattr(model, 'depth_module') and model.depth_module is not None:
        for p in model.depth_module.proj.parameters():
            p.requires_grad = True
        
        # Forward pass
        with torch.enable_grad():
            outputs = model(dummy_left, dummy_right)
            loss = sum(o.sum() for o in outputs[:5])  # Sum of all predictions
            loss.backward()
        
        # Check gradients
        for name, p in model.depth_module.proj.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                print(f"   {name}: grad_norm = {grad_norm:.6f}")
                if grad_norm > 0:
                    print("   ✅ Gradients are flowing through depth features!")
                else:
                    print("   ⚠️  Zero gradients - depth features disconnected?")
    else:
        print("   ❌ No depth module found")


def main():
    parser = argparse.ArgumentParser(description="Check depth feature usage")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--gradients", action="store_true", help="Run gradient analysis")
    parser.add_argument("--with", dest="model_with", type=str, help="Model with depth")
    parser.add_argument("--without", dest="model_without", type=str, help="Model without depth")
    
    args = parser.parse_args()
    
    # Analyze weights
    analyze_depth_weights(args.checkpoint)
    
    # Optional gradient analysis
    if args.gradients:
        gradient_analysis(args.checkpoint)
    
    # Optional comparison
    if args.model_with or args.model_without:
        compare_with_without_depth(args.model_with, args.model_without)


if __name__ == "__main__":
    main()


