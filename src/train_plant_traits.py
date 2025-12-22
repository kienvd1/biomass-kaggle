
"""
Training script for PlantTraits-inspired Biomass Model.
"""
import argparse
import os
import torch
import pandas as pd
from src.config import TrainConfig
from src.dataset import prepare_dataframe, create_folds
from src.device import DeviceType, get_device_type, get_device
from src.trainer_plant_traits import TrainerPlantTraits

def parse_args():
    parser = argparse.ArgumentParser(description="Train PlantTraits Strategy")
    parser.add_argument("--base-path", type=str, default="./data")
    parser.add_argument("--backbone", type=str, default="base", help="dinov3 backbone size: small/base/large")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--train-folds", type=int, nargs="+", default=[0])
    parser.add_argument("--output-dir", type=str, default="outputs/plant_traits")
    parser.add_argument("--device-type", type=str, default="cuda")
    return parser.parse_args()

def main():
    args = parse_args()
    
    cfg = TrainConfig()
    cfg.base_path = args.base_path
    cfg.train_csv = os.path.join(args.base_path, "train.csv")
    cfg.train_image_dir = os.path.join(args.base_path, "train")
    cfg.output_dir = args.output_dir
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    cfg.backbone_name = args.backbone
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.backbone_lr = args.backbone_lr
    cfg.num_folds = args.num_folds
    cfg.train_folds = args.train_folds
    cfg.num_workers = 4
    
    # Device setup
    cfg.device_type = args.device_type
    device_type_enum = DeviceType(args.device_type)
    cfg._device_type_enum = device_type_enum
    cfg.device = get_device(device_type_enum, 0)
    cfg.distributed = False # Simplified for now
    cfg.world_size = 1
    cfg.local_rank = 0
    cfg.gradient_accumulation_steps = 1
    cfg.log_interval = 10
    cfg.use_amp = True
    cfg.amp_dtype = "float16"
    cfg.grid = (2, 2)
    cfg.pretrained = True
    cfg.dropout = 0.0
    cfg.hidden_ratio = 0.25
    cfg.gradient_checkpointing = False
    cfg.compile_model = False
    cfg.scheduler = "cosine"
    cfg.min_lr = 1e-6
    cfg.weight_decay = 1e-4
    cfg.img_size = 518
    cfg.aug_prob = 0.5
    cfg.prefetch_factor = 2
    
    # Enable Depth Features (Inference-Safe via Depth Anything V2)
    cfg.use_depth = True
    cfg.depth_model_size = "small" # 'small' is faster, 'base' is more accurate
    cfg.use_depth_attention = True 
    
    # Data
    df = prepare_dataframe(cfg.train_csv)
    df = create_folds(df, n_folds=cfg.num_folds, cv_strategy="group_month")
    
    # Trainer
    trainer = TrainerPlantTraits(cfg)
    
    print(f"Starting training on {cfg.device}")
    
    for fold in cfg.train_folds:
        print(f"Fold {fold}")
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        valid_df = df[df["fold"] == fold].reset_index(drop=True)
        
        trainer.train_fold(fold, train_df, valid_df)

if __name__ == "__main__":
    main()
