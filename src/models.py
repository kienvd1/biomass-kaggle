"""Model architectures for CSIRO Biomass prediction."""
from typing import List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def _infer_input_res(model: nn.Module) -> int:
    """Infer input resolution from model config."""
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "img_size"):
        isz = model.patch_embed.img_size
        return int(isz if isinstance(isz, (int, float)) else isz[0])
    if hasattr(model, "img_size"):
        isz = model.img_size
        return int(isz if isinstance(isz, (int, float)) else isz[0])
    
    dc = getattr(model, "default_cfg", {}) or {}
    ins = dc.get("input_size", None)
    if ins:
        if isinstance(ins, (tuple, list)) and len(ins) >= 2:
            return int(ins[1])
        return int(ins if isinstance(ins, (int, float)) else 224)
    
    name = dc.get("architecture", "") or str(type(model))
    if "dinov2" in name.lower():
        return 518
    elif "dinov3" in name.lower():
        return 256  # DINOv3 default, but supports any size divisible by 16
    else:
        return 224


def _build_dino_by_name(
    name: str, pretrained: bool = True, gradient_checkpointing: bool = False
) -> Tuple[nn.Module, int, int]:
    """
    Build DINOv2 backbone by name.
    
    Returns:
        model: Backbone model
        feat_dim: Feature dimension
        input_res: Input resolution
    """
    model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    feat_dim = model.num_features
    input_res = _infer_input_res(model)
    
    # Enable gradient checkpointing for memory efficiency
    if gradient_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(enable=True)
    
    return model, feat_dim, input_res


def _make_edges(length: int, parts: int) -> List[Tuple[int, int]]:
    """Create edge indices for tiling."""
    step = length // parts
    edges = []
    start = 0
    for _ in range(parts - 1):
        edges.append((start, start + step))
        start += step
    edges.append((start, length))
    return edges


class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer."""
    
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        hidden = max(32, in_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim * 2),
        )
    
    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context: Context features (B, D)
            
        Returns:
            gamma: Scale factor (B, D)
            beta: Shift factor (B, D)
        """
        gb = self.mlp(context)
        gamma, beta = torch.chunk(gb, 2, dim=1)
        return gamma, beta


class TwoStreamDINOBase(nn.Module):
    """Base class for two-stream DINOv2 models."""
    
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_ratio: float = 0.25,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.backbone, feat, input_res = _build_dino_by_name(
            backbone_name, pretrained, gradient_checkpointing
        )
        self.used_backbone_name = backbone_name
        self.input_res = int(input_res)
        self.feat_dim = feat
        self.combined = feat * 2
        
        hidden = max(8, int(self.combined * hidden_ratio))
        
        def _make_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(self.combined, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        
        self.head_green = _make_head()
        self.head_clover = _make_head()
        self.head_dead = _make_head()
        self.softplus = nn.Softplus(beta=1.0)
    
    def _merge_heads(
        self, f_l: torch.Tensor, f_r: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Merge features from both streams and compute predictions.
        
        Returns:
            green, dead, clover, gdm, total predictions
        """
        f = torch.cat([f_l, f_r], dim=1)
        
        green_pos = self.softplus(self.head_green(f))
        clover_pos = self.softplus(self.head_clover(f))
        dead_pos = self.softplus(self.head_dead(f))
        
        gdm = green_pos + clover_pos
        total = gdm + dead_pos
        
        return green_pos, dead_pos, clover_pos, gdm, total


class TwoStreamDINOPlain(TwoStreamDINOBase):
    """Plain two-stream model without tiling."""
    
    def forward(
        self, x_left: torch.Tensor, x_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f_l = self.backbone(x_left)
        f_r = self.backbone(x_right)
        return self._merge_heads(f_l, f_r)


class TwoStreamDINOTiled(TwoStreamDINOBase):
    """Two-stream model with tiled encoding."""
    
    def __init__(
        self,
        backbone_name: str,
        grid: Tuple[int, int] = (2, 2),
        **kwargs,
    ) -> None:
        super().__init__(backbone_name, **kwargs)
        self.grid = tuple(grid)
    
    def _encode_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image tiles and aggregate features - batched for speed."""
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
                        xt,
                        size=(self.input_res, self.input_res),
                        mode="bilinear",
                        align_corners=False,
                    )
                tiles.append(xt)

        # Stack and process all tiles in one forward pass
        num_tiles = len(tiles)
        tiles = torch.cat(tiles, dim=0)  # (B * num_tiles, C, H, W)
        feats = self.backbone(tiles)     # (B * num_tiles, D)
        feats = feats.view(num_tiles, B, -1).permute(1, 0, 2)  # (B, num_tiles, D)
        feat_stream = feats.mean(dim=1)  # (B, D)
        return feat_stream
    
    def forward(
        self, x_left: torch.Tensor, x_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f_l = self._encode_tiles(x_left)
        f_r = self._encode_tiles(x_right)
        return self._merge_heads(f_l, f_r)


class TwoStreamDINOTiledFiLM(TwoStreamDINOBase):
    """Two-stream model with tiled encoding and FiLM conditioning."""
    
    def __init__(
        self,
        backbone_name: str,
        grid: Tuple[int, int] = (2, 2),
        **kwargs,
    ) -> None:
        super().__init__(backbone_name, **kwargs)
        self.grid = tuple(grid)
        self.film_left = FiLM(self.feat_dim)
        self.film_right = FiLM(self.feat_dim)
    
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
                        xt,
                        size=(self.input_res, self.input_res),
                        mode="bilinear",
                        align_corners=False,
                    )
                tiles.append(xt)

        # Stack and process all tiles in one forward pass
        num_tiles = len(tiles)
        tiles = torch.cat(tiles, dim=0)  # (B * num_tiles, C, H, W)
        feats = self.backbone(tiles)     # (B * num_tiles, D)
        feats = feats.view(num_tiles, B, -1).permute(1, 0, 2)  # (B, num_tiles, D)
        return feats
    
    def _encode_stream(self, x: torch.Tensor, film: FiLM) -> torch.Tensor:
        """Encode stream with FiLM conditioning."""
        tiles = self._tiles_backbone(x)  # (B, num_tiles, D)
        context = tiles.mean(dim=1)  # (B, D)
        
        gamma, beta = film(context)
        # Apply FiLM: x * (1 + gamma) + beta
        tiles = tiles * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        
        feat_stream = tiles.mean(dim=1)  # (B, D)
        return feat_stream
    
    def forward(
        self, x_left: torch.Tensor, x_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f_l = self._encode_stream(x_left, self.film_left)
        f_r = self._encode_stream(x_right, self.film_right)
        return self._merge_heads(f_l, f_r)


def build_model(
    backbone_name: str,
    model_type: str = "tiled_film",
    grid: Tuple[int, int] = (2, 2),
    pretrained: bool = True,
    dropout: float = 0.3,
    hidden_ratio: float = 0.25,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """
    Build model by type.
    
    Args:
        backbone_name: Name of the backbone (timm model)
        model_type: One of "plain", "tiled", "tiled_film"
        grid: Grid size for tiled models
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
        hidden_ratio: Hidden layer ratio
        gradient_checkpointing: Enable gradient checkpointing
        
    Returns:
        Model instance
    """
    kwargs = {
        "backbone_name": backbone_name,
        "pretrained": pretrained,
        "dropout": dropout,
        "hidden_ratio": hidden_ratio,
        "gradient_checkpointing": gradient_checkpointing,
    }
    
    if model_type == "tiled_film":
        return TwoStreamDINOTiledFiLM(grid=grid, **kwargs)
    elif model_type == "tiled":
        return TwoStreamDINOTiled(grid=grid, **kwargs)
    elif model_type == "plain":
        return TwoStreamDINOPlain(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

