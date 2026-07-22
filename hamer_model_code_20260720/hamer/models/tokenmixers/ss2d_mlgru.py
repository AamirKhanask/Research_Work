# hamer/hamer/models/tokenmixers/ss2d_mlgru.py
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlgru_hgrn import MLGRU1D
from .bitlinear_from_parent import BitLinear  # use if you later want proj_in/proj_out

def monitor(var_name, var):  # monitor("Before_Attention vit 163", x)
    # """Prints details about a variable for debugging."""
    # print(f"\n🔍 {var_name} Info:")
    # print(f"Type: {type(var)}")
    # if isinstance(var, torch.Tensor):
    #     print(f"Value: {var}")
    #     print(f"Shape: {var.shape}")
    #     print(f"Dtype: {var.dtype}")
    #     print(f"Size: {var.numel()} elements")
    #     print(f"Min: {var.min().item() if var.numel() > 0 else 'N/A'}")
    #     print(f"Max: {var.max().item() if var.numel() > 0 else 'N/A'}")
    # else:
    #     print(f"Value: {var}")
    pass
    
def _to_grid(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    # x: [B, L, D] -> [B, D, H, W]
    B, L, D = x.shape
    assert L == H * W, f"SS2D needs L=H*W, got {L} vs {H}*{W}"
    return x.transpose(1, 2).reshape(B, D, H, W)

def _to_seq(y: torch.Tensor) -> torch.Tensor:
    # y: [B, D, H, W] -> [B, H*W, D]
    B, D, H, W = y.shape
    return y.reshape(B, D, H * W).transpose(1, 2)

def _cross_scan_4(x_bchw: torch.Tensor) -> torch.Tensor:
    # [B, C, H, W] -> 4 sequences [B, 4, L, C] (row, col, row_rev, col_rev)
    B, C, H, W = x_bchw.shape
    L = H * W
    seqs = x_bchw.flatten(2, 3)                               # row-major: [B, C, L]
    seqs_col = x_bchw.transpose(2, 3).flatten(2, 3)           # col-major: [B, C, L]
    seqs_rev = torch.flip(seqs, dims=[-1])
    seqs_col_rev = torch.flip(seqs_col, dims=[-1])
    # stack and convert to [B, 4, L, C]
    return torch.stack([seqs, seqs_col, seqs_rev, seqs_col_rev], dim=1).transpose(2, 3).contiguous()

def _cross_merge_4(y_b4lcd: torch.Tensor, H: int, W: int) -> torch.Tensor:
    # [B, 4, L, C] -> average the 4 directions back to [B, C, H, W]
    B, K, L, C = y_b4lcd.shape
    assert K == 4
    y_b4cl = y_b4lcd.transpose(2, 3)  # [B, 4, C, L]
    # invert the two reversed sequences
    dir0 = y_b4cl[:, 0]                             # row
    dir1 = y_b4cl[:, 1]                             # col
    dir2 = torch.flip(y_b4cl[:, 2], dims=[-1])      # row_rev -> row
    dir3 = torch.flip(y_b4cl[:, 3], dims=[-1])      # col_rev -> col
    # reshape col paths back to [B,C,H,W]
    d0 = dir0.view(B, C, H, W)
    d2 = dir2.view(B, C, H, W)
    d1 = dir1.view(B, C, W, H).transpose(2, 3).contiguous()
    d3 = dir3.view(B, C, W, H).transpose(2, 3).contiguous()
    return (d0 + d1 + d2 + d3) / 4.0 ###########

class SS2D_MLGRU_Attn(nn.Module):
    """
    Drop-in replacement for ViT's Attention:
    forward(x: [B, N, D]) -> [B, N, D], using SS2D + MLGRU1D (4 directions).
    """
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0,
                 proj_in: bool = False, proj_out: bool = True, use_short_conv: bool = False, 
                 layer_idx: int = 0):  # ← ADD layer_idx parameter
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx  # ← ADD THIS
        self.proj_in = BitLinear(dim, dim, bias=False) if proj_in else nn.Identity()
        self.act_in = nn.SiLU()
        self.mlgru = MLGRU1D(hidden_size=dim, num_heads=num_heads, 
                             use_short_conv=use_short_conv, layer_idx=layer_idx)  # ← Pass layer_idx
        self.act_out = nn.GELU()
        self.proj_out = BitLinear(dim, dim, bias=False) if proj_out else nn.Identity()
        self.drop = nn.Dropout(proj_drop)

    @torch.no_grad()
    def _infer_hw(self, N: int) -> Tuple[int, int]:
        hw = int(N**0.5)
        return hw, N // hw

    def forward(self, x: torch.Tensor, hw: Tuple[int, int] = None,
                lower_bound: Optional[torch.Tensor] = None) -> torch.Tensor:  # ← ADD lower_bound parameter
        # x: [B, N, D]
        B, N, D = x.shape
        
        H, W = self._infer_hw(N) if hw is None else hw
        assert H * W == N, f"L != H*W: got N={N}, H={H}, W={W}"

        # pre-proj + SiLU
        x = self.act_in(self.proj_in(x))
        grid = _to_grid(x, H, W)
        scans = _cross_scan_4(grid)
        scans = scans.reshape(B * 4, N, D)

        # ========== COMPUTE LOWER BOUND ==========
        # For layer 0, lower_bound should be None (no bound, allowing f to go to 0)
        # For layers > 0, we can use a learnable parameter or fixed value
        # The original uses: lower_bound = lower_bounds[i].softmax(0).cumsum(0) - lower_bounds[0]
        # For simplicity, we'll use a learnable parameter per layer
        
        # If lower_bound not provided, compute a default based on layer depth
        if lower_bound is None and self.layer_idx > 0:
            # Default: bound increases with depth (0.1 for deep layers)
            # You can make this learnable by creating a nn.Parameter in __init__
            lower_bound_val = min(0.1 * self.layer_idx / 32, 0.5)  # Cap at 0.5
            lower_bound = torch.tensor(lower_bound_val, device=scans.device, dtype=scans.dtype)
        # =========================================

        # Pass lower_bound to MLGRU (same bound for all 4 directions)
        y = self.mlgru(scans, lower_bound=lower_bound)
        y = y.view(B, 4, N, D)
        y_grid = _cross_merge_4(y, H, W)
        y = _to_seq(y_grid)
        
        y = self.act_out(y)
        y = self.drop(self.proj_out(y))
        return y