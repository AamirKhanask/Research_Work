# hamer/hamer/models/tokenmixers/mlgru_hgrn.py

import os
import torch
import torch.nn as nn
from typing import Optional  # Add at top of file
from ._vendor_mmfree.layers.hgrn_bit import HGRNBitAttention


class MLGRU1D(nn.Module):
    """
    A thin wrapper that exposes the parent repo's HGRNBitAttention
    as a simple 1D token mixer:
        in:  [B, L, D]   out: [B, L, D]
    """

    def __init__(self, hidden_size: int, num_heads: int, use_short_conv: bool = False, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx  # ← ADD THIS
        self.core = HGRNBitAttention(
            mode='fused_recurrent',
            hidden_size=hidden_size,
            num_heads=num_heads,
            expand_ratio=1,
            use_short_conv=use_short_conv,
            layer_idx=layer_idx,
        )

    def forward(self, x: torch.Tensor, lower_bound: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pass lower_bound through to HGRNBitAttention
        y, _, _ = self.core(
            x, 
            attention_mask=None, 
            past_key_values=None, 
            use_cache=False, 
            output_attentions=False,
            lower_bound=lower_bound  # ← ADD THIS
        )
        return y