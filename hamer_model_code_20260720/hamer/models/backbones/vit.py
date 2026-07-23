# Copyright (c) OpenMMLab. All rights reserved.
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

# SS2D + MLGRU token mixer (drop-in): forward([B,N,D], hw=(H,W)) -> [B,N,D]
from hamer.models.tokenmixers.ss2d_mlgru import SS2D_MLGRU_Attn

def monitor(var_name, var):  # monitor("Before_Attention vit 163", x)
    # """Prints details about a variable for debugging."""
    # print(f"\n\U0001f50d {var_name} Info:")
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

def vit(cfg):
    # Read token_mixer_type from config, default to "attention" for backward compatibility
    token_mixer_type = cfg.MODEL.BACKBONE.get('TOKEN_MIXER_TYPE', 'attention')
    token_mixer_heads = cfg.MODEL.BACKBONE.get('TOKEN_MIXER_HEADS', 16)
    # NEW: explicit config-driven control of SS2D_MLGRU_Attn's proj_in/proj_out.
    # Defaults preserve prior behavior (proj_in=False, proj_out=True) so existing
    # YAMLs that don't set these keys are unaffected.
    token_mixer_proj_in = cfg.MODEL.BACKBONE.get('TOKEN_MIXER_PROJ_IN', False)
    token_mixer_proj_out = cfg.MODEL.BACKBONE.get('TOKEN_MIXER_PROJ_OUT', True)

    print(f"🔧 Initializing ViT with token_mixer_type={token_mixer_type}, heads={token_mixer_heads}, "
          f"proj_in={token_mixer_proj_in}, proj_out={token_mixer_proj_out}")

    return ViT(
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.55,
        token_mixer_type=token_mixer_type,
        token_mixer_heads=token_mixer_heads,
        token_mixer_proj_in=token_mixer_proj_in,
        token_mixer_proj_out=token_mixer_proj_out,
    )


def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    """
    Resize absolute positional embeddings to (h, w) if needed; optionally keep cls token.
    abs_pos: (1, L, C)
    returns: (1, L', C)
    """
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, 0:1]
        abs_pos = abs_pos[:, 1:]

    if ori_h != h or ori_w != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(B, -1, C)
    else:
        new_abs_pos = abs_pos

    if cls_token is not None:
        new_abs_pos = torch.cat([cls_token, new_abs_pos], dim=1)
    return new_abs_pos


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return f"p={self.drop_prob}"


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_head_dim=None,
                 token_mixer_type="attention", token_mixer_heads=None,
                 layer_idx=0,
                 token_mixer_proj_in=False, token_mixer_proj_out=True):  # NEW
        super().__init__()
        self.norm1 = norm_layer(dim)
        self._uses_hw = False  # whether attn module needs (H,W)
        self.layer_idx = layer_idx

        if token_mixer_type == "attention":
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
            )
        else:
            # SS2D + MLGRU token mixer (keeps [B,N,D] I/O)
            from ..tokenmixers.ss2d_mlgru import SS2D_MLGRU_Attn
            self.attn = SS2D_MLGRU_Attn(
                dim=dim, num_heads=token_mixer_heads or num_heads, proj_drop=drop,
                layer_idx=layer_idx,
                proj_in=token_mixer_proj_in,    # NEW: now config-driven, no longer hardcoded
                proj_out=token_mixer_proj_out,  # NEW: now config-driven, no longer hardcoded
            )
            self._uses_hw = True

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, Hp, Wp, lower_bound=None):
        if self._uses_hw:
            monitor("Before_attn block vit 171", x)
            x = x + self.drop_path(self.attn(self.norm1(x), hw=(Hp, Wp), lower_bound=lower_bound))
            monitor("Before_mlp block vit 173", x)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        monitor("after_mlp block vit 177", x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (
            int(img_size[0] // patch_size[0] * ratio),
            int(img_size[1] // patch_size[1] * ratio),
        )
        self.origin_patch_shape = (
            int(img_size[0] // patch_size[0]),
            int(img_size[1] // patch_size[1]),
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # keep your original conv-based patch embed (note: padding/stride per ratio)
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=(patch_size[0] // ratio),
            padding=4 + 2 * (ratio // 2 - 1),
        )

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, L, D]
        return x, (Hp, Wp)


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding: extract backbone feature map and project to embedding dim. """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ViT(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 use_checkpoint=False, frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False,
                 token_mixer_type: str = "attention", token_mixer_heads: int = None,
                 token_mixer_proj_in: bool = False, token_mixer_proj_out: bool = True):  # NEW
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth
        self.token_mixer_type = token_mixer_type
        self.token_mixer_heads = token_mixer_heads or num_heads
        self.token_mixer_proj_in = token_mixer_proj_in
        self.token_mixer_proj_out = token_mixer_proj_out

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        # since the pretraining model has class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # ========== ADD LEARNABLE LOWER BOUNDS FOR MatMul-Free ==========
        if token_mixer_type != "attention":
            # Learnable lower bounds per layer (one scalar per layer)
            # Initialized to small values that will be converted via softmax + cumsum
            self.lower_bounds = nn.Parameter(torch.zeros(depth, 1))
        else:
            self.lower_bounds = None
        # ================================================================

        # build blocks with mixer selection decided at construction
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                token_mixer_type=self.token_mixer_type,
                token_mixer_heads=self.token_mixer_heads,
                layer_idx=i,
                token_mixer_proj_in=self.token_mixer_proj_in,    # NEW
                token_mixer_proj_out=self.token_mixer_proj_out,  # NEW
            )
            for i in range(depth)
        ])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters based on config flags."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for p in self.patch_embed.parameters():
                p.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval(); m.norm1.eval()
                for p in m.attn.parameters():
                    p.requires_grad = False
                for p in m.norm1.parameters():
                    p.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for p in self.patch_embed.parameters():
                p.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval(); m.norm2.eval()
                for p in m.mlp.parameters():
                    p.requires_grad = False
                for p in m.norm2.parameters():
                    p.requires_grad = False

    def init_weights(self):
        """Initialize weights."""
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        B, C, H, W = x.shape
        monitor("Before_patch_embed vit 365", x)
        x, (Hp, Wp) = self.patch_embed(x)
        monitor("Before_pos_embed vit 367", x)
        if self.pos_embed is not None:
            # keep your original absolute pos embed addition
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        monitor("Before_blk vit 371", x)

        # ========== COMPUTE PER-LAYER LOWER BOUNDS ==========
        # Following original MatMul-Free LM pattern:
        # lower_bounds = self.lower_bounds.softmax(0)  # Positive weights summing to 1
        # lower_bounds = lower_bounds.cumsum(0) - lower_bounds[0]  # Increasing from 0
        # This ensures: lb_0 = 0, lb_1 > 0, lb_2 > lb_1, ...
        if self.lower_bounds is not None:
            lb_weights = F.softmax(self.lower_bounds, dim=0)  # [depth, 1]
            lower_bounds = torch.cumsum(lb_weights, dim=0) - lb_weights[0]  # Start at 0, increase
            lower_bounds = lower_bounds.squeeze(1)  # [depth]
        else:
            lower_bounds = [None] * len(self.blocks)
        # ====================================================

        for i, blk in enumerate(self.blocks):
            lb = lower_bounds[i] if i < len(lower_bounds) else None
            if self.use_checkpoint:
                # thread (Hp, Wp) and lower_bound through checkpoint wrapper
                x = checkpoint.checkpoint(lambda t, b=blk, lb=lb: b(t, Hp, Wp, lower_bound=lb), x)
            else:
                x = blk(x, Hp, Wp, lower_bound=lb)
        monitor("Before_last_norm vit 378", x)
        x = self.last_norm(x)
        # return feature map [B, D, Hp, Wp] (HaMeR expects gridified tokens)
        monitor("Before_permute vit 381", x)
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        monitor("final_out vit 383", x)
        return xp

    def forward(self, x):
        return self.forward_features(x)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
