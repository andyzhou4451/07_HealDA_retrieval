# -*- coding: utf-8 -*-
"""ViT backbone for HealDA-style retrieval.

The class name keeps the HPX terminology used by HealDA.  When HPX dependencies
are unavailable, the model operates on the 181x360 lat-lon fallback grid while
retaining patch encode / Transformer / patch decode structure.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        random.floor_()
        return x.div(keep) * random


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with query/key RMS-normalized attention input."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        # RMS-normalize Q/K/V input for bf16 stability; this is a lightweight approximation
        # of HealDA's q/k RMS normalization.
        y = y / y.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LatLonViTBackbone(nn.Module):
    """Patch encode -> global Transformer -> patch decode backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 26,
        img_size: Sequence[int] = (181, 360),
        patch_size: Sequence[int] = (6, 6),
        dim: int = 512,
        depth: int = 12,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.05,
        drop_path: float = 0.1,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.dim = int(dim)
        self.use_checkpoint = bool(use_checkpoint)
        self.patch_encode = nn.Conv2d(in_channels, dim, kernel_size=self.patch_size, stride=self.patch_size)
        h, w = self.img_size
        ph, pw = self.patch_size
        self.pad_h = (math.ceil(h / ph) * ph) - h
        self.pad_w = (math.ceil(w / pw) * pw) - w
        gh = (h + self.pad_h) // ph
        gw = (w + self.pad_w) // pw
        self.grid_tokens = (gh, gw)
        self.pos_embed = nn.Parameter(torch.zeros(1, gh * gw, dim))
        dpr = torch.linspace(0, drop_path, depth).tolist() if depth > 0 else []
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.patch_decode = nn.ConvTranspose2d(dim, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.out_proj = nn.Sequential(nn.GroupNorm(8 if dim % 8 == 0 else 1, dim), nn.SiLU(), nn.Conv2d(dim, out_channels, kernel_size=1))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        if self.pad_h or self.pad_w:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h))
        x = self.patch_encode(x)
        gh, gw = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2)
        if tokens.shape[1] != self.pos_embed.shape[1]:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2).view(1, self.dim, *self.grid_tokens),
                size=(gh, gw), mode="bilinear", align_corners=False,
            ).flatten(2).transpose(1, 2)
        else:
            pos = self.pos_embed
        tokens = tokens + pos
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                tokens = checkpoint.checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        tokens = self.norm(tokens)
        x = tokens.transpose(1, 2).view(b, self.dim, gh, gw)
        x = self.patch_decode(x)
        x = x[..., :h, :w]
        return self.out_proj(x)


class HPXViTBackbone(LatLonViTBackbone):
    """Compatibility alias for the HealDA HPX ViT backbone."""

    pass
