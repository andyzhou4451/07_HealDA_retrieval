"""Per-obs 表示观测嵌入网络 ``XiChenRepresentationObsEmbedding`` (ROE)。

本模块实现 ``XiChenRepresentationObsEmbedding`` 类, 是 multimodal DA Solver
的下游: 接收 ``(xb, grad)`` (其中 ``grad = ∂J/∂xb`` 已经过 L2 归一化),
通过 patch embedding + Swin-V2 encoder + Swin-V2 latent (cross-attn with
``xb``) + Swin-V2 decoder 把每个 obs 源的 ``(xb, grad)`` 编码为 ``(B, L, D)``
表示 ``roe``, 送入 ``XiChenFusion`` 做跨 obs 源 Perceiver 融合。

网络结构与 ``XiChenDA`` 几乎相同(同名 ``forward_encoder`` / ``forward_latent``
/ ``forward_decoder``), 唯一区别:
- 本类不构造 ``self.final`` / ``self.log_var`` / ``self.out_norm`` 等上采样头,
  也不构造 ``self.enc_norm`` / ``self.latent_norm`` / ``self.dec_norm`` 的
  列表合并(每个 norm 用 ``add_module(f"enc_norm{i}", ...)`` 单独保存,
  ``forward`` 时通过 ``getattr`` 取);
- ``xb`` 与 ``grad`` 分别通过 ``self.patch_embed`` 与 ``self.patch_embed_grad`` 两个
  独立的 ``PatchEmbed`` 模块 tokenize（两个分支独立权重）;
- ``get_var_ids`` 加了 ``@lru_cache`` 装饰, 避免同一 ``(vars, device)`` 反复
  ``np.array`` + ``torch.from_numpy``。
"""
from functools import partial, lru_cache
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
import collections.abc
from einops import repeat, rearrange
import torch.nn.functional as F

from src.layers.patch_embed import PatchEmbed
from src.layers.swin_attn import SwinLayer

class XiChenRepresentationObsEmbedding(nn.Module):
    """Per-obs ``(xb, grad)`` 编码网络 (multimodal DA Solver 使用)。

    把单 obs 源的 ``(xb, grad)`` 编码为 ``(B, L, D)`` 表示 ``roe``。与
    ``XiChenDA`` 的区别: 本类没有上采样头, 输出直接交给 ``XiChenFusion``。

    Args:
        default_vars (list): list of default variables to be used for training
        img_size (list): image size of the input data
        patch_size (int): patch size of the input data
        embed_dim (int): embedding dimension
        depth (int): number of transformer layers
        num_blocks (int): number of fno blocks
        mlp_ratio (float): ratio of mlp hidden dimension to embedding dimension
        drop_path (float): stochastic depth rate
        drop_rate (float): dropout rate
        double_skip (bool): whether to use residual twice
    """

    def __init__(
        self,
        default_vars,
        img_size=[181, 360],
        window_size=[6, 12],
        patch_size=[5, 4],
        patch_stride=[4, 4],
        embed_dim=768,
        num_heads=12,
        encoder_depths=[2, 2, 2],
        latent_depths=[4, 4, 4],
        decoder_depths=[2, 2, 2],
        mlp_ratio=4,
        drop_path=0.2,
        drop_rate=0.2,
        attn_drop=0.,
    ):
        """初始化 ``XiChenRepresentationObsEmbedding``。

        与 ``XiChenDA.__init__`` 结构完全一致, 但不构造 ``final`` / ``log_var``
        / ``out_norm`` 等上采样头(本类输出 ``(B, L, D)`` 表示, 由
        ``XiChenFusion`` 统一上采样)。

        Args:
            default_vars (list[str]): 状态变量名列表(69 通道)。
            img_size (list[int]): 输入网格尺寸, 默认 ``[181, 360]``。
            window_size (list[int]): Swin 窗口大小, 默认 ``[6, 12]``。
            patch_size (list[int]): patch 卷积核, 默认 ``[5, 4]``。
            patch_stride (list[int]): patch 卷积步长, 默认 ``[4, 4]``。
            embed_dim (int): 隐向量维度, 默认 768。
            num_heads (int): 注意力头数, 默认 12。
            encoder_depths (list[int]): encoder 每段 SwinBlock 数。
            latent_depths (list[int]): latent 每段 SwinBlock 数。
            decoder_depths (list[int]): decoder 每段 SwinBlock 数。
            mlp_ratio (float): FFN 隐层倍率, 默认 4。
            drop_path (float): stochastic depth, 默认 0.2。
            drop_rate (float): dropout 率, 默认 0.2。
            attn_drop (float): 注意力 dropout, 默认 0。
        """
        super().__init__()

        # TODO: remove time_history parameter
        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.c = len(self.default_vars)
        self.h = self.img_size[0] // patch_stride[0]
        self.w = self.img_size[1] // patch_stride[1]
        self.embed_dim = embed_dim
        self.num_enc_layers = len(encoder_depths)
        self.num_latent_layers = len(latent_depths)
        self.num_dec_layers = len(decoder_depths)
        self.feat_size = [self.h, self.w]

        # variable tokenization: separate embedding layer for each input variable
        self.var_map = self.create_var_map()
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            patch_stride=patch_stride,
            in_chans=self.c, 
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches

        self.patch_embed_grad = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            patch_stride=patch_stride,
            in_chans=self.c, 
            embed_dim=embed_dim
        )

        # --------------------------------------------------------------------------

        encoder_layers = []
        for i in range(self.num_enc_layers):
            layer = SwinLayer(
                embed_dim,
                self.feat_size,
                window_size,
                depth=encoder_depths[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=drop_path,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                condition=False,
            )
            encoder_layers.append(layer)
            self.add_module(f"enc_norm{i}", norm_layer(embed_dim, eps=1e-6))

        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        self.enc_fpn = nn.Sequential(
            nn.Linear(embed_dim * self.num_enc_layers, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        latent_layers = []
        for i in range(self.num_latent_layers):
            layer = SwinLayer(
                embed_dim,
                self.feat_size,
                window_size,
                depth=latent_depths[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=drop_path,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                condition=True,
            )
            latent_layers.append(layer)
            self.add_module(f"latent_norm{i}", norm_layer(embed_dim, eps=1e-6))

        self.latent_layers = nn.ModuleList(latent_layers)

        self.latent_fpn = nn.Sequential(
            nn.Linear(embed_dim * self.num_latent_layers, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        decoder_layers = []
        for i in range(self.num_dec_layers):
            layer = SwinLayer(
                embed_dim,
                self.feat_size,
                window_size,
                depth=decoder_depths[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=drop_path,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                condition=False,
            )
            decoder_layers.append(layer)
            self.add_module(f"dec_norm{i}", norm_layer(embed_dim, eps=1e-6))

        self.decoder_layers = nn.ModuleList(decoder_layers)

        self.dec_fpn = nn.Sequential(
            nn.Linear(embed_dim * self.num_dec_layers, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.initialize_weights()

    def initialize_weights(self):
        """初始化 PatchEmbed + 遍历 apply ``_init_weights``。"""
        # token embedding layer
        w = self.patch_embed.proj.weight.data
        trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """递归初始化 ``nn.Linear`` (trunc_normal std=0.02) 与 ``nn.LayerNorm`` (常数)。"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_map(self):
        """构造 ``{var_name: idx}`` 字典, 用于 ``get_var_ids``。"""
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        """把变量名列表转成 ``torch.Tensor`` 索引, 移到 ``device`` (带 ``lru_cache``)。

        与 ``XiChenDA.get_var_ids`` 相比加了 ``@lru_cache(maxsize=None)`` 装饰:
        multimodal 训练时 ``variables`` 元组(69 通道全集)在每个 batch 都一样,
        ``lru_cache`` 避免重复 ``np.array + torch.from_numpy`` 拷贝。

        Args:
            vars (tuple[str] or list[str]): 变量名列表, 元素需在 ``self.var_map`` 中。
            device (torch.device): 输出张量所在设备。

        Returns:
            Tensor: 形状 ``(len(vars),)`` 的 int64 索引张量。
        """
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def forward_encoder(self, xb: torch.Tensor, grad: torch.Tensor, use_checkpoint=False):
        """Encoder 阶段(与 ``XiChenDA.forward_encoder`` 行为一致)。

        流程:
            1. ``patch_embed(xb)`` token 化 xb, ``patch_embed_grad(grad)`` 用
               独立的 patch 投影 (不与 xb 共享权重) token 化 grad;
            2. ``encoder_layers`` (无 condition) 逐层对 ``grad`` 编码;
            3. 收集每层 ``enc_norm{i}`` 输出, 拼接后经 ``enc_fpn`` 融合;
            4. 加回 ``grad`` 残差, 与 token 化的 ``xb`` 一起返回。

        Args:
            xb (Tensor): 背景场, 形状 ``(B, V, H, W)``。
            grad (Tensor): ``∂J/∂xb`` (L2 归一化), 形状 ``(B, V, H, W)``。
            use_checkpoint (bool): 是否对 SwinBlock 使用 ``torch.utils.checkpoint``。

        Returns:
            tuple[Tensor, Tensor]: ``(hb, hg)``, ``hb`` = ``patch_embed(xb)``
                形状 ``(B, L, D)``, ``hg`` = encoder 输出的 ``grad`` 表示。
        """
        # x: `[B, V, H, W]` shape.
        # (B, 8, H, W)
        xb = self.patch_embed(xb)

        grad = self.patch_embed_grad(grad)
        residual = grad

        # attention (swin or vit)
        outs = []
        for i, blk in enumerate(self.encoder_layers):
            if use_checkpoint:
                grad = checkpoint(blk, grad, use_reentrant=False)
            else:
                grad = blk(grad)
            out = getattr(self, f"enc_norm{i}")(grad)
            outs.append(out)

        if use_checkpoint:
            grad = checkpoint(self.enc_fpn, torch.cat(outs, dim=-1), use_reentrant=False)
        else:
            grad = self.enc_fpn(torch.cat(outs, dim=-1))

        grad = grad + residual

        return xb, grad

    def forward_latent(self, grad: torch.Tensor, xb: torch.Tensor, use_checkpoint=False):
        """Latent 阶段: ``condition=True`` 的 SwinBlock, 以 ``xb`` 为 cross-attn 条件。

        流程:
            1. 逐层 ``latent_layers`` 调 ``blk(grad, xb)`` 把 xb 作为 condition;
            2. 收集每层 ``latent_norm{i}`` 输出, ``latent_fpn`` 融合;
            3. 加回 ``grad`` 残差, 返回融合后的表示。

        Args:
            grad (Tensor): encoder 输出的 ``(B, L, D)`` 梯度表示。
            xb (Tensor): encoder 输出的 ``(B, L, D)`` 背景 patch 表示(作为 condition)。
            use_checkpoint (bool): 是否对 SwinBlock 使用 ``torch.utils.checkpoint``。

        Returns:
            Tensor: latent 融合后的 ``(B, L, D)`` 表示。
        """
        residual = grad

        # attention (swin or vit)
        outs = []
        for i, blk in enumerate(self.latent_layers):
            if use_checkpoint:
                grad = checkpoint(blk, grad, xb, use_reentrant=False)
            else:
                grad = blk(grad, xb)
            out = getattr(self, f"latent_norm{i}")(grad)
            outs.append(out)

        if use_checkpoint:
            grad = checkpoint(self.latent_fpn, torch.cat(outs, dim=-1), use_reentrant=False)
        else:
            grad = self.latent_fpn(torch.cat(outs, dim=-1))

        grad = grad + residual

        return grad

    def forward_decoder(self, h: torch.Tensor, use_checkpoint=False):
        """Decoder 阶段: SwinBlock + FPN 还原 patch 网格表示(无 ``out_norm``)。

        与 ``XiChenDA.forward_decoder`` 的区别: 本类不加 ``self.out_norm``,
        因为输出直接交给 ``XiChenFusion``。

        Args:
            h (Tensor): latent 输出的 ``(B, L, D)`` 表示。
            use_checkpoint (bool): 是否对 SwinBlock 使用 ``torch.utils.checkpoint``。

        Returns:
            Tensor: decoder 输出的 ``(B, L, D)`` 表示。
        """
        # x: `[B, V, H, W]` shape.
        # (B, 8, H, W)
        residual = h
        # attention (swin or vit)
        outs = []
        for i, blk in enumerate(self.decoder_layers):
            if use_checkpoint:
                h = checkpoint(blk, h, use_reentrant=False)
            else:
                h = blk(h)
            out = getattr(self, f"dec_norm{i}")(h)
            outs.append(out)

        if use_checkpoint:
            h = checkpoint(self.dec_fpn, torch.cat(outs, dim=-1), use_reentrant=False)
        else:
            h = self.dec_fpn(torch.cat(outs, dim=-1))

        h = h + residual

        return h

    def forward(self, xb, grad, variables, use_checkpoint=False):
        """前向计算: ``(xb, grad) -> ha`` 表示, 由 ``XiChenFusion`` 进一步处理。

        流程:
            1. ``forward_encoder`` 编码 (xb, grad) -> (hb, hg);
            2. ``forward_latent(hg, hb)`` 用 xb 作为 cross-attn 条件融合;
            3. ``forward_decoder`` 还原 patch 表示;
            4. 返回 ``(B, L, D)`` 形状的 roe 表示(不取 ``variables`` 子集,
               子集选择在 ``XiChenFusion`` 统一上采样之后做)。

        Args:
            xb (Tensor): 背景场, 形状 ``(B, C, H, W)``。
            grad (Tensor): ``∂J/∂xb`` 归一化梯度, 形状 ``(B, C, H, W)``, 与 xb 一致。
            variables (list[str] | tuple[str]): 占位参数, 本类不取子集 (子集选择
                在 ``XiChenFusion`` 统一上采样之后做), 保留签名仅为接口对齐。
            use_checkpoint (bool): 是否对 SwinBlock / CrossAttention 使用
                ``torch.utils.checkpoint``。默认 False。

        Returns:
            Tensor: 形状 ``(B, L, D)`` 的 roe 表示, ``L = (H/patch_stride[0]) * (W/patch_stride[1])``,
            ``D = embed_dim``。
        """
        hb, hg = self.forward_encoder(xb, grad, use_checkpoint)  # B, L, D

        ha = self.forward_latent(hg, hb, use_checkpoint)

        ha = self.forward_decoder(ha, use_checkpoint)

        return ha
