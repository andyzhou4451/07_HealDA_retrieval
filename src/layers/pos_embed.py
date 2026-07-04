"""
2D 位置编码工具（sin-cos）与 checkpoint 插值器

为视觉 transformer / Swin 提供：

- ``get_2d_sincos_pos_embed``：按 (grid_h, grid_w) 生成 2D 固定位置编码；
  ``embed_dim`` 必须为偶数，前半编码 grid_h、后半编码 grid_w。
- ``get_2d_sincos_pos_embed_from_grid`` / ``get_1d_sincos_pos_embed_from_grid``：
  上述的内部实现。
- ``interpolate_pos_embed``：当推理分辨率高于预训练时，对
  ``pos_embed`` 做双三次插值（DeiT 风格），以避免重复预训练。
- ``interpolate_channel_embed``：当输入通道数减少时裁剪
  ``channel_embed``（如去掉某些卫星通道）。

上游依赖：MAE / MoCo v3 / DeiT 官方实现；``timm.models.layers`` 提供
``drop_path, to_2tuple, trunc_normal_``。
下游调用：被各 backbone ``arch.py`` 的 ``PatchEmbed`` 之后立即调用，
以生成 ``net.pos_embed``；checkpoint 加载脚本（在 ``inference`` 中）会
通过 ``interpolate_pos_embed`` 适配高分辨率推理。
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------


import numpy as np
import torch
import timm
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """生成 2D 固定（sin-cos）位置编码。

    关键约定：

    - 将 ``grid_h`` 的编码放在前 ``embed_dim / 2`` 维，将 ``grid_w`` 的
      编码放在后 ``embed_dim / 2`` 维，最后沿通道维拼接。
    - ``np.meshgrid`` 中 width 在前，与原 TensorFlow 实现保持一致。
    - ``cls_token=True`` 时在编码最前面拼接一行 ``0``，给 ``[CLS]`` token
      留一个"无位置"占位（与 ViT / MAE 一致）。

    Args:
        embed_dim: 编码维度（必须为偶数）。
        grid_size: ``int`` 或 ``(int, int)``，分别表示 grid 高 / 宽。
        cls_token: 是否在第一行预留 ``[CLS]`` 位置。

    Returns:
        ``np.ndarray``，shape 为 ``(grid_h*grid_w, embed_dim)``，
        若 ``cls_token=True`` 则为 ``(1+grid_h*grid_w, embed_dim)``。
    """
    grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    # import pdb
    # pdb.set_trace()
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """把 ``grid``（shape ``(2, 1, H, W)``）拆为 h / w 两组 1D 编码再拼接。

    Args:
        embed_dim: 总维度（必须为偶数）。
        grid: shape ``(2, 1, H, W)`` 的坐标网格。

    Returns:
        ``np.ndarray``，shape ``(H*W, embed_dim)``。
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """经典 1D sin-cos 位置编码（Transformer 风格）。

    频率以 ``10000 ** (-2k/d)`` 等比递减；输出拼接 ``[sin, cos]``，
    形状为 ``(M, embed_dim)``。

    Args:
        embed_dim: 输出维度（必须为偶数）。
        pos: 位置数组，shape ``(M,)``。

    Returns:
        ``np.ndarray``，shape ``(M, embed_dim)``。
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model, new_size=(64, 128)):
    """把预训练 ``pos_embed`` 插值到 ``new_size``（按 patch 数对齐）。

    算法流程：

    1. 从 ``checkpoint_model["net.pos_embed"]`` 拿到原形状
       ``(1, orig_num_patches, embed_dim)``，并通过 ``w_h_ratio=2`` 假设
       推回原始 ``(orig_h, orig_w)``（项目里所有 backbone 都按 2:1 设计）。
    2. ``new_size`` 按 ``patch_size`` 折算为新 patch 数 ``(new_h, new_w)``。
    3. 若 ``(orig_h, orig_w) != (new_h, new_w)``，对 ``pos_embed`` 做
       ``bicubic`` 插值，再 reshape 回 ``(1, new_h*new_w, embed_dim)``
       写回 ``checkpoint_model``。

    Args:
        model: 当前 ``nn.Module``；仅用于读取 ``patch_size``。
        checkpoint_model: state_dict 字典（原地修改）。
        new_size: 推理时新的 ``(H, W)``（未除 patch 之前的物理像素）。

    Note:
        ``w_h_ratio=2`` 是 XiChen 的"硬编码假设"——所有模型都按
        ``2:1`` 的宽高比训练。当新输入不符合 2:1 时，需要扩展此处。
    """
    if "net.pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["net.pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        orig_num_patches = pos_embed_checkpoint.shape[-2]
        patch_size = model.patch_size
        w_h_ratio = 2
        orig_h = int((orig_num_patches // w_h_ratio) ** 0.5)
        orig_w = w_h_ratio * orig_h
        orig_size = (orig_h, orig_w)
        new_size = (new_size[0] // patch_size, new_size[1] // patch_size)
        # print (orig_size)
        # print (new_size)
        if orig_size != new_size:
            print("Interpolate PEs from %dx%d to %dx%d" % (orig_size[0], orig_size[1], new_size[0], new_size[1]))
            pos_tokens = pos_embed_checkpoint.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(
                0, 3, 1, 2
            )
            new_pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size[0], new_size[1]), mode="bicubic", align_corners=False
            )
            new_pos_tokens = new_pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            checkpoint_model["net.pos_embed"] = new_pos_tokens


def interpolate_channel_embed(checkpoint_model, new_len):
    """就地裁剪 ``net.channel_embed`` 以适配更少的输入通道。

    当推理只用到部分通道（例：去掉某些卫星通道）时，把
    ``channel_embed_checkpoint[:, :new_len]`` 替换原值。若 ``new_len > old_len``
    则保持原值不修改（依赖外层逻辑决定是否报错）。

    Args:
        checkpoint_model: state_dict 字典。
        new_len: 目标通道数。
    """
    if "net.channel_embed" in checkpoint_model:
        channel_embed_checkpoint = checkpoint_model["net.channel_embed"]
        old_len = channel_embed_checkpoint.shape[1]
        if new_len <= old_len:
            checkpoint_model["net.channel_embed"] = channel_embed_checkpoint[:, :new_len]