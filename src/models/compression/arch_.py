"""XiChen 数据压缩自编码器 **遗留 / 扩展变体** (compression / arch_.py)。

.. warning::

    该文件是 ``arch.py`` 的 **遗留 / 扩展变体**,目前不在活跃训练路径中。
    活跃实现请参见 :mod:`src.models.compression.arch` 中的
    :class:`XiChenAutoEncoder`。本文件保留仅供历史对照与外部依赖 (``xichen_latent``
    自定义 Attention / Mlp 等) 的回滚参考,**不应再被新增训练任务引用**。

内容概览:
    - ``exists`` / ``to_2tuple``: 轻量工具函数。
    - :class:`PatchEmbed`: 局部 ``Conv2d`` patch tokenize,会保留 mask 信息,
      支持可变 patch grid。
    - :class:`Norm2d`: 通道维 LayerNorm,内部 permute 到 NHWC 再 permute 回来。
    - :class:`Block`: 单个 transformer block,可切换全局 Attention 与窗口
      :class:`WindowAttention`,带可选 relative-position-bias。
    - :class:`Encoder` / :class:`Decoder`: 单段(Swin 风格 half-depth)结构,
      ``encoder`` 段包含 ``quan_mlp`` 量化瓶颈,``decoder`` 段包含对称的
      ``post_quan_mlp``。
    - :class:`CRA5`: encoder + decoder 组合包装,提供 ``compress_to_latent``
      / ``decompress_from_latent`` / ``forward`` 三个对外接口。

历史:
    该实现引入了相对位置偏置、window 切换策略与 ``QuickGELU`` 激活等额外
    自由度;新一代 ``arch.py`` 将 Swin-V2 替换为统一的 :class:`SwinLayer` 抽象
    并默认开启 ``ending_norm``。本文件仍被 ``CRA5``/``Encoder``/``Decoder`` 通过
    显式 import 复用（见 ``src.models.compression.arch``）。
"""

from functools import partial, lru_cache
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import collections.abc
from einops import rearrange
import torch.nn.functional as F
import math
from xichen_latent.utils.model_utils import load_constant
from xichen_latent.models.modules.pos_embed import get_2d_sincos_pos_embed
from xichen_latent.models.modules.attention import Attention, WindowAttention, Mlp, QuickGELU

def exists(val):
    """判断值是否非 ``None``,作为 ``x is not None`` 的轻量封装。

    Args:
        val (Any): 待判断的任意 Python 对象。

    Returns:
        bool: ``val is not None`` 的布尔结果。
    """
    return val is not None

def to_2tuple(x):
    """将输入规范化为长度为 2 的 ``tuple``。

    若输入本身就是可迭代对象则原样返回,否则复制一份构成长度 2 的 ``tuple``。
    主要用于将 ``img_size`` / ``patch_size`` 等参数统一为 ``(h, w)`` 形式。

    Args:
        x (Any): 单值或可迭代对象。

    Returns:
        tuple: 长度为 2 的 tuple。
    """
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)

class PatchEmbed(nn.Module):
    """Image to Patch Embedding

    使用单层 :class:`nn.Conv2d` 作为 ``patch stride`` 的投影,输出
    ``[B, N, embed_dim]`` 形式的 token 序列,并按 ``patch_stride`` 同步处理
    可选 ``mask``,支持变分辨率的 patch grid (因 ``patch_shape`` 可动态调整)。

    Attributes:
        img_size (tuple[int, int]): 输入 ``(H, W)``。
        patch_shape (tuple[int, int]): token 化的网格尺寸 ``(Hp, Wp)``。
        num_patches (int): ``Hp * Wp``。
        patch_size (tuple[int, int]): 单个 patch 的 ``(kH, kW)``,由
            ``patch_size`` 和 ``patch_stride`` 解耦决定。
        proj (nn.Conv2d): 实际的 patch 投影卷积。
    """
    def __init__(
            self, 
            img_size=224, 
            patch_size=16,
            patch_stride=16, 
            in_chans=3, 
            embed_dim=768
        ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.img_size = img_size
        self.patch_shape = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])  # could be dynamic
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]  # could be dynamic
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride)

    def forward(self, x, mask=None, **kwargs):
        """执行 patch token 化,可选地同步插值 mask。

        Args:
            x (torch.Tensor): ``[B, C, H, W]`` 形状的输入。
            mask (torch.Tensor | None): 与 ``x`` 空间对齐的可选 0/1 mask,
                会被插值到 token grid 并二值化为 ``bool``。
            **kwargs: 兼容未来扩展,当前忽略。

        Returns:
            tuple[torch.Tensor, tuple[int, int], torch.Tensor | None]:
                分别为 ``[B, N, embed_dim]`` 的 token 序列、``(Hp, Wp)``
                网格尺寸,以及插值后的 bool mask (输入 ``mask=None`` 时返回 ``None``)。
        """
        x = self.proj(x) 
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        if mask is not None:
            mask = F.interpolate(mask[None].float(), size=(Hp, Wp)).to(torch.bool)[0]

        return x, (Hp, Wp), mask

class Norm2d(nn.Module):
    """对 ``[B, C, H, W]`` 张量在通道维做 :class:`nn.LayerNorm` 的便利包装。

    通过 ``permute(0, 2, 3, 1)`` 把通道维移到末尾进行 LN,再 permute 回原
    布局并保证 ``contiguous``。
    """

    def __init__(self, embed_dim):
        """初始化。

        Args:
            embed_dim (int): 通道维大小,等于 :class:`nn.LayerNorm` 的
                ``normalized_shape``。
        """
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        """对 ``[B, C, H, W]`` 张量做通道维 LN。

        Args:
            x (torch.Tensor): ``[B, C, H, W]`` 形状的张量。

        Returns:
            torch.Tensor: ``[B, C, H, W]`` 形状的归一化结果。
        """
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class Block(nn.Module):
    """单段 transformer block (全局 Attention / 窗口 Attention 可切换)。

    采用 Pre-LN 结构 + stochastic depth (``DropPath``) + GEGLU/MLP 块。窗口
    模式时启用 :class:`WindowAttention` 配合相对位置偏置,全局模式时退化为
    标准 :class:`Attention`。
    """

    def __init__(
            self, 
            dim, 
            num_heads, 
            mlp_ratio=4., 
            qkv_bias=False,
            drop_path=0., 
            act_layer=nn.GELU, 
            norm_layer=nn.LayerNorm,
            window_size=None, 
            window=False, 
            rel_pos_spatial=False
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if not window:
            self.attn = Attention(
                dim, 
                num_heads=num_heads, 
                qkv_bias=qkv_bias,
                window_size=window_size, 
                rel_pos_spatial=rel_pos_spatial
            )
        else:
            self.attn = WindowAttention(
                dim, 
                num_heads=num_heads, 
                qkv_bias=qkv_bias,
                window_size=window_size, 
                rel_pos_spatial=rel_pos_spatial,
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer
        )
       
    def forward(self, x, H, W, mask=None):
        """执行 Pre-LN 残差 block 的前向计算。

        Args:
            x (torch.Tensor): ``[B, N, C]`` 形状的 token 序列。
            H (int): patch grid 的高度 ``Hp``。
            W (int): patch grid 的宽度 ``Wp``。
            mask (torch.Tensor | None): 可选注意力 mask (当前未使用,保留以
                兼容后续 NestedTensor 化场景)。

        Returns:
            torch.Tensor: ``[B, N, C]`` 形状的输出 token 序列。
        """
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

norm_layer = partial(nn.LayerNorm, eps=1e-6)

class Encoder(nn.Module):
    """CRA5 自编码器的 encoder 单段实现 (legacy)。

    与新一代 :class:`src.models.compression.arch.XiChenAutoEncoder` 内部
    encoder 相比,本类只承载 ``depth // 2`` 段 block,并保留了 ``ending_norm``
    / ``quan_mlp`` 量化瓶颈以及 ``use_abs_pos_emb`` (sin-cos 2D 绝对位置编
    码) 等老式选项。``const_dir`` 启用时会从磁盘载入常数特征并与输入拼接。

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
        img_size=224,
        patch_size=16, 
        patch_stride=16, 
        z_dim=None,
        embed_dim=768, 
        depth=12,
        num_heads=12, 
        mlp_ratio=4.,
        qkv_bias=False, 
        window_size=(14,14),
        drop_path_rate=0.,
        norm_layer=None,
        window=True,
        use_abs_pos_emb=False,
        interval=3, 
        bn_group=None, 
        test_pos_mode='simple_interpolate',
        learnable_pos=False, 
        rel_pos_spatial=False, 
        lms_checkpoint_train=False, 
        pad_attn_mask=False, 
        freeze_iters=0,
        act_layer='GELU', 
        pre_ln=False,
        mask_input=False, 
        ending_norm=True,
        round_padding=False,
        const_dir=None,
    ):
        super().__init__()

        # TODO: remove time_history parameter
        self.default_vars = default_vars
        self.in_chans = len(self.default_vars)
        self.pad_attn_mask = pad_attn_mask  # only effective for detection task input w/ NestedTensor wrapping
        self.lms_checkpoint_train = lms_checkpoint_train
        self.freeze_iters = freeze_iters
        self.mask_input = mask_input
        self.ending_norm = ending_norm
        self.round_padding = round_padding
        self.patch_size = patch_size
        self.img_size = img_size
        self.depth = depth
        self.num_heads =num_heads
        self.Hp, self.Wp = 0, 0
        self.ori_Hp, self.ori_Hw = img_size[0] // patch_stride[0], \
                                   img_size[1] // patch_stride[1]

        if const_dir is not None:
            self.constant = torch.from_numpy(load_constant(const_dir))
        else:
            self.constant = None

        # variable tokenization: separate embedding layer for each input variable
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.z_dim = z_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            patch_stride=patch_stride,
            in_chans=self.in_chans, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=learnable_pos)
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.patch_shape, cls_token=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            raise

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList()
        for i in range(0, depth // 2):
            which_win = min(i%interval, len(window_size)-1)
            block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_path=dpr[i], norm_layer=norm_layer,
                window_size=window_size[which_win] if ((i + 1) % interval != 0) else self.patch_embed.patch_shape,
                window=((i + 1) % interval != 0) if window else False,
                rel_pos_spatial=rel_pos_spatial,
                act_layer=QuickGELU if act_layer == 'QuickGELU' else nn.GELU
            )
            self.blocks.append(block)

        self.ln_pre = norm_layer(embed_dim) if pre_ln else nn.Identity()  # for clip model only
        
        if self.z_dim is not None:
            self.norm = norm_layer(embed_dim) if ending_norm else nn.Identity()  # for clip model only
            self.quan_mlp = Mlp(
                in_features=embed_dim,
                hidden_features=int(np.sqrt(embed_dim//z_dim))*z_dim,
                out_features=z_dim
            )

        ### duplicated init, only affects network weights and has no effect given pretrain
        self.apply(self._init_weights)
        self.fix_init_weight()
        self.test_pos_mode = test_pos_mode

    def fix_init_weight(self):
        """按 ``1/sqrt(2 * layer_id)`` 重新缩放每层 attention proj 与 MLP fc2。

        修正深层 residual 叠加带来的方差漂移,常用于 ViT/Swin 风格的深度网络。
        """
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        """模块级权重初始化钩子,由 ``self.apply`` 触发。

        Args:
            m (:class:`nn.Module`): 当前遍历到的子模块,仅处理
                :class:`nn.Linear` 与 :class:`nn.LayerNorm`。
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        """返回 ``blocks`` 的层数。

        Returns:
            int: ``len(self.blocks)``,用于外部诊断 / 训练统计。
        """
        return len(self.blocks)

    def embedding_forward(self, x, mask=None, use_checkpoint=False):
        """执行 patch token 化 + 绝对位置编码 + 可选前置 LN 的嵌入阶段。

        若 ``self.constant`` 不为 ``None``,则将常数特征沿通道维拼接到输入
        再做 patch token 化;否则仅 token 化输入。位置编码使用 sin-cos 2D
        绝对位置编码 ``self.pos_embed`` (在 ``__init__`` 中初始化)。

        Args:
            x (torch.Tensor): ``[B, C, H, W]`` 形状的输入。
            mask (torch.Tensor | None): 与 ``x`` 空间对齐的可选 mask。
            use_checkpoint (bool): 是否启用梯度检查点 (当前未使用,保留接口)。

        Returns:
            torch.Tensor: ``[B, N, embed_dim]`` 形状的嵌入序列。
        """
        if self.constant is not None:
            constant = torch.repeat_interleave(self.constant, x.shape[0], dim=0).to(x.device, dtype=x.dtype)
            x, (self.Hp, self.Wp), mask = self.patch_embed(torch.concat([x, constant], dim=1), mask)    
        else:
            x, (self.Hp, self.Wp), mask = self.patch_embed(x, mask)    
        x = self.ln_pre(x) + self.pos_embed      #get_abs_pos(pos_embed, False, (self.ori_Hp, self.ori_Hw), patch_shape)
        
        return x
    
    def encoder_forward(self, x, use_checkpoint=False):
        """堆叠 ``len(blocks) - 1`` 个 :class:`Block`,不包含最后一个 block。

        注意: 最后一个 block 实例化但本方法不执行 — 当前实现未在
        ``encoder_forward`` 之后使用其输出（dead code 行为）；如需启用，
        可在 ``__init__`` 把 ``range(0, depth // 2)`` 改为 ``depth // 2 - 1``。

        Args:
            x (torch.Tensor): ``[B, N, embed_dim]`` 形状的嵌入序列。
            use_checkpoint (bool): 是否对每个 block 使用梯度检查点。

        Returns:
            torch.Tensor: ``[B, N, embed_dim]`` 形状的中间表示。
        """
        # x = self.ln_pre(x)  # effective for clip model only, otherwise nn.Identity
        for i in range(len(self.blocks)-1):
            if use_checkpoint:
                x = checkpoint(self.blocks[i], x, self.Hp, self.Wp)
            else:
                x = self.blocks[i](x, self.Hp, self.Wp)

        return x

    def forward(self, x, use_checkpoint=False):
        """Encoder 端到端前向:嵌入 + 堆叠 block + 可选量化瓶颈 + 重塑到 NCHW。

        ``z_dim`` 不为 ``None`` 时,会先经 ``norm`` (当 ``ending_norm=True``)
        再经 ``quan_mlp`` 投影到 ``z_dim`` 维 latent;否则保持 ``embed_dim``。

        Args:
            x (torch.Tensor): ``[B, C, H, W]`` 形状的输入气象场。
            use_checkpoint (bool): 是否启用梯度检查点。

        Returns:
            torch.Tensor: ``[B, C', Hp, Wp]`` 形状的潜在表示, ``C'`` 为
            ``z_dim`` (启用瓶颈) 或 ``embed_dim``。
        """
        x = self.embedding_forward(x, use_checkpoint=use_checkpoint)
        
        x = self.encoder_forward(x, use_checkpoint=use_checkpoint)
        
        if self.z_dim is not None:
            x = self.norm(x)
            x = self.quan_mlp(x)

        B, N, C = x.shape
        x = x.reshape(B, self.Hp, self.Wp, C).permute(0,3,1,2)

        return x

class Decoder(nn.Module):
    """CRA5 自编码器的 decoder 单段实现 (legacy)。

    与 :class:`Encoder` 对偶,只承载 ``depth // 2`` 段 block,包含
    ``post_quan_mlp`` 还原 bottleneck latent 到 ``embed_dim`` 维,以及可选
    ``ending_norm``。
    ``img_size == (721, 1440)`` 时使用 :class:`nn.ConvTranspose2d` 上采样,
    其他分辨率使用 ``Linear + einops.rearrange``。

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
        img_size=224,
        patch_size=16, 
        patch_stride=16, 
        z_dim=None,
        embed_dim=768, 
        depth=12,
        num_heads=12, 
        mlp_ratio=4.,
        qkv_bias=False, 
        window_size=(14,14),
        drop_path_rate=0.,
        norm_layer=None,
        window=True,
        use_abs_pos_emb=False,
        interval=3, 
        bn_group=None, 
        test_pos_mode='simple_interpolate',
        learnable_pos=False, 
        rel_pos_spatial=False, 
        lms_checkpoint_train=False, 
        pad_attn_mask=False, 
        freeze_iters=0,
        act_layer='GELU', 
        pre_ln=False,
        mask_input=False, 
        ending_norm=True,
        round_padding=False,
    ):
        super().__init__()
        # TODO: remove time_history parameter
        self.default_vars = default_vars
        self.pad_attn_mask = pad_attn_mask  # only effective for detection task input w/ NestedTensor wrapping
        self.lms_checkpoint_train = lms_checkpoint_train
        self.freeze_iters = freeze_iters
        self.mask_input = mask_input
        self.ending_norm = ending_norm
        self.round_padding = round_padding
        self.patch_size = patch_size
        self.img_size = img_size
        self.depth = depth
        self.num_heads =num_heads
        self.Hp, self.Wp = img_size[0] // patch_stride[0], \
                                   img_size[1] // patch_stride[1]

        # variable tokenization: separate embedding layer for each input variable
        self.var_map = self.create_var_map()

        self.patch_shape = (self.Hp, self.Wp)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.z_dim=z_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if z_dim is not None:
            self.post_quan_mlp = Mlp(
                in_features=z_dim,
                hidden_features=int(np.sqrt(embed_dim//z_dim))*z_dim,
                out_features=embed_dim
            )

        self.blocks = nn.ModuleList()
        for i in range(depth//2, depth):
            which_win = min(i%interval, len(window_size)-1)
            block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_path=dpr[i], norm_layer=norm_layer,
                window_size=window_size[which_win] if ((i + 1) % interval != 0) else self.patch_shape,
                window=((i + 1) % interval != 0) if window else False,
                rel_pos_spatial=rel_pos_spatial,
                act_layer=QuickGELU if act_layer == 'QuickGELU' else nn.GELU
            )
            self.blocks.append(block)

        self.ln_pre = norm_layer(embed_dim) if pre_ln else nn.Identity()  # for clip model only
        self.norm = norm_layer(embed_dim)

        if self.img_size==(721, 1440):
            self.final = nn.ConvTranspose2d(
                in_channels=embed_dim, 
                out_channels=len(default_vars),
                kernel_size=patch_size, 
                stride=patch_stride, 
                bias=False
            )
        else:
            self.final = nn.Linear(
                embed_dim, 
                len(default_vars)*patch_size[-1]*patch_size[-2], 
                bias=False
            )

        ### duplicated init, only affects network weights and has no effect given pretrain
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        """按 ``1/sqrt(2 * layer_id)`` 重新缩放每层 attention proj 与 MLP fc2。

        修正深层 residual 叠加带来的方差漂移,常用于 ViT/Swin 风格的深度网络。
        """
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        """模块级权重初始化钩子,由 ``self.apply`` 触发。

        Args:
            m (:class:`nn.Module`): 当前遍历到的子模块,仅处理
                :class:`nn.Linear` 与 :class:`nn.LayerNorm`。
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_map(self):
        """构建变量名 → 通道索引的映射字典。

        Returns:
            dict: 键为变量名字符串、值为该变量在输入张量通道维上的索引。供
            :meth:`get_var_ids` 反查使用。
        """
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        """根据变量名列表解析出对应的通道索引,并缓存到目标设备上。

        使用 ``lru_cache`` 缓存已解析结果,可显著加速训练时同一变量集合的反
        复查询。

        Args:
            vars (tuple[str]): 变量名可迭代对象,会被 hash 后缓存。
            device (torch.device | str): 返回张量所在的目标设备。

        Returns:
            torch.Tensor: 长度为 ``len(vars)`` 的 ``int64`` 索引张量。
        """
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def decoder_forward(self, x, use_checkpoint=False):
        """堆叠 ``len(blocks)`` 个 :class:`Block`,可选尾部 ``ending_norm``。

        Args:
            x (torch.Tensor): ``[B, N, embed_dim]`` 形状的输入序列。
            use_checkpoint (bool): 是否对每个 block 使用梯度检查点。

        Returns:
            torch.Tensor: ``[B, N, embed_dim]`` 形状的解码输出。
        """
        x = self.ln_pre(x)  # effective for clip model only, otherwise nn.Identity
        
        for i, blk in enumerate(self.blocks):
            if use_checkpoint:
                x = checkpoint(blk, x, self.Hp, self.Wp)
            else:
                x = blk(x, self.Hp, self.Wp)

        if self.ending_norm:
            x = self.norm(x)  # b h*w c

        return x
    
    def up_forward(self, x):
        """将 ``[B, N, embed_dim]`` 上采样到 ``[B, V, H, W]`` 物理网格。

        根据 ``img_size`` 是否等于 ``(721, 1440)`` 区分两套实现:
            - HR (721, 1440): 使用 :class:`nn.ConvTranspose2d`。
            - 其他分辨率: 使用 :class:`nn.Linear` + :func:`einops.rearrange`。
        注:与 ``arch.py`` 不同,本类只输出 ``res``,不输出 ``log_var``。

        Args:
            x (torch.Tensor): ``[B, N, embed_dim]`` 形状的解码特征。

        Returns:
            torch.Tensor: ``[B, V, H, W]`` 形状的重建场。
        """
        x = x.view(x.size(0), self.Hp, self.Wp,-1)
        if self.img_size==(721, 1440):
            res = self.final(x.permute(0, 3, 1, 2))
            return res
        else:
            x = self.final(x)
            res = rearrange(
                x,
                "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                p1=self.patch_size[-2],
                p2=self.patch_size[-1],
                h=self.img_size[0] // self.patch_size[-2],
                w=self.img_size[1] // self.patch_size[-1],
            )
            return res

    def forward(self, feat, vars=None, use_checkpoint=False):
        """Decoder 端到端前向:NCHW 输入 → 还原 latent → 堆叠 block → 上采样。

        ``z_dim`` 不为 ``None`` 时,先经 ``post_quan_mlp`` 还原到
        ``embed_dim`` 维;最后若 ``vars`` 不为 ``None``,按变量名裁剪到目标
        通道。

        Args:
            feat (torch.Tensor): ``[B, C, Hp, Wp]`` 形状的潜在表示。
            vars (list[str] | None): 期望输出的变量名列表, ``None`` 时输出
                全部通道。
            use_checkpoint (bool): 是否启用梯度检查点。

        Returns:
            torch.Tensor: ``[B, V', H, W]`` 形状的重建结果, ``V'`` 由 ``vars``
            决定。
        """
        B, C, H,W = feat.shape
        x = feat.reshape(B, C, -1).permute(0,2,1)

        if self.z_dim is not None:
            x = self.post_quan_mlp(x)

        out = self.decoder_forward(x, use_checkpoint=use_checkpoint)

        out = self.up_forward(out)

        if vars is not None:
            out_var_ids = self.get_var_ids(tuple(vars), out.device)
            out = out[:, out_var_ids]

        return out

class CRA5(nn.Module):
    """CRA5 自编码器顶层模型 (legacy)。

    将 :class:`Encoder` 与 :class:`Decoder` 配对组合,提供三个对外入口:
        - :meth:`compress_to_latent`: 仅跑 encoder。
        - :meth:`decompress_from_latent`: 仅跑 decoder。
        - :meth:`forward`: encoder → decoder 端到端重构。

    注意:本类不输出 ``log_var``,仅返回 ``preds``。

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
        img_size=224,
        patch_size=16, 
        patch_stride=16, 
        z_dim=None,
        embed_dim=768, 
        depth=12,
        num_heads=12, 
        mlp_ratio=4.,
        qkv_bias=False, 
        window_size=(14,14),
        drop_path_rate=0.,
        norm_layer=None,
        window=True,
        use_abs_pos_emb=False,
        interval=3, 
        bn_group=None, 
        test_pos_mode='simple_interpolate',
        learnable_pos=False, 
        rel_pos_spatial=False, 
        lms_checkpoint_train=False, 
        pad_attn_mask=False, 
        freeze_iters=0,
        act_layer='GELU', 
        pre_ln=False,
        out_ln=False, 
        mask_input=False, 
        ending_norm=True,
        round_padding=False,
        const_dir=None,
    ):
        super().__init__()

        self.encoder = Encoder(
            default_vars=default_vars,
            img_size=img_size,
            patch_size=patch_size, 
            patch_stride=patch_stride, 
            z_dim=z_dim,
            embed_dim=embed_dim, 
            depth=depth,
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, 
            window_size=window_size,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            window=window,
            use_abs_pos_emb=use_abs_pos_emb,
            interval=interval, 
            bn_group=bn_group, 
            test_pos_mode=test_pos_mode,
            learnable_pos=learnable_pos, 
            rel_pos_spatial=rel_pos_spatial, 
            lms_checkpoint_train=lms_checkpoint_train, 
            pad_attn_mask=pad_attn_mask, 
            freeze_iters=freeze_iters,
            act_layer=act_layer, 
            pre_ln=pre_ln,
            mask_input=mask_input, 
            ending_norm=ending_norm,
            round_padding=round_padding,
            const_dir=const_dir,
        )

        self.decoder = Decoder(
            default_vars=default_vars,
            img_size=img_size,
            patch_size=patch_size, 
            patch_stride=patch_stride, 
            z_dim=z_dim,
            embed_dim=embed_dim, 
            depth=depth,
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, 
            window_size=window_size,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            window=window,
            use_abs_pos_emb=use_abs_pos_emb,
            interval=interval, 
            bn_group=bn_group, 
            test_pos_mode=test_pos_mode,
            learnable_pos=learnable_pos, 
            rel_pos_spatial=rel_pos_spatial, 
            lms_checkpoint_train=lms_checkpoint_train, 
            pad_attn_mask=pad_attn_mask, 
            freeze_iters=freeze_iters,
            act_layer=act_layer, 
            pre_ln=pre_ln,
            mask_input=mask_input, 
            ending_norm=ending_norm,
            round_padding=round_padding,
        )

    def compress_to_latent(self, x, use_checkpoint=False):
        """仅执行 encoder,得到量化后的 latent 表示。

        Args:
            x (torch.Tensor): ``[B, V, H, W]`` 形状的输入气象场。
            use_checkpoint (bool): 是否启用梯度检查点。

        Returns:
            torch.Tensor: ``[B, C, Hp, Wp]`` 形状的 latent 表示。
        """
        latent = self.encoder(x, use_checkpoint=use_checkpoint)

        return latent

    def decompress_from_latent(self, latent, vars=None, use_checkpoint=False):
        """仅执行 decoder,从 latent 重建多通道气象场。

        Args:
            latent (torch.Tensor): ``[B, C, Hp, Wp]`` 形状的潜在表示。
            vars (list[str] | None): 期望输出的变量名列表, ``None`` 时输出
                全部通道。
            use_checkpoint (bool): 是否启用梯度检查点。

        Returns:
            torch.Tensor: ``[B, V', H, W]`` 形状的重建结果。
        """
        recon = self.decoder(latent, vars, use_checkpoint=use_checkpoint)

        return recon

    def forward(self, x, vars=None, use_checkpoint=False):
        """端到端前向:encoder → 量化 → decoder → 单步重建。

        Args:
            x (`torch.Tensor`): `[B, V, H, W]` 输入。
            vars (`list[str]`): 输入变量名列表 (用于 ``create_var_map``)。
            use_checkpoint (bool): SwinBlock 是否启用 gradient checkpointing。

        Returns:
            torch.Tensor: `[B, V, H, W]` 重建张量。
        """
        latent = self.encoder(x, use_checkpoint=use_checkpoint)

        recon = self.decoder(latent, vars, use_checkpoint=use_checkpoint)

        return recon
