"""XiChen 数据压缩自编码器架构 (compression / arch.py)。

该模块定义了 XiChen 数据同化与存档链路中的自编码模型
:class:`XiChenAutoEncoder`,其骨干沿用与预报任务相同的 Swin-Transformer V2
``SwinLayer``,但在 encoder / decoder 之间插入一对 ``quan_mlp`` /
``post_quan_mlp`` 量化瓶颈,以及可选的 ``quan_norm`` / ``post_quan_norm`` 层
归一化 (``ending_norm``)。

设计要点:
    - 输入 69 通道 (4 表面 + 13 等压面层 × 5 变量),默认 1.0° 网格 ``181x360``。
    - 瓶颈 latent 维度默认 ``z_dim=69``,与输入通道数对齐,便于与同分辨率
      预报模型进行表示空间对齐。
    - 与 forecast 任务不同,本模型不在 latent 段注入 lead-time 信息,仅通过
      encoder / decoder 完成 ``x -> z -> x'`` 的有损压缩重构。
    - 对外暴露 :meth:`compress` 与 :meth:`decompress` 两个独立接口,可在无监
      督表征学习、数据同化 cost 计算以及存档存储场景中分别使用。

历史:
    较新的 ``arch.py`` 取代了早期的 ``arch_.py`` 实现,后者被保留为遗留/扩
    展变体,不在活跃训练路径中。
"""

from functools import partial, lru_cache
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.layers import trunc_normal_
import collections.abc
from einops import rearrange
import torch.nn.functional as F

from src.layers.patch_embed import PatchEmbed
from src.layers.swin_attn import SwinLayer
from src.layers.mlp import Mlp

class XiChenAutoEncoder(nn.Module):
    """基于 Swin-Transformer V2 + 量化瓶颈的多通道气象场自编码器。

    默认参数 ``embed_dim=768``、``z_dim=69``、``num_heads=12``、``mlp_ratio=4``、
    ``drop_path=0.2``。encoder 通过 ``quan_mlp`` (可选前置 ``quan_norm``)
    将 ``embed_dim`` 维 token 投影到 ``z_dim`` 维 latent,decoder 通过
    ``post_quan_mlp`` (可选 ``post_quan_norm``) 还原到原始维度,再经
    ``up_forward`` 上采样回 ``[B, V, H, W]``。

    训练时输出 ``(preds, log_var)``,``log_var`` 由 ``softplus`` 双向裁剪到
    ``[-10, 10]``。推理或存档场景可调用 :meth:`compress` / :meth:`decompress`
    仅走 encoder 或 decoder 路径。

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
        z_dim=69,
        num_heads=12,
        encoder_depths=[2, 2, 2],
        latent_depths=[4, 4, 4],
        decoder_depths=[2, 2, 2],
        mlp_ratio=4,
        drop_path=0.2,
        drop_rate=0.2,
        attn_drop=0.,
        ending_norm=False,
    ):
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
        self.z_dim = z_dim
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
        
        if self.z_dim is not None:
            self.quan_norm = norm_layer(embed_dim) if ending_norm else nn.Identity()  # for clip model only
            self.quan_mlp = Mlp(
                in_features=embed_dim,
                hidden_features=int(np.sqrt(embed_dim//z_dim))*z_dim,
                out_features=z_dim
            )
            
            self.post_quan_mlp = Mlp(
                in_features=z_dim,
                hidden_features=int(np.sqrt(embed_dim//z_dim))*z_dim,
                out_features=embed_dim
            )
            self.post_quan_norm = norm_layer(embed_dim) if ending_norm else nn.Identity()  # for clip model only

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

        self.out_norm = norm_layer(embed_dim, eps=1e-6)
        # --------------------------------------------------------------------------

        # prediction head
        if self.img_size[0] % 2 == 1:
            self.final = nn.ConvTranspose2d(
                in_channels=embed_dim, 
                out_channels=len(default_vars),
                kernel_size=patch_size, 
                stride=patch_stride, 
                bias=False
            )
            self.log_var = nn.ConvTranspose2d(
                in_channels=embed_dim, 
                out_channels=len(default_vars),
                kernel_size=patch_size, 
                stride=patch_stride, 
                bias=False
            )
        else:
            self.final = nn.Linear(
                embed_dim, 
                len(default_vars) * patch_size[-1] * patch_size[-2], 
                bias=False
            )
            self.log_var = nn.Linear(
                embed_dim, 
                len(default_vars) * patch_size[-1] * patch_size[-2], 
                bias=False
            )

        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        """初始化模型中所有可学习参数。

        流程:
            1. 对 ``PatchEmbed`` 的卷积权重按 ``std=0.02`` 做截断正态初始化;
            2. 调用 ``self.apply`` 触发 ``_init_weights`` 钩子,统一初始化所有
               :class:`nn.Linear` 与 :class:`nn.LayerNorm`。
        """
        # token embedding layer
        w = self.patch_embed.proj.weight.data
        trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
        
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """模块级权重初始化钩子,由 ``self.apply`` 触发。

        Args:
            m (:class:`nn.Module`): 当前遍历到的子模块,仅处理
                :class:`nn.Linear` 与 :class:`nn.LayerNorm`。
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
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

    def forward_encoder(self, x: torch.Tensor, use_checkpoint=False):
        """编码前向过程。

        将多通道气象场 token 化后,堆叠 ``num_enc_layers`` 段 Swin-V2 窗口自
        注意力层,再经 ``enc_fpn`` 将各段归一化输出拼接融合,最后加回 ``residual``
        残差。

        Args:
            x (torch.Tensor): ``[B, V, H, W]`` 形状的输入气象场。
            use_checkpoint (bool): 是否对每段使用梯度检查点以节省显存。

        Returns:
            torch.Tensor: ``[B, L, D]`` 形状的编码器输出潜在表示。
        """
        # x: `[B, V, H, W]` shape.
        # tokenize each variable separately
        # (B, 8, H, W)
        h = self.patch_embed(x)    
        residual = h

        # attention (swin or vit)
        outs = []
        for i, blk in enumerate(self.encoder_layers):
            if use_checkpoint:
                h = checkpoint(blk, h, use_reentrant=False)
            else:
                h = blk(h)
            out = getattr(self, f"enc_norm{i}")(h)
            outs.append(out)
            
        if use_checkpoint:
            h = checkpoint(self.enc_fpn, torch.cat(outs, dim=-1), use_reentrant=False)
        else:
            h = self.enc_fpn(torch.cat(outs, dim=-1))

        h = h + residual

        return h

    def forward_decoder(self, h: torch.Tensor, use_checkpoint=False):
        """解码前向过程。

        堆叠 ``num_dec_layers`` 段 Swin-V2 自注意力层,``dec_fpn`` 融合多段归
        一化输出,加回残差后经 ``out_norm`` 归一化,得到与输入空间同分辨率的
        解码特征。

        Args:
            h (torch.Tensor): ``[B, L, D]`` 形状的解码器输入 (例如 ``post_quan_mlp``
                输出)。
            use_checkpoint (bool): 是否对每段使用梯度检查点以节省显存。

        Returns:
            torch.Tensor: ``[B, L, D]`` 形状的解码特征。
        """
        # x: `[B, V, H, W]` shape.
        # tokenize each variable separately
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
        
        h = self.out_norm(h)

        return h

    def up_forward(self, x):
        """将潜在特征 ``[B, L, D]`` 上采样到 ``[B, V, H, W]`` 物理网格。

        根据 ``img_size[0]`` 的奇偶性分两套实现:
            - 奇数高度(例如 ``181``):使用 ``ConvTranspose2d`` 直接上采样。
            - 偶数高度:使用 :class:`nn.Linear` 沿 patch 维展开,再经
              :func:`einops.rearrange` 还原到 ``(B, V, H, W)``。

        Args:
            x (torch.Tensor): ``[B, L, D]`` 形状的解码特征。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 分别为 ``preds`` 与 ``log_var``,
            形状 ``[B, V, H, W]``。
        """
        x = x.view(x.size(0), self.h, self.w,-1)
        if self.img_size[0] % 2 == 1:
            res = self.final(x.permute(0, 3, 1, 2))
            log_var = self.log_var(x.permute(0, 3, 1, 2))
            return res, log_var
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
            log_var = self.log_var(x)
            log_var = rearrange(
                log_var,
                "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                p1=self.patch_size[-2],
                p2=self.patch_size[-1],
                h=self.img_size[0] // self.patch_size[-2],
                w=self.img_size[1] // self.patch_size[-1],
            )
            return res, log_var

    def compress(self, x, use_checkpoint=False):
        """仅执行编码 + 量化瓶颈,得到低维 latent 表示。

        用于离线压缩、存档存储以及无监督表征学习的 encoder-only 场景。

        Args:
            x (torch.Tensor): ``[B, V, H, W]`` 形状的输入气象场。
            use_checkpoint (bool): 是否对每段使用梯度检查点以节省显存。

        Returns:
            torch.Tensor: ``[B, L, z_dim]`` 形状的量化后 latent 表示;若
            ``self.z_dim`` 为 ``None`` 则返回 ``[B, L, embed_dim]``。

        Note:
            压缩结果未经变量子集筛选;``variables`` 参数在 ``decompress`` 时生效。
        """
        z = self.forward_encoder(x, use_checkpoint)  # B, L, D

        if self.z_dim is not None:
            z = self.quan_mlp(self.quan_norm(z))

        return z

    def decompress(self, z, variables, use_checkpoint=False):
        """从 latent 表示重建多通道气象场。

        当 ``self.z_dim`` 不为 ``None`` 时,先经 ``post_quan_norm`` 与
        ``post_quan_mlp`` 还原到 ``embed_dim`` 维,再走 decoder → 上采样路径。
        ``log_var`` 同样经过 ``softplus`` 双向裁剪到 ``[-10, 10]``,并按
        ``variables`` 列表裁剪到目标通道。

        Args:
            z (torch.Tensor): ``[B, L, z_dim]`` 或 ``[B, L, embed_dim]`` 形状
                的 latent 表示,具体形状取决于上游是否走过 ``quan_mlp``。
            variables (list[str]): 期望输出的变量名列表,按顺序裁剪通道。
            use_checkpoint (bool): 是否对每段使用梯度检查点以节省显存。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 分别为重建场 ``preds`` 与对
            数方差 ``log_var``,形状 ``[B, V', H, W]``。
        """
        if self.z_dim is not None:
            z = self.post_quan_mlp(self.post_quan_norm(z))

        z = self.forward_decoder(z, use_checkpoint)

        preds, log_var = self.up_forward(z)  # B, L, V*p*p
        
        log_var= -10 + F.softplus(log_var + 10)
        log_var = 10 - F.softplus(10 - log_var)

        out_var_ids = self.get_var_ids(tuple(variables), preds.device)
        preds = preds[:, out_var_ids]
        log_var = log_var[:, out_var_ids]

        return preds, log_var

    def forward(self, x, variables, use_checkpoint=False):
        """端到端前向传播:encoder → 量化瓶颈 → decoder → 上采样。

        与预报任务不同,该模型不带 lead-time 条件化,仅做 ``x → z → x'`` 的
        有损压缩重构。当 ``self.z_dim`` 不为 ``None`` 时,在 encoder 与 decoder
        之间串入 ``quan_mlp`` / ``post_quan_mlp`` 量化瓶颈(可选前置
        ``quan_norm`` / ``post_quan_norm``)。``log_var`` 由 ``up_forward`` 同步
        输出,并用 ``softplus`` 双向裁剪到 ``[-10, 10]`` 稳定区间;最终按
        ``variables`` 列表将 ``preds`` 与 ``log_var`` 裁剪到目标通道。

        Args:
            x (torch.Tensor): ``[B, Vi, H, W]`` 形状的输入 ERA5 状态场。
            variables (Sequence[str]): 目标输出变量名列表;用于按 ``get_var_ids``
                将模型全通道输出裁剪到目标子集 ``[B, Vo, H, W]``。
            use_checkpoint (bool): 是否在 encoder / decoder 内的 ``SwinLayer``
                上启用梯度检查点以节省显存。默认 ``False``。

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - **preds** (``torch.Tensor``): ``[B, Vo, H, W]`` 形状,按
                  ``variables`` 裁剪后的重构结果。
                - **log_var** (``torch.Tensor``): ``[B, Vo, H, W]`` 形状,与
                  ``preds`` 对齐的高斯观测-误差对数方差(已裁剪到 ``[-10, 10]``),
                  用于 CRPS-Gaussian 等概率损失。
        """
        z = self.forward_encoder(x, use_checkpoint)  # B, L, D

        if self.z_dim is not None:
            z = self.quan_mlp(self.quan_norm(z))
            
            z = self.post_quan_norm(self.post_quan_mlp(z))

        z = self.forward_decoder(z, use_checkpoint)

        preds, log_var = self.up_forward(z)  # B, L, V*p*p
        
        log_var= -10 + F.softplus(log_var + 10)
        log_var = 10 - F.softplus(10 - log_var)

        out_var_ids = self.get_var_ids(tuple(variables), preds.device)
        preds = preds[:, out_var_ids]
        log_var = log_var[:, out_var_ids]

        return preds, log_var
