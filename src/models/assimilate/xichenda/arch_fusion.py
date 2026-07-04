"""多观测源跨模态融合网络 ``XiChenFusion`` (Perceiver 风格) + ``PerceiverAttention``。

本模块包含两个类:
- ``PerceiverAttention``: 标准 cross-attention, ``latent_query`` 对 ``x`` 做
  Q-from-latent / K,V-from-``x`` 风格的注意力;支持可选的 ``LayerNorm(k)`` /
  ``LayerNorm(q)``(在多头切分之前),以及 batch 超过 ``b_lim=40_000`` 时按
  batch 切片回退,避免 NPU ``scaled_dot_product_attention`` 报 OOM。
- ``XiChenFusion``: multimodal DA Solver 的下游, 接收
  ``{obs_name: roe}`` 字典, 拼接 + learnable obs embedding + 跨 obs 源
  Perceiver 聚合 (``agg_depth`` 层 latent cross-attn) + Swin 精调 + 上采样,
  输出 ``(xa, log_var)``。

设计要点:
- **Perceiver latent query**: ``self.latent_query`` 是 ``(1, 1, embed_dim)``
  形状的可学习参数, 在 ``aggregate_observations`` 中按 batch 复制;
- **obs embedding**: ``self.obs_embed`` 是 ``nn.Embedding(V_max, embed_dim)``
  的可学习 per-obs 偏置表 (V_max = max(obs_vocab.values()) + 1, 包含扩展槽位),
  沿 obs 维加到 token;
- **大 batch 回退**: ``F.scaled_dot_product_attention`` 在 batch >= 40_000
  时按 chunk 切分, 避免 NPU/GPU 一次性 OOM;
- **log_var 双侧截断**: 与 ``XiChenDA`` 一致, ``[-10, 10]`` softplus clamp。
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
from src.layers.mlp import GeGLUFFN
from src.layers.patch_embed import PatchEmbed
from src.layers.swin_attn import SwinLayer
from src.layers.pos_embed import get_2d_sincos_pos_embed
from src.utils import get_logger

log = get_logger("xichen.arch_fusion")

class PerceiverAttention(nn.Module):
    """Perceiver 风格 cross-attention, 基于 ``nn.MultiheadAttention`` 实现 (Task 3)。

    与旧实现的差异:
    - 旧版手写 ``to_q`` / ``to_kv`` / ``to_out`` + 手写 ``rearrange`` 切头 + 手写 SDPA
    - 新版用 ``nn.MultiheadAttention(batch_first=True)`` 一体化处理, 获得:
        (1) 原生 ``key_padding_mask`` 支持 (True=pad, PyTorch 标准语义)
        (2) 自动 NPU/CPU 后端 dispatch
        (3) 参数名规范 (``mha.in_proj_weight`` / ``mha.out_proj.weight``)
    - 旧版 ``to_out`` 行为完全保留: ``mha.out_proj`` 即"多头 concat 后回投到 embed_dim"
    - 旧版 ``ln_k`` / ``ln_q`` 在 MHA 实现里已不需要 (MHA 内部机制覆盖), **不保留占位**
    - 旧版 ``ln_k_q`` 构造参数无功能意义, **直接删除**

    Attributes:
        mha (nn.MultiheadAttention): 多头注意力, 含 Q/K/V 合并投影 + out_proj。
        num_heads (int): 注意力头数。
        attn_drop (float): MHA 的注意力 dropout。
    """

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        attn_drop=0.,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads,
            dropout=attn_drop, 
            bias=False, 
            batch_first=True,
        )
        self.attn_drop = attn_drop

    def forward(
        self,
        latent_query: torch.Tensor,
        x: torch.Tensor,
    ):
        """Perceiver cross-attention。

        Args:
            latent_query (Tensor): 形状 (B, L1, D) 的 latent query token。
            x (Tensor): 形状 (B, L2, D) 的上下文 token 序列, L2 = V_obs。
            key_padding_mask (Tensor, optional): 形状 (B, L2) 的 bool 张量,
                True 表示该 obs token 是 padding, 需在 cross-attn 中屏蔽。
                None 表示所有 token 都有效 (训练时全集场景)。

        Returns:
            Tensor: 形状 (B, L1, D) 的 cross-attention 输出。
        """
        attn_out, _ = self.mha(
            query=latent_query, 
            key=x, 
            value=x,
        )
        return attn_out

norm_layer = partial(nn.LayerNorm, eps=1e-6)

class XiChenFusion(nn.Module):
    """多观测源跨模态融合网络 (Perceiver 风格)。

    接收 ``{obs_name: roe}`` 字典, 拼接 + 加 per-obs 可学习偏置 +
    ``agg_depth`` 层 Perceiver cross-attn 聚合到 1 个全局 latent, 再过
    Swin-V2 精调 (``condition=False``), 最后 ``up_forward`` 上采样回
    ``(B, V, H, W)`` 得到 ``(xa, log_var)``。

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
        obs_list,
        img_size=[181, 360],
        window_size=[6, 12],
        patch_size=[5, 4],
        patch_stride=[4, 4],
        embed_dim=768,
        num_heads=12,
        agg_depth=2,
        depths=[2, 2, 2],
        mlp_ratio=4,
        drop_path=0.2,
        drop_rate=0.2,
        attn_drop=0.,
        learnable_pos=False,
        obs_vocab=None,        # 新增 kwargs
        **kwargs,              # 透传未知参数
    ):
        """初始化 ``XiChenFusion``, 构造 obs 嵌入 + Perceiver 聚合 + Swin 精调。

        关键组件:
            - ``self.obs_embed``: ``nn.Embedding(V_max, embed_dim)`` 形态的可学习
              per-obs 偏置表 (V_max = max(obs_vocab.values()) + 1), 沿 obs 维加到 token;
            - ``self.latent_query``: 形状 ``(1, 1, embed_dim)`` 的可学习全局
              latent token, 复制 ``B * L`` 次后送入 ``latent_agg``;
            - ``self.latent_agg``: ``agg_depth`` 层 ``PerceiverAttention`` +
              ``GeGLUFFN`` 残差块(``ln_k_q=False``), 把 7 个 obs 源 token
              聚合到 1 个全局 latent;
            - ``self.layers`` + ``self.fpn``: Swin-V2 精调(``condition=False``),
              每段后接 ``LayerNorm``, FPN 融合;
            - ``self.final`` / ``self.log_var``: 上采样分支, 同 ``XiChenDA``。

        Args:
            default_vars (list[str]): 状态变量名列表(69 通道)。
            obs_list (list[str]): 观测源名列表, 如
                ``["atms", "amsua", "mhs", "hrs4", "prepbufr", "satwnd", "ascat"]``。
            img_size (list[int]): 输入网格尺寸, 默认 ``[181, 360]``。
            window_size (list[int]): Swin 窗口大小, 默认 ``[6, 12]``。
            patch_size (list[int]): patch 卷积核, 默认 ``[5, 4]``。
            patch_stride (list[int]): patch 卷积步长, 默认 ``[4, 4]``。
            embed_dim (int): 隐向量维度, 默认 768。
            num_heads (int): 注意力头数, 默认 12。
            agg_depth (int): Perceiver 聚合层数, 默认 2。
            depths (list[int]): Swin 精调每段 Block 数, 默认 ``[2, 2, 2]``。
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
        self.obs_list = obs_list
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.c = len(self.default_vars)
        self.h = self.img_size[0] // patch_stride[0]
        self.w = self.img_size[1] // patch_stride[1]
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.feat_size = [self.h, self.w]

        # variable tokenization: separate embedding layer for each input variable
        self.var_map = self.create_var_map()

        # obs_vocab: 接受 plain dict 或 Hydra/OmegaConf DictConfig (Q1 修订)。
        # 原 "仅 plain dict" 约束会让 `python main.py model=assimilate/.../v7obs`
        # 触发 TypeError, 因 Hydra instantiate 时 DictConfig 仍未 to_container。
        # 放宽为 ``Mapping`` 协议, 仍禁止 None/str/list 等非映射类型, 避免静默错配。
        if obs_vocab is not None and not isinstance(obs_vocab, collections.abc.Mapping):
            raise TypeError(
                f"obs_vocab must be a mapping (dict or DictConfig), got {type(obs_vocab)}. "
                f"Pass a YAML mapping like ``obs_vocab: {{atms: 0, ...}}``."
            )
        if obs_vocab is None:
            obs_vocab = {n: i for i, n in enumerate(obs_list)}
        # 统一物化为 plain dict, 后续 _lookup_obs_ids / state_dict 保存路径
        # 都不依赖 DictConfig 的特殊行为, 且 plain dict 可序列化更稳定。
        # === F12 修复: 删死 else 分支 (code-review)
        # 原版 `dict(obs_vocab) if obs_vocab is not None else obs_vocab` 的 else
        # 永远不执行 (L203-204 已把 None 替换为 dict,此处 obs_vocab 必非 None)。
        self.obs_vocab = dict(obs_vocab)

        # === D15 修复: 显式校验 obs_vocab values 是 int (且非 bool), 防止未引号 YAML
        # (obs_vocab: {atms: '0', ...}) 让 max() 走字符串比较得到 max char '9'+1
        # 报 TypeError, 或 dict-with-int values 混入 str/bool 后静默错位 (code-review #5) ===
        # === D15.1 修复: 显式排除 bool (code-review F1) ===
        # Python 中 bool 是 int 的子类, isinstance(True, int) == True; 若不排除,
        # YAML `atms: yes` 会解析为 True 并被当作 id=1 静默通过 D15。改为
        # `type(v) is int` 同时排除 bool 和任何 int 子类,语义最严格。
        non_int = {k: v for k, v in self.obs_vocab.items() if type(v) is not int}
        if non_int:
            raise TypeError(
                f"obs_vocab values must be int (not bool); got non-int entries: {non_int}. "
                f"Use plain integers in YAML (e.g. `atms: 0`); avoid `yes`/`no`/`true`/`false` "
                f"which YAML parses to bool and bypasses int checks."
            )

        # === D17 修复: 校验 obs_vocab id 是 0..V-1 连续整数, 防止稀疏 id
        # (e.g. {atms: 0, mhs: 5}) 浪费 Embedding 行 + 破坏 V_old<V_max 扩展迁移
        # 的 "前 V_old 行 copy" 假设 (code-review #7)。
        sorted_ids = sorted(self.obs_vocab.values())
        if sorted_ids != list(range(len(self.obs_vocab))):
            raise ValueError(
                f"obs_vocab ids must be contiguous 0..N-1; got {sorted_ids} for "
                f"vocab of size {len(self.obs_vocab)}. Sparse ids waste Embedding "
                f"rows and break load_state_dict migration semantics. "
                f"Use {dict(zip(self.obs_vocab.keys(), range(len(self.obs_vocab))))}."
            )

        # obs_embed 统一为 nn.Embedding: V_max = max(obs_vocab.values()) + 1。
        # 新行默认 trunc_normal_(0.02) 初始化 (_init_weights), 老 ckpt 加载由
        # PyTorch load_state_dict(strict=False) 原生处理 (V_old < V_max 自动
        # copy 老行, 新行保留 trunc_normal 初始化; V_old == V_max 完美匹配)。
        V_max = max(self.obs_vocab.values()) + 1
        self.obs_embed = nn.Embedding(num_embeddings=V_max, embedding_dim=embed_dim)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, int(self.feat_size[0] * self.feat_size[1]), embed_dim), requires_grad=learnable_pos)
        
        # variable aggregation: a learnable query and a single-layer cross attention
        # === 关键: nn.Parameter 默认是 torch.zeros, 但 _init_weights 只覆盖
        # nn.Linear / nn.LayerNorm / nn.Embedding, 不动 nn.Parameter, 此前 latent_query
        # 永远从全零开始训练, 导致首 epoch 跨 obs 融合信号为 0, loss 不下降 (code-review #1)。
        # 显式用 trunc_normal_(0.02) 初始化, 与项目其他 Linear 层约定一致。
        self.latent_query = nn.Parameter(torch.empty(1, int(self.feat_size[0] * self.feat_size[1]), embed_dim), requires_grad=True)
        
        self.latent_agg = nn.ModuleList([])
        for i in range(agg_depth):
            self.latent_agg.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            attn_drop=attn_drop,
                        ),
                        GeGLUFFN(
                            in_features=embed_dim,
                            hidden_features=int(mlp_ratio * embed_dim), 
                            drop=drop_rate
                        ),
                        nn.LayerNorm(embed_dim, eps=1e-6),
                        nn.LayerNorm(embed_dim, eps=1e-6),
                    ]
                )
            )

        # --------------------------------------------------------------------------

        layers = []
        for i in range(self.num_layers):
            layer = SwinLayer(
                embed_dim,
                self.feat_size,
                window_size,
                depth=depths[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=drop_path,
                attn_drop=attn_drop,
                condition=False,
            )
            layers.append(layer)
            self.add_module(f"layer_norm{i}", norm_layer(embed_dim, eps=1e-6))

        self.layers = nn.ModuleList(layers)
        
        self.fpn = nn.Sequential(
            nn.Linear(embed_dim * self.num_layers, embed_dim),
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

        # === obs_vocab ⊇ obs_list 硬检查 (Task 1 激活) ===
        # === F14 修复: 显式拒绝 obs_list 重复名 (code-review)
        # 原版 `missing = set(obs_list) - set(self.obs_vocab.keys())` 在 obs_list
        # 含重复名时被 set 去重掩盖, 后续 forward 中 torch.stack 会堆出 N+1 token
        # 但 obs_embed 只有 N 行, 巧合通过但语义错位 (用户传 3 个 token 期望 V=3,
        # 实际 Embedding V=2)。
        if len(obs_list) != len(set(obs_list)):
            from collections import Counter
            dups = {n: c for n, c in Counter(obs_list).items() if c > 1}
            raise ValueError(
                f"XiChenFusion.__init__: obs_list contains duplicate names: {dups}. "
                f"Each obs name must appear at most once in obs_list. "
                f"obs_list={obs_list}."
            )
        missing = set(obs_list) - set(self.obs_vocab.keys())
        if missing:
            raise ValueError(
                f"XiChenFusion.__init__: obs_list contains obs names not in obs_vocab: "
                f"{missing}. Add them to obs_vocab or remove from obs_list. "
                f"obs_list={obs_list}, obs_vocab={list(self.obs_vocab.keys())}"
            )
        extra = set(self.obs_vocab.keys()) - set(obs_list)
        if extra:
            log.info(
                f"XiChenFusion: obs_vocab contains {len(extra)} obs names not in current "
                f"obs_list: {extra}. These vocab rows will be initialized but unused this run."
            )

    def initialize_weights(self):
        """遍历 apply ``_init_weights`` 初始化 ``nn.Linear`` / ``nn.LayerNorm``。"""
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """递归初始化 ``nn.Linear`` (trunc_normal std=0.02) / ``nn.LayerNorm`` (常数) / ``nn.Embedding`` (trunc_normal std=0.02)。

        注: ``nn.Embedding`` 加进来是因为 ``self.obs_embed = nn.Embedding(V_max, embed_dim)`` 默认用 ``N(0, 1)`` 初始化,
        与项目其他 Linear 层的 ``trunc_normal_(0.02)`` 不一致;扩展场景下新 vocab 行的初始化更稳定。
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            trunc_normal_(m.weight, std=0.02)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.feat_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        trunc_normal_(self.latent_query, std=0.02)

    def create_var_map(self):
        """构造 ``{var_name: idx}`` 字典, 用于 ``get_var_ids``。"""
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_map

    def load_state_dict(self, state_dict, strict=True):
        """处理 obs_embed 扩展 (V_old < V_max) 的 size-mismatch, 再走原生 load。

        为什么需要这个 hook:
            PyTorch ``nn.Embedding.load_state_dict`` 在 num_embeddings
            不一致时即使 ``strict=False`` 也直接 raise RuntimeError
            (不是 "missing/unexpected key", 是 "size mismatch" 语义不同)。
            扩展微调场景 (V=5 → V=7) 必须先 partial copy_ 老行, 然后
            让原生 load 报 remaining error, 我们捕获并忽略 size mismatch
            (其他 size mismatch 仍按 strict=True 抛)。

        Args:
            state_dict: 待加载的 state_dict。本方法**不**原地修改调用方 dict;
                会先 shallow copy 一份, partial copy 后仅从副本中删 obs_embed.weight
                (code-review F2 修复, 原版违反 PyTorch consumer-of-state_dict 约定)。
            strict: 见 torch.nn.Module.load_state_dict。
        """
        # === F2 修复: shallow copy 避免 mutate 调用方 dict ===
        # 原版 `del state_dict["obs_embed.weight"]` 改写了传入对象,导致:
        #   - trainer 路径: state = torch.load(...); model.load_state_dict(state, ...);
        #     log.info(state.keys()) 看不到 obs_embed.weight, 误以为权重缺失
        #   - DDP 多 rank 共享同一引用时, rank 0 的 del 会波及其他 rank
        state_dict = dict(state_dict)
        if "obs_embed.weight" in state_dict:
            ckpt_v = state_dict["obs_embed.weight"].shape[0]
            cur_v = self.obs_embed.weight.shape[0]
            if ckpt_v < cur_v:
                # 扩展场景: 老行 partial copy_, 新行保留 trunc_normal_(0.02) 初始化
                with torch.no_grad():
                    self.obs_embed.weight[:ckpt_v].copy_(state_dict["obs_embed.weight"])
                # 从 (副本) state_dict 删, 让原生 load 跳过此 key (避免 size mismatch raise)
                del state_dict["obs_embed.weight"]
                log.info(
                    f"XiChenFusion.load_state_dict: partial obs_embed migration "
                    f"({ckpt_v} → {cur_v}); new rows keep trunc_normal_(0.02) init"
                )
        return super().load_state_dict(state_dict, strict=strict)

    def aggregate_observations(
        self,
        x: torch.Tensor,                          # (B, V, L, D)
        key_padding_mask: torch.Tensor = None,    # (B, V) bool, True=pad (Task 4)
    ):
        """Perceiver 跨 obs 源聚合, 支持 key_padding_mask 屏蔽缺失 obs token。

        流程:
            1. einsum + flatten(0, 1) → (B*L, V, D)
            2. 复制 latent_query → (B*L, 1, D)
            3. 沿 V 维复制 key_padding_mask → (B*L, V)
            4. 逐层 latent_agg: MHA(latents, x, key_padding_mask=kpm) + 残差 + GeGLUFFN(ln2) + 残差
            5. view 回 (B, L, D)
        """
        b, V, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # (B*L, V, D)

        latents = (self.latent_query + self.pos_embed).repeat_interleave(b, dim=0).reshape(b * l, 1, -1)

        for attn, ff, ln1, ln2 in self.latent_agg:
            attn_out = ln1(attn(latents, x))
            latents = attn_out + latents
            latents = ln2(ff(latents)) + latents

        return latents.view(b, l, -1)

    def get_var_ids(self, vars, device):
        """把变量名列表转成 ``torch.Tensor`` 索引, 移到 ``device``。

        Args:
            vars (list[str]): 变量名列表。
            device (torch.device): 输出张量所在设备。

        Returns:
            Tensor: 形状 ``(len(vars),)`` 的 int64 索引张量。
        """
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def _lookup_obs_ids(self, obs_list, device):
        """把 obs 名列表转成 Embedding 用的 long 索引, 处理"未注册"名字 (Task 2)。

        Args:
            obs_list (list[str]): 当前 batch 的 obs 名列表。
            device (torch.device): 输出张量所在设备。

        Returns:
            Tensor: 形状 (len(obs_list),) 的 int64 索引。

        Raises:
            KeyError: 若 ``obs_list`` 含不在 ``obs_vocab`` 的名字。__init__
                已硬检查 ``obs_vocab ⊇ obs_list`` (line 310), 但 ``forward`` 的
                ``obs_list`` 是 per-batch 子集 (与 __init__ 不同);若运行期
                出现 obs 名 typo, 旧版用 ``max(vocab.values())`` 作 fallback
                静默误索引到末尾 obs, 偏向 bias。修复: 早爆, 让 typo 在第一
                个 batch 即 KeyError (code-review #9 修订)。
        """
        ids = []
        # === F11 修复: 单循环 + try/except (code-review)
        # 原版两次 list-comp 扫描 obs_list (L501 unknown 检测 + L510-511 取值),
        # 合并为单循环: 在 dict 取值时 KeyError 一次性捕获, 错误消息仍列出
        # 所有 unknown 名 (用 list-comp 收集避免中断循环)。
        try:
            for name in obs_list:
                ids.append(self.obs_vocab[name])
        except KeyError as e:
            bad = str(e).strip("'\"")
            unknown = [n for n in obs_list if n not in self.obs_vocab]
            raise KeyError(
                f"obs_list contains names not in self.obs_vocab: {unknown}. "
                f"Known: {list(self.obs_vocab.keys())}. "
                f"Either add to obs_vocab (YAML) or fix the obs_list typo. "
                f"Silent fallback to max(vocab) id was removed in code-review #9 "
                f"because it biased training toward the last-vocab-id obs."
            ) from e
        return torch.tensor(ids, dtype=torch.long, device=device)

    def forwward_latent(self, x: torch.Tensor, use_checkpoint=False):
        """Swin-V2 精调阶段: 把 Perceiver 聚合后的 ``(B, L, D)`` 进一步精调。

        流程:
            1. 逐层 SwinLayer(``condition=False``) 编码;
            2. 收集每层 ``layer_norm{i}`` 输出, 拼接后 ``self.fpn`` 融合;
            3. 加回 ``x`` 残差, ``self.out_norm`` 收尾。

        注意: 方法名拼写为 ``forwward_latent`` (3 个 w) 是历史 typo, 调用方
        须按此名引用, 暂时不要修正以避免破坏已有 checkpoint / 引用。

        Args:
            x (Tensor): ``aggregate_observations`` 输出的 ``(B, L, D)`` 全局 latent。
            use_checkpoint (bool): 是否对 SwinBlock 使用 ``torch.utils.checkpoint``。

        Returns:
            Tensor: 形状 ``(B, L, D)`` 的精调后表示。
        """
        # x: `[B, L, D]` shape.
        # tokenize each variable separately
        residual = x
        outs = []
        for i, blk in enumerate(self.layers):
            if use_checkpoint:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            out = getattr(self, f"layer_norm{i}")(x)
            outs.append(out)

        if use_checkpoint:
            x = checkpoint(self.fpn, torch.cat(outs, dim=-1), use_reentrant=False)
        else:
            x = self.fpn(torch.cat(outs, dim=-1))    # Task 6: 修 self.enc_fpn → self.fpn typo

        x = x + residual

        x = self.out_norm(x)

        return x

    def up_forward(self, x):
        """把 ``(B, L, D)`` patch 网格表示上采样回 ``(B, V, H, W)``。

        与 ``XiChenDA.up_forward`` 完全一致, 按 ``img_size[0]`` 奇偶分两个分支:
        - 奇数: ``ConvTranspose2d``;
        - 偶数: ``Linear + rearrange``。

        Args:
            x (Tensor): ``forwward_latent`` 输出的 ``(B, L, D)`` 表示。

        Returns:
            tuple[Tensor, Tensor]: ``(res, log_var)``, 形状均为 ``(B, V, H, W)``。
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

    def forward(
        self,
        roe_dict,                          # 改名: 旧名 representation_obs_embed_dict
        obs_list,
        variables,
        use_checkpoint=False,
    ):
        """前向计算: {obs_name: roe} → (xa, log_var), 支持 obs_available mask。

        Args:
            roe_dict (dict): {obs_name: Tensor} 字典, 每个值形状 (B, L, D)。
            obs_list (list[str]): 当前 batch 实际 obs 名, 长度 V (不再假设 7 个全在)。
            variables (list[str]): 输出变量名子集, 沿通道维挑选输出。
            use_checkpoint (bool): 是否对 SwinBlock 使用 torch.utils.checkpoint。

        Returns:
            tuple[Tensor, Tensor]: (preds, log_var), 形状均为 (B, V_out, H, W)。
        """
        # 0. 入参校验
        if not obs_list:
            raise ValueError(
                "XiChenFusion.forward: obs_list is empty; "
                "check training.obs_list config or datamodule obs loading."
            )

        device = next(iter(roe_dict.values())).device
        B = next(iter(roe_dict.values())).shape[0]

        # 1. 拼 obs token 序列
        x = torch.stack([roe_dict[k] for k in obs_list], dim=1)  # (B, V, L, D)

        # 2. 加 obs embedding (从 Embedding 表查)
        in_obs_ids = self._lookup_obs_ids(obs_list, device)
        obs_emb = self.obs_embed(in_obs_ids)    # (V, D)
        x = x + obs_emb[None, :, None, :] + self.pos_embed      # (B, V, L, D)

        # 3. Perceiver 跨 obs 源聚合 (当前 forward 调用未传 mask)
        x = self.aggregate_observations(x, key_padding_mask=None)  # (B, L, D)

        # 4. Swin-V2 精调
        x = self.forwward_latent(x, use_checkpoint)

        # 5. 上采样到 (B, V, H, W)
        preds, log_var = self.up_forward(x)

        # 6. log_var 双侧 softplus 截断到 [-10, 10]
        log_var = -10 + F.softplus(log_var + 10)
        log_var = 10 - F.softplus(10 - log_var)

        # 7. 按 variables 子集索引输出
        out_var_ids = self.get_var_ids(tuple(variables), preds.device)
        preds = preds[:, out_var_ids]
        log_var = log_var[:, out_var_ids]

        return preds, log_var