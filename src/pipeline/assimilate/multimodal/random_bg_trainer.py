# -*- coding: utf-8 -*-
"""
RandomBgMultiModalAssimTrainer - 随机背景场多模态资料同化训练器
============================================================

本模块实现了 XiChen 框架中 ``随机背景场 (Random Background)`` + ``多模态 (Multimodal)`` 资料同化任务的训练器。
它编排 ``Solver`` (级联变分求解) + ``ROE`` (各观测源表征编码) + ``XiChenFusion`` (Perceiver 风格
融合 + 上采样回 ``(B, V, H, W)`` 输出分析场 ``xa``, 单网络双职责) 的实例化、预训练权重加载、
分布式包装、AMP 混合精度以及 DDP 多卡训练循环。

与 cascade 方案的关键差异:
    * cascade 是 1 obs → 1 DA → 取均值的并行级联;
    * multimodal 是每个观测源用各自的 ``XiChenRepresentationObsEmbedding`` (ROE) 提取表征,
      然后由 ``XiChenFusion`` (Perceiver cross-attention + Swin stack + 上采样) 统一
      融合并直接输出分析场 ``xa``。**只有 1 个 ``XiChenFusion`` 实例, 它同时承担
      跨 obs 融合 + DA 输出两步, 不存在第二个 DA 网络** (code-review #10 修订)。

观测源列表(默认 7 类): ``atms``、``amsua``、``mhs``、``hrs4``、``prepbufr``、``satwnd``、
``ascat``。扩展微调 / 5-obs 起步可走 ``training.obs_list`` (dict-shape) 子集,
详见 ``v7obs.yaml`` 头注释。

核心职责:
    1. 按 ``model_training_config`` 实例化 ``Solver`` / ``ROE_models`` / ``ObsOp_models`` /
       ``XiChenFusion (da_model)`` / ``forecast_model``,并按 ``trainable`` 标志冻结或解冻;
       (注: ``da_model`` 在本路径下就是 ``XiChenFusion`` 单一实例, 同时承担
       "Perceiver 跨 obs 融合 + 上采样回 (B, V, H, W) 输出 xa" 两步, 无第二个 DA 网络)
    2. 加载 ``forecast / XiChenFusion (da_model) / ROE / ObsOp`` 四类预训练权重,
       支持 ``cross_model_loading`` 跨模型共享;
    3. 在外层自回归 (AR) 滚动预报 ``lead_times`` 步后,调用 ``Solver`` 取得 per-obs 表征
       → 喂入 ``XiChenFusion`` 输出 ``xa``;
    4. 通过 ``hydra.utils.instantiate(self.config.loss_fn)`` 实例化损失函数
       (典型为 ``CRPS-Gaussian`` / ``L1`` 等, 内部利用 ``XiChenFusion`` 输出的
       ``log_var`` 算 sigma, 隐式实现 Kendall 多任务不确定度加权),
       支持 bf16/fp16 AMP 与 ``max_grad_norm`` 梯度裁剪;
    5. 每 epoch 保存 ``last.ckpt`` (断点续训) 和 ``best.ckpt`` (最佳验证损失)。

典型 YAML 配置 (节选)::

    training:
        model_training_config:
            forecast_model: {trainable: false, pretrained_path: ".../forecast.ckpt"}
            da_model:      {trainable: true,  pretrained_path: ".../da.ckpt"}
            roe_models:
                atms:     {trainable: true,  pretrained_path: ".../roe_atms.ckpt"}
                amsua:    {trainable: true,  pretrained_path: ".../roe_amsua.ckpt"}
                ...
            obsop_models:
                atms:     {trainable: false, pretrained_path: ".../obsop_atms.ckpt"}
                ...
        obs_list: {atms: {...}, amsua: {...}, mhs: {...}, hrs4: {...},
                   prepbufr: {...}, satwnd: {...}, ascat: {...}}
        lr: 1e-4
        epochs: 100
        use_amp: true
        precision: {type: bf16}
"""
import os
from abc import ABC, abstractmethod
from typing import Optional
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
import hydra
try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

from src.metrics.weighted_acc_rmse import weighted_rmse_torch
from src.models.assimilate.utils.varcost import Obs_WeighedL2Norm, Model_Var_Cost, Model_H
from src.utils.lr_scheduler import CosineSchedulerWithWarmup
from src.utils import get_logger
from src.utils.device import (
    autocast,
    get_grad_scaler,
    get_device,
)
from src.utils.model import count_parameters_detailed, format_number
from src.pipeline.base.trainer import BaseTrainer
from src.utils.device import empty_cache

class RandomBgMultiModalAssimTrainer(BaseTrainer):
    """随机背景场多模态资料同化训练器。

    负责多模态方案 (multimodal) 下 7 类观测 (atms/amsua/mhs/hrs4/prepbufr/satwnd/ascat) 的资料同化训练。
    每个观测源由一个独立的 ``XiChenRepresentationObsEmbedding`` (ROE) 网络提取表征,
    再由 ``XiChenFusion`` (Perceiver 风格:可学习 latent queries + cross-attention + Swin 堆叠) 统一融合,
    最终由 ``da_model`` 输出分析场 ``xa``。

    Attributes:
        config (DictConfig): Hydra 全局配置。
        device: 训练设备 (NPU 或 CUDA 张量)。
        local_rank (int): 当前进程本地 rank (DDP)。
        world_size (int): 分布式进程总数。
        is_main (bool): 是否为主进程 (rank 0)。
        training_config (DictConfig): ``config.training`` 子树。
        log_dir (str): TensorBoard 日志根目录。
        ckpt_dir (str): Checkpoint 输出目录。
        log: 名称为 ``xichen.trainer`` 的 DDP-rank-filtered logger。
        solver: 实例化后的 ``src.models.assimilate.fdvarsolver.multimodal.Solver``。
        roe_models (nn.ModuleDict): 键为观测源,值为 ``XiChenRepresentationObsEmbedding`` 表征编码器。
        obsop_models (nn.ModuleDict): 键为观测源,值为 ``XiChenObsOp`` 辐射率观测算子。
        da_model: ``XiChenFusion`` (Perceiver 风格) 融合 + DA 网络。
        forecast_model: 预训练自回归预报模型 (默认冻结)。
        H_models (nn.ModuleDict): 每个观测源对应的 ``Model_H(obs_err)`` (不可训练)。
        VarCost_models (nn.ModuleDict): 每个观测源对应的 ``Model_Var_Cost`` (不可训练)。
        obs_list (dict): 当前任务启用的观测源字典 (来自 ``training.obs_list``)。
        optimizer: ``torch.optim.AdamW`` 优化器。
        scheduler: 学习率调度器。
        loss_fn: Hydra 实例化的损失函数 (例如 CRPS-Gaussian)。
        scaler: AMP ``GradScaler``。
        precision_type (``torch.dtype``): 实际使用的 AMP 精度。
        writer: TensorBoard ``SummaryWriter`` (仅主进程)。
        profiler: torch NPU/GPU profiler (可选,仅主进程)。
        start_epoch (int): 断点续训起始 epoch。
        best_loss (float): 历史最佳验证损失。
    """

    def __init__(
        self,
        cfg: DictConfig,
        device,
        local_rank: int,
        world_size: int,
        is_main: bool,
        **kwargs,
    ):
        """初始化 multimodal DA trainer — 全部由基类 __init__ 模板驱动."""
        super().__init__(cfg, device, local_rank, world_size, is_main, **kwargs)

    def _build_models(self) -> None:
        """一次性装配 7 子模型 (solver / ROE / obsop / da_model / forecast / H / VarCost) + 预训练.

        调用顺序严格: 6 个 _init_* → _setup_model_training_states → _load_pretrained_models → _log_model_statistics.
        """
        # 获取模型训练配置
        model_training_config = self.training_config.get("model_training_config", {})

        # 实例化所有模型
        self._init_solver()
        self._init_roe_models(model_training_config)
        self._init_obsop_models(model_training_config)
        self._init_da_model(model_training_config)
        self._init_forecast_model(model_training_config)
        self._init_observation_models()

        # 关键决策点:先设置 requires_grad 标志再 load_state_dict,
        # 以避免预训练权重在加载时被 strict=False 静默丢弃与冻结策略不一致的参数。
        self._setup_model_training_states(model_training_config)

        # 加载预训练权重(必须在 requires_grad 标志确定之后)
        self._load_pretrained_models(model_training_config)

        # 统计参数
        self._log_model_statistics()

    def _init_solver(self):
        """实例化 ``Solver`` (多模态变分求解器) 并迁移到目标设备。

        与 cascade 的差异: cascade 中此 ``Solver`` 输出 per-obs 分析场 ``xa`` 然后取均值;
        多模态下改为输出 per-obs 表征张量, 由 ``XiChenFusion`` 融合。
        """
        # 实例化模型
        self.solver = hydra.utils.instantiate(self.config.model.Solver, _recursive_=False).to(self.device)

    def _init_roe_models(self, model_training_config):
        """实例化每个观测源对应的 ``XiChenRepresentationObsEmbedding`` 表征编码器。

        ROE 网络负责将 ``Solver`` 输出的 per-obs ``(xb, grad)`` 编码为定长表征向量,
        这些表征将由 ``XiChenFusion`` (Perceiver) 统一融合。

        Args:
            model_training_config (dict): ``training.model_training_config.roe_models`` 子树。
        """
        da_config = model_training_config.get("roe_models", {})
        self.roe_models = nn.ModuleDict()

        # === Task 8: 记录扩展模式下的"新 obs 列表" ===
        # === D13 修复: 深搜嵌套位置, 防止 "is_extension 误写到 model_training_config.*" 静默 bypass ===
        # 旧版只读顶层 ``self.training_config.get("is_extension", False)``; 若
        # 用户 typo 写成 ``model_training_config.roe_models.atms: {is_extension: true, ...}``,
        # 顶层读到 False → 校验静默通过, 7 个 ROE 全部按"非扩展"路径训练,
        # 旧 obs 的 trainable=false 仍生效, 但 hrs4/ascat 这两个真要走扩展分支
        # 的新 obs 会按非扩展路径处理, 与 extension_5_to_7.yaml 意图相反。
        # 修复: 1) 深搜 model_training_config 子树, 找到就 warn+override;
        #       2) 顶层缺失时若 deep 找到, 自动用 deep 值 (更宽松, 易用);
        #       3) 顶层 True 但 new_obs_names 空才硬错。
        is_extension = self.training_config.get("is_extension", False)
        # === D13.1 修复: 深搜找到后必须同步到 self.training_config, 否则下游
        # _setup_model_training_states (line 464) / _setup_optimizer_scheduler (line 715)
        # 仍会读到顶层 False, 扩展语义静默失效。
        if not is_extension:
            # 深搜 model_training_config 中的 is_extension (depth=2 即可覆盖所有常见嵌套位置)
            mtc = self.training_config.get("model_training_config", {})
            for parent_key in ("roe_models", "obsop_models", "da_model", "forecast_model"):
                sub = mtc.get(parent_key, {})
                if isinstance(sub, dict) and sub.get("is_extension", False):
                    log_nested = sub.get("is_extension")
                    if self.is_main:
                        self.log.warning(
                            f"is_extension found nested under model_training_config.{parent_key}; "
                            f"this is a known typo-prone location. Promoting to top-level "
                            f"is_extension={log_nested} for this run. "
                            f"Move it to top-level training.is_extension to silence this warning."
                        )
                    is_extension = log_nested
                    self.training_config.is_extension = log_nested
                    break
                # 再深一层: 组件子项的 dict 里 (e.g. roe_models.atms: {is_extension: true})
                if isinstance(sub, dict):
                    for child_name, child_cfg in sub.items():
                        if isinstance(child_cfg, dict) and child_cfg.get("is_extension", False):
                            if self.is_main:
                                self.log.warning(
                                    f"is_extension found nested under model_training_config."
                                    f"{parent_key}.{child_name}; this is a typo-prone location. "
                                    f"Promoting to top-level. Move it to training.is_extension."
                                )
                            is_extension = child_cfg.get("is_extension")
                            self.training_config.is_extension = child_cfg.get("is_extension")
                            break
                    if is_extension:
                        break
        self.new_obs_names = self.training_config.get("new_obs_names", [])
        if is_extension and not self.new_obs_names:
            raise ValueError(
                "is_extension=True requires training.new_obs_names to be non-empty. "
                "List the obs names added in this extension."
            )

        for name, model_config in self.config.model.roe_models.items():
            # 实例化模型
            model = hydra.utils.instantiate(model_config, _recursive_=False).to(self.device)
            self.roe_models[name] = model

            # 记录模型配置
            if name not in da_config:
                da_config[name] = {"trainable": False, "pretrained_path": None}

    def _init_obsop_models(self, model_training_config):
        """实例化每个观测源对应的 ``XiChenObsOp`` 辐射率观测算子。

        Args:
            model_training_config (dict): ``training.model_training_config.obsop_models`` 子树。
        """
        obsop_config = model_training_config.get("obsop_models", {})
        self.obsop_models = nn.ModuleDict()

        for name, model_config in self.config.model.obsop_models.items():
            # 实例化模型
            model = hydra.utils.instantiate(model_config, _recursive_=False).to(self.device)
            self.obsop_models[name] = model

            # 记录模型配置
            if name not in obsop_config:
                obsop_config[name] = {"trainable": False, "pretrained_path": None}

    def _init_da_model(self, model_training_config):
        """实例化 ``da_model`` (Perceiver 风格 ``XiChenFusion``) — 多模态融合与最终分析网络。

        关键决策点 — 与 cascade 的差异:
            * cascade 是 1 obs → 1 DA → 取均值;
            * multimodal 是 7 obs → 7 ROE → ``XiChenFusion`` 融合 → 单个 ``da_model`` 输出 ``xa``。

        Args:
            model_training_config (dict): ``training.model_training_config.da_model`` 子树。
        """
        da_config = model_training_config.get("da_model", {})
        self.da_model = hydra.utils.instantiate(
            self.config.model.da_model, _recursive_=False
        ).to(self.device)
        # 记录模型配置
        if not da_config:
            da_config.update({"trainable": False, "pretrained_path": None})

    def _init_forecast_model(self, model_training_config):
        """实例化自回归背景场预报模型 (``XiChenForecast``)。

        默认冻结 — 在 DA 训练中预报模型作为静态先验,只在 ``Solver`` 的 AR 循环里被调用;
        若 ``model_training_config.forecast_model.trainable=True`` 则会随主训练一起微调。

        Args:
            model_training_config (dict): ``training.model_training_config.forecast_model`` 子树。
        """
        forecast_config = model_training_config.get("forecast_model", {})
        self.forecast_model = hydra.utils.instantiate(
            self.config.model.forecast_model, _recursive_=False
        ).to(self.device)
        # 记录模型配置
        if not forecast_config:
            forecast_config.update({"trainable": False, "pretrained_path": None})

    def _init_observation_models(self):
        """实例化 H 观测算子和 VarCost 变分代价(全部冻结,仅在 ``Solver`` 内部被调用)。

        与 cascade 方案的差别仅在 ``Model_H(obs_err).to(self.device)`` — 多模态下观测算子需
        显式迁移到目标设备,因为 ``Solver`` 内部会按 batch 调用这些小算子。
        """
        H_models = {}
        VarCost_models = {}
        # === 关键: obs_list 在本 codebase 是 dict-shape (e.g. ``atms: {trainable: true}``),
        #  下方循环用 .keys() / [name].trainable 都假设 dict。CLI override 或
        #  model.da_model.obs_list (list-shape) 不能直接喂到这里, 显式校验失败
        #  模式以提供清晰诊断, 避免下游 ``AttributeError: 'list' object has no
        #  attribute 'keys'`` 难定位。
        self.obs_list = self.training_config.get("obs_list", {})
        obs_dir = self.config.paths.get("obs_dir", "data/obs")

        for obs_name in self.obs_list.keys():
            # 加载观测误差数据
            obs_err = self._load_observation_error(obs_name, obs_dir)

            # 创建观测模型
            H_models[obs_name] = Model_H(obs_err).to(self.device)
            m_NormObs = Obs_WeighedL2Norm(obs_err)
            VarCost_models[obs_name] = Model_Var_Cost(m_NormObs).to(self.device)

        # 保存观测模型
        self.H_models = nn.ModuleDict({name: model for name, model in H_models.items()})
        self.VarCost_models = nn.ModuleDict({name: model for name, model in VarCost_models.items()})

    def _load_observation_error(self, obs_name, obs_dir):
        """根据观测源类型加载对应的观测误差 std npz。

        与 cascade 版的区别:多模态下 ``obs_err_list`` 直接保留为 numpy 数组,
        在 ``Model_H(obs_err).to(self.device)`` 阶段统一迁移到设备,
        避免在 CPU 端预转 torch.Tensor 带来的额外拷贝。

        Args:
            obs_name (str): 观测源名 (atms/amsua/mhs/hrs4/prepbufr/satwnd/ascat)。
            obs_dir (str): ``config.paths.obs_dir`` 路径。

        Returns:
            np.ndarray: 形状 ``(C_obs,)`` 的 obs σ 数组。
        """
        if obs_name in ["atms", "amsua", "mhs", "hrs4"]:
            obs_err = np.load(os.path.join(obs_dir, f"1b{obs_name}_merged_npy_1.0deg", "avg_obs_error.npz"))
        elif obs_name == "prepbufr":
            obs_err = np.load(os.path.join(obs_dir, "GDAS_prepbufr_merged_npy_1.0deg/obs_sigma.npz"))
        elif obs_name == "satwnd":
            obs_err = np.load(os.path.join(obs_dir, "satwnd_merged_npy_1.0deg/obs_sigma.npz"))
        elif obs_name == "ascat":
            obs_err = np.load(os.path.join(obs_dir, "ascat_b_merged_npy_1.0deg/obs_sigma.npz"))

        obs_err_list = []
        # === D16 修复: 按 key 字典序排序, 避免 NpzFile.keys() 按文件内部存储顺序
        # 迭代时与物理通道顺序 (ch01..ch22) 错位, 导致 Model_H[ch_idx] 与
        # QC 阈值 5·σ_obs 错配 (code-review #6)。
        # 注: 若 key 是 ch01..ch22 形式, sorted 字典序即数字序 (ch02 < ch10)。
        # 若 key 形如 '0','1',... 则 sorted 与文件原序可能不同, 但物理序优先。 ===
        for i, key in enumerate(sorted(obs_err.keys())):
            obs_err_list.append(obs_err[key])

        return np.stack(obs_err_list, axis=0)

    def _setup_model_training_states(self, model_training_config):
        """按 ``model_training_config`` 中各组件的 ``trainable`` 标志统一设置 ``requires_grad``。

        关键决策点 — 此方法 **必须** 在 ``_load_pretrained_models`` 之前调用。
        因为 ``load_state_dict(strict=False)`` 不会改变参数的 ``requires_grad`` 标志;
        若先加载预训练权重,某些被冻结的子模块可能会保留其原始梯度状态,与配置不一致。

        多模态下需冻结/解冻的组件: ``forecast_model`` / ``da_model`` / ``roe_models[*]`` / ``obsop_models[*]``。

        Args:
            model_training_config (dict): ``training.model_training_config`` 子树。
        """
        forecast_config = model_training_config.get("forecast_model", {})
        da_config = model_training_config.get("da_model", {})
        roe_config = model_training_config.get("roe_models", {})
        obsop_config = model_training_config.get("obsop_models", {})

        # 设置预报模型训练状态
        forecast_trainable = forecast_config.get("trainable", False)
        for param in self.forecast_model.parameters():
            param.requires_grad = forecast_trainable

        # 设置DA模型训练状态
        da_trainable = da_config.get("trainable", False)
        for param in self.da_model.parameters():
            param.requires_grad = da_trainable

        # 设置ROE模型训练状态 (Task 9: is_extension 分支)
        if self.training_config.get("is_extension", False):
            # 扩展微调模式: 旧 obs 冻结, 新 obs 按配置
            for name, model in self.roe_models.items():
                if name in self.new_obs_names:
                    is_trainable = roe_config.get(name, {}).get("trainable", True)
                    for param in model.parameters():
                        param.requires_grad = is_trainable
                    self.log.info(
                        f"  ROE[is_extension] '{name}' (NEW) trainable={is_trainable}"
                    )
                else:
                    # 旧 obs 强制冻结
                    for param in model.parameters():
                        param.requires_grad = False
                    self.log.info(
                        f"  ROE[is_extension] '{name}' (OLD, FROZEN) trainable=False"
                    )
        else:
            # 从头训练: 按配置
            for name, model in self.roe_models.items():
                is_trainable = roe_config.get(name, {}).get("trainable", False)
                for param in model.parameters():
                    param.requires_grad = is_trainable

        # 设置ObsOp模型训练状态
        for name, model in self.obsop_models.items():
            is_trainable = obsop_config.get(name, {}).get("trainable", False)
            for param in model.parameters():
                param.requires_grad = is_trainable

    def _load_pretrained_models(self, model_training_config):
        """按顺序加载 ``forecast → DA → ROE → ObsOp`` 四类预训练权重(多模态扩展)。

        与 cascade 版的差异:多模态多了 ``ROE_models`` 一类(每个观测源一个表征编码器),
        且 ``da_model`` 退化为单一实例(而非 ``DA_models`` 字典)。

        支持 ``cross_model_loading`` 跨模型参数共享:某观测源 ROE 可用另一观测源 ROE 的
        检查点进行初始化(常用于无该卫星训练数据时的迁移)。

        加载顺序约定: 先 ``forecast_model`` → ``da_model`` → ``ROE_models`` → ``ObsOp_models``,
        便于在日志中按依赖链追踪权重来源。

        Args:
            model_training_config (dict): ``training.model_training_config`` 子树。
        """
        roe_config = model_training_config.get("roe_models", {})
        obsop_config = model_training_config.get("obsop_models", {})
        da_config = model_training_config.get("da_model", {})
        forecast_config = model_training_config.get("forecast_model", {})

        # 加载预报模型预训练权重
        if "pretrained_path" in forecast_config and forecast_config["pretrained_path"] is not None:
            self._load_model_checkpoint(
                self.forecast_model,
                forecast_config["pretrained_path"],
                forecast_config.get("trainable", False),
                strict=forecast_config.get("strict", False)
            )

        # 加载DA模型预训练权重
        if "pretrained_path" in da_config and da_config["pretrained_path"] is not None:
            self._load_model_checkpoint(
                self.da_model,
                da_config["pretrained_path"],
                da_config.get("trainable", False),
                strict=da_config.get("strict", False)
            )

        # 加载ROE模型预训练权重
        for name, config in roe_config.items():
            if "pretrained_path" in config and config["pretrained_path"] is not None:
                # 检查是否有跨模型加载配置
                cross_model_config = config.get("cross_model_loading", {})
                if cross_model_config and cross_model_config.get("enabled", False):
                    # 跨模型加载：使用其他模型的检查点
                    source_model = cross_model_config.get("source_model")
                    self._load_model_checkpoint(
                        self.roe_models[name],
                        config[source_model]["pretrained_path"],  # 使用源模型的路径
                        config.get("trainable", False),
                        strict=config.get("strict", False),
                        model_name=source_model  # 指定源模型名称
                    )
                else:
                    # 常规加载：使用自己的检查点
                    self._load_model_checkpoint(
                        self.roe_models[name],
                        config["pretrained_path"],
                        config.get("trainable", False),
                        strict=config.get("strict", False),
                        model_name=name  # 指定模型名称
                    )

        # 加载ObsOp模型预训练权重
        for name, config in obsop_config.items():
            if "pretrained_path" in config and config["pretrained_path"] is not None:
                self._load_model_checkpoint(
                    self.obsop_models[name],
                    config["pretrained_path"],
                    config.get("trainable", False),
                    strict=config.get("strict", False),
                    model_name=name
                )

    def _load_model_checkpoint(self, model, checkpoint_path, is_trainable, strict=True, model_name=None):
        """加载单个模型检查点(支持多种 ckpt 格式 + 跨模型加载),加载后强制刷新 ``requires_grad``。

        支持的 ckpt 格式(按优先级检查):
            1. ``{"model_state_dict": ...}`` — 标准训练器保存格式;
            2. ``{"state_dict": ...}`` — Lightning / DDP 通用格式;
            3. ``{"da_model_state_dict": {"atms": ..., "amsua": ...}}`` — 多观测源 DA 打包格式;
               此时按 ``model_name`` 选择对应子字典,若未指定则取第一个;
            4. 直接 ``torch.save(model.state_dict())`` — 裸 state_dict。

        加载完成后无论 ``strict`` 设置如何,均按 ``is_trainable`` 强制刷新所有参数的
        ``requires_grad`` 标志,确保与 ``model_training_config`` 一致。

        Args:
            model (``nn.Module``): 目标模型。
            checkpoint_path (str): ``.ckpt`` 路径;若不存在则仅 ``log.warning`` 并跳过。
            is_trainable (bool): 加载后是否允许梯度。
            strict (bool): ``load_state_dict`` 的 ``strict`` 参数;默认 ``True``。
            model_name (str, optional): 跨模型加载时,``da_model_state_dict`` 中要取哪一份参数。
        """
        if not os.path.exists(checkpoint_path):
            if self.is_main:
                self.log.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
            return

        if self.is_main:
            self.log.info(f"Loading pretrained model from: {checkpoint_path}")

        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = None

            # 支持多种检查点格式
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "da_model_state_dict" in ckpt:
                # 如果提供了模型名称，尝试加载特定模型的参数
                if model_name and model_name in ckpt["da_model_state_dict"]:
                    state_dict = ckpt["da_model_state_dict"][model_name]
                # 如果没有指定模型名称，尝试加载第一个可用的模型参数
                elif ckpt["da_model_state_dict"]:
                    first_model_name = next(iter(ckpt["da_model_state_dict"].keys()))
                    state_dict = ckpt["da_model_state_dict"][first_model_name]
                    if self.is_main:
                        self.log.info(f"Using parameters from model: {first_model_name}")
                else:
                    raise ValueError("No DA model state dict found in checkpoint")
            else:
                state_dict = ckpt

            # 如果state_dict仍然为空，抛出异常
            if state_dict is None:
                raise ValueError("Could not extract state dictionary from checkpoint")

            # 加载权重
            model.load_state_dict(state_dict, strict=strict)

            # 设置梯度状态
            for param in model.parameters():
                param.requires_grad = is_trainable

            if self.is_main:
                self.log.info(f"Successfully loaded pretrained model from {checkpoint_path}")

        except Exception as e:
            if self.is_main:
                self.log.error(f"Failed to load pretrained model from {checkpoint_path}: {e}")

    def _log_model_statistics(self):
        """打印每个组件的 (总参数, 可训练参数, 比例) 统计信息 (仅主进程)。

        用途:帮助确认 ``model_training_config`` 冻结策略是否按预期生效,以及全局可训练参数量级。
        """
        if self.is_main:
            # 统计所有模型参数
            all_models = {
                "roe_models": self.roe_models,
                "obsop_models": self.obsop_models,
                "da_model": self.da_model,
                "forecast_model": self.forecast_model
            }

            total_params = 0
            trainable_params = 0

            for model_group_name, model_group in all_models.items():
                if isinstance(model_group, nn.ModuleDict):
                    for name, model in model_group.items():
                        model_total = sum(p.numel() for p in model.parameters())
                        model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        total_params += model_total
                        trainable_params += model_trainable

                        self.log.info(
                            f"{model_group_name}.{name}: "
                            f"Total={format_number(model_total)}, "
                            f"Trainable={format_number(model_trainable)}, "
                            f"Ratio={model_trainable/model_total*100:.2f}%"
                        )
                else:
                    model_total = sum(p.numel() for p in model_group.parameters())
                    model_trainable = sum(p.numel() for p in model_group.parameters() if p.requires_grad)
                    total_params += model_total
                    trainable_params += model_trainable

                    self.log.info(
                        f"{model_group_name}: "
                        f"Total={format_number(model_total)}, "
                        f"Trainable={format_number(model_trainable)}, "
                        f"Ratio={model_trainable/model_total*100:.2f}%"
                    )

            self.log.info(
                f"Overall - Total: {format_number(total_params)}, "
                f"Trainable: {format_number(trainable_params)}, "
                f"Ratio: {trainable_params/total_params*100:.2f}%"
            )

    def _setup_optimizer_scheduler(self):
        """装配 AdamW 优化器 + scheduler (Task 10: idempotent + 3 param_groups)。

        param_group 分组:
            - group_1 (lr=lr_roe): 新 obs 的 ROE (扩展模式), 或全部 ROE (从头训练)
            - group_2 (lr=lr_fusion): arch_fusion (da_model) 全参
            - group_3 (lr=lr_roe_old): 旧 obs 的 ROE (扩展模式且 trainable=True, 默认 0 冻结)
        forecast / ObsOp 始终冻结, 不进 optimizer。

        Idempotent: 多次调用安全 (用于 _load_checkpoint 的 lossy resume 重建, 见 Task 11)。
        """
        # === Idempotent: 释放旧 optimizer / scheduler ===
        if hasattr(self, "optimizer") and self.optimizer is not None:
            del self.optimizer
        if hasattr(self, "scheduler") and self.scheduler is not None:
            del self.scheduler

        if self.is_main:
            self.log.info("Initializing optimizer with differential LR...")

        lr_roe = self.training_config.get("lr_roe", 5e-4)
        lr_fusion = self.training_config.get("lr_fusion", 1e-4)
        lr_roe_old = self.training_config.get("lr_roe_old", lr_fusion)
        betas = self.training_config.get("betas", [0.9, 0.95])
        weight_decay = self.training_config.get("weight_decay", 5e-5)

        param_groups = []
        is_extension = self.training_config.get("is_extension", False)
        roe_cfg = self.training_config.get("model_training_config", {}).get("roe_models", {})

        # === group_1: trainable ROE (扩展模式: 仅 new obs; 从头训练: 全部) ===
        for name, model in self.roe_models.items():
            if is_extension and name not in self.new_obs_names:
                continue    # 扩展模式下旧 obs ROE 已被 setup 阶段置 requires_grad=False
            if any(p.requires_grad for p in model.parameters()):
                param_groups.append({
                    "params": [p for p in model.parameters() if p.requires_grad],
                    "lr": lr_roe,
                    "name": f"roe/{name}",
                })
                if self.is_main:
                    self.log.info(f"  group_1 (lr_roe={lr_roe}): roe/{name}")

        # === group_2: arch_fusion (da_model) ===
        if any(p.requires_grad for p in self.da_model.parameters()):
            param_groups.append({
                "params": [p for p in self.da_model.parameters() if p.requires_grad],
                "lr": lr_fusion,
                "name": "da_model",
            })
            if self.is_main:
                self.log.info(f"  group_2 (lr_fusion={lr_fusion}): da_model")

        # === group_2b: solver (D7 修复: 旧版显式收集 solver.parameters(),
        # 新版误以为 "solver 永远冻结" 而完全跳过。若 solver 任何子模块
        # requires_grad=True (例如 var_cost 内部 buffer、forecast 共享 sub-net),
        # 之前会静默丢参, loss 平台期无报错。现统一收进 lr_fusion 组, 与 da_model
        # 同学习率; 用户可通过 model_training_config.solver.trainable=False 显式冻结。)
        if hasattr(self, "solver") and any(p.requires_grad for p in self.solver.parameters()):
            param_groups.append({
                "params": [p for p in self.solver.parameters() if p.requires_grad],
                "lr": lr_fusion,
                "name": "solver",
            })
            if self.is_main:
                self.log.info(f"  group_2b (lr_fusion={lr_fusion}): solver")

        # === group_3: 旧 obs ROE (扩展模式且 trainable=True 时进; 默认 0 冻结) ===
        if is_extension:
            for name, model in self.roe_models.items():
                if name in self.new_obs_names:
                    continue
                if any(p.requires_grad for p in model.parameters()):
                    param_groups.append({
                        "params": [p for p in model.parameters() if p.requires_grad],
                        "lr": lr_roe_old,
                        "name": f"roe/{name} (old)",
                    })
                    if self.is_main:
                        self.log.info(f"  group_3 (lr_roe_old={lr_roe_old}): roe/{name} (old)")

        # === 透明度: 列出"被排除的可训练组件"以便诊断 (D7 透明化) ===
        # forecast_model / obsop_models / H_models / VarCost_models 按项目约定
        # 冻结, 但若 config 误标 trainable=True, 显式 log 出来避免静默丢参。
        excluded_components = []
        for comp_name in ("forecast_model", "obsop_models", "H_models", "VarCost_models"):
            comp = getattr(self, comp_name, None)
            if comp is None:
                continue
            if isinstance(comp, nn.ModuleDict):
                for sub_name, sub in comp.items():
                    if any(p.requires_grad for p in sub.parameters()):
                        excluded_components.append(f"{comp_name}.{sub_name}")
            else:
                if any(p.requires_grad for p in comp.parameters()):
                    excluded_components.append(comp_name)
        if excluded_components and self.is_main:
            self.log.info(
                f"  excluded (frozen by convention): {excluded_components}. "
                f"Override by adding to a param_group explicitly if needed."
            )

        if not param_groups:
            raise ValueError(
                "No trainable parameters found. Check trainable_models configuration. "
                f"is_extension={is_extension}, roe_cfg={list(roe_cfg.keys())}"
            )

        self.optimizer = torch.optim.AdamW(
            param_groups, betas=betas, weight_decay=weight_decay
        )

        # === scheduler (与原版一致) ===
        if self.is_main:
            self.log.info("Initializing scheduler...")
        # 调度器
        self.scheduler = self._build_scheduler(self.optimizer)

    def _load_checkpoint(self):
        """断点续训: 支持 param_groups schema 变化 (Task 11: lossy resume)。

        Lossy resume 语义:
            - ckpt 的 optimizer_state_dict 与当前 param_groups 数量不一致时
              (例如 V=7 (手写 SDPA 时代) → V=8 第一次扩展), 丢弃 opt/sched/scaler 状态
            - start_epoch / best_loss 仍保留
            - 学习率 schedule 从 warmup 起点重新走 (LR 跳变)
            - AdamW 动量清零
        """
        ckpt_path = os.path.join(self.ckpt_dir, "last.ckpt")
        if not os.path.exists(ckpt_path):
            if self.is_main:
                self.log.info("No checkpoint found, training from scratch.")
            return

        if self.is_main:
            self.log.info(f"Resuming training from: {ckpt_path}")

        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)

            # === 1. 加载 model 状态 ===
            if "solver_state_dict" in ckpt:
                self.solver.load_state_dict(ckpt["solver_state_dict"])
            if "roe_model_state_dict" in ckpt:
                self.roe_models.load_state_dict(ckpt["roe_model_state_dict"])
            if "obsop_state_dict" in ckpt:
                self.obsop_models.load_state_dict(ckpt["obsop_state_dict"])
            if "da_model_state_dict" in ckpt:
                # obs_embed 统一为 nn.Embedding (V_max, D) 形态, load_state_dict
                # 原生处理:
                #   - V_old == V_max: 完美匹配, 直接 copy
                #   - V_old < V_max: strict=False 下 PyTorch 自动 copy_(legacy[:V_old])
                #     到 self.obs_embed.weight[:V_old], 新行保持 trunc_normal_(0.02)
                #   - V_old > V_max: 罕见, strict=False 报 size mismatch 警告
                #   - 旧 handcrafted SDPA 时代 3D ``obs_embed`` key 需先转 2D
                #     再保存 (一次性迁移工具, 已通过 v7obs 训练覆盖)
                # mha.in_proj_weight / out_proj.weight 缺失时静默丢弃
                # (首次迁移 attention 重置, 见 spec §10 R1)
                self.da_model.load_state_dict(ckpt["da_model_state_dict"], strict=False)
            if "forecast_state_dict" in ckpt:
                self.forecast_model.load_state_dict(ckpt["forecast_state_dict"])
            if "h_models_state_dict" in ckpt:
                self.H_models.load_state_dict(ckpt["h_models_state_dict"])
            if "varcost_models_state_dict" in ckpt:
                self.VarCost_models.load_state_dict(ckpt["varcost_models_state_dict"])

            # === 1.5 恢复扩展微调元信息 (D13.2 修复: 修复 save/load 不对称) ===
            # save 侧 line 1470-1471 写入 is_extension / new_obs_names, 但
            # 旧 _load_checkpoint 全程未读, 导致用户在 save 后改 YAML 再 resume 时
            # 扩展语义静默丢失。修复: 从 ckpt 恢复并写回 self.training_config,
            # 让 _setup_model_training_states / _setup_optimizer_scheduler 读到正确值。
            if "is_extension" in ckpt:
                self.training_config.is_extension = ckpt["is_extension"]
            if "new_obs_names" in ckpt:
                self.training_config.new_obs_names = ckpt["new_obs_names"]

            # === 2. 重新设置模型梯度 (确保与 config 一致) — 必须在
            #     optimizer 重建前执行,否则 opt rebuild 读到的 requires_grad
            #     是老 ckpt 的状态(由 load_state_dict 保留),不是当前 config 的 ===
            self._setup_model_training_states(self.training_config.get("model_training_config", {}))

            # === 3. schema 检测: optimizer 能否加载? ===
            old_opt = ckpt.get("optimizer_state_dict", None)
            can_load_optimizer = self._check_optimizer_schema_compat(old_opt)
            if not can_load_optimizer:
                if self.is_main:
                    self.log.warning(
                        "optimizer_state_dict schema incompatible with current param_groups "
                        "(old: 1 group, new: 3 groups). Discarding optimizer/scheduler/scaler "
                        "state. Training will resume with rebuilt optimizer; warmup will re-run."
                    )
                # 重新构造 optimizer (idempotent, Task 10)
                self._setup_optimizer_scheduler()
                # === D6 修复: 重建 GradScaler ===
                # 旧 run 的 scaler._growth_tracker 残留在 self.scaler 上, 续训
                # 首轮 batch 会因 scale factor 失配导致跳过更新或梯度爆炸。
                # 重新创建 GradScaler 让 _growth_tracker 归零, 与"lr 走 warmup 起点"
                # 的 lossy 语义对齐。
                if self.training_config.get("use_amp", False):
                    self.scaler = get_grad_scaler(self.device_type)
                    if self.is_main:
                        self.log.info(
                            "  reset GradScaler (lossy resume); _growth_tracker=0"
                        )
            else:
                # schema 兼容, 正常加载
                if old_opt is not None:
                    self.optimizer.load_state_dict(old_opt)
                if self.scheduler and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
                    self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                if self.scaler and "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
                    self.scaler.load_state_dict(ckpt["scaler_state_dict"])

            # === 4. 恢复标量状态 ===
            self.start_epoch = ckpt.get("epoch", -1) + 1  # 兼容无 epoch 字段的旧 ckpt
            self.best_loss = ckpt.get("best_loss", float("inf"))  # 兼容无 best_loss 字段的旧 ckpt

            if self.is_main:
                self.log.info(
                    f"Resumed successfully. Start Epoch: {self.start_epoch}, "
                    f"Best Loss: {self.best_loss:.4f}, "
                    f"optimizer_state={'loaded' if can_load_optimizer else 'discarded (rebuilt)'}"
                )

        except Exception as e:
            if self.is_main:
                self.log.error(f"Failed to load checkpoint: {e}")
            raise

    def _check_optimizer_schema_compat(self, opt_state_dict):
        """检测 ckpt 里 optimizer_state_dict 的 param_groups 是否与当前一致 (Task 11)。

        比较策略 (D18 修复):
            1. 数量一致 — 必须项
            2. name 集合一致 — 防止同 count 但顺序/命名错配 (e.g. 旧 fork 用
               ``["roe/atms", "roe/mhs", "da_model"]`` 与新设计 ``["roe/atms",
               "da_model", "solver"]`` 都是 3 groups, 但 AdamW state 顺序错位
               会让 ROE 动量漏到 da_model)

        Returns:
            bool: True 表示可加载, False 表示需要丢弃并重建。
        """
        if opt_state_dict is None:
            return False
        old_groups = opt_state_dict.get("param_groups", [])
        new_groups = self.optimizer.param_groups
        if len(old_groups) != len(new_groups):
            return False
        # === name 集合比对 (D18 增强) ===
        # 注意: 旧 ckpt 可能没存 'name' 字段 (跨版本兼容), 此时只比数量 (lossy-friendly)
        old_names = [g.get("name") for g in old_groups if g.get("name") is not None]
        new_names = [g.get("name") for g in new_groups if g.get("name") is not None]
        if old_names and new_names:
            # 两边都有 name 字段, 必须按排序后集合一致 (顺序不强制, 但元素集合必须一致)
            if sorted(old_names) != sorted(new_names):
                return False
        return True

    def _wrap_ddp(self):
        """为所有可训练组件执行 DDP 包装(world_size > 1 时)。

        关键决策点:
            * 仅对 ``requires_grad=True`` 的组件包装 — 冻结的子模块不必同步梯度,可显著降低通信开销;
            * H_models / VarCost_models 按项目约定是冻结算子, 默认不会进入 DDP 通信组;
              下方仍按 requires_grad 判定 (与上方总原则一致, 防御误标 trainable=true
              触发 DDP wrap), 与早期 docstring 的"跳过包装"措辞有偏差, 此处统一
              为"requires_grad-driven"语义 (code-review #2 修订);
            * ``find_unused_parameters=True`` — 适配 DDP 中部分参数在某 step 缺席前向图的情况
              (例如冻结的 ROE/ObsOp 模型在某些 batch 不会出现在计算图中)。
        """
        if self.world_size > 1:
            # 只包装需要训练的模型
            models_to_wrap = []

            # 检查solver模型是否需要包装
            if hasattr(self, 'solver') and any(param.requires_grad for param in self.solver.parameters()):
                models_to_wrap.append(("solver", self.solver))

            # 检查预报模型是否需要包装
            if hasattr(self, 'forecast_model') and any(param.requires_grad for param in self.forecast_model.parameters()):
                models_to_wrap.append(("forecast_model", self.forecast_model))

            # 检查DA模型是否需要包装
            if hasattr(self, 'da_model') and any(param.requires_grad for param in self.da_model.parameters()):
                models_to_wrap.append(("da_model", self.da_model))

            # 检查ROE模型是否需要包装
            if hasattr(self, 'roe_models'):
                for name, model in self.roe_models.items():
                    if any(param.requires_grad for param in model.parameters()):
                        models_to_wrap.append((name, model))

            # 检查ObsOp模型是否需要包装
            if hasattr(self, 'obsop_models'):
                for name, model in self.obsop_models.items():
                    if any(param.requires_grad for param in model.parameters()):
                        models_to_wrap.append((name, model))

            # 检查H_models模型是否需要包装
            if hasattr(self, 'H_models'):
                for name, model in self.H_models.items():
                    if any(param.requires_grad for param in model.parameters()):
                        models_to_wrap.append((name, model))

            # 检查VarCost_models模型是否需要包装
            if hasattr(self, 'VarCost_models'):
                for name, model in self.VarCost_models.items():
                    if any(param.requires_grad for param in model.parameters()):
                        models_to_wrap.append((name, model))

            # 包装所有需要训练的模型
            for name, model in models_to_wrap:
                if not isinstance(model, DDP):
                    setattr(self, name, self._wrap_single_ddp(model))

    # 建议抽取的公共方法
    def _prepare_batch(self, batch):
        """将 ``DataLoader`` 输出的 batch 拆分并迁移到目标设备(供 train_epoch / validate 共用)。

        关键决策点:
            * 拆分 14 元 batch: ``(inps, obs_list, obs_data, obs_mask, tgt, lead_times, variables,
              obs_dict, init_time, tgt_time, era5_transforms, microwave_transforms,
              conventional_transforms)``;
            * 微波 (atms/amsua/mhs/hrs4) 用 ``microwave_transforms``,常规 (prepbufr/satwnd/ascat)
              用 ``conventional_transforms`` — 两者只取 ``std`` 作为变量归一化常数;
            * 返回 9 元组以替代 14 元 batch,调用方按需取用。

        Args:
            batch: 来自 ``DataLoader`` 的 14 元 batch。

        Returns:
            Tuple: ``(inps, obs_list, obs_data, obs_mask, tgt, lead_times, std_dict, variables, obs_dict)``。
        """
        inps, obs_list, obs_data, obs_mask, tgt, lead_times, variables, obs_dict, \
        init_time, tgt_time, era5_transforms, microwave_transforms, conventional_transforms = batch

        inps = inps.to(self.device)
        for obs_name in obs_list:
            obs_data[obs_name] = obs_data[obs_name].to(self.device)
            obs_mask[obs_name] = obs_mask[obs_name].to(self.device)
        tgt = tgt.to(self.device)
        lead_times = lead_times.to(self.device)

        std_dict = {}
        std_dict["era5"] = torch.from_numpy(era5_transforms["std"]).to(self.device)
        for name in obs_list:
            if name in microwave_transforms.keys():
                std_dict[name] = torch.from_numpy(microwave_transforms[name]["std"]).to(self.device)
            elif name in conventional_transforms.keys():
                std_dict[name] = torch.from_numpy(conventional_transforms[name]["std"]).to(self.device)

        return inps, obs_list, obs_data, obs_mask, tgt, lead_times, std_dict, variables, obs_dict

    def _run_forecast_steps(self, inps, lead_times, variables):
        """执行外层 AR 预报 ``lead_times`` 步,逐步 ``detach()`` 截断计算图。

        关键决策点 — ``inps, log_var = inps.detach(), log_var.detach()``:
            * 预报模型在 DA 训练中仅作为背景场算子(默认冻结),不接收梯度;
            * ``detach()`` 截断 AR 滚动中的计算图,避免 ``lead_times`` 步累积的中间张量
              全部驻留显存,显著降低 OOM 风险。

        Args:
            inps: 初始 ERA5 状态场 (B, C, H, W)。
            lead_times: 每个迭代步的引导时间 (B, T)。
            variables: 通道变量名列表 (长度 = C)。

        Returns:
            torch.Tensor: 经过 ``T`` 步 AR 滚动后的最终 ``inps``。
        """
        for step in range(lead_times.shape[-1]):
            inps, log_var = self.forecast_model(
                inps,
                lead_times[:, step:step+1],
                variables,
                use_checkpoint=True
            )
            inps, log_var = inps.detach(), log_var.detach()
        return inps

    def train_epoch(self, loader, epoch, epochs):
        """训练一个 epoch:  AR 预报 → Solver per-obs 表征 → XiChenFusion DA → 损失 → 反向传播。

        关键决策点 — 模块模式:
            * ``solver.train()`` / ``da_model.train()`` / ``roe_models[*].train()`` —
              这些模块需要接收梯度(Perceiver 融合 + ROE 表征学习);
            * ``forecast_model.eval()`` / ``obsop_models[*].eval()`` — 静态先验/算子,使用 eval 路径;
            * AR 循环手动 ``detach()`` 截断 ``inps`` 的计算图,避免 AR 滚动过程中显存爆炸;
            * 损失在 ``autocast`` 区域计算,经 ``GradScaler`` 缩放后反传;
            * 多个组件的 ``parameters()`` 列表聚合后做一次 ``clip_grad_norm_`` (max_norm=1.0 默认)。

        Args:
            loader: 训练 ``DataLoader``。
            epoch (int): 当前 epoch 索引。
            epochs (int): 总 epoch 数。

        Returns:
            dict: ``{"train/loss": float}``
        """
        # 设置所有模型为训练模式
        if hasattr(self, 'solver'):
            self.solver.train()
        if hasattr(self, 'forecast_model'):
            self.forecast_model.eval()
        if hasattr(self, 'da_model'):
            self.da_model.train()
        if hasattr(self, 'roe_models'):
            for model in self.roe_models.values():
                model.train()
        if hasattr(self, 'obsop_models'):
            for model in self.obsop_models.values():
                model.eval()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Training epoch {epoch}/{epochs}", leave=False, disable=not self.is_main)

        for batch_idx, batch in enumerate(pbar):
            inps, obs_list, obs_data, obs_mask, tgt, lead_times, std_dict, variables, obs_dict = self._prepare_batch(batch)

            inps = self._run_forecast_steps(inps, lead_times, variables)

            self.optimizer.zero_grad()

            with autocast(self.device_type, dtype=self.precision_type):
                roe = self.solver(
                    self.forecast_model,
                    self.obsop_models,
                    self.roe_models,
                    self.H_models,
                    self.VarCost_models,
                    self.obs_list,
                    inps,
                    obs_data,
                    obs_mask,
                    obs_dict,
                    std_dict,
                    variables
                )

                xa, log_var = self.da_model(
                    roe,
                    self.obs_list,
                    variables,
                    use_checkpoint=True
                )

                loss = self.loss_fn(
                    xa,
                    log_var,
                    tgt,
                    torch.ones_like(log_var).to(self.device, dtype=log_var.dtype)
                )

            # === D14 修复: use_amp=False 时 self.scaler=None, 必须走 fp32 分支
            # 旧版无条件调用 self.scaler.scale(loss).backward(), 在 fp32 debug
            # 路径 (use_amp: false) 上报 AttributeError。修复: 用 if 分流 ===
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()

            all_params = []
            if hasattr(self, 'solver'):
                all_params.extend(self.solver.parameters())
            if hasattr(self, 'roe_models'):
                for model in self.roe_models.values():
                    all_params.extend(model.parameters())
            if hasattr(self, 'obsop_models'):
                for model in self.obsop_models.values():
                    all_params.extend(model.parameters())
            if hasattr(self, 'forecast_model'):
                all_params.extend(self.forecast_model.parameters())
            if hasattr(self, 'da_model'):
                all_params.extend(self.da_model.parameters())

            torch.nn.utils.clip_grad_norm_(
                all_params,
                max_norm=self.training_config.get("max_grad_norm", 1.0)
            )

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            total_loss = total_loss + loss.item()
            pbar.set_postfix(
                {
                    "train_loss": f"{loss.item():.4f}",
                }
            )

            if self.profiler:
                self.profiler.step()

        empty_cache(self.device_type)

        return {"train/loss": total_loss / len(loader)}

    def validate(self, loader, epoch, epochs):
        """验证一个 epoch:全部模块 ``eval()``,无梯度回传;按变量累计 MSE,最终返回平均 RMSE。

        关键决策点:
            * ``xa.detach(), log_var.detach()`` — 验证阶段截断计算图,降低显存;
            * 损失在 ``autocast`` 区域计算;
            * ``val_rmse`` 按变量分通道乘以 std 还原到物理单位 (与训练损失同口径);
            * 进度条同步显示 ``val_loss`` 与 ``val_rmse`` 字符串;
            * TensorBoard 写入放在主进程 + writer 存在的前提下,避免 DDP 重复写。

        Args:
            loader: 验证 ``DataLoader``。
            epoch (int): 当前 epoch 索引。
            epochs (int): 总 epoch 数。

        Returns:
            dict: ``{"val/loss": float, "val/rmse_<var_name>": float, ...}``
            - ``val/loss``: 本 epoch 平均 DA 验证损失
            - ``val/rmse_<var_name>``: 每个变量反归一化加权 RMSE (per-var
              累计),共 ``len(variables)`` 个 key
            空 loader 时返回 ``{"val/loss": float("inf")}``。
        """
        if len(loader) == 0:
            # 空 dataloader (验证日期范围无文件匹配) — 跳过避免 ZeroDivisionError.
            if self.is_main:
                self.log.warning(f"Epoch {epoch}/{epochs}: val_loader is empty, skipping validation.")
            return {"val/loss": float("inf")}
        # 设置所有模型为训练模式
        if hasattr(self, 'solver'):
            self.solver.eval()
        if hasattr(self, 'forecast_model'):
            self.forecast_model.eval()
        if hasattr(self, 'da_model'):
            self.da_model.eval()
        if hasattr(self, 'roe_models'):
            for model in self.roe_models.values():
                model.eval()
        if hasattr(self, 'obsop_models'):
            for model in self.obsop_models.values():
                model.eval()

        total_loss = 0
        total_mse_dict = {}
        pbar = tqdm(loader, desc=f"Validating epoch {epoch}/{epochs}", leave=False, disable=not self.is_main)

        for batch_idx, batch in enumerate(pbar):
            inps, obs_list, obs_data, obs_mask, tgt, lead_times, std_dict, variables, obs_dict = self._prepare_batch(batch)

            inps = self._run_forecast_steps(inps, lead_times, variables)

            with autocast(self.device_type, dtype=self.precision_type):
                roe = self.solver(
                    self.forecast_model,
                    self.obsop_models,
                    self.roe_models,
                    self.H_models,
                    self.VarCost_models,
                    self.obs_list,
                    inps,
                    obs_data,
                    obs_mask,
                    obs_dict,
                    std_dict,
                    variables
                )

                xa, log_var = self.da_model(
                    roe,
                    self.obs_list,
                    variables,
                    use_checkpoint=True
                )

                xa, log_var = xa.detach(), log_var.detach()

                loss = self.loss_fn(
                    xa,
                    log_var,
                    tgt,
                    torch.ones_like(log_var).to(self.device, dtype=log_var.dtype)
                )

                total_loss = total_loss + loss.item()

                # 计算所有变量的RMSE
                val_rmse = std_dict["era5"].to(self.device, dtype=xa.dtype) * weighted_rmse_torch(xa, tgt)

                # 累积每个变量的MSE
                for i, var in enumerate(variables):
                    if var not in total_mse_dict:
                        total_mse_dict[var] = 0
                    total_mse_dict[var] += val_rmse[i].item() ** 2

                # 创建包含所有变量RMSE的字符串用于显示
                rmse_strs = []
                for i, var in enumerate(variables):
                    rmse_strs.append(f"{var}:{val_rmse[i].item():.4f}")
                rmse_str = ", ".join(rmse_strs)

                pbar.set_postfix(
                    {
                        "val_loss": f"{(loss.item()):.4f}",
                        "val_rmse": rmse_str
                    }
                )

        empty_cache(self.device_type)

        # 计算每个变量的平均RMSE
        avg_rmse_per_var = {}
        for var, mse_sum in total_mse_dict.items():
            avg_rmse_per_var[var] = np.sqrt(mse_sum / len(loader))

        # 返回平均损失和所有变量的RMSE字典
        metrics = {"val/loss": total_loss / len(loader)}
        for var, rmse in avg_rmse_per_var.items():
            metrics[f"val/rmse_{var}"] = float(rmse)
        return metrics

    def _save_ckpt(self, ckpt_dir, filename, epoch, val_loss, is_best=False):
        """保存一份完整 Checkpoint(``last.ckpt`` / ``best.ckpt``)以支持断点续训与推理。

        关键决策点 — DDP 安全:
            * 每个 ``nn.Module`` 都要先 ``module.state_dict() if isinstance(m, DDP) else m.state_dict()``
              取出底层模型的 ``state_dict``(避免键名带 ``module.`` 前缀,影响推理侧 ``strict=False`` 加载);
            * 同步保存 ``optimizer`` / ``scheduler`` / ``GradScaler`` 状态,保证续训时优化动量一致;
            * ``config`` 也一并保存,便于推理时还原训练配置;
            * 与 cascade 版的差异:多模态下分别保存 ``solver`` / ``roe_model`` / ``obsop`` / ``da_model``
              (而非 cascade 的 ``DA_models`` 字典)。

        Args:
            ckpt_dir (str): Checkpoint 输出目录。
            filename (str): 文件名(例如 ``"last.ckpt"`` / ``"best.ckpt"``)。
            epoch (int): 当前 epoch 索引。
            val_loss (float): 当前验证损失。
            is_best (bool): 是否为历史最佳。
        """
        # 获取所有模型的状态（处理DDP情况）
        solver_to_save = self.solver.module if isinstance(self.solver, DDP) else self.solver
        roe_model_to_save = self.roe_models.module if isinstance(self.roe_models, DDP) else self.roe_models
        obsop_model_to_save = self.obsop_models.module if isinstance(self.obsop_models, DDP) else self.obsop_models
        da_model_to_save = self.da_model.module if isinstance(self.da_model, DDP) else self.da_model
        forecast_model_to_save = self.forecast_model.module if isinstance(self.forecast_model, DDP) else self.forecast_model
        h_models_to_save = self.H_models.module if isinstance(self.H_models, DDP) else self.H_models
        varcost_models_to_save = self.VarCost_models.module if isinstance(self.VarCost_models, DDP) else self.VarCost_models

        ckpt_dict = {
            "epoch": epoch,
            "solver_state_dict": solver_to_save.state_dict(),
            "roe_model_state_dict": roe_model_to_save.state_dict(),
            "obsop_state_dict": obsop_model_to_save.state_dict(),
            "da_model_state_dict": da_model_to_save.state_dict(),
            "forecast_state_dict": forecast_model_to_save.state_dict(),
            "h_models_state_dict": h_models_to_save.state_dict(),
            "varcost_models_state_dict": varcost_models_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_loss": self.best_loss,
            "config": self.config,  # 新增：保存配置以便恢复
            # === 扩展微调元信息 (供 _load_checkpoint 判定 is_extension 模式) ===
            # obs_vocab 不再存: 它不是 model state_dict 的一部分, 推理侧从
            # obs_list 自动建 (XiChenFusion.__init__ line 202-203); 训练期
            # 也无消费者 (旧 D10 name-keyed 校验随自定义 legacy loader 一起被简化删除)。
            "is_extension": self.training_config.get("is_extension", False),
            "new_obs_names": self.new_obs_names,
        }

        save_path = os.path.join(ckpt_dir, filename)
        torch.save(ckpt_dict, save_path, _use_new_zipfile_serialization=False)
        if self.is_main:
            self.log.info(f"Checkpoint saved to {save_path}")
