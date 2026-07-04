# -*- coding: utf-8 -*-
"""
BaseTrainer - 训练器基类 (Hydra兼容)。

本模块定义了所有具体训练器（forecast / compression / obsoperator / DA 等）需要继承的
抽象基类 ``BaseTrainer``。该基类统一封装了训练流程中跨任务复用的"公共管线"：

- 模型 / 优化器 / 调度器 / 损失函数 / AMP GradScaler 的 Hydra 实例化
- 基于 ``resume_ckpt`` 路径的断点续训（模型权重 + 优化器 + 调度器 + AMP 标量）
- 基于 ``torch.nn.parallel.DistributedDataParallel`` 的分布式封装（NPU / GPU 自适应）
- 基于 ``torch.utils.tensorboard.SummaryWriter`` 的训练指标写入
- 基于 ``torch_npu.profiler`` / ``torch.profiler`` 的 profile 跟踪
- 训练主循环 ``fit``：train → validate → scheduler.step → save best/latest

``main.py`` 只需要实例化 ``DataLoader`` 并调用 ``trainer.fit(train_loader, val_loader)``，
所有训练期资源（模型包装、日志、checkpoint、profiler、分布式进程组清理）由本类托管。
"""
import math
import os
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
import hydra

from src.utils import get_logger
from src.utils.device import get_grad_scaler
from src.utils.lr_scheduler import CosineSchedulerWithWarmup

try:
    import torch_npu
except ImportError:
    torch_npu = None

class BaseTrainer(ABC):
    def __init__(
        self,
        config: DictConfig,
        device,
        local_rank: int,
        world_size: int,
        is_main: bool,
        **kwargs,
    ):
        """5 阶段模板: 公共状态 → 子类 hook 链 → 续训 → DDP → Logger/Profiler.

        Args:
            config: Hydra 顶层配置.
            device: 训练设备 (torch.device).
            local_rank: 当前进程本地 rank (DDP).
            world_size: 全局进程数.
            is_main: 是否为主进程 (rank 0).
            **kwargs: 透传参数, 当前识别 log_dir (默认 'logs').
        """
        # === Phase 0: 公共状态 (基类统一管理) ===
        self.config = config
        self.device = device
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = is_main
        self.training_config = config.get("training", {})
        self.log_dir = kwargs.get("log_dir", "logs")
        output_dir = self.config.paths.get("output_dir", "logs/output")
        self.ckpt_dir = os.path.join(output_dir, "checkpoints")
        self.log = kwargs.get("log", None) or get_logger("xichen.trainer")
        if self.is_main:
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # 仅标量字段在 Phase 0 预设; 模型 / 训练基础设施字段由对应 hook / helper 创建.
        self.start_epoch = 0
        self.best_loss = float("inf")
        # 以下字段在 Phase 4 由 _setup_logger / _setup_profiler / _set_precision 填充;
        # 预设 None 以便 cleanup() / fit() 在 is_main=False 或 use_amp=False 时安全访问.
        self.writer = None
        self.profiler = None
        self.scaler = None

        # === Phase 1: 子类实例化 (Hook 链) ===
        if self.is_main:
            self.log.info("Initializing components...")
        self._build_models()                  # HOOK — 子类 override
        self._build_loss_fn()                 # HOOK — 默认 hydra.instantiate
        self._setup_optimizer_scheduler()     # HOOK — 子类 override
        self._set_precision()                 # helper — AMP GradScaler

        # === Phase 2: 续训 (Hook, 子类可完全 override) ===
        if self.training_config.get("resume_ckpt", False):
            self._load_checkpoint()           # HOOK — 子类 override

        # === Phase 3: DDP 包装 (Hook) ===
        self._wrap_ddp()                      # HOOK — 子类 override

        # === Phase 4: Logger + Profiler ===
        self._setup_logger()
        self._setup_profiler()

    """训练器基类，支持 Hydra 实例化与分布式训练。

    该基类按以下顺序完成训练环境的初始化：
    1. ``_build_models`` / ``_build_loss_fn``：通过 Hydra 实例化模型与损失函数
       （子类在 ``_build_models`` 中按需加载多个子模型）；
    2. ``_setup_optimizer_scheduler``：构造 ``optimizer`` 与可选 ``scheduler``，
       并按需构造 ``torch.amp.GradScaler``；
    3. ``_set_precision``：根据 ``training.precision`` 与 ``device_type`` 设定
       ``autocast`` / ``GradScaler`` 的 dtype；
    4. ``_load_checkpoint``：若 ``config.training.resume_ckpt`` 指向有效路径，
       则恢复 ``model_state_dict`` / ``optimizer_state_dict`` /
       ``scheduler_state_dict`` / ``scaler_state_dict`` 以及 ``start_epoch`` /
       ``best_loss``，注意此处必须在 DDP 包装之前完成；
    5. ``_wrap_ddp``：当 ``world_size > 1`` 时，使用 ``DistributedDataParallel``
       包装裸模型，并在 NPU / CUDA 设备上挂载 ``device_ids`` 与
       ``output_device``；
    6. ``_setup_logger`` / ``_setup_profiler``：仅主进程（``is_main=True``）启用
       ``SummaryWriter`` 与可选的 profiler。

    子类只需实现 ``train_epoch`` 与 ``validate`` 两个抽象方法即可获得完整的
    训练循环、日志记录、checkpoint 保存、分布式协同与资源清理能力。

    Attributes:
        config (DictConfig): Hydra 顶层配置。
        device (torch.device | str): 当前进程绑定的设备（NPU 或 CUDA）。
        local_rank (int): 当前进程在本节点内的本地 rank。
        world_size (int): 总进程数；>1 时启用分布式。
        is_main (bool): 是否为主进程（仅主进程写日志与保存 checkpoint）。
        log_dir (str): TensorBoard 与 checkpoint 的根目录。
        log: 由 ``src.utils.get_logger`` 创建的 logger。
        model: 实例化后的模型，可能为裸 ``nn.Module`` 或被 ``DDP`` 包装。
        optimizer: 实例化后的优化器。
        scheduler: 实例化后的学习率调度器，可能为 ``None``。
        loss_fn: 实例化后的损失函数。
        scaler: AMP ``GradScaler``；若未启用 AMP 则为 ``None``。
        writer: ``SummaryWriter``；非主进程为 ``None``。
        start_epoch (int): 当前训练起始 epoch（断点续训时 ``>0``）。
        best_loss (float): 截至当前 epoch 为止的最优验证损失。
        profiler: 可选的 profiler 对象；非主进程或未开启 profile 时为 ``None``。
    """

    def _wrap_single_ddp(self, model):
        """对单模型做 ``DistributedDataParallel`` 包装（自动绑定 device_ids / output_device）。

        这是 6 个 trainer 子类中重复的 6 行 DDP 构造逻辑的公共实现。
        调用方负责把返回值赋回原字段，例如::

            self.model = self._wrap_single_ddp(self.model)            # 单模型 trainer
            setattr(self, name, self._wrap_single_ddp(model))         # 多模型 trainer (DA)

        Args:
            model: 待包装的 ``torch.nn.Module``。

        Returns:
            ``torch.nn.parallel.DistributedDataParallel``: DDP 包装后的模型。
        """
        is_accel = ("cuda" in str(self.device)) or ("npu" in str(self.device))
        auto_move_inputs = bool(self.training_config.get("ddp_auto_move_inputs", True))
        # Retrieval batches contain large CPU-side point-cloud dictionaries.  When
        # device_ids is set, PyTorch DDP recursively moves every tensor in the input
        # structure to the target GPU before forward(), which defeats the retrieval
        # model's lazy per-sensor transfer and can OOM before the first layer runs.
        # Setting ddp_auto_move_inputs=false keeps DDP synchronization but leaves
        # placement to the training loop/model, as allowed by DDP when device_ids=None.
        ddp_device_ids = [self.local_rank] if (is_accel and auto_move_inputs) else None
        ddp_output_device = self.local_rank if (is_accel and auto_move_inputs) else None
        ddp_kwargs = {
            "device_ids": ddp_device_ids,
            "output_device": ddp_output_device,
            "find_unused_parameters": self.training_config.get("find_unused_parameters", False),
            "broadcast_buffers": self.training_config.get("broadcast_buffers", False),
            "gradient_as_bucket_view": self.training_config.get("gradient_as_bucket_view", True),
            "static_graph": self.training_config.get("static_graph", False),
        }
        bucket_cap_mb = self.training_config.get("bucket_cap_mb", None)
        if bucket_cap_mb is not None:
            ddp_kwargs["bucket_cap_mb"] = int(bucket_cap_mb)
        return DDP(model, **ddp_kwargs)

    def _setup_logger(self):
        """初始化 Logger（仅主进程启用）。"""
        if self.is_main:
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def _setup_profiler(self):
        """初始化 Profiler（仅主进程启用）。

        当 ``config.profile`` 为真时启用 profiler：NPU 设备下使用
        ``torch_npu.profiler``，CUDA 设备下使用 ``torch.profiler``。

        调度策略为 ``schedule(wait=1, warmup=1, active=3, repeat=1)``，并通过
        ``tensorboard_trace_handler`` 直接写出可在 TensorBoard "Profile" 页查看的
        trace 文件。若初始化失败则捕获异常并退化为关闭状态，不影响主训练流程。
        """
        profile_enabled = bool(self.config.get("profile", self.training_config.get("profile", False)))
        if not self.is_main or not profile_enabled:
            return
        try:
            is_npu = "npu" in str(self.device).lower()
            if is_npu and torch_npu is None:
                raise RuntimeError("NPU profiler requested but torch_npu is not installed")
            profiler_lib = torch_npu.profiler if is_npu else torch.profiler

            self.profiler = profiler_lib.profile(
                activities=[
                    profiler_lib.ProfilerActivity.CPU,
                    profiler_lib.ProfilerActivity.NPU if is_npu else profiler_lib.ProfilerActivity.CUDA
                ],
                schedule=profiler_lib.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=profiler_lib.tensorboard_trace_handler(os.path.join(self.log_dir, "profiler")),
                record_shapes=True,
                with_stack=True
            )
            self.profiler.start()
            self.log.info(f"Profiler enabled on {self.device}")
        except Exception as e:
            self.log.warning(f"Profiler failed to initialize (safe fallback): {e}")
            self.profiler = None

    def _set_precision(self):
        """设置 AMP ``GradScaler`` 与 ``precision_type``。

        - 当 ``self.training_config.use_amp`` 为真时，按 ``device_type`` 选择 NPU 或 CUDA
          上的 ``GradScaler``，并解析 ``self.training_config.precision.type``
          （支持 ``bf16`` / ``fp16`` / ``fp32`` 及其长名）映射到 ``torch.dtype``；
          字符串大小写不敏感（内部先 ``.lower()`` 归一化再查表）；
        - 否则 ``self.scaler = None`` 且 ``self.precision_type = torch.float32``，
          后续 ``autocast`` 走 fp32 路径。

        同时设置 ``self.device_type`` (``"npu"`` 或 ``"cuda"``)，供 ``train_epoch`` /
        ``validate`` 在 ``autocast(self.device_type, ...)`` 调用时使用。

        调用约定: 子类在自身的 ``_setup_components`` 中实例化模型 / 优化器 /
        scheduler / loss_fn 之后调用本方法。
        """
        if self.is_main:
            self.log.info("Initializing AMP...")
        use_amp = self.training_config.get("use_amp", False)
        device_str = str(self.device).lower()
        if "npu" in device_str:
            self.device_type = "npu"
        elif "cuda" in device_str:
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"
        precision_cfg = self.training_config.get("precision", {"type": "bf16"})
        precision_type = "bf16"
        if isinstance(precision_cfg, dict):
            precision_type = precision_cfg.get("type", "bf16")
        elif hasattr(precision_cfg, "get"):
            precision_type = precision_cfg.get("type", "bf16")
        elif isinstance(precision_cfg, str):
            precision_type = precision_cfg
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "32": torch.float32,
        }
        self.precision_type = dtype_map.get(str(precision_type).lower(), torch.bfloat16) if use_amp else torch.float32
        if self.device_type == "cuda" and self.precision_type is torch.bfloat16 and hasattr(torch.cuda, "is_bf16_supported"):
            if not torch.cuda.is_bf16_supported():
                self.log.warning("CUDA device does not report BF16 support; falling back to FP16 AMP")
                self.precision_type = torch.float16
        self.scaler = get_grad_scaler(self.device_type, dtype=self.precision_type) if use_amp else None

    def _build_scheduler(self, optimizer):
        """根据 ``self.training_config.scheduler_type`` 构建 LR 调度器。

        支持以下三种类型:
            * ``cosine_warmup`` (默认) — ``CosineSchedulerWithWarmup`` (带 warmup 的余弦退火);
            * ``cosine`` — ``CosineAnnealingLR`` (无 warmup);
            * 其他 — 不创建调度器，返回 ``None``。

        Args:
            optimizer: 已实例化的 ``torch.optim.Optimizer``，绑定到 ``self.model.parameters()``。

        Returns:
            ``torch.optim.lr_scheduler.LRScheduler`` 或 ``None``。

        调用约定: 子类在自身的 ``_setup_optimizer_scheduler`` 中创建完
        ``self.optimizer`` 后调用本方法,例如::

            self.optimizer = torch.optim.AdamW(self.model.parameters(), ...)
            self.scheduler = self._build_scheduler(self.optimizer)
        """
        scheduler_type = self.training_config.get("scheduler_type", "cosine_warmup")
        warmup_epochs = self.training_config.get("warmup_epochs", 10)
        epochs = self.training_config.get("epochs", 100)
        if scheduler_type == "cosine_warmup":
            return CosineSchedulerWithWarmup(
                optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=epochs,
            )
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=0,
            )
        return None

    def _build_models(self) -> None:
        """实例化模型组件, 赋值给 self.model / self.models / self.solver / 等具名属性。

        默认: raise NotImplementedError. 子类必须 override (基类不知道子类用什么字段名)。

        调用约定: 基类 __init__ 在 Phase 1 调一次.
        """
        raise NotImplementedError(
            "Subclasses must implement _build_models. BaseTrainer cannot infer "
            "the model attribute name (self.model / self.models / self.solver / etc.)."
        )

    def _load_pretrain_ckpt(self) -> None:
        """加载预训练 checkpoint (pretrain→finetune 范式).

        默认: pass (4 个单模型子类在 _build_models 末尾按需调用, DA 子类不走此路径).

        调用约定: 由子类 _build_models 在模型实例化之后自行调用; 基类 __init__ 不显式触发.
        """
        pass

    def _build_loss_fn(self) -> None:
        """实例化损失函数。

        默认: hydra.instantiate(self.config.loss_fn). 所有 6 子类都用此默认.
        """
        self.loss_fn = hydra.utils.instantiate(self.config.loss_fn, _recursive_=False)

    def _setup_optimizer_scheduler(self) -> None:
        """构造 optimizer + scheduler.

        默认: raise NotImplementedError. 子类必须 override.

        调用约定: 基类 __init__ 在 Phase 1 (loss_fn 之后, _set_precision 之前) 调一次.
        """
        raise NotImplementedError(
            "Subclasses must implement _setup_optimizer_scheduler."
        )

    def _wrap_ddp(self) -> None:
        """DDP 包装。

        默认: raise NotImplementedError. 基类不知子类用什么字段名存模型,
        无法提供通用默认. 4 单模型子类 (forecast / compression / obsop / forecast_obconstraint)
        各自 override; DA 子类 (cascade / multimodal) 保留原 6-组件 sweep 实现.

        调用约定: 基类 __init__ 在 Phase 4 调一次 (在 _load_checkpoint 之后).
        """
        raise NotImplementedError(
            "Subclasses must implement _wrap_ddp. BaseTrainer cannot infer "
            "the model attribute name."
        )

    def _load_checkpoint(self) -> None:
        """断点续训. 从 {ckpt_dir}/last.ckpt 恢复.

        默认: raise NotImplementedError. 基类不知子类用什么字段名存模型, 且
        不同子类 ckpt schema 差异大 (单 model vs 6/7 state_dict).

        调用约定: 基类 __init__ 在 Phase 3 调一次 (仅当 training.resume_ckpt 为真).
        """
        raise NotImplementedError(
            "Subclasses must implement _load_checkpoint. BaseTrainer cannot infer "
            "the ckpt schema or the model attribute name."
        )

    def _save_ckpt(self, ckpt_dir, filename, epoch, val_loss, is_best=False) -> None:
        """保存 checkpoint. 子类负责写 ckpt_dict 字段 (因 6 子类 schema 完全不同).

        默认: raise NotImplementedError. fit() 模板通过 _save_epoch_ckpts 调用此方法.

        Args:
            ckpt_dir: checkpoint 输出目录.
            filename: 文件名 (例如 'last.ckpt' / 'best.ckpt').
            epoch: 当前 epoch 编号.
            val_loss: 当前验证损失.
            is_best: 是否为最佳 (仅用于日志标记).
        """
        raise NotImplementedError(
            "Subclasses must implement _save_ckpt. BaseTrainer cannot infer "
            "the ckpt schema or the model attribute name."
        )

    def _log_epoch_scalars(self, epoch, metrics) -> None:
        """默认: 遍历 metrics dict 写 TensorBoard (仅 is_main + writer).

        子类通常无需 override; 仅当需要特殊处理 (4D tensor 直方图等) 时再 override.

        注意: 本 spec 重构后, 参数 train_loss + val_outputs 改为统一的 metrics dict.
        子类按 spec §4 迁移后, 默认实现自动覆盖所有 metrics.
        """
        if self.is_main and self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, epoch)

    def _on_epoch_start(self, epoch: int) -> None:
        """每个 epoch 开始前的钩子。默认: pass。"""
        pass

    def _on_epoch_end(self, epoch: int) -> None:
        """每个 epoch 结束后的钩子。默认: pass。"""
        pass

    @abstractmethod
    def train_epoch(self, loader, epoch, epochs):
        """单个 epoch 的训练逻辑（由子类实现）。

        Args:
            loader: 训练数据加载器。
            epoch (int): 当前 epoch 编号（从 0 开始）。
            epochs (int): 总 epoch 数，用于日志展示。

        Returns:
            dict: ``{"train/loss": float, ...}``，需含 ``train/loss`` key。
        """
        pass

    @abstractmethod
    def validate(self, loader, epoch, epochs):
        """单个 epoch 的验证逻辑（由子类实现）。

        Args:
            loader: 验证数据加载器。
            epoch (int): 当前 epoch 编号（从 0 开始）。
            epochs (int): 总 epoch 数，用于日志展示。

        Returns:
            dict: ``{"val/loss": float, "val/rmse_<var>": float, ...}``，需含
            ``val/loss`` key；空 loader 时返回 ``{"val/loss": float("inf")}``。
        """
        pass

    def fit(self, train_loader, val_loader) -> float:
        """主训练循环 (模板方法). 6 子类共用.

        流程:
            for epoch in range(self.start_epoch, epochs):
                if world_size > 1: sampler.set_epoch(epoch)
                _on_epoch_start(epoch)
                # train_metrics / val_metrics 必须为 dict,含 ``train/loss`` /
                # ``val/loss`` 等 key;不满足时 ``fit`` 内部会抛 ``TypeError``。
                train_metrics = self.train_epoch(train_loader, epoch, epochs)
                val_metrics = self.validate(val_loader, epoch, epochs)
                merged = {**train_metrics, **val_metrics}
                _log_epoch_scalars(epoch, merged)
                if scheduler: scheduler.step()
                if profiler: profiler.step()
                if is_main:
                    lr = _current_lr()
                    log.info(_format_log_line(epoch, epochs, lr, merged))
                    if not math.isinf(merged["val/loss"]):
                        _save_epoch_ckpts(epoch, merged["val/loss"])
            if is_main:
                writer.close()
                log.info(f"Training completed. Best val_loss={self.best_loss:.4f}")
        return self.best_loss

        Args:
            train_loader: 训练 DataLoader.
            val_loader: 验证 DataLoader.

        Returns:
            float: 历史最佳验证损失 self.best_loss.
        """
        epochs = self.training_config.epochs

        for epoch in range(self.start_epoch, epochs):
            if self.world_size > 1 and hasattr(getattr(train_loader, "sampler", None), "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            self._on_epoch_start(epoch)

            train_metrics = self.train_epoch(train_loader, epoch, epochs)
            val_metrics = self.validate(val_loader, epoch, epochs)

            # Contract: subclasses MUST return a dict with "val/loss" key.
            if not isinstance(val_metrics, dict) or "val/loss" not in val_metrics:
                raise TypeError(
                    f"{type(self).__name__}.validate() must return a dict containing "
                    f"'val/loss' key, got: {type(val_metrics).__name__}"
                )

            if not isinstance(train_metrics, dict) or "train/loss" not in train_metrics:
                raise TypeError(
                    f"{type(self).__name__}.train_epoch() must return a dict containing "
                    f"'train/loss' key, got: {type(train_metrics).__name__}"
                )

            merged = {**train_metrics, **val_metrics}

            if hasattr(self, "_append_epoch_metrics_jsonl"):
                self._append_epoch_metrics_jsonl(epoch, merged)

            if self.is_main and self.writer is not None:
                self._log_epoch_scalars(epoch, merged)

            if self.scheduler is not None and getattr(self, "scheduler_step_unit", "epoch") == "epoch":
                self.scheduler.step()
            if self.profiler is not None:
                self.profiler.step()

            if self.is_main:
                lr = self._current_lr()
                self.log.info(self._format_log_line(epoch, epochs, lr, merged))
                # 空 loader 时跳过 ckpt 保存: 空 loader 表示配置错误 (验证日期范围无匹配文件),
                # 不应该写损坏的 inf 值到 last.ckpt (会污染下次 resume).
                if not math.isinf(merged["val/loss"]):
                    self._save_epoch_ckpts(epoch, merged["val/loss"])

        if self.is_main and self.writer is not None:
            self.writer.close()

        if self.is_main:
            self.log.info(f"Training completed. Best val_loss={self.best_loss:.4f}")

        return self.best_loss

    def _save_epoch_ckpts(self, epoch: int, val_loss: float) -> None:
        """总是保存 last.ckpt + 仅在 val_loss 改善时保存 best.ckpt (R1 决策).

        行为等价证明: 6 子类当前 fit 中
            if val_loss < best:
                best = val_loss
                _save_ckpt(last); _save_ckpt(best)
            else:
                _save_ckpt(last)
        代数化简为: _save_ckpt(last) 永远; val<best 时多存一次 best. 与本方法完全一致.
        """
        self._save_ckpt(self.ckpt_dir, "last.ckpt", epoch, val_loss)
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self._save_ckpt(self.ckpt_dir, "best.ckpt", epoch, val_loss, is_best=True)
            self.log.info(f"New best model saved! Best Loss: {self.best_loss:.4f}")

    def _current_lr(self) -> float:
        """获取 optimizer 当前学习率 (含 param_groups 支持)."""
        return self.optimizer.param_groups[0]['lr']

    def _format_log_line(self, epoch: int, epochs: int, lr: float, metrics: dict) -> str:
        """默认: lr 单独保留 + metrics 字典自动展开 (val/loss 排在最后).

        示例输出:
            Epoch 5/100: lr=0.00010000, train/loss=0.1234,
                          val/rmse_z-500=1.2345, ..., val/loss=0.5678

        子类通常无需 override.

        注意: 本 spec 重构后, 参数 train_loss + val_outputs 改为统一的 metrics dict.
        """
        parts = [f"Epoch {epoch}/{epochs}: lr={lr:.8f}"]
        for key, value in metrics.items():
            if key == "val/loss":
                continue  # 排在最后
            parts.append(f"{key}={value:.4f}")
        parts.append(f"val/loss={metrics['val/loss']:.4f}")
        return ", ".join(parts)

    def cleanup(self):
        """清理资源。

        训练/异常退出时统一调用，依次关闭 profiler、关闭
        ``SummaryWriter``、销毁分布式进程组。
        """
        if self.profiler is not None:
            self.profiler.stop()
        if self.writer is not None:
            self.writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()