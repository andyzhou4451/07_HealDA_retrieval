# -*- coding: utf-8 -*-
"""HealDA-style 13 层温湿廓线反演训练器。"""

from __future__ import annotations

import csv
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, Mapping

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from tqdm import tqdm

from src.metrics.retrieval_metrics import retrieval_metrics
from src.pipeline.base.trainer import BaseTrainer
from src.utils.device import autocast
from src.utils.performance import JsonlWriter, StepTimer, cuda_memory_stats, query_nvidia_smi_utilization


class RetrievalTrainer(BaseTrainer):
    """XiChen BaseTrainer 兼容的单卡/多卡反演训练器。"""

    def __init__(self, cfg: DictConfig, device, local_rank: int, world_size: int, is_main: bool, **kwargs) -> None:
        super().__init__(cfg, device, local_rank, world_size, is_main, **kwargs)
        output_dir = str(self.config.paths.get("output_dir", "logs/output"))
        perf_dir = str(self.training_config.get("performance_log_dir", self.config.paths.get("performance_log_dir", os.path.join("outputs", "logs"))))
        self.metrics_csv = os.path.join(output_dir, "metrics.csv")
        self.log_every_n_steps = int(self.training_config.get("log_every_n_steps", 20) or 20)
        self.performance_log_every_n_steps = int(self.training_config.get("performance_log_every_n_steps", 1) or 1)
        self.performance_writer = JsonlWriter(os.path.join(perf_dir, "performance.jsonl"), enabled=self.is_main)
        self.epoch_metrics_writer = JsonlWriter(os.path.join(perf_dir, "epoch_metrics.jsonl"), enabled=self.is_main)
        self.global_step = 0
        self.amp_fallback = False
        self.oom_fallback = False
        self.nan_or_inf_seen = False

    def _build_models(self) -> None:
        """实例化反演模型，并按配置启用 channels_last / torch.compile。"""
        model_cfg = self.config.model.get("net", self.config.model)
        self.model = hydra.utils.instantiate(model_cfg, _recursive_=False).to(self.device)
        if bool(self.training_config.get("channels_last", self.config.get("precision", {}).get("channels_last", False))) and hasattr(self.model, "to"):
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception as exc:
                if self.is_main:
                    self.log.warning(f"channels_last conversion failed safely; continuing with contiguous format: {exc}")
        compile_cfg = self.training_config.get("compile", self.config.get("precision", {}).get("compile", {}))
        compile_enabled = bool(compile_cfg.get("enabled", compile_cfg if isinstance(compile_cfg, bool) else False)) if hasattr(compile_cfg, "get") else bool(compile_cfg)
        compile_target = str(compile_cfg.get("target", "backbone")) if hasattr(compile_cfg, "get") else "backbone"
        if compile_enabled and hasattr(torch, "compile"):
            mode = str(compile_cfg.get("mode", self.config.get("precision", {}).get("compile_mode", "max-autotune"))) if hasattr(compile_cfg, "get") else "max-autotune"
            try:
                if compile_target == "backbone" and hasattr(self.model, "backbone"):
                    self.model.backbone = torch.compile(self.model.backbone, mode=mode)
                elif compile_target == "model":
                    self.model = torch.compile(self.model, mode=mode)
                if self.is_main:
                    self.log.info(f"torch.compile enabled: target={compile_target}, mode={mode}")
            except Exception as exc:
                if self.is_main:
                    self.log.warning(f"torch.compile failed; continuing without compile: {exc}")
        if self.is_main and hasattr(self.model, "estimate_vram_gb"):
            try:
                est = self.model.estimate_vram_gb(batch_size=int(self.config.datamodule.get("batch_size", 1)))
                self.log.info(f"Approximate upper-bound VRAM estimate: {est:.2f} GB")
            except Exception as exc:
                self.log.warning(f"VRAM estimate failed safely: {exc}")

    def _setup_optimizer_scheduler(self) -> None:
        """构建 AdamW 和学习率调度器；H100 配置优先读取顶层 optimizer/scheduler。"""
        optimizer_cfg = self.config.get("optimizer", {})
        lr = float(optimizer_cfg.get("lr", self.training_config.get("lr", 1e-4))) if hasattr(optimizer_cfg, "get") else float(self.training_config.get("lr", 1e-4))
        wd = float(optimizer_cfg.get("weight_decay", self.training_config.get("weight_decay", 0.05))) if hasattr(optimizer_cfg, "get") else float(self.training_config.get("weight_decay", 0.05))
        betas = tuple(optimizer_cfg.get("betas", self.training_config.get("betas", [0.9, 0.95]))) if hasattr(optimizer_cfg, "get") else tuple(self.training_config.get("betas", [0.9, 0.95]))
        eps = float(optimizer_cfg.get("eps", self.training_config.get("eps", 1e-8))) if hasattr(optimizer_cfg, "get") else float(self.training_config.get("eps", 1e-8))
        use_fused = bool(self.training_config.get("fused_adamw", False)) if self.device_type_is_cuda_like() else False
        try:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=wd, eps=eps, fused=use_fused)
        except TypeError:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=wd, eps=eps)
        scheduler_cfg = self.config.get("scheduler", {})
        scheduler_name = str(scheduler_cfg.get("name", self.training_config.get("scheduler_type", "cosine_warmup"))) if hasattr(scheduler_cfg, "get") else str(self.training_config.get("scheduler_type", "cosine_warmup"))
        if scheduler_name == "cosine" and hasattr(scheduler_cfg, "get"):
            warmup_steps = int(scheduler_cfg.get("warmup_steps", self.training_config.get("warmup_steps", 1000)) or 0)
            max_steps = int(scheduler_cfg.get("total_steps", self.training_config.get("max_steps", 100000)) or 100000)
            min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.05))

            def lr_lambda(step: int) -> float:
                if warmup_steps > 0 and step < warmup_steps:
                    return max(float(step + 1) / float(warmup_steps), 1e-8)
                denom = max(max_steps - warmup_steps, 1)
                progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)
                return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            self.scheduler_step_unit = "step"
        else:
            self.scheduler = self._build_scheduler(self.optimizer)
            self.scheduler_step_unit = "epoch"

    def device_type_is_cuda_like(self) -> bool:
        """判断当前设备是否为 CUDA；构造 fused AdamW 时需要该信息。"""
        return "cuda" in str(self.device).lower()

    def _wrap_ddp(self) -> None:
        """在 world_size>1 时使用 DDP 包装模型。"""
        if self.world_size > 1:
            self.model = self._wrap_single_ddp(self.model)

    def _load_checkpoint(self) -> None:
        """从 last.ckpt 或指定路径恢复模型、优化器、调度器和 AMP scaler。"""
        ckpt_path = self.training_config.get("resume_ckpt", None)
        if isinstance(ckpt_path, bool):
            ckpt_path = os.path.join(self.ckpt_dir, "last.ckpt") if ckpt_path else None
        if not ckpt_path or not os.path.exists(str(ckpt_path)):
            if self.is_main:
                self.log.warning(f"No checkpoint found at {ckpt_path}; training from scratch")
            return
        ckpt = torch.load(str(ckpt_path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self.scaler is not None and ckpt.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.best_loss = float(ckpt.get("best_loss", self.best_loss))
        self.global_step = int(ckpt.get("global_step", 0))
        if self.is_main:
            self.log.info(f"Resumed retrieval checkpoint {ckpt_path} at epoch {self.start_epoch}")

    def _save_ckpt(self, ckpt_dir, filename, epoch, val_loss, is_best=False) -> None:
        """仅 rank0 调用：保存 last.ckpt/best.ckpt。"""
        os.makedirs(ckpt_dir, exist_ok=True)
        model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "best_loss": min(self.best_loss, float(val_loss)),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "target_vars": getattr(model, "target_vars", None),
            "pressure_levels": getattr(model, "pressure_levels", None),
            "output_shape": "[B, 26, 181, 360]",
            "active_grid_backend": getattr(model, "active_grid_backend", None),
        }
        torch.save(ckpt, os.path.join(ckpt_dir, filename))

    def _append_metrics_csv(self, epoch: int, metrics: Dict[str, float]) -> None:
        """rank0 追加写 metrics.csv，便于 Slurm 任务结束后直接汇总。"""
        if not self.is_main:
            return
        os.makedirs(os.path.dirname(self.metrics_csv), exist_ok=True)
        row = {"epoch": int(epoch), **metrics}
        write_header = not os.path.exists(self.metrics_csv)
        with open(self.metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _append_epoch_metrics_jsonl(self, epoch: int, metrics: Dict[str, float]) -> None:
        """rank0 写 epoch_metrics.jsonl，字段更适合机器解析。"""
        if not self.is_main:
            return
        row = {"epoch": int(epoch), "lr": self._current_lr(), **metrics}
        row.update(cuda_memory_stats(self.device))
        row.update({"torch_version": torch.__version__, "precision": str(self.precision_type).replace("torch.", "")})
        self.epoch_metrics_writer.write(row)

    def _move_batch_target(self, batch: Mapping[str, object]) -> dict:
        """只提前搬运监督标签；变长观测点云在 sensor embedder 内按需搬到当前 GPU。"""
        out = dict(batch)
        out["target"] = out["target"].to(self.device, non_blocking=True)
        return out

    def _maybe_no_sync(self, should_sync: bool):
        """DDP 梯度累积时只在真正 optimizer step 前同步梯度。"""
        if should_sync or self.world_size <= 1 or not hasattr(self.model, "no_sync"):
            return nullcontext()
        return self.model.no_sync()

    def _all_reduce_sums(self, sums: Mapping[str, float], seen: int) -> tuple[Dict[str, float], int]:
        """把各 rank 的加权和与样本数做 all_reduce，返回全局求和结果。"""
        keys = sorted(sums.keys())
        values = [float(sums[k]) for k in keys] + [float(seen)]
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced = {k: float(tensor[i].item()) for i, k in enumerate(keys)}
        reduced_seen = int(tensor[-1].item())
        return reduced, reduced_seen

    @staticmethod
    def _average_sums(prefix: str, sums: Mapping[str, float], seen: int) -> Dict[str, float]:
        """把加权和转换成平均指标，并追加 train/ 或 val/ 前缀。"""
        denom = max(int(seen), 1)
        return {f"{prefix}/{k}": float(v) / denom for k, v in sums.items()}

    def _log_performance_step(self, epoch: int, step: int, bsz: int, step_time: float, timer: StepTimer, losses: Mapping[str, torch.Tensor]) -> None:
        """写单步性能日志，不强制 CUDA 同步。"""
        if not self.is_main or (step + 1) % self.performance_log_every_n_steps != 0:
            return
        mem = cuda_memory_stats(self.device)
        gpu_idx = torch.device(self.device).index or 0
        gpu_util = query_nvidia_smi_utilization(gpu_idx) if os.environ.get("PERF_QUERY_NVIDIA_SMI", "0") == "1" else None
        grad_accum = max(int(self.training_config.get("gradient_accumulation_steps", 1) or 1), 1)
        row = {
            "epoch": int(epoch),
            "step": int(step),
            "global_step": int(self.global_step),
            "batch_size": int(bsz),
            "effective_batch_size": int(bsz * grad_accum),
            "precision": str(self.precision_type).replace("torch.", ""),
            "step_time_sec": float(step_time),
            "samples_per_second": float(bsz / max(step_time, 1e-12)),
            "data_time_sec": float(timer.data_time),
            "forward_time_sec": float(timer.forward_time),
            "backward_time_sec": float(timer.backward_time),
            "optimizer_time_sec": float(timer.optimizer_time),
            "lr": float(self._current_lr()),
            "train_loss": float(losses["total_loss"].detach().float().cpu()),
            "temperature_loss": float(losses["temperature_loss"].detach().float().cpu()),
            "humidity_loss": float(losses["humidity_loss"].detach().float().cpu()),
            "gpu_utilization": gpu_util,
            "torch_version": torch.__version__,
            "amp_fallback": bool(self.amp_fallback),
            "oom_fallback": bool(self.oom_fallback),
            "nan_or_inf": bool(self.nan_or_inf_seen),
        }
        row.update(mem)
        self.performance_writer.write(row)

    def train_epoch(self, loader, epoch, epochs):
        """执行一个训练 epoch，返回跨 rank 平均后的训练指标。"""
        self.model.train()
        if torch.cuda.is_available() and "cuda" in str(self.device):
            torch.cuda.reset_peak_memory_stats(self.device)
        grad_accum = max(int(self.training_config.get("gradient_accumulation_steps", 1) or 1), 1)
        limit = self.training_config.get("limit_train_batches", None)
        sums = {"loss": 0.0, "temperature_loss": 0.0, "humidity_loss": 0.0}
        seen = 0
        max_steps = len(loader) if limit is None else min(len(loader), int(limit))
        pbar = tqdm(loader, desc=f"Train {epoch + 1}/{epochs}", disable=not self.is_main)
        self.optimizer.zero_grad(set_to_none=True)
        timer = StepTimer()
        for step, batch in enumerate(pbar):
            if limit is not None and step >= int(limit):
                break
            timer.mark_batch_ready()
            cycle_start = (step // grad_accum) * grad_accum
            cycle_end = min(cycle_start + grad_accum, max_steps)
            accum_denom = max(cycle_end - cycle_start, 1)
            should_step = ((step + 1) % grad_accum == 0) or ((step + 1) >= max_steps)
            batch = self._move_batch_target(batch)
            with self._maybe_no_sync(should_sync=should_step):
                forward_start = time.perf_counter()
                with autocast(self.device_type, dtype=self.precision_type):
                    pred = self.model(batch)
                    losses = self.loss_fn(pred, batch["target"])
                    loss = losses["total_loss"] / accum_denom
                timer.mark_forward_done(forward_start)
                if not torch.isfinite(loss.detach()):
                    self.nan_or_inf_seen = True
                    raise FloatingPointError(f"Non-finite retrieval loss at epoch={epoch}, step={step}: {float(loss.detach().cpu())}")
                backward_start = time.perf_counter()
                use_scaler = self.scaler is not None and self.precision_type is torch.float16
                if use_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                timer.mark_backward_done(backward_start)
            optimizer_start = time.perf_counter()
            if should_step:
                max_norm = float(self.training_config.get("max_grad_norm", self.config.get("training", {}).get("grad_clip_norm", 0.0)) or 0.0)
                if max_norm > 0:
                    if use_scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                if use_scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                if self.scheduler is not None and getattr(self, "scheduler_step_unit", "epoch") == "step":
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
            timer.mark_optimizer_done(optimizer_start)
            step_time = timer.finalize()
            bsz = int(batch["target"].shape[0])
            seen += bsz
            sums["loss"] += float(losses["total_loss"].detach()) * bsz
            sums["temperature_loss"] += float(losses["temperature_loss"].detach()) * bsz
            sums["humidity_loss"] += float(losses["humidity_loss"].detach()) * bsz
            self._log_performance_step(epoch, step, bsz, step_time, timer, losses)
            if self.is_main and (step + 1) % self.log_every_n_steps == 0:
                pbar.set_postfix(loss=sums["loss"] / max(seen, 1), sps=f"{bsz / max(step_time, 1e-12):.2f}")
        reduced, reduced_seen = self._all_reduce_sums(sums, seen)
        return self._average_sums("train", reduced, reduced_seen)

    @torch.inference_mode()
    def validate(self, loader, epoch, epochs):
        """执行一个验证 epoch，返回跨 rank 平均后的验证 loss 与诊断指标。"""
        self.model.eval()
        sums: Dict[str, float] = {"loss": 0.0, "temperature_loss": 0.0, "humidity_loss": 0.0}
        seen = 0
        limit = self.training_config.get("limit_val_batches", None)
        pbar = tqdm(loader, desc=f"Val {epoch + 1}/{epochs}", disable=not self.is_main)
        for step, batch in enumerate(pbar):
            if limit is not None and step >= int(limit):
                break
            batch = self._move_batch_target(batch)
            with autocast(self.device_type, dtype=self.precision_type):
                pred = self.model(batch)
                losses = self.loss_fn(pred, batch["target"])
            bsz = int(batch["target"].shape[0])
            seen += bsz
            sums["loss"] += float(losses["total_loss"].detach()) * bsz
            sums["temperature_loss"] += float(losses["temperature_loss"].detach()) * bsz
            sums["humidity_loss"] += float(losses["humidity_loss"].detach()) * bsz
            metric = retrieval_metrics(pred.detach(), batch["target"].detach(), batch.get("pressure_levels"))
            for key, value in metric.items():
                sums[key] = sums.get(key, 0.0) + float(value) * bsz
        reduced, reduced_seen = self._all_reduce_sums(sums, seen)
        if reduced_seen == 0:
            return {"val/loss": float("inf")}
        out = self._average_sums("val", reduced, reduced_seen)
        self._append_metrics_csv(epoch, out)
        return out
