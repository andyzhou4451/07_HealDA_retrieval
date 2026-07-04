# -*- coding: utf-8 -*-
"""H100 单卡训练性能日志与显存工具。"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch


def now_sec() -> float:
    """返回高精度墙钟时间，训练 profiler 用它避免强制 CUDA 同步。"""
    return time.perf_counter()


def safe_float(value: Any, default: float | None = None) -> float | None:
    """把张量、NumPy 标量和普通数值安全转为 JSON 可写 float。"""
    try:
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().item()
        return float(value)
    except Exception:
        return default


def cuda_memory_stats(device: torch.device | str | None = None) -> dict[str, float | str | None]:
    """采集当前 CUDA 显存峰值；CPU 环境返回空值但保持字段完整。"""
    if not torch.cuda.is_available():
        return {
            "gpu_name": None,
            "cuda_version": torch.version.cuda,
            "max_memory_allocated_gb": 0.0,
            "max_memory_reserved_gb": 0.0,
            "memory_utilization": 0.0,
            "free_memory_gb": None,
            "total_memory_gb": None,
        }
    dev = torch.device(device or torch.cuda.current_device())
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    total = float(props.total_memory) / 1024**3
    reserved = float(torch.cuda.max_memory_reserved(index)) / 1024**3
    allocated = float(torch.cuda.max_memory_allocated(index)) / 1024**3
    mem_get_info = getattr(torch.cuda, "mem_get_info", None)
    free = None
    if mem_get_info is not None:
        try:
            free_bytes, _ = mem_get_info(index)
            free = float(free_bytes) / 1024**3
        except Exception:
            free = None
    return {
        "gpu_name": torch.cuda.get_device_name(index),
        "cuda_version": torch.version.cuda,
        "max_memory_allocated_gb": allocated,
        "max_memory_reserved_gb": reserved,
        "memory_utilization": reserved / max(total, 1e-12),
        "free_memory_gb": free,
        "total_memory_gb": total,
    }


def query_nvidia_smi_utilization(gpu_index: int = 0) -> float | None:
    """通过 nvidia-smi 查询 GPU 利用率；失败时返回 None，不影响训练。"""
    try:
        cmd = [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=2).strip().splitlines()
        return float(out[0]) if out else None
    except Exception:
        return None


class JsonlWriter:
    """极简 JSONL 写入器；每次写入 flush，便于 Slurm 中途排查。"""

    def __init__(self, path: str | os.PathLike[str] | None, enabled: bool = True) -> None:
        self.path = Path(path) if path is not None else None
        self.enabled = bool(enabled) and self.path is not None
        self._fh = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        if not self.enabled or self._fh is None:
            return
        clean = {}
        for key, value in row.items():
            if isinstance(value, torch.Tensor):
                value = safe_float(value)
            clean[key] = value
        self._fh.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


@dataclass
class StepTimer:
    """记录 data/forward/backward/optimizer 各阶段耗时。"""

    last_end: float = field(default_factory=now_sec)
    data_time: float = 0.0
    forward_time: float = 0.0
    backward_time: float = 0.0
    optimizer_time: float = 0.0
    step_start: float = 0.0

    def mark_batch_ready(self) -> None:
        now = now_sec()
        self.data_time = now - self.last_end
        self.step_start = now

    def mark_forward_done(self, start: float) -> None:
        self.forward_time = now_sec() - start

    def mark_backward_done(self, start: float) -> None:
        self.backward_time = now_sec() - start

    def mark_optimizer_done(self, start: float) -> None:
        self.optimizer_time = now_sec() - start

    def finalize(self) -> float:
        step_time = now_sec() - self.step_start
        self.last_end = now_sec()
        return step_time


def summarize_step_times(step_times: Iterable[float]) -> dict[str, float]:
    """汇总 step time 的 mean/p50/p90，空列表时返回 0。"""
    values = [float(x) for x in step_times]
    if not values:
        return {"step_time_mean": 0.0, "step_time_p50": 0.0, "step_time_p90": 0.0}
    sorted_values = sorted(values)
    p90_index = min(len(sorted_values) - 1, int(round(0.9 * (len(sorted_values) - 1))))
    return {
        "step_time_mean": float(statistics.mean(values)),
        "step_time_p50": float(statistics.median(values)),
        "step_time_p90": float(sorted_values[p90_index]),
    }
