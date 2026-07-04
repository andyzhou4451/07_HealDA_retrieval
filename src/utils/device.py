# -*- coding: utf-8 -*-
"""统一设备与分布式工具。

该模块现在以 NVIDIA GPU/CUDA/H100 为默认运行路径，同时保留旧 XiChen NPU
环境的兼容入口。用户可以在配置中写 ``cuda`` 或 ``gpu``；二者都会规范化为
``cuda``。只有明确传入 ``npu`` 时才尝试导入 ``torch_npu``。
"""

from __future__ import annotations

import importlib.util
import os
import random
from contextlib import nullcontext
from typing import Any, Callable, Literal

import numpy as np
import torch

DeviceType = Literal["cuda", "gpu", "npu", "cpu"]


def normalize_device_type(device_type: str | None) -> Literal["cuda", "npu", "cpu"]:
    """把用户配置中的 ``gpu``/``cuda:N`` 等写法统一成内部设备名。"""
    value = str(device_type or "cuda").strip().lower()
    if value.startswith("cuda") or value == "gpu":
        return "cuda"
    if value.startswith("npu"):
        return "npu"
    if value == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device type {device_type!r}; expected cuda/gpu/npu/cpu")


def is_npu_available() -> bool:
    """检测 torch-npu 是否安装且 NPU 可用；CUDA 环境不会强制导入 NPU 包。"""
    spec = importlib.util.find_spec("torch_npu")
    if spec is None:
        return False
    try:
        import torch_npu  # type: ignore

        return bool(torch_npu.is_available())
    except (ImportError, AttributeError):
        return False


def get_device_type() -> Literal["cuda", "npu"]:
    """自动选择可用加速器；默认优先 CUDA/H100，其次才兼容旧 NPU。"""
    if torch.cuda.is_available():
        return "cuda"
    if is_npu_available():
        return "npu"
    raise RuntimeError("No accelerator available: neither CUDA torch nor torch-npu is available")


def configure_accelerator_performance(config: Any | None = None, device_type: str = "cuda", log_fn: Callable[[str], None] | None = None) -> None:
    """设置 H100 友好的 PyTorch 性能开关。

    默认启用 TF32 和 high float32 matmul precision；确定性训练可通过
    ``training.deterministic=true`` 关闭 cudnn benchmark 并启用确定性算法。
    """
    device_type = normalize_device_type(device_type)
    if device_type != "cuda" or not torch.cuda.is_available():
        return

    training = config.get("training", {}) if config is not None and hasattr(config, "get") else {}
    deterministic = bool(training.get("deterministic", False)) if hasattr(training, "get") else False
    benchmark = bool(training.get("cudnn_benchmark", not deterministic)) if hasattr(training, "get") else (not deterministic)
    matmul_precision = str(training.get("float32_matmul_precision", "high")) if hasattr(training, "get") else "high"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = benchmark
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        torch.set_float32_matmul_precision(matmul_precision)
    except Exception as exc:  # pragma: no cover - 老版本 torch 可能没有该接口。
        if log_fn is not None:
            log_fn(f"torch.set_float32_matmul_precision failed safely: {exc}")


def init_distributed(device_type: str, local_rank: int) -> None:
    """初始化 DDP 进程组，并保证每个进程只绑定一张本地 GPU/NPU。"""
    device_type = normalize_device_type(device_type)
    if device_type == "npu":
        import torch_npu  # type: ignore  # noqa: F401
        from torch_npu.distributed import is_hccl_available  # type: ignore

        if not is_hccl_available():
            raise RuntimeError("HCCL is not available; check CANN, HCCL driver and torch-npu/CANN version match")
        torch.npu.set_device(local_rank)
        backend = "hccl"
    elif device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device=cuda/gpu was requested but torch.cuda.is_available() is False")
        visible = torch.cuda.device_count()
        if local_rank >= visible:
            raise RuntimeError(f"LOCAL_RANK={local_rank} but only {visible} CUDA devices are visible")
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    elif device_type == "cpu":
        backend = "gloo"
    else:  # pragma: no cover - normalize_device_type 已经兜底。
        raise ValueError(f"Unsupported distributed device_type={device_type!r}")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend, init_method="env://")


def get_grad_scaler(device_type: str, dtype: torch.dtype | None = None):
    """只在 FP16 混合精度下创建 GradScaler；BF16/H100 默认不需要 loss scaling。"""
    device_type = normalize_device_type(device_type)
    if dtype is not torch.float16:
        return None
    if device_type == "npu":
        return torch.npu.amp.GradScaler()
    if device_type == "cuda":
        grad_scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
        if grad_scaler is not None:
            try:
                return grad_scaler("cuda")
            except TypeError:
                pass
        return torch.cuda.amp.GradScaler()
    return None


def autocast(device_type: str, dtype: torch.dtype = torch.bfloat16):
    """返回适配 CUDA/NPU/CPU 的 autocast 上下文；FP32 时退化为 no-op。"""
    device_type = normalize_device_type(device_type)
    if dtype is torch.float32 or device_type == "cpu":
        return nullcontext()
    if device_type == "npu":
        return torch.npu.amp.autocast(dtype=dtype)
    if device_type == "cuda":
        amp_autocast = getattr(getattr(torch, "amp", None), "autocast", None)
        if amp_autocast is not None:
            try:
                return amp_autocast("cuda", dtype=dtype)
            except TypeError:
                pass
        return torch.cuda.amp.autocast(dtype=dtype)
    return nullcontext()


def manual_seed(device_type: str, seed: int, rank: int = 0) -> None:
    """设置 Python、NumPy、PyTorch 与加速器随机种子；rank 偏移避免 worker 序列重复。"""
    device_type = normalize_device_type(device_type)
    final_seed = int(seed) + int(rank)
    os.environ["PYTHONHASHSEED"] = str(final_seed)
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    if device_type == "npu":
        torch.npu.manual_seed(final_seed)
    elif device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(final_seed)
        torch.cuda.manual_seed_all(final_seed)


def get_device(device_type: str, local_rank: int = 0) -> torch.device:
    """返回当前进程绑定的 torch.device，并支持 ``gpu`` 作为 ``cuda`` 别名。"""
    device_type = normalize_device_type(device_type)
    if device_type == "npu":
        return torch.device(f"npu:{local_rank}")
    if device_type == "cuda":
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def empty_cache(device_type: str | None = None) -> None:
    """按设备类型安全释放加速器缓存；CUDA 环境不会访问 ``torch.npu``。"""
    device_type = normalize_device_type(device_type or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()


def synchronize(device_type: str | None = None) -> None:
    """按设备类型安全同步；仅在确有对应加速器时执行。"""
    device_type = normalize_device_type(device_type or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def destroy_process_group() -> None:
    """安全销毁分布式进程组，异常退出和正常退出都可重复调用。"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
