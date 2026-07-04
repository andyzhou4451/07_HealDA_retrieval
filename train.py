#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""兼容用户命令的训练入口封装。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


def _config_name_from_path(path: str | Path) -> str:
    """把 configs/xxx.yaml 转成 Hydra --config-name 可识别的相对名称。"""
    p = Path(path)
    if not p.exists():
        fallback = PROJECT_ROOT / "configs" / "train_h100_80gb_single_gpu.yaml"
        return str(fallback.relative_to(PROJECT_ROOT / "configs"))
    try:
        rel = p.resolve().relative_to((PROJECT_ROOT / "configs").resolve())
        return str(rel)
    except ValueError:
        return "train_h100_80gb_single_gpu.yaml"


def _load_external_config(path: Path) -> tuple[str, list[str]]:
    """读取 tuner 输出的 recommended_config.yaml，返回 base_config 与 overrides。"""
    if not path.exists():
        return "train_h100_80gb_single_gpu.yaml", []
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_config = data.get("base_config", "configs/train_h100_80gb_single_gpu.yaml")
    overrides = list(data.get("overrides", []))
    if "recommended" in data and isinstance(data["recommended"], dict):
        rec = data["recommended"]
        mapping = {
            "batch_size": "datamodule.batch_size",
            "num_workers": "datamodule.num_workers",
            "prefetch_factor": "datamodule.prefetch_factor",
            "pin_memory": "datamodule.pin_memory",
            "persistent_workers": "datamodule.persistent_workers",
            "precision": "training.precision.type",
            "patch_size": "model.net.patch_size",
            "compile": "training.compile.enabled",
            "channels_last": "model.net.channels_last",
            "lr": "optimizer.lr",
            "weight_decay": "optimizer.weight_decay",
            "warmup_steps": "scheduler.warmup_steps",
            "grad_clip_norm": "training.max_grad_norm",
        }
        for key, override_key in mapping.items():
            if key not in rec:
                continue
            value = rec[key]
            if isinstance(value, list):
                value = "[" + ",".join(str(v) for v in value) + "]"
            elif isinstance(value, bool):
                value = str(value).lower()
            overrides.append(f"{override_key}={value}")
    return _config_name_from_path(base_config), overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="XiChen/HealDA H100 单卡训练入口")
    parser.add_argument("--config", default="configs/train_h100_80gb_single_gpu.yaml", help="configs 下的 Hydra 配置，或 tuner 生成的 recommended_config.yaml")
    parser.add_argument("--device", default="cuda:0", help="目标设备；单卡 H100 默认 cuda:0")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="继续透传给 Hydra 的 key=value 覆盖项")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists() and not str(config_path).startswith(str(PROJECT_ROOT / "configs")) and "outputs" in config_path.parts:
        config_name, loaded_overrides = _load_external_config(config_path)
    else:
        config_name = _config_name_from_path(config_path)
        loaded_overrides = []
    device_type = "cuda" if str(args.device).startswith("cuda") or str(args.device) == "gpu" else str(args.device)
    overrides = [
        "hardware.single_gpu=true",
        "training.single_gpu=true",
        f"hardware.device={args.device}",
        f"training.device={device_type}",
    ]
    overrides.extend(loaded_overrides)
    overrides.extend([x for x in args.overrides if x != "--"])
    cmd = [sys.executable, str(PROJECT_ROOT / "main.py"), f"--config-name={config_name}", *overrides]
    raise SystemExit(subprocess.call(cmd, cwd=PROJECT_ROOT))


if __name__ == "__main__":
    main()
