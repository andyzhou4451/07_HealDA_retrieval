#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于推荐 batch 配置的 LR/weight_decay/warmup/grad_clip 短跑 sweep。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "h100_tuning"


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _config_name_from_base(base_config: str) -> str:
    p = Path(base_config)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        p = PROJECT_ROOT / "configs" / "train_h100_80gb_single_gpu.yaml"
    try:
        return str(p.resolve().relative_to((PROJECT_ROOT / "configs").resolve()))
    except ValueError:
        return "train_h100_80gb_single_gpu.yaml"


def _load_recommended(path: Path) -> tuple[str, list[str], dict[str, Any]]:
    if not path.exists():
        return "train_h100_80gb_single_gpu.yaml", [], {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _config_name_from_base(data.get("base_config", "configs/train_h100_80gb_single_gpu.yaml")), list(data.get("overrides", [])), dict(data.get("recommended", {}))


def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return 10 ** rng.uniform(math.log10(lo), math.log10(hi))


def _parse_epoch_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {}
    row = rows[-1]
    return {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}


def _run_trial(config_name: str, base_overrides: list[str], trial_dir: Path, trial: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        lr = trial["lr"]
        stability = 1.0 if 3e-5 <= lr <= 8e-4 else 0.0
        score = 1.0 / (abs(math.log10(lr) - math.log10(1e-4)) + 1.0) + stability
        return {**trial, "returncode": 0, "oom": False, "nan_or_inf": False, "val/overall_rmse": 2.0 - score, "samples_per_second": 1.0}
    trial_dir.mkdir(parents=True, exist_ok=True)
    overrides = [
        *base_overrides,
        "hardware.single_gpu=true",
        "training.single_gpu=true",
        "training.device=cuda",
        f"hardware.device={args.device}",
        f"optimizer.lr={trial['lr']}",
        f"training.lr={trial['lr']}",
        f"optimizer.weight_decay={trial['weight_decay']}",
        f"training.weight_decay={trial['weight_decay']}",
        f"scheduler.warmup_steps={trial['warmup_steps']}",
        f"training.warmup_steps={trial['warmup_steps']}",
        f"training.max_grad_norm={trial['grad_clip_norm'] if trial['grad_clip_norm'] is not None else 0.0}",
        f"training.limit_train_batches={args.short_run_steps}",
        "training.limit_val_batches=4",
        "training.epochs=1",
        f"training.performance_log_dir={trial_dir / 'logs'}",
        f"hydra.run.dir={trial_dir / 'hydra'}",
    ]
    cmd = [sys.executable, str(PROJECT_ROOT / "main.py"), f"--config-name={config_name}", *overrides]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (trial_dir / "run.log").write_text(proc.stdout, encoding="utf-8")
    metrics = _parse_epoch_metrics(trial_dir / "logs" / "epoch_metrics.jsonl")
    oom = "out of memory" in proc.stdout.lower()
    nan_or_inf = "nan" in proc.stdout.lower() or "inf" in proc.stdout.lower()
    return {**trial, **metrics, "returncode": int(proc.returncode), "oom": bool(oom), "nan_or_inf": bool(nan_or_inf)}


def main() -> None:
    parser = argparse.ArgumentParser(description="H100 单卡 LR/WD/warmup/clip 短跑 sweep")
    parser.add_argument("--config", default=str(OUT_DIR / "recommended_config.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-trials", type=int, default=12)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--max-lr", type=float, default=3e-3)
    parser.add_argument("--weight-decays", default="0.0,1e-5,3e-5,1e-4,3e-4,1e-3")
    parser.add_argument("--warmups", default="0,500,1000,2000,4000")
    parser.add_argument("--grad-clips", default="none,1.0,5.0")
    parser.add_argument("--short-run-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_name, base_overrides, recommended = _load_recommended(Path(args.config))
    rng = random.Random(args.seed)
    weight_decays = [float(x) for x in args.weight_decays.split(",") if x.strip()]
    warmups = [int(x) for x in args.warmups.split(",") if x.strip()]
    clips: list[float | None] = []
    for x in args.grad_clips.split(","):
        x = x.strip().lower()
        clips.append(None if x in {"none", "null"} else float(x))
    rows: list[dict[str, Any]] = []
    for i in range(args.num_trials):
        trial = {
            "lr": _log_uniform(rng, args.min_lr, args.max_lr),
            "weight_decay": rng.choice(weight_decays),
            "warmup_steps": rng.choice(warmups),
            "grad_clip_norm": rng.choice(clips),
        }
        trial_dir = OUT_DIR / "hparam_trials" / f"trial_{i:04d}"
        print(f"[hparam {i:04d}] {trial}", flush=True)
        rows.append(_run_trial(config_name, base_overrides, trial_dir, trial, args))
    stable = [r for r in rows if r.get("returncode", 1) == 0 and not r.get("oom") and not r.get("nan_or_inf")]
    primary = "val/overall_rmse"
    if stable and primary in stable[0]:
        best = min(stable, key=lambda r: float(r.get(primary, float("inf"))))
    else:
        best = stable[0] if stable else (rows[0] if rows else {})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with (OUT_DIR / "hparam_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "hparam_sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    # 更新 recommended_config.yaml，保留 batch sweep 的推荐项并加入 hparam 结果。
    rec_path = OUT_DIR / "recommended_config.yaml"
    data = yaml.safe_load(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {"base_config": "configs/train_h100_80gb_single_gpu.yaml", "overrides": [], "recommended": recommended}
    data.setdefault("recommended", {}).update({
        "lr": best.get("lr", 1e-4),
        "weight_decay": best.get("weight_decay", 3e-4),
        "warmup_steps": best.get("warmup_steps", 1000),
        "grad_clip_norm": best.get("grad_clip_norm", 1.0),
    })
    data["hparam_sweep_best"] = best
    data.setdefault("overrides", [])
    data["overrides"].extend([
        f"optimizer.lr={data['recommended']['lr']}",
        f"training.lr={data['recommended']['lr']}",
        f"optimizer.weight_decay={data['recommended']['weight_decay']}",
        f"training.weight_decay={data['recommended']['weight_decay']}",
        f"scheduler.warmup_steps={data['recommended']['warmup_steps']}",
        f"training.warmup_steps={data['recommended']['warmup_steps']}",
        f"training.max_grad_norm={data['recommended']['grad_clip_norm'] if data['recommended']['grad_clip_norm'] is not None else 0.0}",
    ])
    rec_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"updated_recommended_config={rec_path}")


if __name__ == "__main__":
    main()
