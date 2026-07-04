#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单张 H100 80GB batch/dataloader/precision/patch 自动短跑搜索。"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_h100_80gb_single_gpu.yaml"
OUT_DIR = PROJECT_ROOT / "outputs" / "h100_tuning"


def _parse_csv_ints(value: str, default: list[int]) -> list[int]:
    if not value:
        return default
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_strs(value: str, default: list[str]) -> list[str]:
    if not value:
        return default
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_patch_sizes(value: str) -> list[list[int]]:
    if not value:
        return [[6, 6], [8, 8], [10, 10], [12, 12]]
    out: list[list[int]] = []
    for item in value.split(","):
        item = item.strip().lower().replace("x", " ")
        if not item:
            continue
        parts = [int(x) for x in item.split()]
        if len(parts) == 1:
            parts = [parts[0], parts[0]]
        out.append(parts[:2])
    return out


def _config_name_from_path(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        p = DEFAULT_CONFIG
    try:
        return str(p.resolve().relative_to((PROJECT_ROOT / "configs").resolve()))
    except ValueError:
        return "train_h100_80gb_single_gpu.yaml"


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _trial_overrides(trial_dir: Path, config: dict[str, Any], args: argparse.Namespace) -> list[str]:
    ph, pw = config["patch_size"]
    overrides = [
        "hardware.single_gpu=true",
        "training.single_gpu=true",
        "training.device=cuda",
        f"hardware.device={args.device}",
        "training.gradient_accumulation_steps=1",
        f"training.limit_train_batches={args.warmup_steps + args.benchmark_steps}",
        "training.limit_val_batches=1",
        "training.epochs=1",
        f"datamodule.batch_size={config['batch_size']}",
        f"datamodule.num_workers={config['num_workers']}",
        f"datamodule.prefetch_factor={config['prefetch_factor']}",
        f"datamodule.pin_memory={_bool_text(config['pin_memory'])}",
        f"datamodule.persistent_workers={_bool_text(config['persistent_workers'])}",
        f"datamodule.data.max_points_per_sensor={args.max_points_per_sensor}",
        f"training.precision.type={config['precision']}",
        f"precision.amp_dtype={config['precision']}",
        f"training.compile.enabled={_bool_text(config['compile'])}",
        f"precision.compile.enabled={_bool_text(config['compile'])}",
        f"model.net.channels_last={_bool_text(config['channels_last'])}",
        f"training.channels_last={_bool_text(config['channels_last'])}",
        f"model.net.patch_size=[{ph},{pw}]",
        f"training.performance_log_dir={trial_dir / 'logs'}",
        f"hydra.run.dir={trial_dir / 'hydra'}",
    ]
    return overrides


def _read_performance_jsonl(path: Path, warmup_steps: int) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    measured = rows[warmup_steps:] if len(rows) > warmup_steps else rows
    if not measured:
        return {}
    step_times = [float(r.get("step_time_sec", 0.0)) for r in measured]
    sps = [float(r.get("samples_per_second", 0.0)) for r in measured]
    mem_reserved = [float(r.get("max_memory_reserved_gb", 0.0)) for r in rows]
    mem_alloc = [float(r.get("max_memory_allocated_gb", 0.0)) for r in rows]
    mem_util = [float(r.get("memory_utilization", 0.0)) for r in rows]
    nan_or_inf = any(bool(r.get("nan_or_inf", False)) for r in rows)
    p90_idx = min(len(step_times) - 1, int(round(0.9 * (len(step_times) - 1))))
    return {
        "step_time_mean": sum(step_times) / max(len(step_times), 1),
        "step_time_p50": sorted(step_times)[len(step_times) // 2],
        "step_time_p90": sorted(step_times)[p90_idx],
        "samples_per_second": sum(sps) / max(len(sps), 1),
        "max_memory_reserved_gb": max(mem_reserved) if mem_reserved else 0.0,
        "max_memory_allocated_gb": max(mem_alloc) if mem_alloc else 0.0,
        "memory_utilization": max(mem_util) if mem_util else 0.0,
        "nan_or_inf": nan_or_inf,
        "gpu_name": rows[-1].get("gpu_name"),
        "cuda_version": rows[-1].get("cuda_version"),
        "torch_version": rows[-1].get("torch_version"),
    }


def _run_trial(config_name: str, trial_dir: Path, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        bs = config["batch_size"]
        patch_area = config["patch_size"][0] * config["patch_size"][1]
        speed = 1.0 + math.log2(bs + 1) * 0.8 + patch_area / 100.0
        mem = min(79.0, 12.0 + bs * 7.0 + 900.0 / patch_area)
        return {**config, "oom": mem > 75.0, "nan_or_inf": False, "returncode": 0, "samples_per_second": speed, "max_memory_reserved_gb": mem, "max_memory_allocated_gb": mem * 0.85, "memory_utilization": mem / 80.0, "step_time_mean": bs / max(speed, 1e-6), "step_time_p50": bs / max(speed, 1e-6), "step_time_p90": bs / max(speed, 1e-6) * 1.1}
    trial_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(PROJECT_ROOT / "main.py"), f"--config-name={config_name}", *_trial_overrides(trial_dir, config, args)]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    env.setdefault("OMP_NUM_THREADS", "4")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (trial_dir / "run.log").write_text(proc.stdout, encoding="utf-8")
    oom = "out of memory" in proc.stdout.lower() or "cuda error: out of memory" in proc.stdout.lower()
    perf = _read_performance_jsonl(trial_dir / "logs" / "performance.jsonl", args.warmup_steps)
    row = {**config, **perf, "oom": bool(oom), "returncode": int(proc.returncode)}
    row["nan_or_inf"] = bool(row.get("nan_or_inf", False)) or "nan" in proc.stdout.lower() or "inf" in proc.stdout.lower()
    return row


def _score_candidate(row: dict[str, Any], args: argparse.Namespace) -> float:
    if row.get("oom") or row.get("returncode", 1) != 0 or row.get("nan_or_inf"):
        return -1.0
    mem_util = float(row.get("memory_utilization", 0.0) or 0.0)
    if mem_util > args.max_memory_utilization:
        return -1.0
    return float(row.get("samples_per_second", 0.0) or 0.0)


def _write_results(rows: list[dict[str, Any]], recommended: dict[str, Any], args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with (OUT_DIR / "batch_size_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / "batch_size_sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    rec = {
        "base_config": "configs/train_h100_80gb_single_gpu.yaml",
        "generated_by": "scripts/tune_h100_single_gpu.py",
        "selection_rule": "highest samples_per_second among stable non-OOM trials under max_memory_utilization",
        "recommended": recommended,
        "overrides": [
            f"datamodule.batch_size={recommended.get('batch_size', 1)}",
            f"datamodule.num_workers={recommended.get('num_workers', 8)}",
            f"datamodule.prefetch_factor={recommended.get('prefetch_factor', 4)}",
            f"datamodule.pin_memory={_bool_text(recommended.get('pin_memory', True))}",
            f"datamodule.persistent_workers={_bool_text(recommended.get('persistent_workers', True))}",
            f"training.precision.type={recommended.get('precision', 'bf16')}",
            f"precision.amp_dtype={recommended.get('precision', 'bf16')}",
            f"model.net.patch_size=[{recommended.get('patch_size', [8, 8])[0]},{recommended.get('patch_size', [8, 8])[1]}]",
            f"training.compile.enabled={_bool_text(recommended.get('compile', False))}",
            f"precision.compile.enabled={_bool_text(recommended.get('compile', False))}",
            f"model.net.channels_last={_bool_text(recommended.get('channels_last', True))}",
            f"training.channels_last={_bool_text(recommended.get('channels_last', True))}",
        ],
    }
    (OUT_DIR / "recommended_config.yaml").write_text(yaml.safe_dump(rec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    lines = [
        "# H100 single GPU tuning summary",
        "",
        f"dry_run: {args.dry_run}",
        f"num_trials: {len(rows)}",
        f"recommended: `{recommended}`",
        "",
        "## Top stable trials",
    ]
    stable = [r for r in rows if _score_candidate(r, args) >= 0]
    stable = sorted(stable, key=lambda r: float(r.get("samples_per_second", 0.0) or 0.0), reverse=True)[:10]
    for r in stable:
        lines.append(f"- bs={r.get('batch_size')}, patch={r.get('patch_size')}, precision={r.get('precision')}, workers={r.get('num_workers')}, sps={float(r.get('samples_per_second', 0.0) or 0.0):.4f}, reserved={float(r.get('max_memory_reserved_gb', 0.0) or 0.0):.2f}GB")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="H100 80GB 单卡 batch/dataloader/precision 自动搜索")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Hydra 配置路径；缺失时自动使用 configs/train_h100_80gb_single_gpu.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--patch-sizes", default="8x8,10x10,12x12")
    parser.add_argument("--num-workers", default="4,8,12,16")
    parser.add_argument("--prefetch-factors", default="2,4,8")
    parser.add_argument("--precisions", default="bf16,fp16,fp32")
    parser.add_argument("--compile-options", default="false,true")
    parser.add_argument("--channels-last-options", default="true,false")
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--persistent-workers", action="store_true", default=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--benchmark-steps", type=int, default=30)
    parser.add_argument("--target-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-memory-utilization", type=float, default=0.94)
    parser.add_argument("--min-free-memory-gb", type=float, default=4.0)
    parser.add_argument("--max-points-per-sensor", type=int, default=100000)
    parser.add_argument("--max-trials", type=int, default=48, help="限制组合数量，避免一次 sweep 过长")
    parser.add_argument("--dry-run", action="store_true", help="不启动训练，生成 mock 结果以测试脚本与输出格式")
    args = parser.parse_args()

    config_name = _config_name_from_path(args.config)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_DIR.exists() and not args.dry_run:
        # 不删除根目录，只清理 trials，保留历史 csv/json 的覆盖行为清晰。
        shutil.rmtree(OUT_DIR / "trials", ignore_errors=True)
    batches = _parse_csv_ints(args.batch_sizes, [1, 2, 4, 8, 16, 32])
    patches = _parse_patch_sizes(args.patch_sizes)
    workers = _parse_csv_ints(args.num_workers, [4, 8, 12, 16])
    prefetch = _parse_csv_ints(args.prefetch_factors, [2, 4, 8])
    precisions = _parse_csv_strs(args.precisions, ["bf16", "fp16", "fp32"])
    compile_opts = [x.lower() == "true" for x in _parse_csv_strs(args.compile_options, ["false", "true"])]
    channel_opts = [x.lower() == "true" for x in _parse_csv_strs(args.channels_last_options, ["true", "false"])]

    rows: list[dict[str, Any]] = []
    combinations = itertools.product(batches, patches, workers, prefetch, precisions, compile_opts, channel_opts)
    for trial_id, (bs, patch, nw, pf, prec, comp, chlast) in enumerate(combinations):
        if trial_id >= args.max_trials:
            break
        cfg = {"batch_size": bs, "effective_batch_size": bs, "patch_size": patch, "num_workers": nw, "prefetch_factor": pf, "pin_memory": args.pin_memory, "persistent_workers": args.persistent_workers, "precision": prec, "compile": comp, "channels_last": chlast}
        trial_dir = OUT_DIR / "trials" / f"trial_{trial_id:04d}"
        print(f"[trial {trial_id:04d}] {cfg}", flush=True)
        row = _run_trial(config_name, trial_dir, cfg, args)
        rows.append(row)
        print(f"    return={row.get('returncode')} oom={row.get('oom')} sps={row.get('samples_per_second')} reserved={row.get('max_memory_reserved_gb')}", flush=True)
    stable = [r for r in rows if _score_candidate(r, args) >= 0]
    if stable:
        best = max(stable, key=lambda r: float(r.get("samples_per_second", 0.0) or 0.0))
    else:
        best = max(rows, key=lambda r: float(r.get("samples_per_second", 0.0) or 0.0)) if rows else {"batch_size": 1, "patch_size": [8, 8], "precision": "bf16", "num_workers": 8, "prefetch_factor": 4, "pin_memory": True, "persistent_workers": True, "compile": False, "channels_last": True}
    _write_results(rows, best, args)
    print(f"recommended_config={OUT_DIR / 'recommended_config.yaml'}")


if __name__ == "__main__":
    main()
