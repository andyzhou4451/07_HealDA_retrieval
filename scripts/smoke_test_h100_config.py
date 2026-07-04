#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""H100 单卡最终配置 smoke test。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="单卡 H100 配置小步数 smoke test")
    parser.add_argument("--config", default="configs/train_h100_80gb_single_gpu.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="只检查命令构造，不启动训练")
    args = parser.parse_args()
    perf_dir = PROJECT_ROOT / "outputs" / "logs"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "train.py"),
        "--config",
        args.config,
        "--device",
        args.device,
        "debug=true",
        "model.model_size=tiny",
        "datamodule.batch_size=1",
        "datamodule.num_workers=0",
        "training.compile.enabled=false",
        "precision.compile.enabled=false",
        f"training.limit_train_batches={args.max_steps}",
        "training.limit_val_batches=1",
        "training.epochs=1",
        f"training.performance_log_dir={perf_dir}",
    ]
    if args.dry_run:
        print(json.dumps({"cmd": cmd}, ensure_ascii=False, indent=2))
        return
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    required = [perf_dir / "performance.jsonl", perf_dir / "epoch_metrics.jsonl"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"smoke test finished but required logs are missing: {missing}")
    print("smoke_status=ok")


if __name__ == "__main__":
    main()
