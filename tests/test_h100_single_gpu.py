# -*- coding: utf-8 -*-
"""H100 单卡调参脚本轻量测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_tune_h100_dry_run_generates_recommended_config() -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "tune_h100_single_gpu.py"),
        "--dry-run",
        "--max-trials",
        "2",
        "--batch-sizes",
        "1,2",
        "--patch-sizes",
        "8x8",
        "--num-workers",
        "4",
        "--prefetch-factors",
        "2",
        "--precisions",
        "bf16",
        "--compile-options",
        "false",
        "--channels-last-options",
        "true",
    ]
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)
    assert (PROJECT_ROOT / "outputs" / "h100_tuning" / "batch_size_sweep.csv").exists()
    assert (PROJECT_ROOT / "outputs" / "h100_tuning" / "recommended_config.yaml").exists()


def test_smoke_command_dry_run() -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "smoke_test_h100_config.py"),
        "--dry-run",
        "--max_steps",
        "1",
    ]
    out = subprocess.check_output(cmd, cwd=PROJECT_ROOT, text=True)
    payload = json.loads(out)
    assert "train.py" in " ".join(payload["cmd"])
