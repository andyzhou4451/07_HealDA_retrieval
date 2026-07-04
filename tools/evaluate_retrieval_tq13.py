#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate a HealDA-XiChen T/Q retrieval checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

from src.utils.device import normalize_device_type

from src.metrics.retrieval_metrics import retrieval_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit_batches", type=int, default=0)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    overrides = [
        "datamodule=retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml",
        "model=retrieval/healda_xichen_tq13.yaml",
        "pipeline=retrieval/trainer.yaml",
        "training=retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml",
        "loss_fn=retrieval_tq_huber.yaml",
        "paths=retrieval_public02.yaml",
    ] + args.overrides
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(args.config_dir)):
        cfg = compose(config_name="train.yaml", overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    requested_device = normalize_device_type(args.device)
    device = torch.device(requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu")
    datamodule = hydra.utils.instantiate(cfg.datamodule, distributed=False, _recursive_=False)
    loader = datamodule.val_dataloader() if args.split == "val" else datamodule.test_dataloader()
    model = hydra.utils.instantiate(cfg.model.net, _recursive_=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    accum = {}; seen = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="evaluate")):
            if args.limit_batches and step >= args.limit_batches:
                break
            batch["target"] = batch["target"].to(device)
            pred = model(batch)
            m = retrieval_metrics(pred, batch["target"], batch.get("pressure_levels"))
            b = batch["target"].shape[0]
            seen += b
            for k, v in m.items():
                accum[k] = accum.get(k, 0.0) + v * b
    if seen == 0:
        raise RuntimeError("No evaluation samples were loaded")
    for k in sorted(accum):
        print(f"{k}: {accum[k] / seen:.6g}")


if __name__ == "__main__":
    main()
