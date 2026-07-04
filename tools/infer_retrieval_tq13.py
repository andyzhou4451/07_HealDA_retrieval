#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run inference for one or more HealDA-XiChen retrieval batches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

from src.utils.device import normalize_device_type


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--limit_batches", type=int, default=1)
    parser.add_argument("--as_profile", action="store_true", help="save [B,2,13,181,360] instead of [B,26,181,360]")
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
    dm = hydra.utils.instantiate(cfg.datamodule, distributed=False, _recursive_=False)
    loader = dm.val_dataloader() if args.split == "val" else dm.test_dataloader()
    model = hydra.utils.instantiate(cfg.model.net, _recursive_=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    preds = []; times = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="infer")):
            if args.limit_batches and step >= args.limit_batches:
                break
            pred = model(batch, as_profile=args.as_profile).detach().cpu().numpy()
            preds.append(pred)
            times.extend(batch["target_time"])
    if not preds:
        raise RuntimeError("No inference samples were loaded")
    arr = np.concatenate(preds, axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez(args.output, prediction=arr, target_time=np.asarray(times), output_shape=str(arr.shape))
    print(f"saved {args.output}, prediction shape={arr.shape}")


if __name__ == "__main__":
    main()
