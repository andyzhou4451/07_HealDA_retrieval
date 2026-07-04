# CLAUDE.md — XiChen (XiChenPaper)

> **Doc status**: This file is the canonical project doc. It **supersedes** the existing `AGENTS.md` (commit `48fbbe0`, 2026-05-04) and `README.md` — both are forecast-only snapshots and are out of date.
> **Working tree**: branch `dev`, heavy WIP, many uncommitted changes. Last verified against tree: **2026-06-18**.

Deep-learning weather forecasting + data-assimilation framework. PyTorch DDP + Hydra, supports Huawei Ascend NPU and NVIDIA GPU from a single code-path.

---

## 1. Project Overview

- **Domain**: medium-range global weather forecasting + observation-space data assimilation. Five sibling tasks share a Swin-V2 transformer backbone.
- **Backbone** (`src/layers/swin_attn.py`): cosine-attention Swin-V2 with continuous-relative-position-bias, optional `WindowCrossAttentionV2` for conditioning (lead-time, satellite context, gradients). Used by every model.
- **Inputs**: 69 channels (4 surface + 13 pressure levels × 5 vars), 1.0° lat/lon grid (`181 × 360`) by default. Output channels mirror inputs; per-task outputs add satellite brightness-temperatures, gradients, or compressed latents.
- **Hardware**: single code-path runs on either NPU (`torch_npu`) or GPU. Device is selected by `device: "cuda" | "gpu"` in `configs/train.yaml`; the DDP backend auto-selects (`hccl` for NPU, `nccl` for GPU) via `src/utils/device.py`.
- **Precision**: bf16 mixed precision by default; `GradScaler` from `src.utils.device.get_grad_scaler`.
- **Probabilistic**: every network outputs `(preds, log_var)` — log-variance is softplus-clipped to `[-10, 10]`. Default loss is CRPS-Gaussian.

---

## 2. Quick Start

> `requirements.txt` is currently **missing** from the working tree (see §10). Install packages manually until it's restored.

```bash
# Install (manual until requirements.txt returns)
pip install torch torch-npu==2.1.0 hydra-core hydra-colorlog omegaconf \
    torchmetrics==0.9.3 tensorboard tqdm pyyaml pyrootutils matplotlib seaborn \
    numpy scipy dask python-dateutil click timm einops
pip install -e .

# Required env vars (see configs/paths/default.yaml)
export DATA_DIR=/mnt/xichen/data/era5_lr_1p0deg
export SCALE_DIR=/mnt/xichen/data/normalized_mean_std
# Optional: PROJECT_ROOT, MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK

# Single-card (forecast, the default task)
bash scripts/example.sh

# Multi-card
bash scripts/example.sh --nproc 4

# Swap task via Hydra override (obs-operator example)
python -m torch.distributed.run --nproc_per_node=4 main.py \
    datamodule=obsoperator/atms \
    model=obsoperator/xichen_atms_obsoperator \
    pipeline=obsoperator/xichen_obsoperator \
    training=train_obsop
```

---

## 3. The Five Task Families

Each family has a paired `pipeline/`, `models/`, `datamodules/`, `configs/`, and `scripts/` subtree. They all consume the same `src/layers/` building blocks.

### 3.1 forecast — `XiChenForecast`

| Aspect | Value |
|---|---|
| Model class | `src.models.forecast.arch.XiChenForecast` |
| Pipeline trainer | `src.pipeline.forecast.trainer.ForecastTrainer` (+ `trainer_obconstraint.py` variant) |
| Datamodule | `src.datamodules.forecast.{state,obs}_datamodule.StateForecastDataModule` |
| Config groups | `datamodule=forecast/xichen_state_forecast[.obconstraint].yaml`, `model=forecast/xichen_state_forecast.yaml`, `pipeline=forecast/xichen_state_forecast.yaml` |
| Loss | `loss_fn/crps_gaussian.yaml` or `loss_fn/l1.yaml` |
| Lead-time conditioning | `lead_time_embed` injected via `WindowCrossAttentionV2` (`condition=True`) on the latent stack |
| Autoregressive | trainer rolls `iter_num` steps with `detach_iter_num` graph truncation; inference scripts under `inference/era5_*_forecast.py` |
| Scripts | `scripts/forecast/train/{pretrain_xichen_state.sh, finetune_xichen_state_ar{2,3,6,8,10,12,15}.sh, ...}`, `scripts/forecast/debug/...` |
| Checkpoints | `logs/<task_name>/runs/checkpoints/{best,latest}.ckpt` |

### 3.2 compression — `XiChenAutoEncoder`

| Aspect | Value |
|---|---|
| Model class | `src.models.compression.arch.XiChenAutoEncoder` |
| Pipeline trainer | `src.pipeline.compression.trainer.CompressionTrainer` |
| Datamodule | `src.datamodules.compression.state_datamodule.StateCompressionDataModule` |
| Config groups | `datamodule=compression/xichenae_lr.yaml`, `model=compression/xichenae_lr.yaml`, `pipeline=compression/xichenae.yaml` |
| Latent | `z_dim=69` with `quan_mlp` / `post_quan_mlp` quantizer pair, optional `ending_norm` (LayerNorm) |
| Scripts | `scripts/compression/train/train_xichenae_lr_{wnorm,wonorm}.sh`, `scripts/compression/debug/...` |
| Notes | `src/models/compression/arch_.py` is a legacy/extended variant; the active model is `arch.py` |

### 3.3 obsoperator — `XiChenObsOp` (atms / amsua / mhs / hrs4)

| Aspect | Value |
|---|---|
| Model class | `src.models.obsoperator.arch.XiChenObsOp` |
| Pipeline trainer | `src.pipeline.obsoperator.trainer.ObsOperatorTrainer` |
| Datamodule | `src.datamodules.obsoperator.{atms,amsua,mhs,hrs4}.npydatamodule.{ATMS,AMSUA,MHS,HRS4}DataModule` |
| Config groups | `datamodule=obsoperator/<sat>.yaml`, `model=obsoperator/xichen_<sat>_obsoperator.yaml`, `pipeline=obsoperator/xichen_obsoperator.yaml` |
| Inputs | ERA5 state + per-satellite auxiliary fields (cos(zenith), azimuth, scan/fov/orbit, satellite_height) + scan mask |
| Outputs | brightness temperatures `out_sat_vars` + observation-error log-variance, masked by `sat_mask` |
| Loss | CRPS-Gaussian on `tmbrs` (or L1) |
| Scripts | `scripts/obsoperator/{atms,amsua,mhs,hrs4}.sh`, `scripts/obsoperator/debug/{hrs4_bf16,hrs4_fp32}.sh` |

### 3.4 assimilate (cascade) — `Solver` + `XiChenDA`

| Aspect | Value |
|---|---|
| Model classes | `src.models.assimilate.fdvarsolver.cascade.Solver` + per-obs `src.models.assimilate.xichenda.arch.XiChenDA` |
| Pipeline trainer | `src.pipeline.assimilate.cascade.random_bg_trainer.RandomBgCascadeAssimTrainer` |
| Datamodule | `src.datamodules.assimilate.random_bg.npydatamodule.RandomBgAssimDataModule` |
| Config groups | `datamodule=assimilate/random_bg/atms.yaml` (single-obs) or `atms_amsua_mhs_hrs4_prepbufr_satwnd_ascat.yaml` (multi-obs), `model=assimilate/random_bg/{cascade,atms}.yaml`, `pipeline=assimilate/cascade/random_bg_trainer.yaml`, `training=cascade_da/random_bg_atms.yaml` |
| Solver loop | for each obs in `obs_list`: auto-regressive forecast (`1h/3h/6h/12h/24h` sub-models) → `H(x)` (`ObsOp_models[obs_name]` for radiances, identity for conventional) → `VarCost_models[obs_name]` (autograd through time) → gradient-normalized `(xb, grad)` → `DA_models[obs_name]` → analysis `xa`. Returns mean of per-obs `xa` |
| VarCost | `src.models.assimilate.utils.varcost.Obs_WeighedL2Norm` weighted by `R⁻¹ σ²`; QC mask `OmB > 5·σ_obs` → 0 |
| Scripts | `scripts/assimilate/cascade/{train,debug}/atms.sh` |
| Checkpoint strategy | per-component `model_training_config` decides what's frozen vs trainable (forecast frozen, DA trainable, obsop frozen by default) |

### 3.5 assimilate (multimodal) — `XiChenRepresentationObsEmbedding` + `XiChenFusion`

| Aspect | Value |
|---|---|
| Model classes | `src.models.assimilate.xichenda.arch_roe.XiChenRepresentationObsEmbedding` + `src.models.assimilate.xichenda.arch_fusion.XiChenFusion` (Perceiver-style) |
| Pipeline trainer | `src.pipeline.assimilate.multimodal.random_bg_trainer.RandomBgMultiModalAssimTrainer` |
| Config groups | `datamodule=assimilate/random_bg/atms_amsua_mhs_hrs4_prepbufr_satwnd_ascat.yaml`, `model=assimilate/random_bg/multimodal/<full>.yaml`, `pipeline=assimilate/multimodal/random_bg_trainer.yaml`, `training=multimodal_da/random_bg_atms_amsua_mhs_hrs4_prepbufr_satwnd_ascat.yaml` |
| Solver loop | for each obs, encode `(xb, grad)` into a representation via `ROE_models[obs_name]`; concatenate per-obs representations and fuse with `XiChenFusion` (learnable latent queries + cross-attention + Swin stack) |
| Obs list | 7 obs sources: `atms`, `amsua`, `mhs`, `hrs4`, `prepbufr`, `satwnd`, `ascat` |
| Scripts | `scripts/assimilate/multimodal/{train,debug}/atms_amsua_mhs_hrs4_prepbufr_satwnd_ascat.sh` |

> Dead-code register (do not edit): `src/models/assimilate/fdvarsolver/old/`, `src/models/assimilate/fdvarsolver/multimodal copy.py` (backup of the active multimodal solver).
>
> Kendall multi-task uncertainty: `src/layers/uncertanty.py` — note typo "uncertanty" (used by DA trainers to balance forecast / observation cost / analysis losses).

---

## 4. Project Structure

```
XiChenPaper/
├── main.py                          Hydra entry; DDP + logger + datamodule + trainer.fit()
├── configs/
│   ├── train.yaml                   defaults: datamodule/model/loss_fn/paths/hydra/pipeline/training
│   ├── datamodule/{forecast,compression,obsoperator,assimilate}/
│   ├── model/{forecast,compression,obsoperator,assimilate}/
│   ├── pipeline/{forecast,compression,obsoperator,assimilate}/
│   ├── loss_fn/                     # NOTE: README.md incorrectly says "loss/" — directory is "loss_fn/"
│   │   ├── l1.yaml
│   │   └── crps_gaussian.yaml       # NOTE: typo "guassian" — kept for back-compat
│   ├── training/                    Per-task training-loop overrides
│   │   ├── default.yaml             # epochs=20, lr=1e-6, bf16, device=cuda
│   │   ├── pretrain_forecast.yaml   # epochs=100, lr=1e-3, detach_iter_num=4
│   │   ├── finetune_forecast{,_obconstraint}.yaml
│   │   ├── train_compression.yaml
│   │   ├── train_obsop.yaml
│   │   ├── cascade_da/random_bg_atms.yaml
│   │   └── multimodal_da/random_bg_atms_amsua_mhs_hrs4_prepbufr_satwnd_ascat.yaml
│   ├── paths/default.yaml           Env-resolved: era5_hr_dir, era5_lr_dir, scale_dir, obs_dir
│   └── hydra/default.yaml
├── src/
│   ├── layers/                      Building blocks
│   │   ├── mlp.py                   GEGLU / GeGLUFFN / Mlp
│   │   ├── patch_embed.py           PatchEmbed (Conv2d tokenizer)
│   │   ├── pos_embed.py             sin-cos 2D positional encoding + interpolators
│   │   ├── swin_attn.py             WindowAttentionV2, WindowCrossAttentionV2, SwinBlock, SwinLayer
│   │   └── uncertanty.py            Kendall multi-task uncertainty (typo in filename)
│   ├── models/                      One package per family (forecast/compression/obsoperator/assimilate)
│   ├── datamodules/                 One package per family; pure PyTorch DataLoaders (no Lightning)
│   ├── pipeline/
│   │   ├── base/trainer.py          BaseTrainer (DDP, AMP, checkpoint, SummaryWriter, profiler)
│   │   └── <family>/                ForecastTrainer, CompressionTrainer, ObsOperatorTrainer, ...
│   ├── losses/crps_gaussian_loss.py # Masked CRPS for Gaussian predictive distribution
│   ├── metrics/                     crps.py + weighted_acc_rmse.py (NumPy + Torch variants)
│   └── utils/                       device, logger, lr_scheduler, model, parse_config, tqdm_logger
├── scripts/
│   ├── example.sh                   Minimal launcher (forecast only)
│   ├── forecast/{train,debug}/
│   ├── compression/{train,debug}/
│   ├── obsoperator/{atms,amsua,mhs,hrs4}.sh + debug/
│   ├── assimilate/{cascade,da_cycle,multimodal}/{train,debug}/
│   └── inference/{obsop,state_forecast/{1p0deg,interp_0p25deg}}/
├── inference/                       Production-time evaluation (separate package, not under src/)
│   ├── era5_lr_forecast.py          # 272 LOC
│   ├── era5_interp_forecast.py      # 272 LOC — near-duplicate of era5_lr_forecast.py
│   ├── obsoperator.py               Per-satellite OMB evaluation
│   ├── configs/{amsua,atms,hrs4,mhs}_obsop.json + xichen_forecast.json
│   └── utils/
│       ├── data_utils.py            738 LOC; schemas + per-sat prep/get helpers + geographic interpolation
│       └── model_utils.py           load_forecast_ckpt, load_obsop_ckpt
├── data_factory/                    Off-pipeline data prep (click CLIs)
│   ├── calculate_era5_deviation.py  48h ERA5 background deviations
│   ├── npy_prepbufr_qc.py           Conventional obs σ QC
│   └── npy_satwnd_qc.py             Satellite-wind σ QC
├── plots/                           Visualization
│   ├── plot_forecast_metrics.py     per-lead-time RMSE/ACC/activity
│   └── plot_obsop_metrics.py        OMB histograms per satellite
├── logs/                            Outputs (gitignored)
└── figures/                         Outputs (gitignored; currently empty in working tree)
```

---

## 5. Conventions & Coding Style

### 5.1 Logging — **no `print()`**

```python
from src.utils import setup_logger, get_logger
log = get_logger("xichen.<module>")
log.info(...)
```

- DDP-rank-filtered: rank > 0 logs are suppressed.
- `setup_logger(rank=local_rank, log_file=...)` writes to console + `logs/tensorboard/<task_name>.log`.
- `tqdm` is monkey-patched via `src/utils/tqdm_logger.py:patch_tqdm_for_logger` so progress bars go through the logger.

### 5.2 DDP

- Driven by `LOCAL_RANK` / `WORLD_SIZE` env vars (set by `torch.distributed.run`).
- `init_distributed(device_type, local_rank)` from `src/utils/device.py:40` picks backend (`hccl`/`nccl`) and calls `set_device`.
- All scripts under `scripts/` invoke `python -m torch.distributed.run --nproc_per_node=N main.py ...`.
- DDP wrap: `model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)` (in trainers).

### 5.3 Mixed precision

- bf16 by default; fp32 available for debug (`scripts/obsoperator/debug/hrs4_fp32.sh`).
- `autocast(device_type, dtype=torch.bfloat16)` and `get_grad_scaler(device_type)` from `src/utils/device.py`.

### 5.4 Checkpointing

- Always DDP-aware: `state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()`.
- Path: `logs/<task_name>/runs/checkpoints/{best,latest}.ckpt`.
- `best.pt` = lowest validation loss; chosen by `BaseTrainer._save_ckpt`.
- Inference loads with `strict=False` via `inference/utils/model_utils.py`.

### 5.5 Hydra instantiation

```python
datamodule = hydra.utils.instantiate(
    config.datamodule,
    distributed=(world_size > 1),
    num_replicas=world_size,
    rank=local_rank,
    _recursive_=False,   # required for datamodule and pipeline
)
trainer = hydra.utils.instantiate(config.pipeline, cfg=config, device=device, ..., _recursive_=False)
```

### 5.6 Reproducibility

`manual_seed(device_type, seed)` from `src/utils/device.py:101` seeds Python + NumPy + PyTorch + accelerator. `seed: 1024` in `configs/train.yaml`.

### 5.7 No tests

There are no `pytest` / `unittest` files anywhere in this repo. The validation loop inside `BaseTrainer.fit` is the only formal gate. Smoke-test changes manually before claiming done.

---

## 6. Configuration System

**Composition order** (declared in `configs/train.yaml:defaults`, applied left-to-right):

```
train.yaml → datamodule/* → model/* → loss_fn/* → paths/* → hydra/* → pipeline/* → training/*
```

**OmegaConf patterns**:

| Pattern | Meaning |
|---|---|
| `${oc.env:VAR}` | read env var (e.g. `${oc.env:DATA_DIR}`) |
| `${paths.root_dir}` | relative OmegaConf interpolation |
| `${hydra:runtime.output_dir}` | Hydra runtime path |
| `_target_: module.path.ClassName` | Hydra instantiation |
| `_recursive_=False` | required for datamodule / pipeline instantiation |

**Required paths** (`configs/paths/default.yaml`, hardcoded to `/mnt/xichen/...` in the working tree — some older scripts still reference `/public/home/studentresearch/XiChen/...`):

- `era5_hr_dir` — high-resolution 0.25° ERA5 (`721 × 1440`).
- `era5_lr_dir` — low-resolution 1.0° ERA5 (`181 × 360`); default training resolution.
- `scale_dir` — `normalize_mean.npz` / `normalize_std.npz`.
- `obs_dir` — per-satellite `*-auxiliary_value.npy`, `*-brightness_temperature_value.npy`, `*-mask.npy`.

**Switching tasks** = swapping all four config groups on the CLI:

```bash
# Cascade DA on ATMS only
python -m torch.distributed.run --nproc_per_node=4 main.py \
    datamodule=assimilate/random_bg/atms \
    model=assimilate/random_bg/cascade/atms \
    pipeline=assimilate/cascade/random_bg_trainer \
    training=cascade_da/random_bg_atms

# Multimodal DA on 7 obs sources
python -m torch.distributed.run --nproc_per_node=8 main.py \
    datamodule=assimilate/random_bg/atms_amsua_mhs_hrs4_prepbufr_satwnd_ascat \
    model=assimilate/random_bg/multimodal/atms_amsua_mhs_hrs4_prepbufr_satwnd_ascat \
    pipeline=assimilate/multimodal/random_bg_trainer \
    training=multimodal_da/random_bg_atms_amsua_mhs_hrs4_prepbufr_satwnd_ascat
```

---

## 7. Inference Pipeline

`inference/` is a **separate package** (not under `src/`) with its own JSON configs.

| Script | Purpose | Config |
|---|---|---|
| `inference/era5_lr_forecast.py` | AR rollout of `XiChenForecast` against ERA5 truth on 1.0° grid | `inference/configs/xichen_forecast.json` |
| `inference/era5_interp_forecast.py` | Same, but inputs interpolated to 0.25° via `geographic_interpolate("lr2hr")` | `inference/configs/xichen_forecast.json` |
| `inference/obsoperator.py` | Per-satellite OMB evaluation (atms/amsua/mhs/hrs4) | `inference/configs/<sat>_obsop.json` |

- Checkpoint loader: `inference/utils/model_utils.py` → `load_forecast_ckpt`, `load_obsop_ckpt` (`strict=False`, `map_location="cpu"`).
- Data utils: `inference/utils/data_utils.py` (738 LOC) — defines `VARIABLES` (69 elements), `conv_vars`, `sat_auxiliary_vars`, `sat_tmbrs_vars` schemas, `geographic_interpolate`, `get_normalize`, `get_climatology`, and per-satellite `prepare_{atms,amsua,mhs,hrs4,prepbufr,satwnd,ascat}` + `get_{...}` helpers.
- Plots: `plots/plot_forecast_metrics.py` (per-lead-time RMSE/ACC/activity), `plots/plot_obsop_metrics.py` (OMB histograms).
- Entry scripts: `scripts/inference/obsop/eval_{atms,amsua,mhs,hrs4}.sh`, `scripts/inference/state_forecast/{1p0deg,interp_0p25deg}/finetune_xichen_state_ar{2,4,6,8,10,12,15}.sh`.

---

## 8. Data Factory & Plots

- **`data_factory/calculate_era5_deviation.py`** — click CLI computing 48h ERA5 background deviations (`era5_48h_deviation.npz`).
- **`data_factory/npy_prepbufr_qc.py`** / **`npy_satwnd_qc.py`** — per-obs σ quality-control; both save `obs_sigma_qc_<year>.npz` + `variable_id.npz`.
- **`plots/plot_forecast_metrics.py`** — `plot_forecast_metrics(rmse, acc, activity, variables, title, ...)` → 3 PNGs (RMSE/ACC/activity) of 4×5 lead-time line plots over `['z-300','t-300','u-300','v-300','q-300','z-500','t-500','u-500','v-500','q-500','z-850','t-850','u-850','v-850','q-850','t2m','u10','v10','msl']`.
- **`plots/plot_obsop_metrics.py`** — `plot_obsop_omb(tgt, out, mask, var, dir)` → OMB histograms with overlaid Gaussian KDE (`.jpg` + `.pdf`).

---

## 9. Known Bugs & Tech Debt

> **Stale docs** (kept for archeology, do NOT use as source of truth): `AGENTS.md` (commit `48fbbe0`, 2026-05-04, forecast-only) and `README.md` (forecast-only).

| # | Location | Issue | Severity |
|---|---|---|---|
| 1 | `inference/utils/data_utils.py:26` | `conv_vars: {...}` is a syntax error — should be `conv_vars = {...}` (colon vs `=`). The file currently fails to import. | **High** |
| 2 | `data_factory/calculate_era5_deviation.py` | Imports `from utils.data_utils import ...` but `data_factory/utils/` does not exist. Module is currently broken. | **High** |
| 3 | `requirements.txt` | Deleted from working tree; README and AGENTS.md still reference it. Manual install required (see §2). | Medium |
| 4 | `README.md` | Documents `configs/loss/`; actual directory is `configs/loss_fn/`. | Low |
| 5 | `src/losses/crps_gaussian_loss.py` + `loss_fn/crps_gaussian.yaml` | Typo: `guassian` → `gaussian`. Functional but mis-named. | Low |
| 6 | `src/layers/uncertanty.py` | Typo: `uncertanty` → `uncertainty`. Used by DA trainers via the filename. | Low |
| 7 | `main.py:34` | `log_file = os.path.join(log_dir, ...)` uses `log_dir = config.paths.get("output_dir", "logs/chekpoints")` — typo `chekpoints`. | Low |
| 8 | `src/models/assimilate/fdvarsolver/old/` | Dead legacy solvers (amsua/atms/hrs4/mhs/multimodal/prepbufr/satwnd copies). Don't edit. | Low |
| 9 | `src/models/assimilate/fdvarsolver/multimodal copy.py` | Backup of the active multimodal solver. Don't edit. | Low |
| 10 | `inference/era5_interp_forecast.py` ↔ `inference/era5_lr_forecast.py` | 272-line near-duplicates differing only by data loader. Extract a shared `eval_forecast()`. | Medium |
| 11 | `data_factory/npy_prepbufr_qc.py` ↔ `data_factory/npy_satwnd_qc.py` | Identical `compute_sigma` helper duplicated. Extract to a shared module. | Medium |
| 12 | Hardcoded paths | `data_factory/` scripts use `/public/home/studentresearch/XiChen/...`; `inference/era5_*_forecast.py` uses `/mnt/xichen/...`. Resolve via `configs/paths/default.yaml` instead. | Medium |
| 13 | Working tree | `git status` shows many uncommitted config edits, deleted figure artifacts, modified `.prettierrc` / `.gitignore`. Branch `dev`. Verify before committing. | High |

---

## 10. Common Commands

### Single-launcher (forecast)

```bash
bash scripts/example.sh                         # 1 card, 100 epochs (default task)
bash scripts/example.sh --nproc 4 --epochs 50   # 4 cards, 50 epochs
```

### Per-family recipes

```bash
# Forecast
bash scripts/forecast/train/pretrain_xichen_state.sh
bash scripts/forecast/train/finetune_xichen_state_ar15.sh

# Forecast with obs-constraint loss
bash scripts/forecast/train/finetune_xichen_state_obconstraint_ar2.sh

# Compression
bash scripts/compression/train/train_xichenae_lr_wnorm.sh
bash scripts/compression/train/train_xichenae_lr_wonorm.sh

# Observation operators (one script per satellite)
bash scripts/obsoperator/atms.sh
bash scripts/obsoperator/amsua.sh
bash scripts/obsoperator/mhs.sh
bash scripts/obsoperator/hrs4.sh
bash scripts/obsoperator/debug/hrs4_bf16.sh    # explicit precision

# Cascade DA (ATMS only)
bash scripts/assimilate/cascade/train/atms.sh

# Multimodal DA (7 obs sources)
bash scripts/assimilate/multimodal/train/atms_amsua_mhs_hrs4_prepbufr_satwnd_ascat.sh

# Inference
bash scripts/inference/obsop/eval_atms.sh
bash scripts/inference/state_forecast/1p0deg/finetune_xichen_state_ar12.sh
bash scripts/inference/state_forecast/interp_0p25deg/finetune_xichen_state_ar15.sh

# Utilities
bash scripts/kill_job.sh          # find + kill straggler distributed runs
```

### Direct Hydra override (any task)

```bash
python -m torch.distributed.run --nproc_per_node=8 main.py \
    task_name=my_run \
    training.epochs=50 \
    training.lr=5e-5 \
    datamodule=obsoperator/amsua \
    model=obsoperator/xichen_amsua_obsoperator \
    pipeline=obsoperator/xichen_obsoperator
```

---

## 11. Verification Checklist

Before claiming a change is done:

1. **Config resolves.** `python main.py task_name=smoke_test training.epochs=1 device=cuda` should reach `trainer.fit()` without raising. (Skip the actual training loop with `Ctrl-C` once you've confirmed datamodule + trainer instantiate.)
2. **Forward shapes logged.** Add a `log.info` of input/output shapes on one train step; confirm `B × 69 × 181 × 360` for state and per-satellite channels for obs-ops.
3. **DDP clean.** No `torch.distributed` warnings in `logs/tensorboard/<task>.log`; `init_distributed` returns without raising.
4. **Checkpoint appears.** After 1 epoch: `ls logs/<task_name>/runs/checkpoints/` shows both `best.ckpt` and `latest.ckpt`.
5. **DDP-aware save.** Confirm `state_dict = model.module.state_dict()` is used in trainer — never `model.state_dict()` directly.
6. **Inference keys align.** `inference/utils/model_utils.load_*_ckpt` uses `strict=False`, but the keys must overlap with `src/models/*/arch.*Model` `state_dict()`.
7. **DA tasks**: verify that the obs-op checkpoint loaded by `Solver` is a `XiChenObsOp` checkpoint (not a `XiChenForecast` one) — see `model_training_config` in the cascade YAML.
8. **No tests exist.** Manual smoke is the only gate.
9. **`git status` clean** before commit (working tree currently has many uncommitted changes — coordinate before pushing).

---

## 12. See Also / Pointers

- **Backbone details**: `src/layers/swin_attn.py` — `WindowAttentionV2`, `WindowCrossAttentionV2`, `SwinBlock`, `SwinLayer` (cosine attention + continuous-relative-position-bias + `_max_logit` clamp).
- **Forecast arch**: `src/models/forecast/arch.py` — `XiChenForecast` (embed_dim=768, 12 heads, mlp_ratio=4, drop_path=0.2, lead-time cross-attention).
- **DA solver**: `src/models/assimilate/fdvarsolver/cascade.py` (cascade `Solver`), `src/models/assimilate/fdvarsolver/multimodal.py` (multimodal `Solver`).
- **Perceiver fusion**: `src/models/assimilate/xichenda/arch_fusion.py` — `XiChenFusion` + `PerceiverAttention`.
- **Loss**: `src/losses/crps_gaussian_loss.py` + `configs/loss_fn/crps_gaussian.yaml`.
- **Multi-task uncertainty**: `src/layers/uncertanty.py` (typo filename).
- **Hyperparameter knobs**: `configs/training/` (per-task overrides).
- **Hydra entry contract**: `main.py`.
- **NPU/GPU abstraction**: `src/utils/device.py`.
- **Stale docs (don't trust)**: `AGENTS.md`, `README.md`.