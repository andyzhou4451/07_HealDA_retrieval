# HealDA-XiChen T/Q13 Retrieval

This extension adds an observation-only, background-free HealDA-style retrieval task to XiChenPaper:

```text
ATMS + AMSU-A + MHS + HIRS4 + GDAS_prebufr_corrected_npy_1.0deg
  -> sensor-specific scalar observation tokenizer
  -> metadata encoder + scatter-reduce aggregation + observability mask
  -> sensor fusion
  -> HPX/lat-lon ViT backbone
  -> 26-channel ERA5 T/Q profile
```

The forward output is always either:

```text
[B, 26, 181, 360]
```

with channel order:

```text
t-50, t-100, t-150, t-200, t-250, t-300, t-400, t-500, t-600, t-700, t-850, t-925, t-1000,
q-50, q-100, q-150, q-200, q-250, q-300, q-400, q-500, q-600, q-700, q-850, q-925, q-1000
```

or, at inference time with `--as_profile`,

```text
[B, 2, 13, 181, 360]
```

where axis 1 is `[temperature, specific_humidity]`, axis 2 is the 13 pressure levels.

ERA5 T/Q is used only as the supervised label. It is never passed to the model input.

## Data paths

Default public02 paths are configured in `configs/paths/retrieval_public02.yaml`:

```yaml
obs_dir: /public02/data/Observation/observation_npy/
era5_dir: /public02/data/era5_np181x360_level13
era5_lr_dir: /public02/data/era5_np181x360_level13
scale_dir: /public02/data/era5_np181x360_level13/normalized_mean_std
```


ERA5 labels are now supported in both XiChen layouts:

```text
# full-state file per time
/public02/data/era5_np181x360_level13/2021/2021-01-01/22:00:00.npy

# per-variable file per time, as on your public02 tree
/public02/data/era5_np181x360_level13/2021/2021-01-01/22:00:00-t-1000.npy
/public02/data/era5_np181x360_level13/2021/2021-01-01/22:00:00-q-1000.npy
```

For the per-variable layout the dataloader requires all 26 target files at a target time:
`t-50 ... t-1000` and `q-50 ... q-1000`. It stacks them into `[26,181,360]` in the configured target order.

If your files are not exactly at 00/06/12/18 UTC, set:

```bash
datamodule.data.strict_time_index=true datamodule.data.dt_data=1
```

or let the dataset fallback scan find the complete target times automatically when the regular 6-hour grid is empty.

The dataloader searches these observation directory aliases:

```text
atms: 1batms_merged_npy_1.0deg, ATMS, atms
amsua: 1bamsua_merged_npy_1.0deg, AMSU-A, AMSUA, amsua
mhs: 1bmhs_merged_npy_1.0deg, MHS, mhs
hrs4: 1bhrs4_merged_npy_1.0deg, HIRS4, HRS4, hrs4
gdas_prebufr: GDAS_prebufr_corrected_npy_1.0deg, GDAS_prepbufr_merged_npy_1.0deg, gdas_prebufr
```

`satwnd` and `ascat` are not used by this task.

## Data inspection

Run before training:

```bash
python tools/check_retrieval_data.py \
  --obs_dir /public02/data/Observation/observation_npy/ \
  --era5_dir /public02/data/era5_np181x360_level13 \
  --sensors atms amsua mhs hrs4 gdas_prebufr
```

The script prints file counts, time ranges, sample shapes, missing-value ratios, value ranges, schema files, GDAS PREPBUFR diagnostics, ERA5 target status, and exact time matching.

## Generate target and observation normalization

If `/public02/data/era5_np181x360_level13/normalized_mean_std` does not exist, run:

```bash
python tools/generate_retrieval_mean_std.py \
  --era5_dir /public02/data/era5_np181x360_level13 \
  --scale_dir /public02/data/era5_np181x360_level13/normalized_mean_std \
  --target_vars \
    t-50 t-100 t-150 t-200 t-250 t-300 t-400 t-500 t-600 t-700 t-850 t-925 t-1000 \
    q-50 q-100 q-150 q-200 q-250 q-300 q-400 q-500 q-600 q-700 q-850 q-925 q-1000 \
  --dt_data 1
```

Optional observation statistics:

```bash
python tools/generate_retrieval_mean_std.py \
  --era5_dir /public02/data/era5_np181x360_level13 \
  --scale_dir /public02/data/era5_np181x360_level13/normalized_mean_std \
  --include_obs_stats \
  --obs_dir /public02/data/Observation/observation_npy/ \
  --sensors atms amsua mhs hrs4 gdas_prebufr
```

The dataloader fails with a clear error when target `normalize_mean.npz` or `normalize_std.npz` is missing and `normalize_target=true`.

## Debug training

```bash
python main.py \
  --config-name=train.yaml \
  datamodule=retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml \
  model=retrieval/healda_xichen_tq13.yaml \
  pipeline=retrieval/trainer.yaml \
  loss_fn=retrieval_tq_huber.yaml \
  training=retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml \
  paths=retrieval_public02.yaml \
  debug=true \
  model.model_size=tiny \
  datamodule.batch_size=1 \
  datamodule.num_workers=0 \
  training.limit_train_batches=2 \
  training.limit_val_batches=2
```

## Two-GPU training

```bash
torchrun \
  --nproc_per_node=2 \
  --master_port=${MASTER_PORT:-29501} \
  main.py \
  --config-name=train.yaml \
  datamodule=retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml \
  model=retrieval/healda_xichen_tq13.yaml \
  pipeline=retrieval/trainer.yaml \
  loss_fn=retrieval_tq_huber.yaml \
  training=retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml \
  paths=retrieval_public02.yaml \
  paths.obs_dir=/public02/data/Observation/observation_npy/ \
  paths.era5_dir=/public02/data/era5_np181x360_level13 \
  paths.scale_dir=/public02/data/era5_np181x360_level13/normalized_mean_std \
  training.device=cuda \
  training.precision.type=bf16 \
  model.model_size=base \
  task_name=healda_xichen_retrieval_atms_amsua_mhs_hrs4_gdas_tq13
```

SLURM:

```bash
sbatch scripts/retrieval/train_healda_xichen_tq13.slurm
```

## Evaluation

```bash
python tools/evaluate_retrieval_tq13.py \
  --checkpoint outputs/.../checkpoints/best.ckpt \
  --split test \
  --device cuda \
  paths.obs_dir=/public02/data/Observation/observation_npy/ \
  paths.era5_dir=/public02/data/era5_np181x360_level13 \
  paths.scale_dir=/public02/data/era5_np181x360_level13/normalized_mean_std
```

Metrics include overall RMSE, T/Q RMSE, T/Q MAE, T/Q bias, per-level T/Q RMSE, and latitude-weighted RMSE.

## Inference

```bash
python tools/infer_retrieval_tq13.py \
  --checkpoint outputs/.../checkpoints/best.ckpt \
  --output retrieval_tq13_predictions.npz \
  --split test \
  --device cuda \
  --limit_batches 4
```

Use `--as_profile` to save `[B,2,13,181,360]` instead of `[B,26,181,360]`.

## HPX / lat-lon regridding

```bash
python tools/regrid_hpx_latlon.py --mode check --input sample_latlon.npy --nside 64 --output_grid 181 360
python tools/regrid_hpx_latlon.py --mode latlon_to_hpx --input sample_latlon.npy --output sample_hpx.npy --nside 64
python tools/regrid_hpx_latlon.py --mode hpx_to_latlon --input sample_hpx.npy --output sample_latlon_back.npy --nside 64 --output_grid 181 360
```

If `earth2grid` is installed, the regrid functions use it. Otherwise the code falls back to safe nearest/bilinear lat-lon approximations and reports the limitation.

## Model sizes

```yaml
tiny:
  dim: 256
  depth: 6
  heads: 4
  obs_token_dim: 32
  sensor_embed_dim: 128
base:
  dim: 512
  depth: 12
  heads: 8
  obs_token_dim: 32
  sensor_embed_dim: 256
full_healda_like:
  dim: 1024
  depth: 24
  heads: 16
  obs_token_dim: 32
  sensor_embed_dim: 512
```

Use `model.model_size=tiny` for debug and `model.model_size=base` for normal training. The full HealDA-like option is provided but may require substantially larger GPUs.

## Troubleshooting

1. Missing `normalized_mean_std`: run `tools/generate_retrieval_mean_std.py`.
2. Missing observation directory: run `tools/check_retrieval_data.py` and verify the alias list above.
3. OOM: use `model.model_size=tiny`, lower `datamodule.data.max_points_per_sensor`, set `datamodule.batch_size=1`, and increase `training.gradient_accumulation_steps`.
4. `earth2grid` missing: keep `model.net.fallback_grid_backend=latlon` or install `earth2grid` in the HPC environment.
5. Hydra config resolves to the old forecast task: pass all retrieval groups explicitly, especially `loss_fn=retrieval_tq_huber.yaml`.
6. Accidentally reading `satwnd` or `ascat`: the retrieval dataset raises an error if those sensors are listed.

## Validation checklist

```bash
python -m compileall src tools
python tools/check_retrieval_data.py --obs_dir /public02/data/Observation/observation_npy/ --era5_dir /public02/data/era5_np181x360_level13 --sensors atms amsua mhs hrs4 gdas_prebufr
python main.py --config-name=train.yaml datamodule=retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml model=retrieval/healda_xichen_tq13.yaml pipeline=retrieval/trainer.yaml loss_fn=retrieval_tq_huber.yaml training=retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml paths=retrieval_public02.yaml debug=true model.model_size=tiny datamodule.batch_size=1 datamodule.num_workers=0
```

Expected forward output: `[B,26,181,360]`.
