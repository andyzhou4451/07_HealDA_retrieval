# H100 工程级审计与优化交付报告

## 一、项目理解

项目类型：PyTorch + Hydra 训练项目，新增任务是 HealDA-style 多源观测到 ERA5 13 层温湿廓线反演。主入口为 `main.py`，训练配置入口为 `configs/train.yaml` 加 retrieval 配置组，超算提交入口为 `scripts/retrieval/train_healda_xichen_tq13.slurm`。

输入观测：ATMS、AMSU-A/AMSUA、MHS、HIRS4/HRS4、GDAS_prebufr_corrected_npy_1.0deg。输出：`[B,26,181,360]`，通道顺序为 `t-50...t-1000,q-50...q-1000`。ERA5 只作为监督标签，不作为模型输入。

依赖文件：`setup.py` 已修正，新增 `requirements-h100.txt`。日志目录由 `paths.log_dir` 控制，Hydra 输出目录由 `paths.output_dir=${hydra:runtime.output_dir}` 控制，checkpoint 位于 `${paths.output_dir}/checkpoints`。

## 二、问题总览表

| 文件 | 问题 | 严重程度 | 修改方案 |
|---|---|---:|---|
| `main.py` | rank0 判断只看 `LOCAL_RANK`，多进程日志和主进程判断不够严谨 | 高 | 改为 `RANK==0`，同时保留 `LOCAL_RANK` 绑定本地 GPU |
| `main.py` | 默认设备可能落回旧 NPU 路径 | 高 | 明确优先读取 `training.device`，retrieval 默认 CUDA |
| `src/utils/device.py` | BF16 下仍创建 GradScaler，CUDA autocast 接口较旧 | 中 | BF16 不启用 scaler，统一使用 `torch.amp.autocast`，FP32 no-op |
| `src/utils/device.py` | H100 TF32、cudnn benchmark、matmul precision 未集中配置 | 中 | 增加 `configure_accelerator_performance` |
| `src/pipeline/base/trainer.py` | DDP 默认 `find_unused_parameters=True` 会增加开销 | 中 | 默认 false，并支持 `gradient_as_bucket_view`、`bucket_cap_mb`、`broadcast_buffers` |
| `src/pipeline/retrieval/trainer.py` | DDP 梯度累积每个 micro-step 都同步 | 高 | 使用 `no_sync()`，只在 optimizer step 前同步 |
| `src/pipeline/retrieval/trainer.py` | 训练/验证指标没有跨 rank 汇总 | 高 | 对 loss 和 metrics 做 `dist.all_reduce` |
| `src/pipeline/retrieval/trainer.py` | 频繁 `.cpu()`/`.item()` 造成训练同步 | 中 | 训练主循环只在必要处转 float，进度条降低刷新频率 |
| `src/datamodules/retrieval/healda_dataset.py` | 严格扫描 ERA5 时递归 glob 大量文件，容易压垮并行文件系统 | 高 | 用 `os.scandir` 扫 `YYYY/YYYY-MM-DD`，按时间聚合目标变量 |
| `src/datamodules/retrieval/healda_dataset.py` | ERA5 标签和经纬度网格重复加载/生成 | 中 | 增加 target LRU cache 和 lat/lon grid cache |
| `src/datamodules/retrieval/healda_datamodule.py` | worker seed 未加入 rank，DDP 下随机序列可能重复 | 中 | worker seed 加 `rank*100000+worker_id` |
| `configs/datamodule/retrieval/*.yaml` | `dt_data=6` 不适合实际 `22:00:00-t-1000.npy` 文件 | 高 | 默认改为 `dt_data=1`、`strict_time_index=true` |
| `configs/training/retrieval/*.yaml` | H100 性能参数不足 | 中 | 增加 BF16、DDP bucket、matmul precision、梯度累积、可选 compile |
| `setup.py` | `install_requires=["pytorch"]` 包名错误 | 高 | 改为 `torch>=2.1` 等实际依赖 |
| `scripts/retrieval/train_healda_xichen_tq13.slurm` | 未显式传入 public02 实际时间索引参数 | 高 | 增加 `datamodule.data.strict_time_index=true datamodule.data.dt_data=1` |

## 三、性能优化总览表

| 优化点 | 作用 | 风险 | 验证方式 |
|---|---|---|---|
| DDP + torchrun | 两张 H100 各跑一个进程，数据并行 | NCCL 环境变量需匹配集群网络 | 日志出现 world_size=2，`nvidia-smi` 两卡都有进程 |
| BF16 autocast | H100 上提升吞吐并降低激活显存 | 个别算子可能数值敏感 | 观察 loss 无 NaN，必要时切 FP32/FP16 |
| TF32 matmul | FP32 路径加速矩阵乘 | 极小数值差异 | 对比 debug loss 和指标 |
| DDP `no_sync` | 梯度累积时减少不必要 all-reduce | 残余 micro-step 必须同步 | debug limit 不是累积步整数也能 optimizer step |
| `gradient_as_bucket_view` | 减少梯度 bucket 额外拷贝 | 旧版 PyTorch 兼容性 | 若 DDP 初始化报错则关掉该项 |
| scandir ERA5 索引 | 降低 metadata server 压力 | 非标准文件名需 check 脚本确认 | `tools/check_retrieval_data.py` 看 complete target times |
| target LRU cache | persistent workers 下减少重复标签读取 | worker 内存增加 | 调整 `target_cache_size` |
| DataLoader 8 workers/rank | 提升 CPU 解码和文件读取吞吐 | 共享文件系统压力增大 | 看 GPU util 与 dataloader wait |
| 可选 torch.compile backbone | 加速 Transformer backbone | 动态点云全模型不适合 compile | 默认关；单独打开 `training.compile.enabled=true` 对比 step time |
| 可选 gradient checkpointing | 显存不足时降低激活显存 | 计算时间增加 | OOM 时打开 `training.use_gradient_checkpointing=true` |

## 四、修改后的完整项目文件树

```text
./.gitignore
./.prettierrc
./AGENTS.md
./CLAUDE.md
./MODIFIED_FILES_FULL_CODE_ANNOTATED.md
./README.md
./README_H100_AUDIT.md
./README_HEALDA_RETRIEVAL.md
./configs/datamodule/assimilate/random_bg/atms.yaml
./configs/datamodule/assimilate/random_bg/atms_amsua_mhs_hrs4_prepbufr_satwnd_ascat.yaml
./configs/datamodule/assimilate/random_bg/atms_amsua_mhs_prepbufr_satwnd_ascat.yaml
./configs/datamodule/compression/xichenae_lr.yaml
./configs/datamodule/forecast/xichen_state_forecast.yaml
./configs/datamodule/forecast/xichen_state_forecast_obconstraint.yaml
./configs/datamodule/obsoperator/amsua.yaml
./configs/datamodule/obsoperator/atms.yaml
./configs/datamodule/obsoperator/hrs4.yaml
./configs/datamodule/obsoperator/mhs.yaml
./configs/datamodule/retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml
./configs/hydra/default.yaml
./configs/loss_fn/crps_gaussian.yaml
./configs/loss_fn/l1.yaml
./configs/loss_fn/retrieval_tq_huber.yaml
./configs/model/assimilate/random_bg/atms.yaml
./configs/model/assimilate/random_bg/cascade/atms.yaml
./configs/model/assimilate/random_bg/multimodal/atms_amsua_mhs_hrs4_prepbufr_satwnd_ascat.yaml
./configs/model/assimilate/random_bg/multimodal/atms_amsua_mhs_prepbufr_satwnd_ascat.yaml
./configs/model/compression/xichenae_lr.yaml
./configs/model/forecast/default.yaml
./configs/model/forecast/xichen_state_forecast.yaml
./configs/model/obsoperator/xichen_amsua_obsoperator.yaml
./configs/model/obsoperator/xichen_atms_obsoperator.yaml
./configs/model/obsoperator/xichen_hrs4_obsoperator.yaml
./configs/model/obsoperator/xichen_mhs_obsoperator.yaml
./configs/model/retrieval/healda_xichen_tq13.yaml
./configs/paths/default.yaml
./configs/paths/retrieval_public02.yaml
./configs/pipeline/assimilate/cascade/random_bg_trainer.yaml
./configs/pipeline/assimilate/multimodal/random_bg_trainer.yaml
./configs/pipeline/compression/xichenae.yaml
./configs/pipeline/forecast/xichen_state_forecast.yaml
./configs/pipeline/forecast/xichen_state_forecast_obconstraint.yaml
./configs/pipeline/obsoperator/xichen_obsoperator.yaml
./configs/pipeline/retrieval/trainer.yaml
./configs/train.yaml
./configs/training/cascade_da/random_bg_atms.yaml
./configs/training/compression/train_compression.yaml
./configs/training/default.yaml
./configs/training/forecast/finetune_forecast.yaml
./configs/training/forecast/pretrain_forecast.yaml
./configs/training/multimodal_da/random_bg_atms_amsua_mhs_hrs4_prepbufr_satwnd_ascat.yaml
./configs/training/multimodal_da/random_bg_atms_amsua_mhs_prepbufr_satwnd_ascat.yaml
./configs/training/obsop/train_obsop.yaml
./configs/training/retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml
./data_factory/__init__.py
./data_factory/_sigma_utils.py
./data_factory/calculate_era5_deviation.py
./data_factory/npy_prepbufr_qc.py
./data_factory/npy_satwnd_qc.py
./fusion_result.json
./inference/__init__.py
./inference/configs/amsua_obsop.json
./inference/configs/atms_obsop.json
./inference/configs/hrs4_obsop.json
./inference/configs/mhs_obsop.json
./inference/configs/xichen_forecast.json
./inference/era5_forecast_core.py
./inference/era5_interp_forecast.py
./inference/era5_lr_forecast.py
./inference/obsoperator.py
./inference/utils/__init__.py
./inference/utils/data_utils.py
./inference/utils/model_utils.py
./main.py
./plots/__init__.py
./plots/plot_forecast_metrics.py
./plots/plot_obsop_metrics.py
./requirements-h100.txt
./scripts/assimilate/cascade/debug/atms.sh
./scripts/assimilate/cascade/train/atms.sh
./scripts/assimilate/multimodal/debug/atms_amsua_mhs_prepbufr_satwnd_ascat.sh
./scripts/assimilate/multimodal/train/atms_amsua_mhs_prepbufr_satwnd_ascat.sh
./scripts/compression/debug/debug_finetune_ar2.sh
./scripts/compression/debug/debug_xichenae_lr.sh
./scripts/compression/train/train_xichenae_lr_wnorm.sh
./scripts/compression/train/train_xichenae_lr_wonorm.sh
./scripts/example.sh
./scripts/forecast/debug/debug_finetune_ar2.sh
./scripts/forecast/debug/debug_pretrain.sh
./scripts/forecast/debug/finetune_xichen_state_ar15.sh
./scripts/forecast/pretrain_xichen_state_stage2.sh
./scripts/forecast/train/finetune_xichen_state_ar10.sh
./scripts/forecast/train/finetune_xichen_state_ar12.sh
./scripts/forecast/train/finetune_xichen_state_ar15.sh
./scripts/forecast/train/finetune_xichen_state_ar15_20260616.sh
./scripts/forecast/train/finetune_xichen_state_ar15_20260624.sh
./scripts/forecast/train/finetune_xichen_state_ar3.sh
./scripts/forecast/train/finetune_xichen_state_ar6.sh
./scripts/forecast/train/finetune_xichen_state_ar8.sh
./scripts/forecast/train/finetune_xichen_state_obconstarint_ar2.sh
./scripts/forecast/train/pretrain_xichen_state.sh
./scripts/inference/obsop/eval_amsua.sh
./scripts/inference/obsop/eval_atms.sh
./scripts/inference/obsop/eval_hrs4.sh
./scripts/inference/obsop/eval_mhs.sh
./scripts/inference/state_forecast/1p0deg/finetune_xichen_state_ar10.sh
./scripts/inference/state_forecast/1p0deg/finetune_xichen_state_ar12.sh
./scripts/inference/state_forecast/1p0deg/finetune_xichen_state_ar15.sh
./scripts/inference/state_forecast/1p0deg/finetune_xichen_state_ar2.sh
./scripts/inference/state_forecast/1p0deg/finetune_xichen_state_ar4.sh
./scripts/inference/state_forecast/1p0deg/finetune_xichen_state_ar6.sh
./scripts/inference/state_forecast/1p0deg/finetune_xichen_state_ar8.sh
./scripts/inference/state_forecast/1p0deg/pretrain_xichen_state.sh
./scripts/inference/state_forecast/interp_0p25deg/finetune_ar12.sh
./scripts/inference/state_forecast/interp_0p25deg/finetune_ar15.sh
./scripts/inference/state_forecast/interp_0p25deg/pretrain_xichen_state.sh
./scripts/kill_job.sh
./scripts/obsoperator/amsua.sh
./scripts/obsoperator/atms.sh
./scripts/obsoperator/debug/hrs4_bf16.sh
./scripts/obsoperator/debug/hrs4_fp32.sh
./scripts/obsoperator/hrs4.sh
./scripts/obsoperator/mhs.sh
./scripts/retrieval/train_healda_xichen_tq13.slurm
./scripts/tmux.sh
./setup.py
./src/MODULE_INDEX.md
./src/__init__.py
./src/datamodules/__init__.py
./src/datamodules/assimilate/__init__.py
./src/datamodules/assimilate/da_cycle/__init__.py
./src/datamodules/assimilate/random_bg/__init__.py
./src/datamodules/assimilate/random_bg/npydatamodule.py
./src/datamodules/assimilate/random_bg/npydataset.py
./src/datamodules/compression/state_datamodule.py
./src/datamodules/compression/state_dataset.py
./src/datamodules/forecast/state_datamodule.py
./src/datamodules/forecast/state_dataset.py
./src/datamodules/obsoperator/__init__.py
./src/datamodules/obsoperator/amsua/__init__.py
./src/datamodules/obsoperator/amsua/npydatamodule.py
./src/datamodules/obsoperator/amsua/npydataset.py
./src/datamodules/obsoperator/atms/__init__.py
./src/datamodules/obsoperator/atms/npydatamodule.py
./src/datamodules/obsoperator/atms/npydataset.py
./src/datamodules/obsoperator/hrs4/__init__.py
./src/datamodules/obsoperator/hrs4/npydatamodule.py
./src/datamodules/obsoperator/hrs4/npydataset.py
./src/datamodules/obsoperator/mhs/__init__.py
./src/datamodules/obsoperator/mhs/npydatamodule.py
./src/datamodules/obsoperator/mhs/npydataset.py
./src/datamodules/retrieval/__init__.py
./src/datamodules/retrieval/healda_datamodule.py
./src/datamodules/retrieval/healda_dataset.py
./src/layers/__init__.py
./src/layers/mlp.py
./src/layers/patch_embed.py
./src/layers/pos_embed.py
./src/layers/swin_attn.py
./src/layers/uncertainty.py
./src/losses/__init__.py
./src/losses/crps_gaussian_loss.py
./src/losses/retrieval_tq_loss.py
./src/metrics/__init__.py
./src/metrics/crps.py
./src/metrics/retrieval_metrics.py
./src/metrics/weighted_acc_rmse.py
./src/models/__init__.py
./src/models/assimilate/__init__.py
./src/models/assimilate/fdvarsolver/__init__.py
./src/models/assimilate/fdvarsolver/cascade.py
./src/models/assimilate/fdvarsolver/multimodal.py
./src/models/assimilate/fdvarsolver/old/amsua.py
./src/models/assimilate/fdvarsolver/old/atms.py
./src/models/assimilate/fdvarsolver/old/hrs4.py
./src/models/assimilate/fdvarsolver/old/mhs.py
./src/models/assimilate/fdvarsolver/old/multimodal.py
./src/models/assimilate/fdvarsolver/old/prepbufr.py
./src/models/assimilate/fdvarsolver/old/satwnd.py
./src/models/assimilate/utils/__init__.py
./src/models/assimilate/utils/forecast.py
./src/models/assimilate/utils/varcost.py
./src/models/assimilate/xichenda/__init__.py
./src/models/assimilate/xichenda/arch.py
./src/models/assimilate/xichenda/arch_fusion.py
./src/models/assimilate/xichenda/arch_roe.py
./src/models/compression/__init__.py
./src/models/compression/arch.py
./src/models/compression/arch_.py
./src/models/forecast/__init__.py
./src/models/forecast/arch.py
./src/models/obsoperator/__init__.py
./src/models/obsoperator/arch.py
./src/models/retrieval/__init__.py
./src/models/retrieval/healda_hpx_vit.py
./src/models/retrieval/healda_obs_encoder.py
./src/models/retrieval/healda_regrid.py
./src/models/retrieval/healda_sensor_embedder.py
./src/models/retrieval/healda_xichen_retrieval.py
./src/models/retrieval/retrieval_decoder.py
./src/pipeline/__init__.py
./src/pipeline/assimilate/__init__.py
./src/pipeline/assimilate/cascade/__init__.py
./src/pipeline/assimilate/cascade/random_bg_trainer.py
./src/pipeline/assimilate/multimodal/__init__.py
./src/pipeline/assimilate/multimodal/random_bg_trainer.py
./src/pipeline/base/__init__.py
./src/pipeline/base/trainer.py
./src/pipeline/compression/__init__.py
./src/pipeline/compression/trainer.py
./src/pipeline/forecast/__init__.py
./src/pipeline/forecast/trainer.py
./src/pipeline/obsoperator/__init__.py
./src/pipeline/obsoperator/trainer.py
./src/pipeline/retrieval/__init__.py
./src/pipeline/retrieval/trainer.py
./src/utils/__init__.py
./src/utils/device.py
./src/utils/logger.py
./src/utils/lr_scheduler.py
./src/utils/model.py
./src/utils/parse_config.py
./src/utils/tqdm_logger.py
./tools/check_retrieval_data.py
./tools/evaluate_retrieval_tq13.py
./tools/generate_retrieval_mean_std.py
./tools/infer_retrieval_tq13.py
./tools/regrid_hpx_latlon.py
./tools/smoke_retrieval_model.py
```

## 五、修改文件列表

```text
configs/datamodule/retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml
configs/model/retrieval/healda_xichen_tq13.yaml
configs/training/retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml
main.py
requirements-h100.txt
scripts/retrieval/train_healda_xichen_tq13.slurm
setup.py
src/datamodules/retrieval/healda_datamodule.py
src/datamodules/retrieval/healda_dataset.py
src/models/retrieval/healda_hpx_vit.py
src/models/retrieval/healda_xichen_retrieval.py
src/pipeline/base/trainer.py
src/pipeline/retrieval/trainer.py
src/utils/device.py
tools/smoke_retrieval_model.py
```

## 六、Slurm 提交脚本

使用 `scripts/retrieval/train_healda_xichen_tq13.slurm`。该脚本保留用户指定的 Slurm 模板、`conda activate xichen_v1` 和 NCCL 环境变量，只在 `torchrun main.py` 参数中增加真实 public02 ERA5 小时文件布局所需的 `strict_time_index=true` 与 `dt_data=1`。

## 七、运行命令

数据检查：

```bash
python tools/check_retrieval_data.py   --obs_dir /public02/data/Observation/observation_npy/   --era5_dir /public02/data/era5_np181x360_level13   --sensors atms amsua mhs hrs4 gdas_prebufr
```

生成 mean/std：

```bash
python tools/generate_retrieval_mean_std.py   --era5_dir /public02/data/era5_np181x360_level13   --scale_dir /public02/data/era5_np181x360_level13/normalized_mean_std   --dt_data 1
```

CPU 语法和轻量模型冒烟：

```bash
python -m compileall -q main.py src tools
python tools/smoke_retrieval_model.py --device cpu --fast_cpu --grid 12 24 --points 8
```

单 GPU debug：

```bash
python main.py   --config-name=train.yaml   datamodule=retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml   model=retrieval/healda_xichen_tq13.yaml   pipeline=retrieval/trainer.yaml   loss_fn=retrieval_tq_huber.yaml   training=retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml   paths=retrieval_public02.yaml   debug=true   model.model_size=tiny   datamodule.batch_size=1   datamodule.num_workers=0   datamodule.data.strict_time_index=true   datamodule.data.dt_data=1
```

双 H100 debug：

```bash
torchrun --nproc_per_node=2 --master_port=29501 main.py   --config-name=train.yaml   datamodule=retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml   model=retrieval/healda_xichen_tq13.yaml   pipeline=retrieval/trainer.yaml   loss_fn=retrieval_tq_huber.yaml   training=retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml   paths=retrieval_public02.yaml   debug=true   model.model_size=tiny   datamodule.batch_size=1   datamodule.num_workers=2   datamodule.data.strict_time_index=true   datamodule.data.dt_data=1   training.device=cuda   training.precision.type=bf16
```

正式提交：

```bash
sbatch scripts/retrieval/train_healda_xichen_tq13.slurm
```

## 八、验证和排错指南

正常日志应包含：`DDP initialized`、`world_size=2`、`precision bf16`、`active_grid_backend=latlon`、`forward output [B,26,181,360]`、`last.ckpt` 与 `best.ckpt` 保存信息。

检查两卡是否使用：运行 `watch -n 1 nvidia-smi`，应看到两个 Python/torchrun 进程分别占用 GPU0/GPU1；训练 step 中 GPU-Util 应持续上升。

检查 NCCL：`NCCL_DEBUG=INFO` 时日志会显示 rank、channel、transport。若卡在 init，先确认端口未冲突、`CUDA_VISIBLE_DEVICES` 是否包含 2 张卡、`MASTER_PORT` 是否被占用。

判断 DataLoader 瓶颈：若 GPU-Util 周期性掉到 0 且 CPU 或文件系统等待高，先把 `datamodule.num_workers` 从 8 调到 4/12 做对比，观察 samples/sec 与 step time；若共享文件系统压力大，降低 `target_cache_size` 以外的随机读取并使用本地缓存目录。

常见错误：

- CUDA OOM：先 `model.model_size=tiny`，再降 `datamodule.data.max_points_per_sensor`，然后增大 `training.gradient_accumulation_steps`；仍不够时打开 `training.use_gradient_checkpointing=true`。
- NCCL timeout：检查 `NCCL_IB_DISABLE=1` 是否符合集群网络；有 IB/RoCE 时可尝试 `NCCL_IB_DISABLE=0` 并设置正确 `NCCL_SOCKET_IFNAME`。
- 找不到 ERA5：确认路径为 `/public02/data/era5_np181x360_level13/YYYY/YYYY-MM-DD/HH:MM:SS-t-1000.npy`，并使用 `datamodule.data.dt_data=1`。
- unknown sensor alias satellite：已修复 Hydra `DictConfig` 分组展开，只会把叶子 sensor 传给 dataset。
- loss NaN：先关闭 q log transform，使用 BF16；若仍 NaN，临时 `training.use_amp=false` 定位异常观测范围。

## 九、性能调优建议

建议初始组合：`model_size=base`、每卡 batch size 1、梯度累积 2、每 rank 8 workers、BF16。若两张 H100 显存占用低且 GPU-Util 高，可以尝试每卡 batch size 2；若 GPU-Util 低，先提高 workers/prefetch，再检查观测点云读取耗时。`torch.compile` 默认关闭，稳定后只编译 backbone：`training.compile.enabled=true training.compile.target=backbone`。

## 十、最终检查清单

- [x] `python -m compileall -q main.py src tools` 通过。
- [x] synthetic CPU smoke test 通过：`forward_shape=(1,26,12,24)`。
- [x] fake nested ERA5 per-variable layout 读取通过，target shape 为 `[26,181,360]`。
- [x] DDP rank、BF16、DataLoader、checkpoint、metrics all-reduce 代码已补强。
- [x] Slurm 脚本保留用户模板并传入 public02 实际 ERA5 时间索引参数。
