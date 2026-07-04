# H100 80GB 单卡调参与训练说明

本版本面向 `ATMS + AMSU-A + MHS + HRS4 + GDAS_prebufr -> ERA5 13 层 T/Q` 反演任务，默认使用单张 H100 80GB。模型输出为 `[B, 26, 181, 360]`，通道顺序为 `t-50..t-1000, q-50..q-1000`。

## 新增入口

```bash
python scripts/tune_h100_single_gpu.py --config configs/train_h100_80gb_single_gpu.yaml --device cuda:0
python scripts/tune_hparams.py --config outputs/h100_tuning/recommended_config.yaml --device cuda:0
python train.py --config configs/train_h100_80gb_single_gpu.yaml --device cuda:0
python scripts/smoke_test_h100_config.py --config configs/train_h100_80gb_single_gpu.yaml --device cuda:0 --max_steps 5
```

## 推荐默认初始配置

真实 H100 sweep 会覆盖 `outputs/h100_tuning/recommended_config.yaml`。在没有实际 H100 benchmark 前，安全初始值为：

```yaml
batch_size: 1
precision: bf16
patch_size: [8, 8]
num_workers: 8
prefetch_factor: 4
pin_memory: true
persistent_workers: true
compile: true
channels_last: true
lr: 1.0e-4
weight_decay: 3.0e-4
warmup_steps: 1000
grad_clip_norm: 1.0
gradient_accumulation_steps: 1
max_points_per_sensor: 100000
```

这个初始值优先保证稳定；`tune_h100_single_gpu.py` 会根据真实 step time、samples/sec、OOM、NaN/Inf 和显存峰值选择吞吐更好的配置，而不是单纯选择显存占用最高的配置。

## 输出文件

```text
outputs/h100_tuning/batch_size_sweep.csv
outputs/h100_tuning/batch_size_sweep.json
outputs/h100_tuning/recommended_config.yaml
outputs/h100_tuning/hparam_sweep.csv
outputs/h100_tuning/hparam_sweep.json
outputs/h100_tuning/summary.md
outputs/logs/performance.jsonl
outputs/logs/epoch_metrics.jsonl
```

## 性能日志字段

`performance.jsonl` 每步记录：`epoch, step, global_step, batch_size, effective_batch_size, precision, max_memory_allocated_gb, max_memory_reserved_gb, memory_utilization, step_time_sec, samples_per_second, data_time_sec, forward_time_sec, backward_time_sec, optimizer_time_sec, gpu_name, cuda_version, torch_version, lr, amp_fallback, oom_fallback, nan_or_inf`。

## 单卡 Slurm

```bash
sbatch scripts/retrieval/train_h100_single_gpu.slurm
```

## 注意事项

1. 不要用 `torchrun --nproc_per_node=2` 跑 `configs/train_h100_80gb_single_gpu.yaml`，该配置设置了 `hardware.single_gpu=true`，检测到多进程会直接报错防止两个 rank 抢同一张卡。
2. 如果 `torch.compile` 首轮编译太慢或失败，训练器会 fallback 到 eager；也可以显式设置 `training.compile.enabled=false`。
3. 如果 BF16 报设备不支持，会 fallback 到 FP16 AMP + GradScaler；若仍出现 NaN/Inf，可设置 `training.precision.type=fp32 training.use_amp=false` 定位。
4. 如果 DataLoader 成为瓶颈，优先比较 `num_workers=4/8/12/16`、`prefetch_factor=2/4/8`，不要先增加梯度累积。
