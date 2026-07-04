# OOM / DDP point-cloud 修复说明

## 现象

训练在 `pred = self.model(batch)` 处进入 `torch.nn.parallel.DistributedDataParallel.forward`，随后在 `torch/distributed/utils.py -> obj.to(target_gpu)` 报 CUDA out of memory。

同时日志里有：

```text
WARNING: torch.distributed.run: Setting OMP_NUM_THREADS environment variable ...
RuntimeWarning: Filtered ERA5 target times by observation-window availability ...
```

其中 OMP 是 torchrun 的线程提示，不是失败原因；Filtered ERA5 只是说明 dataset 按观测窗口过滤了没有观测的 ERA5 标签时次，也不是失败原因。

## 根因

本项目的 retrieval batch 是嵌套结构：

```python
{
  "target": Tensor[B,26,181,360],
  "observations": {
      "atms": [dict of point tensors],
      "amsua": [dict of point tensors],
      ...
  }
}
```

旧 DDP 包装使用：

```python
DDP(model, device_ids=[local_rank], output_device=local_rank)
```

这会让 DDP 在进入模型 forward 前递归调用 `.to(target_gpu)`，把 `observations` 里的所有点云张量一次性搬到 GPU。该行为绕过了模型内部“逐 sensor、按需搬运”的节省显存设计，因此在第一层之前就 OOM。

另外旧 dataset 的 `max_points_per_sensor` 是在每个观测文件上限流，而 HealDA 窗口 `[-21,+3]` 默认会读取 9 个时间文件；所以 `250000` 实际可能膨胀成 `9 * 250000 = 2250000` 个点/传感器/样本。

## 修复

1. 新增 `training.ddp_auto_move_inputs=false`，DDP 仍同步梯度，但不再自动搬运整个嵌套 batch。
2. BaseTrainer 在该开关关闭时使用 `device_ids=None, output_device=None` 包装 DDP。
3. 观测点上限改为在整个 observation window 拼接后再执行，因此 `max_points_per_sensor=100000` 就是真正每个 target_time、每个 sensor 的上限。
4. 默认开启 `training.use_gradient_checkpointing=true`，降低 Transformer backbone 训练显存。
5. Slurm 增加 `OMP_NUM_THREADS/MKL_NUM_THREADS`，消除 torchrun 默认线程提示；增加 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256` 缓解碎片化。
6. 只在 `RANK=0` 输出 target-time 过滤 warning，避免双卡日志重复刷屏。

## 推荐验证命令

```bash
python -m compileall -q main.py src tools inference data_factory

python tools/smoke_retrieval_model.py --device cpu --fast_cpu --grid 12 24 --points 8
```

双卡 debug：

```bash
torchrun --nproc_per_node=2 --master_port=29501 main.py \
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
  datamodule.data.max_points_per_sensor=20000 \
  training.device=cuda \
  training.precision.type=bf16 \
  training.ddp_auto_move_inputs=false
```

正式训练保持脚本：

```bash
sbatch scripts/retrieval/train_healda_xichen_tq13.slurm
```

如果 `base` 仍然 OOM，先不要改代码，按顺序降低：

```bash
datamodule.data.max_points_per_sensor=50000
model.model_size=tiny
model.net.patch_size=[8,8]
```
