# XiChen Weather Forecasting Training

华为昇腾NPU / NVIDIA GPU 天气预报模型训练框架，基于原生PyTorch DDP和Hydra配置管理。

**通过配置文件切换加速卡**: 仅需修改 `device: "cuda"` 或 `device: "cuda"`

## 环境要求

- Python 3.8+
- PyTorch >= 1.7.0
- torch-npu == 2.1.0 (NPU训练)
- Hydra >= 1.2.0

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

## 快速开始

### 1. 配置数据路径

在 `configs/paths/default.yaml` 中设置数据目录，或设置环境变量：

```bash
export DATA_DIR=/path/to/data
```

### 2. 单卡训练

```bash
bash scripts/example.sh
```

### 3. 多卡训练

```bash
bash scripts/example.sh --nproc 4   # 4卡
bash scripts/example.sh --nproc 8   # 8卡
```

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `scripts/example.sh` | 单卡/多卡分布式训练 (使用 torch.distributed.run) |

## 目录结构

```
.
├── main.py                    # 主程序入口
├── configs/
│   ├── train.yaml             # 主配置
│   ├── loss/                  # 损失函数配置
│   │   └── l1.yaml            # L1 Loss (默认)
│   ├── datamodule/            # 数据模块配置
│   │   └── forecast/
│   │       └── state_forecast.yaml
│   ├── model/                 # 模型配置
│   │   └── forecast/
│   │       └── default.yaml
│   ├── paths/                 # 路径配置
│   │   └── default.yaml
│   ├── pipeline/              # 训练流程配置
│   │   └── forecast/
│   │       └── trainer.yaml
│   └── hydra/                 # Hydra日志配置
├── scripts/
│   └── example.sh             # 分布式训练启动脚本
└── src/
    ├── datamodules/           # 数据加载
    │   └── forecast/
    │       └── state_forecast.py
    ├── models/                # 模型定义
    │   └── forecast/
    │       └── arch.py
    ├── pipeline/              # 训练流程
    │   ├── base/              # 基类
    │   │   └── trainer.py     # BaseTrainer
    │   └── forecast/
    │       └── trainer.py     # ForecastTrainer
    └── utils/                 # 工具函数
        ├── __init__.py
        └── device.py          # 设备抽象层 (NPU/GPU)
```

## 核心特性

| 特性 | 实现方式 |
|------|----------|
| 设备切换 | `device: "cuda"` 或 `device: "cuda"` (仅改配置) |
| 分布式训练 | `torch.nn.parallel.DistributedDataParallel` |
| 混合精度 | `torch.{npu,cuda}.amp` (bf16) |
| 通信后端 | `hccl` (NPU) / `nccl` (GPU) 自动选择 |
| 配置管理 | Hydra + OmegaConf |
| 日志 | 统一日志系统 (控制台 + 文件) |
| 性能分析 | `torch_npu.profiler` (NPU) / `torch.profiler` (GPU) |

## 配置说明

### 主配置 (`configs/train.yaml`)

```yaml
# @package _global_

defaults:
  - _self_
  - datamodule: forecast/state_forecast.yaml
  - model: forecast/default.yaml
  - loss: loss/l1.yaml
  - paths: default.yaml
  - hydra: default.yaml
  - pipeline: forecast/trainer.yaml

# 任务
task_name: "xichen_forecast"

# 训练参数
epochs: 100
seed: 1024
ckpt_path: ${hydra:runtime.output_dir}/checkpoints
lr: 1e-4

# 学习率调度器
warmup_epochs: 10
scheduler_type: cosine_warmup  # cosine_warmup | cosine | none

# 日志
log_dir: ${hydra:runtime.output_dir}/tensorboard
profile: false

# 数值精度
precision:
  type: "bf16"

# 加速卡类型: "cuda" 或 "gpu"
device: "cuda"
```

### 路径配置 (`configs/paths/default.yaml`)

```yaml
root_dir: ${oc.env:PROJECT_ROOT}
data_dir: ${oc.env:DATA_DIR}      # 必须设置环境变量 DATA_DIR
log_dir: ${paths.root_dir}/logs
output_dir: ${hydra:runtime.output_dir}
work_dir: ${hydra:runtime.cwd}
```

### 损失函数配置 (`configs/loss/l1.yaml`)

```yaml
_target_: torch.nn.L1Loss
reduction: mean
```

## 数据格式

数据目录结构（由 `root_dir` 指定）：

```
data_dir/
├── 2010/
│   ├── 2010-01-01T00:00:00.npy
│   ├── 2010-01-01T06:00:00.npy
│   └── ...
├── 2011/
│   └── ...
├── ...
└── normalized_mean_std/
    ├── normalize_mean.npz
    └── normalize_std.npz
```

- 训练集: 2010-2021
- 验证集: 2022
- 测试集: 2023

## 命令行参数覆盖

所有配置都可通过命令行覆盖：

```bash
# 修改训练轮数
bash scripts/example.sh --epochs 200

# 修改卡数
bash scripts/example.sh --nproc 4

# 直接使用 torch.distributed.run + 命令行覆盖
python -m torch.distributed.run \
    --nproc_per_node=4 \
    main.py \
    epochs=200 \
    device=cuda
```

## 训练输出

```
output_dir/checkpoints/
└── best.pt    # 最佳模型 (最低验证loss)

output_dir/tensorboard/
└── *.tfevents.*  # TensorBoard 日志
```

## 依赖

```
torch>=1.7.0
torch-npu==2.1.0
torchmetrics==0.9.3
hydra-core>=1.2.0
hydra-colorlog>=1.2.0
tqdm
tensorboard
pyyaml
omegaconf
pyrootutils
```