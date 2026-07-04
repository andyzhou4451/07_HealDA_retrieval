# AGENTS.md - XiChen Weather Forecasting (NPU/GPU)

**Generated:** 2026-05-04
**Commit:** 48fbbe0
**Branch:** master

> Deep learning training template using PyTorch + Hydra + OmegaConf.
> Supports both Huawei Ascend NPU and NVIDIA GPU via config switching.
> Supports DDP (DistributedDataParallel) distributed training and bf16 mixed precision training.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# 单卡训练
bash scripts/example.sh --data /path/to/data --scale /path/to/scale

# 多卡训练 (torch.distributed.run)
bash scripts/example.sh --data /path/to/data --scale /path/to/scale --nproc 4
```

---

## CUDA/GPU + DDP + bf16 训练

### 核心配置项 (`configs/train.yaml`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `epochs` | 训练轮数 | `100` |
| `seed` | 随机种子 | `1024` |
| `lr` | 学习率 | `1e-4` |
| `ckpt_path` | checkpoint保存路径 | `logs/checkpoints` |
| `precision.type` | 数值精度 | `"bf16"` |
| `device` | 加速卡类型 | `"cuda"` |
| `profile` | 启用torch_tb_profiler | `false` |

### torchrun 关键参数

- `--nproc_per_node`: 每个节点的进程数(等于GPU卡数)
- `--master_port`: 主进程通信端口
- `--use_env`: 让子进程从环境变量获取rank

---

## 项目结构

```
hydra-pytorch-npu-template/
├── main.py              # 入口：分布式初始化 + Hydra配置 + 训练
├── configs/
│   ├── train.yaml              # 主配置
│   ├── datamodule/forecast/   # 数据模块配置
│   ├── model/forecast/       # 模型配置
│   ├── paths/default.yaml     # 路径配置 (环境变量)
│   └── hydra/                 # Hydra日志配置
├── src/
│   ├── datamodules/forecast/state_forecast.py  # NpyDataset + StateForecastDataModule
│   ├── models/forecast/arch.py                # XiChenForecast CNN
│   ├── pipeline/forecast/trainer.py           # Trainer类
│   └── utils/
│       ├── __init__.py                     # seed_torch(), setup_logger(), get_logger()
│       └── device.py                       # 设备抽象层 (NPU/GPU)
├── scripts/
│   └── example.sh            # torch.distributed.run 启动脚本
└── requirements.txt
```

---

## Hydra Configuration System

**Config hierarchy** (applied in order):
```
train.yaml → datamodule/*.yaml → model/*.yaml → paths/*.yaml → hydra/*.yaml
```

**OmegaConf patterns:**
- `${oc.env:VAR}` — 环境变量读取
- `${..param}` — relative variable interpolation
- `_target_: module.path.ClassName` — class instantiation

**路径配置** (`configs/paths/default.yaml`):
```yaml
data_dir: ${oc.env:DATA_DIR}      # 必须设置环境变量
scale_dir: ${oc.env:SCALE_DIR}    # 必须设置环境变量
```

---

## Device Abstraction (NPU/GPU)

### 设备切换 (仅需修改配置)

```yaml
# configs/train.yaml
device: "cuda"   # 或 "cuda"
```

### 设备抽象层 (`src/utils/device.py`)

| 函数 | 说明 |
|------|------|
| `get_device_type()` | 自动检测可用设备 |
| `init_distributed()` | 初始化分布式 (hccl/nccl自动选择) |
| `get_grad_scaler()` | 获取混合精度GradScaler |
| `autocast()` | 混合精度上下文管理器 |
| `get_device()` | 获取torch.device |

### NPU/GPU 代码对比

| 功能 | NPU | GPU |
|------|-----|-----|
| 分布式后端 | `hccl` | `nccl` |
| 设备标识 | `npu:0` | `cuda:0` |
| AMP API | `torch.npu.amp` | `torch.cuda.amp` |

### 模型保存(DDP兼容)
```python
state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
```

---

## 核心组件

### 数据模块 (`src/datamodules/forecast/state_forecast.py`)

- **NpyDataset**: 加载.npy格式气象数据，按年组织
- **StateForecastDataModule**: 管理 train/val/test DataLoader
- **数据格式**:
  ```
  data_dir/
  ├── 2010/
  │   ├── 2010-01-01T00:00:00.npy
  │   └── ...
  scale_dir/
  ├── normalize_mean.npz
  └── normalize_std.npz
  ```
- **数据划分**: 训练2010-2021, 验证2022, 测试2023

### 模型 (`src/models/forecast/arch.py`)

- **XiChenForecast**: 简化CNN编解码器
- 输入/输出通道: 69 (气象变量数)
- embed_dim: 256

### 训练器 (`src/pipeline/forecast/trainer.py`)

- fit(): 完整训练流程 + TensorBoard日志
- train_epoch(): bf16前向 + GradScaler后向
- validate(): bf16验证

---

## Important Conventions

### Reproducibility
- `src.utils.seed_torch(seed)` sets seeds for Python/NumPy/PyTorch/CUDA
- Seed配置在 `configs/train.yaml` 的 `seed` 字段

### Logger (统一日志系统)
```python
from src.utils import setup_logger, get_logger

# main.py 入口初始化
log = setup_logger(rank=local_rank, log_file="logs/train.log")

# 各模块获取logger
log = get_logger("xichen.trainer")
log.info("Training started")
log.debug("Detailed info")
```
- **DDP过滤**: 非主进程 (rank>0) 的日志自动过滤
- **双输出**: 控制台 + 文件 (`logs/train.log`)
- **禁止使用print()**: 统一使用 `log.info/debug/warning/error`

### 数据加载 (Hydra实例化)
```python
datamodule = hydra.utils.instantiate(
    config.datamodule,
    root_dir=config.paths.data_dir,
    scale_dir=config.paths.scale_dir,
    distributed=(world_size > 1),
    _recursive_=False  # 必须禁用递归
)
```

### Model Checkpoints
- 最佳模型保存在: `logs/checkpoints/best.pt`
- 基于最低验证loss保存

---

## 依赖版本

- `torch >= 1.7.0`
- `torch-npu == 2.1.0` (Ascend Extension for PyTorch)
- `hydra-core >= 1.2.0`
- `torchmetrics == 0.9.3`
- `tensorboard`
- `tqdm`

---

## 常见问题

**Q: 如何在NPU和GPU之间切换？**
A: 修改 `configs/train.yaml` 中的 `device` 字段: `device: "cuda"` 或 `device: "gpu"`。

**Q: DDP训练时卡死怎么办？**
A: 添加 `--standalone` 参数: `torchrun --nproc_per_node=8 --standalone ...`

**Q: 如何选择bf16还是fp32？**
A: bf16混合精度训练速度更快，显存更省，建议优先使用bf16。

**Q: 多节点训练如何配置？**
A: 设置环境变量 `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`。
