# XiChenPaper src/ 模块导航

> 本文件由 docs/superpowers/specs/2026-06-20-xichen-comments-design.md Task 11 生成。
> 用于帮助新成员快速定位模块,识别上下游依赖。详细参数请直接看每个文件内的 docstring。

## 模块地图

### 1. 基础设施层 (`src/layers/`, `src/losses/`, `src/metrics/`, `src/utils/`)

| 子包 | 核心类/函数 | 一句话用途 |
|---|---|---|
| `src/layers/mlp.py` | `GEGLU` / `GeGLUFFN` / `Mlp` | 三种 MLP 变体 |
| `src/layers/patch_embed.py` | `PatchEmbed` | Conv2d patch 嵌入 |
| `src/layers/pos_embed.py` | (函数式) | sin-cos 2D 位置编码 + 插值器 |
| `src/layers/swin_attn.py` | `WindowAttentionV2` / `WindowCrossAttentionV2` / `SwinBlock` / `SwinLayer` | Swin-V2 注意力(含 cross-attn 变体) |
| `src/layers/uncertanty.py` | (函数式) | Kendall 多任务不确定性权重 |
| `src/losses/crps_gaussian_loss.py` | (函数式) | 掩码 CRPS-Gaussian loss |
| `src/metrics/crps.py` | (函数式) | CRPS 指标(num/torch) |
| `src/metrics/weighted_acc_rmse.py` | (函数式) | 加权 ACC/RMSE(纬度权重) |
| `src/utils/device.py` | `init_distributed` / `get_grad_scaler` / `manual_seed` | NPU/GPU 抽象层 |
| `src/utils/logger.py` | `setup_logger` / `get_logger` | 日志系统 |
| `src/utils/lr_scheduler.py` | (函数式) | 学习率调度 |
| `src/utils/model.py` | (函数式) | 模型工具 |
| `src/utils/parse_config.py` | (函数式) | 配置解析 |
| `src/utils/tqdm_logger.py` | `patch_tqdm_for_logger` | tqdm 补丁 |

### 2. 预报模型 (`src/models/forecast/`)

| 子包 | 核心类 | 一句话用途 |
|---|---|---|
| `src/models/forecast/arch.py` | `XiChenForecast` | 基于 Swin-V2 的 AR 预报模型 |

### 3. 压缩模型 (`src/models/compression/`)

| 子包 | 核心类 | 一句话用途 |
|---|---|---|
| `src/models/compression/arch.py` | `XiChenAutoEncoder` | 状态压缩自编码器(活跃) |
| `src/models/compression/arch_.py` | (legacy) | 扩展/遗留变体 |

### 4. 观测算子 (`src/models/obsoperator/`, `src/datamodules/obsoperator/`)

| 子包 | 核心类 | 一句话用途 |
|---|---|---|
| `src/models/obsoperator/arch.py` | `XiChenObsOp` | 通用观测算子骨架 H(x) |
| `src/datamodules/obsoperator/atms/` | (ATMS DataModule) | ATMS 卫星数据 |
| `src/datamodules/obsoperator/amsua/` | (AMSUA DataModule) | AMSUA 卫星数据 |
| `src/datamodules/obsoperator/mhs/` | (MHS DataModule) | MHS 卫星数据 |
| `src/datamodules/obsoperator/hrs4/` | (HRS4 DataModule) | HRS4 卫星数据 |

### 5. 数据同化模型 (`src/models/assimilate/`)

| 子包 | 核心类 | 一句话用途 |
|---|---|---|
| `src/models/assimilate/fdvarsolver/cascade.py` | `Solver` | 级联 DA 求解器 |
| `src/models/assimilate/fdvarsolver/multimodal.py` | `Solver` | 多模态 DA 求解器 |
| `src/models/assimilate/utils/varcost.py` | `Obs_WeighedL2Norm` | 观测代价(R⁻¹σ² 加权) |
| `src/models/assimilate/xichenda/arch.py` | `XiChenDA` | 单 obs DA 模型 |
| `src/models/assimilate/xichenda/arch_fusion.py` | `XiChenFusion` | Perceiver 风格多 obs 融合 |
| `src/models/assimilate/xichenda/arch_roe.py` | `XiChenRepresentationObsEmbedding` | obs 表征嵌入 |

### 6. 数据加载 (`src/datamodules/`)

| 子包 | 核心类 | 一句话用途 |
|---|---|---|
| `src/datamodules/forecast/state_datamodule.py` | `StateForecastDataModule` | 状态场预报数据模块 |
| `src/datamodules/forecast/obs_datamodule.py` | (obs datamodule) | 含 obs 信息的预报数据 |
| `src/datamodules/forecast/state_dataset.py` | (state Dataset) | 状态场 Dataset |
| `src/datamodules/forecast/obs_dataset.py` | (obs Dataset) | 含 obs 的 Dataset |
| `src/datamodules/compression/state_datamodule.py` | `StateCompressionDataModule` | 压缩数据模块 |
| `src/datamodules/compression/state_dataset.py` | (state Dataset) | 压缩 Dataset |
| `src/datamodules/assimilate/random_bg/npydatamodule.py` | `RandomBgAssimDataModule` | 随机背景 DA 数据模块 |
| `src/datamodules/assimilate/random_bg/npydataset.py` | (random_bg Dataset) | 随机背景 Dataset(538 行,大文件) |

### 7. 训练流程 (`src/pipeline/`)

| 子包 | 核心类 | 一句话用途 |
|---|---|---|
| `src/pipeline/base/trainer.py` | `BaseTrainer` | 训练流程基类(DDP/AMP/checkpoint) |
| `src/pipeline/forecast/trainer.py` | `ForecastTrainer` | 预报训练器(含 AR 滚动) |
| `src/pipeline/forecast/trainer_obconstraint.py` | (variant) | 含 obs-constraint loss 变体 |
| `src/pipeline/compression/trainer.py` | `CompressionTrainer` | 压缩训练器 |
| `src/pipeline/obsoperator/trainer.py` | `ObsOperatorTrainer` | 观测算子训练器 |
| `src/pipeline/assimilate/cascade/random_bg_trainer.py` | `RandomBgCascadeAssimTrainer` | 级联 DA 训练器(918 行) |
| `src/pipeline/assimilate/multimodal/random_bg_trainer.py` | `RandomBgMultiModalAssimTrainer` | 多模态 DA 训练器(978 行) |

## 任务家族入口(快速对照)

| 任务家族 | 模型入口 | 数据入口 | 训练器入口 | 主入口 |
|---|---|---|---|---|
| forecast | `XiChenForecast` | `StateForecastDataModule` | `ForecastTrainer` | `bash scripts/example.sh` |
| compression | `XiChenAutoEncoder` | `StateCompressionDataModule` | `CompressionTrainer` | `bash scripts/compression/train/train_xichenae_lr_wnorm.sh` |
| obsoperator | `XiChenObsOp` | `<sat>/<sat>DataModule` | `ObsOperatorTrainer` | `bash scripts/obsoperator/<sat>.sh` |
| assimilate (cascade) | `Solver` + `XiChenDA` | `RandomBgAssimDataModule` | `RandomBgCascadeAssimTrainer` | `bash scripts/assimilate/cascade/train/atms.sh` |
| assimilate (multimodal) | `XiChenFusion` + ROEs | `RandomBgAssimDataModule` | `RandomBgMultiModalAssimTrainer` | `bash scripts/assimilate/multimodal/train/atms_amsua_mhs_hrs4_prepbufr_satwnd_ascat.sh` |

## 已废弃代码

- `src/models/assimilate/fdvarsolver/old/` —— 7 个历史求解器,标记为废弃,保留仅用于历史对比,不要编辑。

## 更多信息

- 完整架构与配置说明：`/workspace/XiChenPaper/CLAUDE.md`
- 注释规范设计：`/workspace/XiChenPaper/docs/superpowers/specs/2026-06-20-xichen-comments-design.md`
- 实施计划：`/workspace/XiChenPaper/docs/superpowers/plans/2026-06-20-xichen-comments-enhancement.md`