# 修改文件完整代码与逐行中文说明

说明：下列内容覆盖本次 H100 优化涉及的关键代码/配置/脚本文件。每行后方给出中文说明，便于人工审计。

## main.py

```text
0001: # -*- coding: utf-8 -*-    # 说明：中文/配置注释，说明相邻代码或参数用途。
0002: """XiChen/HealDA 训练入口。    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0003:     # 说明：空行，用于分隔逻辑块，提高可读性。
0004: 该入口保留 XiChen 原有 Hydra 配置方式，同时补强单机 2×H100 DDP 训练需要的    # 说明：保留该行以完成当前代码块的语法结构。
0005: rank 解析、CUDA 性能开关、日志目录、debug 参数下传与异常清理逻辑。    # 说明：保留该行以完成当前代码块的语法结构。
0006: """    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0007:     # 说明：空行，用于分隔逻辑块，提高可读性。
0008: from __future__ import annotations    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0009:     # 说明：空行，用于分隔逻辑块，提高可读性。
0010: import os    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0011: import sys    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0012: from pathlib import Path    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0013:     # 说明：空行，用于分隔逻辑块，提高可读性。
0014: try:    # 说明：进入异常保护块，保证超算任务失败时可诊断。
0015:     import pyrootutils    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0016:     # 说明：空行，用于分隔逻辑块，提高可读性。
0017:     root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0018: except ImportError:    # 说明：捕获异常并提供安全回退或清晰报错。
0019:     root = Path(__file__).resolve().parent    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0020:     if str(root) not in sys.path:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0021:         sys.path.insert(0, str(root))    # 说明：调用函数或方法，执行具体工程动作。
0022:     # 说明：空行，用于分隔逻辑块，提高可读性。
0023: import hydra    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0024: from omegaconf import DictConfig, OmegaConf    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0025:     # 说明：空行，用于分隔逻辑块，提高可读性。
0026: try:  # 兼容原 XiChen NPU 环境；CUDA/H100 环境未安装 torch_npu 时安全跳过。    # 说明：进入异常保护块，保证超算任务失败时可诊断。
0027:     import torch_npu  # type: ignore  # noqa: F401    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0028:     import torch_npu.distributed  # type: ignore  # noqa: F401    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0029: except ImportError:    # 说明：捕获异常并提供安全回退或清晰报错。
0030:     pass    # 说明：保留该行以完成当前代码块的语法结构。
0031:     # 说明：空行，用于分隔逻辑块，提高可读性。
0032: from src.utils import setup_logger    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0033: from src.utils.device import (    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0034:     configure_accelerator_performance,    # 说明：保留该行以完成当前代码块的语法结构。
0035:     get_device,    # 说明：保留该行以完成当前代码块的语法结构。
0036:     init_distributed,    # 说明：保留该行以完成当前代码块的语法结构。
0037:     manual_seed,    # 说明：保留该行以完成当前代码块的语法结构。
0038: )    # 说明：调用函数或方法，执行具体工程动作。
0039: from src.utils.tqdm_logger import patch_tqdm_for_logger    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0040:     # 说明：空行，用于分隔逻辑块，提高可读性。
0041:     # 说明：空行，用于分隔逻辑块，提高可读性。
0042: def _env_int(name: str, default: int) -> int:    # 说明：定义函数，复用项目中的关键流程。
0043:     """从环境变量读取整数；Slurm/torchrun 未设置时使用安全默认值。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0044:     try:    # 说明：进入异常保护块，保证超算任务失败时可诊断。
0045:         return int(os.environ.get(name, default))    # 说明：返回当前函数计算得到的结果。
0046:     except (TypeError, ValueError):    # 说明：捕获异常并提供安全回退或清晰报错。
0047:         return int(default)    # 说明：返回当前函数计算得到的结果。
0048:     # 说明：空行，用于分隔逻辑块，提高可读性。
0049:     # 说明：空行，用于分隔逻辑块，提高可读性。
0050: def _resolve_device_type(config: DictConfig) -> str:    # 说明：定义函数，复用项目中的关键流程。
0051:     """按 training.device 优先解析设备类型，避免默认落回旧 NPU 配置。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0052:     training = config.get("training", {})    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0053:     value = training.get("device", config.get("device", "cuda")) if hasattr(training, "get") else config.get("device", "cuda")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0054:     return str(value).lower()    # 说明：返回当前函数计算得到的结果。
0055:     # 说明：空行，用于分隔逻辑块，提高可读性。
0056:     # 说明：空行，用于分隔逻辑块，提高可读性。
0057: def _propagate_debug_overrides(config: DictConfig) -> None:    # 说明：定义函数，复用项目中的关键流程。
0058:     """把顶层 debug=true 下传到 datamodule/training，保证冒烟测试只跑少量 batch。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0059:     if not config.get("debug", False):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0060:         return    # 说明：保留该行以完成当前代码块的语法结构。
0061:     OmegaConf.set_struct(config, False)    # 说明：调用函数或方法，执行具体工程动作。
0062:     if "datamodule" in config:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0063:         config.datamodule.debug = True    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0064:         config.datamodule.batch_size = int(config.datamodule.get("batch_size", 1) or 1)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0065:         config.datamodule.num_workers = int(config.datamodule.get("num_workers", 0) or 0)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0066:     if "training" in config:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0067:         config.training.epochs = int(config.training.get("max_epochs", config.training.get("epochs", 1)) or 1)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0068:         config.training.limit_train_batches = int(config.training.get("limit_train_batches", 2) or 2)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0069:         config.training.limit_val_batches = int(config.training.get("limit_val_batches", 2) or 2)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0070:         config.training.profile = False    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0071:     # 说明：空行，用于分隔逻辑块，提高可读性。
0072:     # 说明：空行，用于分隔逻辑块，提高可读性。
0073: @hydra.main(version_base=None, config_path="configs", config_name="train.yaml")    # 说明：装饰器，修改函数/方法行为或启用框架入口。
0074: def main(config: DictConfig) -> None:    # 说明：定义函数，复用项目中的关键流程。
0075:     """Hydra 主函数：构建 DataModule、Trainer，并执行训练循环。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0076:     local_rank = _env_int("LOCAL_RANK", 0)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0077:     global_rank = _env_int("RANK", local_rank)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0078:     world_size = _env_int("WORLD_SIZE", 1)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0079:     is_main = global_rank == 0    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0080:     device_type = _resolve_device_type(config)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0081:     # 说明：空行，用于分隔逻辑块，提高可读性。
0082:     _propagate_debug_overrides(config)    # 说明：调用函数或方法，执行具体工程动作。
0083:     configure_accelerator_performance(config=config, device_type=device_type, log_fn=None)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0084:     # 说明：空行，用于分隔逻辑块，提高可读性。
0085:     task_name = str(config.get("task_name", "debug"))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0086:     log_dir = str(config.paths.get("log_dir", "logs"))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0087:     output_dir = str(config.paths.get("output_dir", "logs/output"))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0088:     os.makedirs(log_dir, exist_ok=True)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0089:     os.makedirs(output_dir, exist_ok=True)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0090:     log_file = os.path.join(log_dir, f"{task_name}.rank{global_rank}.log") if is_main else None    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0091:     log = setup_logger(rank=global_rank, log_file=log_file)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0092:     patch_tqdm_for_logger(log)    # 说明：调用函数或方法，执行具体工程动作。
0093:     # 说明：空行，用于分隔逻辑块，提高可读性。
0094:     if is_main:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0095:         log.info("=" * 80)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0096:         log.info("XiChen HealDA-style T/Q13 retrieval training")    # 说明：调用函数或方法，执行具体工程动作。
0097:         log.info(f"device={device_type}, global_rank={global_rank}, local_rank={local_rank}, world_size={world_size}")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0098:         log.info(f"hydra_output_dir={output_dir}")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0099:         log.info("=" * 80)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0100:     # 说明：空行，用于分隔逻辑块，提高可读性。
0101:     if world_size > 1:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0102:         init_distributed(device_type=device_type, local_rank=local_rank)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0103:         if is_main:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0104:             log.info(f"DDP initialized with backend={'nccl' if device_type == 'cuda' else 'hccl'} and world_size={world_size}")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0105:     # 说明：空行，用于分隔逻辑块，提高可读性。
0106:     device = get_device(device_type=device_type, local_rank=local_rank)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0107:     manual_seed(device_type=device_type, seed=int(config.get("seed", config.get("training", {}).get("seed", 1024))), rank=global_rank)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0108:     # 说明：空行，用于分隔逻辑块，提高可读性。
0109:     if is_main:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0110:         log.info("Initializing retrieval datamodule")    # 说明：调用函数或方法，执行具体工程动作。
0111:     datamodule = hydra.utils.instantiate(    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0112:         config.datamodule,    # 说明：保留该行以完成当前代码块的语法结构。
0113:         distributed=(world_size > 1),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0114:         num_replicas=world_size,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0115:         rank=global_rank,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0116:         _recursive_=False,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0117:     )    # 说明：调用函数或方法，执行具体工程动作。
0118:     train_loader = datamodule.train_dataloader()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0119:     val_loader = datamodule.val_dataloader()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0120:     # 说明：空行，用于分隔逻辑块，提高可读性。
0121:     if is_main:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0122:         log.info("Initializing retrieval trainer")    # 说明：调用函数或方法，执行具体工程动作。
0123:     trainer = hydra.utils.instantiate(    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0124:         config.pipeline,    # 说明：保留该行以完成当前代码块的语法结构。
0125:         cfg=config,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0126:         device=device,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0127:         local_rank=local_rank,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0128:         world_size=world_size,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0129:         is_main=is_main,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0130:         output_dir=output_dir,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0131:         log_dir=log_dir,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0132:         log=log,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0133:         _recursive_=False,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0134:     )    # 说明：调用函数或方法，执行具体工程动作。
0135:     # 说明：空行，用于分隔逻辑块，提高可读性。
0136:     if is_main:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0137:         log.info(    # 说明：调用函数或方法，执行具体工程动作。
0138:             "Training config: "    # 说明：保留该行以完成当前代码块的语法结构。
0139:             f"epochs={config.training.epochs}, lr={config.training.lr}, "    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0140:             f"batch_size_per_rank={config.datamodule.get('batch_size', 1)}, "    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0141:             f"num_workers_per_rank={config.datamodule.get('num_workers', 0)}"    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0142:         )    # 说明：调用函数或方法，执行具体工程动作。
0143:     # 说明：空行，用于分隔逻辑块，提高可读性。
0144:     try:    # 说明：进入异常保护块，保证超算任务失败时可诊断。
0145:         trainer.fit(train_loader=train_loader, val_loader=val_loader)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0146:     finally:    # 说明：保留该行以完成当前代码块的语法结构。
0147:         trainer.cleanup()    # 说明：调用函数或方法，执行具体工程动作。
0148:     # 说明：空行，用于分隔逻辑块，提高可读性。
0149:     if is_main:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0150:         log.info("Training completed")    # 说明：调用函数或方法，执行具体工程动作。
0151:     # 说明：空行，用于分隔逻辑块，提高可读性。
0152:     # 说明：空行，用于分隔逻辑块，提高可读性。
0153: if __name__ == "__main__":    # 说明：执行条件分支，处理不同运行环境或配置情况。
0154:     main()    # 说明：调用函数或方法，执行具体工程动作。
```

## src/utils/device.py

```text
0001: # -*- coding: utf-8 -*-    # 说明：中文/配置注释，说明相邻代码或参数用途。
0002: """统一设备与分布式工具。    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0003:     # 说明：空行，用于分隔逻辑块，提高可读性。
0004: 该模块兼容原 XiChen 的 NPU 路径，同时为 NVIDIA H100/CUDA 训练补强：NCCL    # 说明：保留该行以完成当前代码块的语法结构。
0005: 进程组初始化、BF16/FP16 autocast、TF32 开关、确定性开关和 rank 相关随机种子。    # 说明：保留该行以完成当前代码块的语法结构。
0006: """    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0007:     # 说明：空行，用于分隔逻辑块，提高可读性。
0008: from __future__ import annotations    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0009:     # 说明：空行，用于分隔逻辑块，提高可读性。
0010: import importlib.util    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0011: import os    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0012: import random    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0013: from contextlib import nullcontext    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0014: from typing import Any, Callable, Literal    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0015:     # 说明：空行，用于分隔逻辑块，提高可读性。
0016: import numpy as np    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0017: import torch    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0018:     # 说明：空行，用于分隔逻辑块，提高可读性。
0019:     # 说明：空行，用于分隔逻辑块，提高可读性。
0020: def is_npu_available() -> bool:    # 说明：定义函数，复用项目中的关键流程。
0021:     """检测 torch-npu 是否安装且 NPU 可用，CUDA 环境不会强制导入 NPU 包。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0022:     spec = importlib.util.find_spec("torch_npu")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0023:     if spec is None:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0024:         return False    # 说明：返回当前函数计算得到的结果。
0025:     try:    # 说明：进入异常保护块，保证超算任务失败时可诊断。
0026:         import torch_npu  # type: ignore    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0027:     # 说明：空行，用于分隔逻辑块，提高可读性。
0028:         return bool(torch_npu.is_available())    # 说明：返回当前函数计算得到的结果。
0029:     except (ImportError, AttributeError):    # 说明：捕获异常并提供安全回退或清晰报错。
0030:         return False    # 说明：返回当前函数计算得到的结果。
0031:     # 说明：空行，用于分隔逻辑块，提高可读性。
0032:     # 说明：空行，用于分隔逻辑块，提高可读性。
0033: def get_device_type() -> Literal["npu", "cuda"]:    # 说明：定义函数，复用项目中的关键流程。
0034:     """自动选择可用加速器，优先保留 NPU 兼容，其次使用 CUDA。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0035:     if is_npu_available():    # 说明：执行条件分支，处理不同运行环境或配置情况。
0036:         return "npu"    # 说明：返回当前函数计算得到的结果。
0037:     if torch.cuda.is_available():    # 说明：执行条件分支，处理不同运行环境或配置情况。
0038:         return "cuda"    # 说明：返回当前函数计算得到的结果。
0039:     raise RuntimeError("No accelerator available: neither torch-npu nor CUDA torch is available")    # 说明：调用函数或方法，执行具体工程动作。
0040:     # 说明：空行，用于分隔逻辑块，提高可读性。
0041:     # 说明：空行，用于分隔逻辑块，提高可读性。
0042: def configure_accelerator_performance(config: Any | None = None, device_type: str = "cuda", log_fn: Callable[[str], None] | None = None) -> None:    # 说明：定义函数，复用项目中的关键流程。
0043:     """设置 H100 友好的 PyTorch 性能开关。    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0044:     # 说明：空行，用于分隔逻辑块，提高可读性。
0045:     默认启用 TF32 和 high float32 matmul precision；确定性训练可通过    # 说明：保留该行以完成当前代码块的语法结构。
0046:     ``training.deterministic=true`` 关闭 cudnn benchmark 并启用确定性算法。    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0047:     """    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0048:     if device_type != "cuda" or not torch.cuda.is_available():    # 说明：执行条件分支，处理不同运行环境或配置情况。
0049:         return    # 说明：保留该行以完成当前代码块的语法结构。
0050:     # 说明：空行，用于分隔逻辑块，提高可读性。
0051:     training = config.get("training", {}) if config is not None and hasattr(config, "get") else {}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0052:     deterministic = bool(training.get("deterministic", False)) if hasattr(training, "get") else False    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0053:     benchmark = bool(training.get("cudnn_benchmark", not deterministic)) if hasattr(training, "get") else (not deterministic)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0054:     matmul_precision = str(training.get("float32_matmul_precision", "high")) if hasattr(training, "get") else "high"    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0055:     # 说明：空行，用于分隔逻辑块，提高可读性。
0056:     torch.backends.cuda.matmul.allow_tf32 = True    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0057:     torch.backends.cudnn.allow_tf32 = True    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0058:     torch.backends.cudnn.benchmark = benchmark    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0059:     if deterministic:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0060:         torch.backends.cudnn.deterministic = True    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0061:         torch.use_deterministic_algorithms(True, warn_only=True)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0062:     try:    # 说明：进入异常保护块，保证超算任务失败时可诊断。
0063:         torch.set_float32_matmul_precision(matmul_precision)    # 说明：调用函数或方法，执行具体工程动作。
0064:     except Exception as exc:  # pragma: no cover - 老版本 torch 可能没有该接口。    # 说明：捕获异常并提供安全回退或清晰报错。
0065:         if log_fn is not None:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0066:             log_fn(f"torch.set_float32_matmul_precision failed safely: {exc}")    # 说明：调用函数或方法，执行具体工程动作。
0067:     # 说明：空行，用于分隔逻辑块，提高可读性。
0068:     # 说明：空行，用于分隔逻辑块，提高可读性。
0069: def init_distributed(device_type: str, local_rank: int) -> None:    # 说明：定义函数，复用项目中的关键流程。
0070:     """初始化 DDP 进程组，并保证每个进程只绑定一张本地 GPU/NPU。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0071:     if device_type == "npu":    # 说明：执行条件分支，处理不同运行环境或配置情况。
0072:         import torch_npu  # type: ignore  # noqa: F401    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0073:         from torch_npu.distributed import is_hccl_available  # type: ignore    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0074:     # 说明：空行，用于分隔逻辑块，提高可读性。
0075:         if not is_hccl_available():    # 说明：执行条件分支，处理不同运行环境或配置情况。
0076:             raise RuntimeError("HCCL is not available; check CANN, HCCL driver and torch-npu/CANN version match")    # 说明：调用函数或方法，执行具体工程动作。
0077:         torch.npu.set_device(local_rank)    # 说明：调用函数或方法，执行具体工程动作。
0078:         backend = "hccl"    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0079:     elif device_type == "cuda":    # 说明：执行备用条件分支，覆盖另一类配置或状态。
0080:         if not torch.cuda.is_available():    # 说明：执行条件分支，处理不同运行环境或配置情况。
0081:             raise RuntimeError("training.device=cuda was requested but torch.cuda.is_available() is False")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0082:         visible = torch.cuda.device_count()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0083:         if local_rank >= visible:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0084:             raise RuntimeError(f"LOCAL_RANK={local_rank} but only {visible} CUDA devices are visible")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0085:         torch.cuda.set_device(local_rank)    # 说明：调用函数或方法，执行具体工程动作。
0086:         backend = "nccl"    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0087:     else:    # 说明：执行默认分支，保证逻辑闭环。
0088:         raise ValueError(f"Unsupported distributed device_type={device_type!r}; expected 'cuda' or 'npu'")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0089:     # 说明：空行，用于分隔逻辑块，提高可读性。
0090:     if not torch.distributed.is_initialized():    # 说明：执行条件分支，处理不同运行环境或配置情况。
0091:         torch.distributed.init_process_group(backend=backend, init_method="env://")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0092:     # 说明：空行，用于分隔逻辑块，提高可读性。
0093:     # 说明：空行，用于分隔逻辑块，提高可读性。
0094: def get_grad_scaler(device_type: str, dtype: torch.dtype | None = None):    # 说明：定义函数，复用项目中的关键流程。
0095:     """只在 FP16 混合精度下创建 GradScaler；BF16/H100 默认不需要 loss scaling。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0096:     if dtype is not torch.float16:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0097:         return None    # 说明：返回当前函数计算得到的结果。
0098:     if device_type == "npu":    # 说明：执行条件分支，处理不同运行环境或配置情况。
0099:         return torch.npu.amp.GradScaler()    # 说明：返回当前函数计算得到的结果。
0100:     if device_type == "cuda":    # 说明：执行条件分支，处理不同运行环境或配置情况。
0101:         return torch.amp.GradScaler("cuda") if hasattr(torch, "amp") else torch.cuda.amp.GradScaler()    # 说明：返回当前函数计算得到的结果。
0102:     return None    # 说明：返回当前函数计算得到的结果。
0103:     # 说明：空行，用于分隔逻辑块，提高可读性。
0104:     # 说明：空行，用于分隔逻辑块，提高可读性。
0105: def autocast(device_type: str, dtype: torch.dtype = torch.bfloat16):    # 说明：定义函数，复用项目中的关键流程。
0106:     """返回适配 CUDA/NPU/CPU 的 autocast 上下文；FP32 时退化为 no-op。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0107:     if dtype is torch.float32 or device_type == "cpu":    # 说明：执行条件分支，处理不同运行环境或配置情况。
0108:         return nullcontext()    # 说明：返回当前函数计算得到的结果。
0109:     if device_type == "npu":    # 说明：执行条件分支，处理不同运行环境或配置情况。
0110:         return torch.npu.amp.autocast(dtype=dtype)    # 说明：返回当前函数计算得到的结果。
0111:     if device_type == "cuda":    # 说明：执行条件分支，处理不同运行环境或配置情况。
0112:         return torch.amp.autocast("cuda", dtype=dtype) if hasattr(torch, "amp") else torch.cuda.amp.autocast(dtype=dtype)    # 说明：返回当前函数计算得到的结果。
0113:     return nullcontext()    # 说明：返回当前函数计算得到的结果。
0114:     # 说明：空行，用于分隔逻辑块，提高可读性。
0115:     # 说明：空行，用于分隔逻辑块，提高可读性。
0116: def manual_seed(device_type: str, seed: int, rank: int = 0) -> None:    # 说明：定义函数，复用项目中的关键流程。
0117:     """设置 Python、NumPy、PyTorch 与加速器随机种子；rank 偏移避免 worker 序列重复。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0118:     final_seed = int(seed) + int(rank)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0119:     os.environ["PYTHONHASHSEED"] = str(final_seed)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0120:     random.seed(final_seed)    # 说明：调用函数或方法，执行具体工程动作。
0121:     np.random.seed(final_seed)    # 说明：调用函数或方法，执行具体工程动作。
0122:     torch.manual_seed(final_seed)    # 说明：调用函数或方法，执行具体工程动作。
0123:     if device_type == "npu":    # 说明：执行条件分支，处理不同运行环境或配置情况。
0124:         torch.npu.manual_seed(final_seed)    # 说明：调用函数或方法，执行具体工程动作。
0125:     elif device_type == "cuda" and torch.cuda.is_available():    # 说明：执行备用条件分支，覆盖另一类配置或状态。
0126:         torch.cuda.manual_seed(final_seed)    # 说明：调用函数或方法，执行具体工程动作。
0127:         torch.cuda.manual_seed_all(final_seed)    # 说明：调用函数或方法，执行具体工程动作。
0128:     # 说明：空行，用于分隔逻辑块，提高可读性。
0129:     # 说明：空行，用于分隔逻辑块，提高可读性。
0130: def get_device(device_type: str, local_rank: int = 0) -> torch.device:    # 说明：定义函数，复用项目中的关键流程。
0131:     """返回当前进程绑定的 torch.device。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0132:     if device_type == "npu":    # 说明：执行条件分支，处理不同运行环境或配置情况。
0133:         return torch.device(f"npu:{local_rank}")    # 说明：返回当前函数计算得到的结果。
0134:     if device_type == "cuda":    # 说明：执行条件分支，处理不同运行环境或配置情况。
0135:         return torch.device(f"cuda:{local_rank}")    # 说明：返回当前函数计算得到的结果。
0136:     return torch.device("cpu")    # 说明：返回当前函数计算得到的结果。
0137:     # 说明：空行，用于分隔逻辑块，提高可读性。
0138:     # 说明：空行，用于分隔逻辑块，提高可读性。
0139: def destroy_process_group() -> None:    # 说明：定义函数，复用项目中的关键流程。
0140:     """安全销毁分布式进程组，异常退出和正常退出都可重复调用。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0141:     if torch.distributed.is_available() and torch.distributed.is_initialized():    # 说明：执行条件分支，处理不同运行环境或配置情况。
0142:         torch.distributed.destroy_process_group()    # 说明：调用函数或方法，执行具体工程动作。
```

## src/pipeline/retrieval/trainer.py

```text
0001: # -*- coding: utf-8 -*-    # 说明：中文/配置注释，说明相邻代码或参数用途。
0002: """HealDA-style 13 层温湿廓线反演训练器。    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0003:     # 说明：空行，用于分隔逻辑块，提高可读性。
0004: 该训练器面向单机 2×H100：使用 DDP、BF16 autocast、梯度累积 no_sync、rank0    # 说明：保留该行以完成当前代码块的语法结构。
0005: checkpoint、跨 rank 指标 all_reduce，并尽量减少训练 step 中的 CPU/GPU 同步。    # 说明：保留该行以完成当前代码块的语法结构。
0006: """    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0007:     # 说明：空行，用于分隔逻辑块，提高可读性。
0008: from __future__ import annotations    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0009:     # 说明：空行，用于分隔逻辑块，提高可读性。
0010: import csv    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0011: import os    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0012: from contextlib import nullcontext    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0013: from typing import Dict, Mapping    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0014:     # 说明：空行，用于分隔逻辑块，提高可读性。
0015: import hydra    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0016: import torch    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0017: import torch.distributed as dist    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0018: from omegaconf import DictConfig    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0019: from tqdm import tqdm    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0020:     # 说明：空行，用于分隔逻辑块，提高可读性。
0021: from src.metrics.retrieval_metrics import retrieval_metrics    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0022: from src.pipeline.base.trainer import BaseTrainer    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0023: from src.utils.device import autocast    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0024:     # 说明：空行，用于分隔逻辑块，提高可读性。
0025:     # 说明：空行，用于分隔逻辑块，提高可读性。
0026: class RetrievalTrainer(BaseTrainer):    # 说明：定义核心类，封装模型、数据或训练职责。
0027:     """XiChen BaseTrainer 兼容的单模型反演训练器。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0028:     # 说明：空行，用于分隔逻辑块，提高可读性。
0029:     def __init__(self, cfg: DictConfig, device, local_rank: int, world_size: int, is_main: bool, **kwargs) -> None:    # 说明：定义函数，复用项目中的关键流程。
0030:         super().__init__(cfg, device, local_rank, world_size, is_main, **kwargs)    # 说明：调用函数或方法，执行具体工程动作。
0031:         self.metrics_csv = os.path.join(self.config.paths.get("output_dir", "logs"), "metrics.csv")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0032:         self.log_every_n_steps = int(self.training_config.get("log_every_n_steps", 20) or 20)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0033:     # 说明：空行，用于分隔逻辑块，提高可读性。
0034:     def _build_models(self) -> None:    # 说明：定义函数，复用项目中的关键流程。
0035:         """实例化反演模型，并可选只编译 Transformer backbone。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0036:         model_cfg = self.config.model.get("net", self.config.model)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0037:         self.model = hydra.utils.instantiate(model_cfg, _recursive_=False).to(self.device)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0038:         compile_cfg = self.training_config.get("compile", {})    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0039:         compile_enabled = bool(compile_cfg.get("enabled", False)) if hasattr(compile_cfg, "get") else False    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0040:         compile_target = str(compile_cfg.get("target", "backbone")) if hasattr(compile_cfg, "get") else "backbone"    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0041:         if compile_enabled and hasattr(torch, "compile"):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0042:             mode = str(compile_cfg.get("mode", "max-autotune-no-cudagraphs"))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0043:             try:    # 说明：进入异常保护块，保证超算任务失败时可诊断。
0044:                 if compile_target == "backbone" and hasattr(self.model, "backbone"):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0045:                     self.model.backbone = torch.compile(self.model.backbone, mode=mode)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0046:                 elif compile_target == "model":    # 说明：执行备用条件分支，覆盖另一类配置或状态。
0047:                     self.model = torch.compile(self.model, mode=mode)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0048:                 if self.is_main:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0049:                     self.log.info(f"torch.compile enabled: target={compile_target}, mode={mode}")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0050:             except Exception as exc:    # 说明：捕获异常并提供安全回退或清晰报错。
0051:                 if self.is_main:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0052:                     self.log.warning(f"torch.compile failed; continuing without compile: {exc}")    # 说明：调用函数或方法，执行具体工程动作。
0053:         if self.is_main and hasattr(self.model, "estimate_vram_gb"):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0054:             try:    # 说明：进入异常保护块，保证超算任务失败时可诊断。
0055:                 est = self.model.estimate_vram_gb(batch_size=int(self.config.datamodule.get("batch_size", 1)))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0056:                 self.log.info(f"Approximate upper-bound VRAM estimate: {est:.2f} GB")    # 说明：调用函数或方法，执行具体工程动作。
0057:             except Exception as exc:    # 说明：捕获异常并提供安全回退或清晰报错。
0058:                 self.log.warning(f"VRAM estimate failed safely: {exc}")    # 说明：调用函数或方法，执行具体工程动作。
0059:     # 说明：空行，用于分隔逻辑块，提高可读性。
0060:     def _setup_optimizer_scheduler(self) -> None:    # 说明：定义函数，复用项目中的关键流程。
0061:         """构建 AdamW 和学习率调度器。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0062:         optim_kwargs = dict(    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0063:             lr=float(self.training_config.get("lr", 1e-4)),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0064:             betas=tuple(self.training_config.get("betas", [0.9, 0.95])),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0065:             weight_decay=float(self.training_config.get("weight_decay", 0.05)),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0066:         )    # 说明：调用函数或方法，执行具体工程动作。
0067:         use_fused = bool(self.training_config.get("fused_adamw", False)) if self.device_type_is_cuda_like() else False    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0068:         try:    # 说明：进入异常保护块，保证超算任务失败时可诊断。
0069:             self.optimizer = torch.optim.AdamW(self.model.parameters(), fused=use_fused, **optim_kwargs)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0070:         except TypeError:    # 说明：捕获异常并提供安全回退或清晰报错。
0071:             self.optimizer = torch.optim.AdamW(self.model.parameters(), **optim_kwargs)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0072:         self.scheduler = self._build_scheduler(self.optimizer)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0073:     # 说明：空行，用于分隔逻辑块，提高可读性。
0074:     def device_type_is_cuda_like(self) -> bool:    # 说明：定义函数，复用项目中的关键流程。
0075:         """判断当前设备是否为 CUDA；构造 fused AdamW 时需要该信息。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0076:         return "cuda" in str(self.device).lower()    # 说明：返回当前函数计算得到的结果。
0077:     # 说明：空行，用于分隔逻辑块，提高可读性。
0078:     def _wrap_ddp(self) -> None:    # 说明：定义函数，复用项目中的关键流程。
0079:         """在 world_size>1 时使用 DDP 包装模型。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0080:         if self.world_size > 1:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0081:             self.model = self._wrap_single_ddp(self.model)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0082:     # 说明：空行，用于分隔逻辑块，提高可读性。
0083:     def _load_checkpoint(self) -> None:    # 说明：定义函数，复用项目中的关键流程。
0084:         """从 last.ckpt 或指定路径恢复模型、优化器、调度器和 AMP scaler。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0085:         ckpt_path = self.training_config.get("resume_ckpt", None)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0086:         if isinstance(ckpt_path, bool):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0087:             ckpt_path = os.path.join(self.ckpt_dir, "last.ckpt") if ckpt_path else None    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0088:         if not ckpt_path or not os.path.exists(str(ckpt_path)):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0089:             if self.is_main:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0090:                 self.log.warning(f"No checkpoint found at {ckpt_path}; training from scratch")    # 说明：调用函数或方法，执行具体工程动作。
0091:             return    # 说明：保留该行以完成当前代码块的语法结构。
0092:         ckpt = torch.load(str(ckpt_path), map_location=self.device)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0093:         self.model.load_state_dict(ckpt["model_state_dict"], strict=True)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0094:         if "optimizer_state_dict" in ckpt:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0095:             self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])    # 说明：调用函数或方法，执行具体工程动作。
0096:         if self.scheduler is not None and ckpt.get("scheduler_state_dict") is not None:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0097:             self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])    # 说明：调用函数或方法，执行具体工程动作。
0098:         if self.scaler is not None and ckpt.get("scaler_state_dict") is not None:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0099:             self.scaler.load_state_dict(ckpt["scaler_state_dict"])    # 说明：调用函数或方法，执行具体工程动作。
0100:         self.start_epoch = int(ckpt.get("epoch", -1)) + 1    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0101:         self.best_loss = float(ckpt.get("best_loss", self.best_loss))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0102:         if self.is_main:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0103:             self.log.info(f"Resumed retrieval checkpoint {ckpt_path} at epoch {self.start_epoch}")    # 说明：调用函数或方法，执行具体工程动作。
0104:     # 说明：空行，用于分隔逻辑块，提高可读性。
0105:     def _save_ckpt(self, ckpt_dir, filename, epoch, val_loss, is_best=False) -> None:    # 说明：定义函数，复用项目中的关键流程。
0106:         """仅 rank0 调用：保存 last.ckpt/best.ckpt。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0107:         os.makedirs(ckpt_dir, exist_ok=True)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0108:         model = self.model.module if hasattr(self.model, "module") else self.model    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0109:         ckpt = {    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0110:             "epoch": int(epoch),    # 说明：调用函数或方法，执行具体工程动作。
0111:             "best_loss": min(self.best_loss, float(val_loss)),    # 说明：调用函数或方法，执行具体工程动作。
0112:             "model_state_dict": model.state_dict(),    # 说明：调用函数或方法，执行具体工程动作。
0113:             "optimizer_state_dict": self.optimizer.state_dict(),    # 说明：调用函数或方法，执行具体工程动作。
0114:             "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,    # 说明：调用函数或方法，执行具体工程动作。
0115:             "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,    # 说明：调用函数或方法，执行具体工程动作。
0116:             "target_vars": getattr(model, "target_vars", None),    # 说明：调用函数或方法，执行具体工程动作。
0117:             "pressure_levels": getattr(model, "pressure_levels", None),    # 说明：调用函数或方法，执行具体工程动作。
0118:             "output_shape": "[B, 26, 181, 360]",    # 说明：保留该行以完成当前代码块的语法结构。
0119:             "active_grid_backend": getattr(model, "active_grid_backend", None),    # 说明：调用函数或方法，执行具体工程动作。
0120:         }    # 说明：保留该行以完成当前代码块的语法结构。
0121:         torch.save(ckpt, os.path.join(ckpt_dir, filename))    # 说明：调用函数或方法，执行具体工程动作。
0122:     # 说明：空行，用于分隔逻辑块，提高可读性。
0123:     def _append_metrics_csv(self, epoch: int, metrics: Dict[str, float]) -> None:    # 说明：定义函数，复用项目中的关键流程。
0124:         """rank0 追加写 metrics.csv，便于 Slurm 任务结束后直接汇总。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0125:         if not self.is_main:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0126:             return    # 说明：保留该行以完成当前代码块的语法结构。
0127:         os.makedirs(os.path.dirname(self.metrics_csv), exist_ok=True)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0128:         row = {"epoch": int(epoch), **metrics}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0129:         write_header = not os.path.exists(self.metrics_csv)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0130:         with open(self.metrics_csv, "a", newline="", encoding="utf-8") as f:    # 说明：使用上下文管理器，安全管理文件、AMP 或 DDP 同步。
0131:             writer = csv.DictWriter(f, fieldnames=list(row.keys()))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0132:             if write_header:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0133:                 writer.writeheader()    # 说明：调用函数或方法，执行具体工程动作。
0134:             writer.writerow(row)    # 说明：调用函数或方法，执行具体工程动作。
0135:     # 说明：空行，用于分隔逻辑块，提高可读性。
0136:     def _move_batch_target(self, batch: Mapping[str, object]) -> dict:    # 说明：定义函数，复用项目中的关键流程。
0137:         """只提前搬运监督标签；变长观测点云在 sensor embedder 内按需搬到当前 GPU。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0138:         out = dict(batch)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0139:         out["target"] = out["target"].to(self.device, non_blocking=True)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0140:         return out    # 说明：返回当前函数计算得到的结果。
0141:     # 说明：空行，用于分隔逻辑块，提高可读性。
0142:     def _maybe_no_sync(self, should_sync: bool):    # 说明：定义函数，复用项目中的关键流程。
0143:         """DDP 梯度累积时只在真正 optimizer step 前同步梯度。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0144:         if should_sync or self.world_size <= 1 or not hasattr(self.model, "no_sync"):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0145:             return nullcontext()    # 说明：返回当前函数计算得到的结果。
0146:         return self.model.no_sync()    # 说明：返回当前函数计算得到的结果。
0147:     # 说明：空行，用于分隔逻辑块，提高可读性。
0148:     def _all_reduce_sums(self, sums: Mapping[str, float], seen: int) -> tuple[Dict[str, float], int]:    # 说明：定义函数，复用项目中的关键流程。
0149:         """把各 rank 的加权和与样本数做 all_reduce，返回全局求和结果。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0150:         keys = sorted(sums.keys())    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0151:         values = [float(sums[k]) for k in keys] + [float(seen)]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0152:         tensor = torch.tensor(values, dtype=torch.float64, device=self.device)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0153:         if self.world_size > 1 and dist.is_available() and dist.is_initialized():    # 说明：执行条件分支，处理不同运行环境或配置情况。
0154:             dist.all_reduce(tensor, op=dist.ReduceOp.SUM)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0155:         reduced = {k: float(tensor[i].item()) for i, k in enumerate(keys)}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0156:         reduced_seen = int(tensor[-1].item())    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0157:         return reduced, reduced_seen    # 说明：返回当前函数计算得到的结果。
0158:     # 说明：空行，用于分隔逻辑块，提高可读性。
0159:     @staticmethod    # 说明：装饰器，修改函数/方法行为或启用框架入口。
0160:     def _average_sums(prefix: str, sums: Mapping[str, float], seen: int) -> Dict[str, float]:    # 说明：定义函数，复用项目中的关键流程。
0161:         """把加权和转换成平均指标，并追加 train/ 或 val/ 前缀。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0162:         denom = max(int(seen), 1)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0163:         return {f"{prefix}/{k}": float(v) / denom for k, v in sums.items()}    # 说明：返回当前函数计算得到的结果。
0164:     # 说明：空行，用于分隔逻辑块，提高可读性。
0165:     def train_epoch(self, loader, epoch, epochs):    # 说明：定义函数，复用项目中的关键流程。
0166:         """执行一个训练 epoch，返回跨 rank 平均后的训练指标。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0167:         self.model.train()    # 说明：调用函数或方法，执行具体工程动作。
0168:         grad_accum = max(int(self.training_config.get("gradient_accumulation_steps", 1) or 1), 1)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0169:         limit = self.training_config.get("limit_train_batches", None)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0170:         sums = {"loss": 0.0, "temperature_loss": 0.0, "humidity_loss": 0.0}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0171:         seen = 0    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0172:         max_steps = len(loader) if limit is None else min(len(loader), int(limit))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0173:         pbar = tqdm(loader, desc=f"Train {epoch + 1}/{epochs}", disable=not self.is_main)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0174:         self.optimizer.zero_grad(set_to_none=True)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0175:         for step, batch in enumerate(pbar):    # 说明：遍历集合或数据流，逐项完成处理。
0176:             if limit is not None and step >= int(limit):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0177:                 break    # 说明：保留该行以完成当前代码块的语法结构。
0178:             should_step = ((step + 1) % grad_accum == 0) or ((step + 1) >= max_steps)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0179:             batch = self._move_batch_target(batch)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0180:             with self._maybe_no_sync(should_sync=should_step):    # 说明：使用上下文管理器，安全管理文件、AMP 或 DDP 同步。
0181:                 with autocast(self.device_type, dtype=self.precision_type):    # 说明：使用上下文管理器，安全管理文件、AMP 或 DDP 同步。
0182:                     pred = self.model(batch)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0183:                     losses = self.loss_fn(pred, batch["target"])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0184:                     loss = losses["total_loss"] / grad_accum    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0185:                 use_scaler = self.scaler is not None and self.precision_type is torch.float16    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0186:                 if use_scaler:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0187:                     self.scaler.scale(loss).backward()    # 说明：调用函数或方法，执行具体工程动作。
0188:                 else:    # 说明：执行默认分支，保证逻辑闭环。
0189:                     loss.backward()    # 说明：调用函数或方法，执行具体工程动作。
0190:             if should_step:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0191:                 max_norm = float(self.training_config.get("max_grad_norm", 0.0) or 0.0)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0192:                 if max_norm > 0:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0193:                     if use_scaler:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0194:                         self.scaler.unscale_(self.optimizer)    # 说明：调用函数或方法，执行具体工程动作。
0195:                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)    # 说明：调用函数或方法，执行具体工程动作。
0196:                 if use_scaler:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0197:                     self.scaler.step(self.optimizer)    # 说明：调用函数或方法，执行具体工程动作。
0198:                     self.scaler.update()    # 说明：调用函数或方法，执行具体工程动作。
0199:                 else:    # 说明：执行默认分支，保证逻辑闭环。
0200:                     self.optimizer.step()    # 说明：调用函数或方法，执行具体工程动作。
0201:                 self.optimizer.zero_grad(set_to_none=True)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0202:             bsz = int(batch["target"].shape[0])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0203:             seen += bsz    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0204:             sums["loss"] += float(losses["total_loss"].detach()) * bsz    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0205:             sums["temperature_loss"] += float(losses["temperature_loss"].detach()) * bsz    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0206:             sums["humidity_loss"] += float(losses["humidity_loss"].detach()) * bsz    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0207:             if self.is_main and (step + 1) % self.log_every_n_steps == 0:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0208:                 pbar.set_postfix(loss=sums["loss"] / max(seen, 1))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0209:         reduced, reduced_seen = self._all_reduce_sums(sums, seen)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0210:         return self._average_sums("train", reduced, reduced_seen)    # 说明：返回当前函数计算得到的结果。
0211:     # 说明：空行，用于分隔逻辑块，提高可读性。
0212:     @torch.no_grad()    # 说明：装饰器，修改函数/方法行为或启用框架入口。
0213:     def validate(self, loader, epoch, epochs):    # 说明：定义函数，复用项目中的关键流程。
0214:         """执行一个验证 epoch，返回跨 rank 平均后的验证 loss 与诊断指标。"""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0215:         self.model.eval()    # 说明：调用函数或方法，执行具体工程动作。
0216:         sums: Dict[str, float] = {"loss": 0.0, "temperature_loss": 0.0, "humidity_loss": 0.0}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0217:         seen = 0    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0218:         limit = self.training_config.get("limit_val_batches", None)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0219:         pbar = tqdm(loader, desc=f"Val {epoch + 1}/{epochs}", disable=not self.is_main)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0220:         for step, batch in enumerate(pbar):    # 说明：遍历集合或数据流，逐项完成处理。
0221:             if limit is not None and step >= int(limit):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0222:                 break    # 说明：保留该行以完成当前代码块的语法结构。
0223:             batch = self._move_batch_target(batch)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0224:             with autocast(self.device_type, dtype=self.precision_type):    # 说明：使用上下文管理器，安全管理文件、AMP 或 DDP 同步。
0225:                 pred = self.model(batch)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0226:                 losses = self.loss_fn(pred, batch["target"])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0227:             bsz = int(batch["target"].shape[0])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0228:             seen += bsz    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0229:             sums["loss"] += float(losses["total_loss"].detach()) * bsz    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0230:             sums["temperature_loss"] += float(losses["temperature_loss"].detach()) * bsz    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0231:             sums["humidity_loss"] += float(losses["humidity_loss"].detach()) * bsz    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0232:             metric = retrieval_metrics(pred.detach(), batch["target"].detach(), batch.get("pressure_levels"))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0233:             for key, value in metric.items():    # 说明：遍历集合或数据流，逐项完成处理。
0234:                 sums[key] = sums.get(key, 0.0) + float(value) * bsz    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0235:         reduced, reduced_seen = self._all_reduce_sums(sums, seen)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0236:         if reduced_seen == 0:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0237:             return {"val/loss": float("inf")}    # 说明：返回当前函数计算得到的结果。
0238:         out = self._average_sums("val", reduced, reduced_seen)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0239:         self._append_metrics_csv(epoch, out)    # 说明：调用函数或方法，执行具体工程动作。
0240:         return out    # 说明：返回当前函数计算得到的结果。
```

## src/datamodules/retrieval/healda_datamodule.py

```text
0001: # -*- coding: utf-8 -*-    # 说明：中文/配置注释，说明相邻代码或参数用途。
0002: """DataModule wrapper for HealDA-style T/Q retrieval."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0003:     # 说明：空行，用于分隔逻辑块，提高可读性。
0004: from __future__ import annotations    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0005:     # 说明：空行，用于分隔逻辑块，提高可读性。
0006: from collections.abc import Mapping, Sequence    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0007: import random    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0008:     # 说明：空行，用于分隔逻辑块，提高可读性。
0009: import numpy as np    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0010: import torch    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0011: from torch.utils.data import DataLoader, DistributedSampler    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0012:     # 说明：空行，用于分隔逻辑块，提高可读性。
0013: from .healda_dataset import HealDARetrievalDataset, TARGET_VARS, PRESSURE_LEVELS, XICHEN_ERA5_ALL_VARS, collate_retrieval_batch    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0014:     # 说明：空行，用于分隔逻辑块，提高可读性。
0015:     # 说明：空行，用于分隔逻辑块，提高可读性。
0016: class HealDARetrievalDataModule:    # 说明：定义核心类，封装模型、数据或训练职责。
0017:     """Pure PyTorch datamodule used by ``src.pipeline.retrieval.trainer``."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0018:     # 说明：空行，用于分隔逻辑块，提高可读性。
0019:     def __init__(    # 说明：定义函数，复用项目中的关键流程。
0020:         self,    # 说明：保留该行以完成当前代码块的语法结构。
0021:         obs_dir: str,    # 说明：保留该行以完成当前代码块的语法结构。
0022:         era5_dir: str,    # 说明：保留该行以完成当前代码块的语法结构。
0023:         scale_dir: str,    # 说明：保留该行以完成当前代码块的语法结构。
0024:         sensors: dict | list,    # 说明：保留该行以完成当前代码块的语法结构。
0025:         target: dict | None = None,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0026:         data: dict | None = None,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0027:         qc: dict | None = None,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0028:         obs_default_normalization: dict | None = None,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0029:         start_train_year: int = 2016,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0030:         start_val_year: int = 2022,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0031:         start_test_year: int = 2023,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0032:         end_year: int = 2024,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0033:         batch_size: int = 1,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0034:         num_workers: int = 4,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0035:         shuffle: bool = True,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0036:         pin_memory: bool = True,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0037:         prefetch_factor: int = 4,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0038:         seed: int = 1024,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0039:         debug: bool = False,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0040:         max_debug_samples: int = 8,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0041:         distributed: bool = False,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0042:         num_replicas: int = 1,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0043:         rank: int = 0,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0044:         **kwargs,    # 说明：保留该行以完成当前代码块的语法结构。
0045:     ) -> None:    # 说明：保留该行以完成当前代码块的语法结构。
0046:         self.obs_dir = obs_dir    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0047:         self.era5_dir = era5_dir    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0048:         self.scale_dir = scale_dir    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0049:         self.sensors_cfg = sensors    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0050:         self.target_cfg = target or {}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0051:         self.data_cfg = data or {}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0052:         self.qc = qc or {}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0053:         self.obs_default_normalization = obs_default_normalization or {}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0054:         self.start_train_year = int(start_train_year)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0055:         self.start_val_year = int(start_val_year)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0056:         self.start_test_year = int(start_test_year)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0057:         self.end_year = int(end_year)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0058:         self.batch_size = int(batch_size)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0059:         self.num_workers = int(num_workers)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0060:         self.shuffle = bool(shuffle)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0061:         self.pin_memory = bool(pin_memory)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0062:         self.prefetch_factor = int(prefetch_factor)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0063:         self.seed = int(seed)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0064:         self.debug = bool(debug)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0065:         self.max_debug_samples = int(max_debug_samples)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0066:         self.distributed = bool(distributed)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0067:         self.num_replicas = int(num_replicas)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0068:         self.rank = int(rank)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0069:         self.extra_kwargs = dict(kwargs)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0070:     # 说明：空行，用于分隔逻辑块，提高可读性。
0071:         self.train_data = None    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0072:         self.val_data = None    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0073:         self.test_data = None    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0074:     # 说明：空行，用于分隔逻辑块，提高可读性。
0075:     @staticmethod    # 说明：装饰器，修改函数/方法行为或启用框架入口。
0076:     def _as_list(value) -> list[str]:    # 说明：定义函数，复用项目中的关键流程。
0077:         """Convert a Hydra/OmegaConf scalar or list-like value into a plain list.    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0078:     # 说明：空行，用于分隔逻辑块，提高可读性。
0079:         OmegaConf ``DictConfig``/``ListConfig`` objects are not plain ``dict``/``list``    # 说明：保留该行以完成当前代码块的语法结构。
0080:         instances.  Using ``list(DictConfig)`` on the configured ``sensors`` block    # 说明：调用函数或方法，执行具体工程动作。
0081:         returns the group keys (``satellite`` and ``conventional``), which are not    # 说明：调用函数或方法，执行具体工程动作。
0082:         real sensors.  This helper keeps the configured leaf sensor names only.    # 说明：保留该行以完成当前代码块的语法结构。
0083:         """    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0084:         if value is None:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0085:             return []    # 说明：返回当前函数计算得到的结果。
0086:         if isinstance(value, str):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0087:             return [value]    # 说明：返回当前函数计算得到的结果。
0088:         if isinstance(value, Mapping):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0089:             out: list[str] = []    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0090:             for nested in value.values():    # 说明：遍历集合或数据流，逐项完成处理。
0091:                 out.extend(HealDARetrievalDataModule._as_list(nested))    # 说明：调用函数或方法，执行具体工程动作。
0092:             return out    # 说明：返回当前函数计算得到的结果。
0093:         if isinstance(value, Sequence):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0094:             return [str(v) for v in value]    # 说明：返回当前函数计算得到的结果。
0095:         return [str(value)]    # 说明：返回当前函数计算得到的结果。
0096:     # 说明：空行，用于分隔逻辑块，提高可读性。
0097:     @property    # 说明：装饰器，修改函数/方法行为或启用框架入口。
0098:     def sensors(self) -> list[str]:    # 说明：定义函数，复用项目中的关键流程。
0099:         """Return flat sensor aliases, never Hydra group names.    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0100:     # 说明：空行，用于分隔逻辑块，提高可读性。
0101:         Accepts either    # 说明：保留该行以完成当前代码块的语法结构。
0102:     # 说明：空行，用于分隔逻辑块，提高可读性。
0103:         ``sensors: [atms, amsua, mhs, hrs4, gdas_prebufr]``    # 说明：保留该行以完成当前代码块的语法结构。
0104:     # 说明：空行，用于分隔逻辑块，提高可读性。
0105:         or the recommended grouped form    # 说明：保留该行以完成当前代码块的语法结构。
0106:     # 说明：空行，用于分隔逻辑块，提高可读性。
0107:         ``sensors: {satellite: [...], conventional: [...]}``.    # 说明：保留该行以完成当前代码块的语法结构。
0108:         """    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0109:         if isinstance(self.sensors_cfg, Mapping):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0110:             flat = []    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0111:             for group in ("satellite", "conventional"):    # 说明：遍历集合或数据流，逐项完成处理。
0112:                 flat.extend(self._as_list(self.sensors_cfg.get(group, [])))    # 说明：调用函数或方法，执行具体工程动作。
0113:             if not flat:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0114:                 # Fallback for custom mappings whose values are already sensor lists.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0115:                 flat = self._as_list(self.sensors_cfg)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0116:         else:    # 说明：执行默认分支，保证逻辑闭环。
0117:             flat = self._as_list(self.sensors_cfg)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0118:     # 说明：空行，用于分隔逻辑块，提高可读性。
0119:         # Preserve order while avoiding accidental duplicates from aliases.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0120:         seen: set[str] = set()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0121:         deduped: list[str] = []    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0122:         for sensor in flat:    # 说明：遍历集合或数据流，逐项完成处理。
0123:             sensor = str(sensor).strip()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0124:             if sensor and sensor not in seen:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0125:                 seen.add(sensor)    # 说明：调用函数或方法，执行具体工程动作。
0126:                 deduped.append(sensor)    # 说明：调用函数或方法，执行具体工程动作。
0127:         return deduped    # 说明：返回当前函数计算得到的结果。
0128:     # 说明：空行，用于分隔逻辑块，提高可读性。
0129:     def _worker_init_fn(self, worker_id: int) -> None:    # 说明：定义函数，复用项目中的关键流程。
0130:         worker_seed = int(self.seed) + int(self.rank) * 100000 + int(worker_id)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0131:         random.seed(worker_seed)    # 说明：调用函数或方法，执行具体工程动作。
0132:         np.random.seed(worker_seed)    # 说明：调用函数或方法，执行具体工程动作。
0133:         torch.manual_seed(worker_seed)    # 说明：调用函数或方法，执行具体工程动作。
0134:     # 说明：空行，用于分隔逻辑块，提高可读性。
0135:     def _get_sampler(self, dataset, shuffle: bool):    # 说明：定义函数，复用项目中的关键流程。
0136:         if self.distributed and self.num_replicas > 1:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0137:             return DistributedSampler(dataset, num_replicas=self.num_replicas, rank=self.rank, shuffle=shuffle, seed=self.seed)    # 说明：返回当前函数计算得到的结果。
0138:         return None    # 说明：返回当前函数计算得到的结果。
0139:     # 说明：空行，用于分隔逻辑块，提高可读性。
0140:     def _make_dataset(self, mode: str, start_year: int, end_year: int) -> HealDARetrievalDataset:    # 说明：定义函数，复用项目中的关键流程。
0141:         target_variables = self.target_cfg.get("target_vars", self.target_cfg.get("variables", TARGET_VARS))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0142:         # Convert target.variables=[t,q] into explicit t-50... q-1000 names.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0143:         if target_variables and set(target_variables).issubset({"t", "q"}):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0144:             levels = self.target_cfg.get("pressure_levels", PRESSURE_LEVELS)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0145:             target_variables = [*(f"t-{p}" for p in levels), *(f"q-{p}" for p in levels)]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0146:         return HealDARetrievalDataset(    # 说明：返回当前函数计算得到的结果。
0147:             obs_dir=self.obs_dir,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0148:             era5_dir=self.era5_dir,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0149:             scale_dir=self.scale_dir,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0150:             mode=mode,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0151:             sensors=self.sensors,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0152:             target_variables=target_variables,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0153:             pressure_levels=self.target_cfg.get("pressure_levels", PRESSURE_LEVELS),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0154:             era5_all_vars=self.data_cfg.get("era5_all_vars", XICHEN_ERA5_ALL_VARS),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0155:             grid_shape=self.data_cfg.get("grid_shape", [181, 360]),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0156:             obs_window=self.data_cfg.get("obs_window", {"start_hours": -21, "end_hours": 3}),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0157:             no_lookahead=self.data_cfg.get("no_lookahead", False),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0158:             no_lookahead_window=self.data_cfg.get("no_lookahead_window", {"start_hours": -24, "end_hours": 0}),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0159:             dt_data=self.data_cfg.get("dt_data", 6),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0160:             dt_obs=self.data_cfg.get("dt_obs", 3),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0161:             start_year=start_year,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0162:             end_year=end_year,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0163:             debug=self.debug,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0164:             max_debug_samples=self.max_debug_samples,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0165:             max_points_per_sensor=self.data_cfg.get("max_points_per_sensor", 250_000),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0166:             strict_time_index=self.data_cfg.get("strict_time_index", False),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0167:             target_cache_size=self.data_cfg.get("target_cache_size", 16),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0168:             normalize_target=self.data_cfg.get("normalize_target", True),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0169:             normalize_obs=self.data_cfg.get("normalize_obs", True),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0170:             require_obs_stats=self.data_cfg.get("require_obs_stats", False),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0171:             qc=self.qc,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0172:             obs_default_normalization=self.obs_default_normalization,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0173:         )    # 说明：调用函数或方法，执行具体工程动作。
0174:     # 说明：空行，用于分隔逻辑块，提高可读性。
0175:     def setup(self) -> None:    # 说明：定义函数，复用项目中的关键流程。
0176:         if self.train_data is None:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0177:             self.train_data = self._make_dataset("train", self.start_train_year, self.start_val_year)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0178:         if self.val_data is None:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0179:             self.val_data = self._make_dataset("val", self.start_val_year, self.start_test_year)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0180:         if self.test_data is None:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0181:             self.test_data = self._make_dataset("test", self.start_test_year, self.end_year)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0182:     # 说明：空行，用于分隔逻辑块，提高可读性。
0183:     def _loader(self, dataset, shuffle: bool, drop_last: bool) -> DataLoader:    # 说明：定义函数，复用项目中的关键流程。
0184:         sampler = self._get_sampler(dataset, shuffle=shuffle)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0185:         kwargs = dict(    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0186:             batch_size=self.batch_size,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0187:             sampler=sampler,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0188:             shuffle=(sampler is None and shuffle),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0189:             num_workers=self.num_workers,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0190:             collate_fn=collate_retrieval_batch,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0191:             pin_memory=self.pin_memory,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0192:             drop_last=drop_last,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0193:             worker_init_fn=self._worker_init_fn,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0194:             persistent_workers=self.num_workers > 0,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0195:         )    # 说明：调用函数或方法，执行具体工程动作。
0196:         if self.num_workers > 0:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0197:             kwargs["prefetch_factor"] = self.prefetch_factor    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0198:         return DataLoader(dataset, **kwargs)    # 说明：返回当前函数计算得到的结果。
0199:     # 说明：空行，用于分隔逻辑块，提高可读性。
0200:     def train_dataloader(self) -> DataLoader:    # 说明：定义函数，复用项目中的关键流程。
0201:         self.setup()    # 说明：调用函数或方法，执行具体工程动作。
0202:         return self._loader(self.train_data, shuffle=self.shuffle, drop_last=True)    # 说明：返回当前函数计算得到的结果。
0203:     # 说明：空行，用于分隔逻辑块，提高可读性。
0204:     def val_dataloader(self) -> DataLoader:    # 说明：定义函数，复用项目中的关键流程。
0205:         self.setup()    # 说明：调用函数或方法，执行具体工程动作。
0206:         return self._loader(self.val_data, shuffle=False, drop_last=False)    # 说明：返回当前函数计算得到的结果。
0207:     # 说明：空行，用于分隔逻辑块，提高可读性。
0208:     def test_dataloader(self) -> DataLoader:    # 说明：定义函数，复用项目中的关键流程。
0209:         self.setup()    # 说明：调用函数或方法，执行具体工程动作。
0210:         return self._loader(self.test_data, shuffle=False, drop_last=False)    # 说明：返回当前函数计算得到的结果。
```

## src/datamodules/retrieval/healda_dataset.py

```text
0001: # -*- coding: utf-8 -*-    # 说明：中文/配置注释，说明相邻代码或参数用途。
0002: """Datasets for HealDA-style multi-source observation -> ERA5 T/Q retrieval.    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0003:     # 说明：空行，用于分隔逻辑块，提高可读性。
0004: The dataset deliberately keeps observations as variable-length point clouds.  It    # 说明：保留该行以完成当前代码块的语法结构。
0005: can read the gridded 1.0 degree XiChen-style NPY files by flattening observed    # 说明：保留该行以完成当前代码块的语法结构。
0006: pixels into scalar observations, and it can also consume point-like npy/npz files    # 说明：保留该行以完成当前代码块的语法结构。
0007: when fields such as ``lat``/``lon``/``observation`` are available.    # 说明：保留该行以完成当前代码块的语法结构。
0008:     # 说明：空行，用于分隔逻辑块，提高可读性。
0009: Returned target shape is always ``[26, 181, 360]`` in this order:    # 说明：保留该行以完成当前代码块的语法结构。
0010: ``t-50 ... t-1000, q-50 ... q-1000``.    # 说明：保留该行以完成当前代码块的语法结构。
0011: """    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0012:     # 说明：空行，用于分隔逻辑块，提高可读性。
0013: from __future__ import annotations    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0014:     # 说明：空行，用于分隔逻辑块，提高可读性。
0015: import json    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0016: import os    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0017: import re    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0018: import warnings    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0019: from collections import OrderedDict    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0020: from dataclasses import dataclass    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0021: from functools import lru_cache    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0022: from datetime import datetime, timedelta    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0023: from glob import glob    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0024: from pathlib import Path    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0025: from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0026:     # 说明：空行，用于分隔逻辑块，提高可读性。
0027: import numpy as np    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0028: import torch    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0029: from torch.utils.data import Dataset    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0030:     # 说明：空行，用于分隔逻辑块，提高可读性。
0031: PRESSURE_LEVELS: List[int] = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0032: TARGET_VARS: List[str] = [*(f"t-{p}" for p in PRESSURE_LEVELS), *(f"q-{p}" for p in PRESSURE_LEVELS)]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0033:     # 说明：空行，用于分隔逻辑块，提高可读性。
0034: SENSOR_ALIAS: Dict[str, str] = {    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0035:     "atms": "atms",    # 说明：保留该行以完成当前代码块的语法结构。
0036:     "amsu-a": "amsua",    # 说明：保留该行以完成当前代码块的语法结构。
0037:     "amsua": "amsua",    # 说明：保留该行以完成当前代码块的语法结构。
0038:     "amsu_a": "amsua",    # 说明：保留该行以完成当前代码块的语法结构。
0039:     "mhs": "mhs",    # 说明：保留该行以完成当前代码块的语法结构。
0040:     "hirs": "hrs4",    # 说明：保留该行以完成当前代码块的语法结构。
0041:     "hirs4": "hrs4",    # 说明：保留该行以完成当前代码块的语法结构。
0042:     "hrs4": "hrs4",    # 说明：保留该行以完成当前代码块的语法结构。
0043:     "gdas_prebufr": "gdas_prebufr",    # 说明：保留该行以完成当前代码块的语法结构。
0044:     "gdas_prepbufr": "gdas_prebufr",    # 说明：保留该行以完成当前代码块的语法结构。
0045:     "prepbufr": "gdas_prebufr",    # 说明：保留该行以完成当前代码块的语法结构。
0046:     "prebufr": "gdas_prebufr",    # 说明：保留该行以完成当前代码块的语法结构。
0047:     "GDAS_prebufr_corrected_npy_1.0deg": "gdas_prebufr",    # 说明：保留该行以完成当前代码块的语法结构。
0048: }    # 说明：保留该行以完成当前代码块的语法结构。
0049:     # 说明：空行，用于分隔逻辑块，提高可读性。
0050: SENSOR_DIR_CANDIDATES: Dict[str, Sequence[str]] = {    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0051:     "atms": ("ATMS", "atms", "1batms_merged_npy_1.0deg", "1batms"),    # 说明：调用函数或方法，执行具体工程动作。
0052:     "amsua": ("AMSU-A", "AMSUA", "amsua", "1bamsua_merged_npy_1.0deg", "1bamsua"),    # 说明：调用函数或方法，执行具体工程动作。
0053:     "mhs": ("MHS", "mhs", "1bmhs_merged_npy_1.0deg", "1bmhs"),    # 说明：调用函数或方法，执行具体工程动作。
0054:     "hrs4": ("HIRS4", "HRS4", "hrs4", "hirs4", "1bhrs4_merged_npy_1.0deg", "1bhrs4"),    # 说明：调用函数或方法，执行具体工程动作。
0055:     "gdas_prebufr": (    # 说明：调用函数或方法，执行具体工程动作。
0056:         "GDAS_prebufr_corrected_npy_1.0deg",    # 说明：保留该行以完成当前代码块的语法结构。
0057:         "GDAS_prepbufr_merged_npy_1.0deg",    # 说明：保留该行以完成当前代码块的语法结构。
0058:         "gdas_prebufr",    # 说明：保留该行以完成当前代码块的语法结构。
0059:         "prepbufr",    # 说明：保留该行以完成当前代码块的语法结构。
0060:     ),    # 说明：保留该行以完成当前代码块的语法结构。
0061: }    # 说明：保留该行以完成当前代码块的语法结构。
0062:     # 说明：空行，用于分隔逻辑块，提高可读性。
0063: SATELLITE_SENSORS = {"atms", "amsua", "mhs", "hrs4"}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0064: CONVENTIONAL_SENSORS = {"gdas_prebufr"}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0065:     # 说明：空行，用于分隔逻辑块，提高可读性。
0066: # XiChen full 13-level state order, used only to select labels when an ERA5 file    # 说明：中文/配置注释，说明相邻代码或参数用途。
0067: # stores all channels.  It is copied from the existing XiChen configs, not from    # 说明：中文/配置注释，说明相邻代码或参数用途。
0068: # file-name guesswork.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0069: XICHEN_ERA5_ALL_VARS: List[str] = [    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0070:     "t2m", "u10", "v10", "msl",    # 说明：保留该行以完成当前代码块的语法结构。
0071:     *(f"z-{p}" for p in PRESSURE_LEVELS),    # 说明：调用函数或方法，执行具体工程动作。
0072:     *(f"u-{p}" for p in PRESSURE_LEVELS),    # 说明：调用函数或方法，执行具体工程动作。
0073:     *(f"v-{p}" for p in PRESSURE_LEVELS),    # 说明：调用函数或方法，执行具体工程动作。
0074:     *(f"t-{p}" for p in PRESSURE_LEVELS),    # 说明：调用函数或方法，执行具体工程动作。
0075:     *(f"q-{p}" for p in PRESSURE_LEVELS),    # 说明：调用函数或方法，执行具体工程动作。
0076: ]    # 说明：保留该行以完成当前代码块的语法结构。
0077:     # 说明：空行，用于分隔逻辑块，提高可读性。
0078: _TIME_PATTERNS = (    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0079:     re.compile(    # 说明：调用函数或方法，执行具体工程动作。
0080:         r"(?P<year>\d{4})[-_](?P<month>\d{2})[-_](?P<day>\d{2})[\\/]+"    # 说明：调用函数或方法，执行具体工程动作。
0081:         r"(?P<hour>\d{2})[:_-](?P<minute>\d{2})[:_-](?P<second>\d{2})"    # 说明：调用函数或方法，执行具体工程动作。
0082:     ),    # 说明：保留该行以完成当前代码块的语法结构。
0083:     re.compile(    # 说明：调用函数或方法，执行具体工程动作。
0084:         r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2}).*?"    # 说明：调用函数或方法，执行具体工程动作。
0085:         r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})"    # 说明：调用函数或方法，执行具体工程动作。
0086:     ),    # 说明：保留该行以完成当前代码块的语法结构。
0087: )    # 说明：调用函数或方法，执行具体工程动作。
0088:     # 说明：空行，用于分隔逻辑块，提高可读性。
0089:     # 说明：空行，用于分隔逻辑块，提高可读性。
0090: def canonical_sensor(name: str) -> str:    # 说明：定义函数，复用项目中的关键流程。
0091:     """Return canonical sensor name and reject satwnd/ascat for this task."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0092:     key = str(name).strip()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0093:     canon = SENSOR_ALIAS.get(key, SENSOR_ALIAS.get(key.lower()))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0094:     if canon is None:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0095:         raise ValueError(f"Unknown sensor alias {name!r}. Allowed aliases: {sorted(SENSOR_ALIAS)}")    # 说明：调用函数或方法，执行具体工程动作。
0096:     if canon in {"satwnd", "ascat"}:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0097:         raise ValueError("satwnd/ascat are intentionally disabled for the T/Q retrieval task")    # 说明：调用函数或方法，执行具体工程动作。
0098:     return canon    # 说明：返回当前函数计算得到的结果。
0099:     # 说明：空行，用于分隔逻辑块，提高可读性。
0100:     # 说明：空行，用于分隔逻辑块，提高可读性。
0101: def parse_datetime_from_path(path: str | os.PathLike[str]) -> Optional[datetime]:    # 说明：定义函数，复用项目中的关键流程。
0102:     """Parse XiChen-style ``YYYY-MM-DD/HH:MM:SS`` or compact variants from a path."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0103:     text = str(path)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0104:     for pattern in _TIME_PATTERNS:    # 说明：遍历集合或数据流，逐项完成处理。
0105:         for match in pattern.finditer(text):    # 说明：遍历集合或数据流，逐项完成处理。
0106:             gd = {k: int(v) for k, v in match.groupdict().items()}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0107:             try:    # 说明：进入异常保护块，保证超算任务失败时可诊断。
0108:                 return datetime(gd["year"], gd["month"], gd["day"], gd["hour"], gd["minute"], gd["second"])    # 说明：返回当前函数计算得到的结果。
0109:             except ValueError:    # 说明：捕获异常并提供安全回退或清晰报错。
0110:                 continue    # 说明：保留该行以完成当前代码块的语法结构。
0111:     return None    # 说明：返回当前函数计算得到的结果。
0112:     # 说明：空行，用于分隔逻辑块，提高可读性。
0113:     # 说明：空行，用于分隔逻辑块，提高可读性。
0114: def datetime_path(root: str | os.PathLike[str], t: datetime, suffix: str) -> str:    # 说明：定义函数，复用项目中的关键流程。
0115:     """XiChen-style time path helper."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0116:     return os.path.join(    # 说明：返回当前函数计算得到的结果。
0117:         str(root),    # 说明：调用函数或方法，执行具体工程动作。
0118:         f"{t.year:04d}",    # 说明：保留该行以完成当前代码块的语法结构。
0119:         f"{t.year:04d}-{t.month:02d}-{t.day:02d}",    # 说明：保留该行以完成当前代码块的语法结构。
0120:         f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}-{suffix}",    # 说明：保留该行以完成当前代码块的语法结构。
0121:     )    # 说明：调用函数或方法，执行具体工程动作。
0122:     # 说明：空行，用于分隔逻辑块，提高可读性。
0123:     # 说明：空行，用于分隔逻辑块，提高可读性。
0124: def datetime_era5_path(root: str | os.PathLike[str], t: datetime) -> str:    # 说明：定义函数，复用项目中的关键流程。
0125:     """XiChen-style full-state ERA5 file path helper.    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0126:     # 说明：空行，用于分隔逻辑块，提高可读性。
0127:     Some XiChen datasets store a complete state in one file named    # 说明：保留该行以完成当前代码块的语法结构。
0128:     ``HH:MM:SS.npy``.  The retrieval dataset also supports the per-variable    # 说明：保留该行以完成当前代码块的语法结构。
0129:     layout used on ``/public02`` where each target is stored separately, for    # 说明：保留该行以完成当前代码块的语法结构。
0130:     example ``HH:MM:SS-t-1000.npy``.    # 说明：保留该行以完成当前代码块的语法结构。
0131:     """    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0132:     return os.path.join(    # 说明：返回当前函数计算得到的结果。
0133:         str(root),    # 说明：调用函数或方法，执行具体工程动作。
0134:         f"{t.year:04d}",    # 说明：保留该行以完成当前代码块的语法结构。
0135:         f"{t.year:04d}-{t.month:02d}-{t.day:02d}",    # 说明：保留该行以完成当前代码块的语法结构。
0136:         f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}.npy",    # 说明：保留该行以完成当前代码块的语法结构。
0137:     )    # 说明：调用函数或方法，执行具体工程动作。
0138:     # 说明：空行，用于分隔逻辑块，提高可读性。
0139:     # 说明：空行，用于分隔逻辑块，提高可读性。
0140: def datetime_era5_variable_path(root: str | os.PathLike[str], t: datetime, var: str) -> str:    # 说明：定义函数，复用项目中的关键流程。
0141:     """Path for per-variable ERA5 labels, e.g. ``22:00:00-t-1000.npy``."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0142:     return os.path.join(    # 说明：返回当前函数计算得到的结果。
0143:         str(root),    # 说明：调用函数或方法，执行具体工程动作。
0144:         f"{t.year:04d}",    # 说明：保留该行以完成当前代码块的语法结构。
0145:         f"{t.year:04d}-{t.month:02d}-{t.day:02d}",    # 说明：保留该行以完成当前代码块的语法结构。
0146:         f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}-{var}.npy",    # 说明：保留该行以完成当前代码块的语法结构。
0147:     )    # 说明：调用函数或方法，执行具体工程动作。
0148:     # 说明：空行，用于分隔逻辑块，提高可读性。
0149:     # 说明：空行，用于分隔逻辑块，提高可读性。
0150: def _era5_day_dir(root: str | os.PathLike[str], t: datetime) -> str:    # 说明：定义函数，复用项目中的关键流程。
0151:     return os.path.join(str(root), f"{t.year:04d}", f"{t.year:04d}-{t.month:02d}-{t.day:02d}")    # 说明：返回当前函数计算得到的结果。
0152:     # 说明：空行，用于分隔逻辑块，提高可读性。
0153:     # 说明：空行，用于分隔逻辑块，提高可读性。
0154: def _era5_stamp(t: datetime) -> str:    # 说明：定义函数，复用项目中的关键流程。
0155:     return f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}"    # 说明：返回当前函数计算得到的结果。
0156:     # 说明：空行，用于分隔逻辑块，提高可读性。
0157:     # 说明：空行，用于分隔逻辑块，提高可读性。
0158: def find_era5_fullstate_file(root: str | os.PathLike[str], t: datetime) -> Optional[str]:    # 说明：定义函数，复用项目中的关键流程。
0159:     """Return a complete-state ERA5 file for ``t`` if one exists.    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0160:     # 说明：空行，用于分隔逻辑块，提高可读性。
0161:     This intentionally does not match ``HH:MM:SS-t-1000.npy``; per-variable    # 说明：保留该行以完成当前代码块的语法结构。
0162:     files are handled by :func:`find_era5_variable_file`.    # 说明：保留该行以完成当前代码块的语法结构。
0163:     """    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0164:     exact = datetime_era5_path(root, t)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0165:     if os.path.exists(exact):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0166:         return exact    # 说明：返回当前函数计算得到的结果。
0167:     base = _era5_day_dir(root, t)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0168:     if not os.path.isdir(base):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0169:         return None    # 说明：返回当前函数计算得到的结果。
0170:     stamp = _era5_stamp(t)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0171:     for name in (f"{stamp}.npy", f"{stamp}-era5.npy", f"{stamp}-state.npy", f"{stamp}-all.npy"):    # 说明：遍历集合或数据流，逐项完成处理。
0172:         path = os.path.join(base, name)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0173:         if os.path.exists(path):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0174:             return path    # 说明：返回当前函数计算得到的结果。
0175:     return None    # 说明：返回当前函数计算得到的结果。
0176:     # 说明：空行，用于分隔逻辑块，提高可读性。
0177:     # 说明：空行，用于分隔逻辑块，提高可读性。
0178: def find_era5_variable_file(root: str | os.PathLike[str], t: datetime, var: str) -> Optional[str]:    # 说明：定义函数，复用项目中的关键流程。
0179:     """Return one per-variable ERA5 file for ``t`` and ``var`` if present."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0180:     exact = datetime_era5_variable_path(root, t, var)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0181:     if os.path.exists(exact):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0182:         return exact    # 说明：返回当前函数计算得到的结果。
0183:     base = _era5_day_dir(root, t)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0184:     if not os.path.isdir(base):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0185:         return None    # 说明：返回当前函数计算得到的结果。
0186:     stamp = _era5_stamp(t)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0187:     candidates = [    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0188:         f"{stamp}-{var}.npy",    # 说明：保留该行以完成当前代码块的语法结构。
0189:         f"{stamp}_{var}.npy",    # 说明：保留该行以完成当前代码块的语法结构。
0190:         f"{stamp}-{var.replace('-', '_')}.npy",    # 说明：调用函数或方法，执行具体工程动作。
0191:         f"{stamp}_{var.replace('-', '_')}.npy",    # 说明：调用函数或方法，执行具体工程动作。
0192:     ]    # 说明：保留该行以完成当前代码块的语法结构。
0193:     for name in candidates:    # 说明：遍历集合或数据流，逐项完成处理。
0194:         path = os.path.join(base, name)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0195:         if os.path.exists(path):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0196:             return path    # 说明：返回当前函数计算得到的结果。
0197:     # Last-resort support for prefixes such as HH:MM:SS-era5-t-1000.npy.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0198:     patterns = [    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0199:         os.path.join(base, f"{stamp}*{var}.npy"),    # 说明：调用函数或方法，执行具体工程动作。
0200:         os.path.join(base, f"{stamp}*{var.replace('-', '_')}.npy"),    # 说明：调用函数或方法，执行具体工程动作。
0201:     ]    # 说明：保留该行以完成当前代码块的语法结构。
0202:     for pattern in patterns:    # 说明：遍历集合或数据流，逐项完成处理。
0203:         matches = sorted(glob(pattern))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0204:         if matches:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0205:             return matches[0]    # 说明：返回当前函数计算得到的结果。
0206:     return None    # 说明：返回当前函数计算得到的结果。
0207:     # 说明：空行，用于分隔逻辑块，提高可读性。
0208:     # 说明：空行，用于分隔逻辑块，提高可读性。
0209: def era5_time_has_targets(root: str | os.PathLike[str], t: datetime, target_vars: Sequence[str]) -> bool:    # 说明：定义函数，复用项目中的关键流程。
0210:     """True when either a full-state file or all per-variable target files exist."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0211:     if find_era5_fullstate_file(root, t) is not None:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0212:         return True    # 说明：返回当前函数计算得到的结果。
0213:     return all(find_era5_variable_file(root, t, var) is not None for var in target_vars)    # 说明：返回当前函数计算得到的结果。
0214:     # 说明：空行，用于分隔逻辑块，提高可读性。
0215:     # 说明：空行，用于分隔逻辑块，提高可读性。
0216: def _era5_variable_name_from_file(path: str | os.PathLike[str]) -> Optional[str]:    # 说明：定义函数，复用项目中的关键流程。
0217:     """Extract ``t-1000`` from ``HH:MM:SS-t-1000.npy`` style names."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0218:     name = Path(path).name    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0219:     m = re.match(r"^\d{2}:\d{2}:\d{2}[-_](?P<var>.+)\.npy$", name)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0220:     if not m:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0221:         return None    # 说明：返回当前函数计算得到的结果。
0222:     var = m.group("var")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0223:     # Undo the common underscore variant only for variable prefixes.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0224:     if re.match(r"^[a-zA-Z]+_\d+$", var):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0225:         head, lev = var.rsplit("_", 1)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0226:         var = f"{head}-{lev}"    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0227:     return var    # 说明：返回当前函数计算得到的结果。
0228:     # 说明：空行，用于分隔逻辑块，提高可读性。
0229:     # 说明：空行，用于分隔逻辑块，提高可读性。
0230: def collect_era5_target_times(    # 说明：定义函数，复用项目中的关键流程。
0231:     root: str | os.PathLike[str],    # 说明：保留该行以完成当前代码块的语法结构。
0232:     target_vars: Sequence[str],    # 说明：保留该行以完成当前代码块的语法结构。
0233:     start_year: int,    # 说明：保留该行以完成当前代码块的语法结构。
0234:     end_year: int,    # 说明：保留该行以完成当前代码块的语法结构。
0235:     dt_data: int = 1,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0236: ) -> List[datetime]:    # 说明：保留该行以完成当前代码块的语法结构。
0237:     """Scan nested ERA5 labels and return times with complete T/Q targets.    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0238:     # 说明：空行，用于分隔逻辑块，提高可读性。
0239:     The implementation walks ``YYYY/YYYY-MM-DD`` directories with ``os.scandir``    # 说明：保留该行以完成当前代码块的语法结构。
0240:     instead of materialising a multi-million-file recursive glob.  This is much    # 说明：保留该行以完成当前代码块的语法结构。
0241:     friendlier to shared HPC filesystems when ``dt_data=1`` and the dataset has    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0242:     one ``*.npy`` per variable per hour.    # 说明：保留该行以完成当前代码块的语法结构。
0243:     """    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0244:     root = str(root)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0245:     if not os.path.isdir(root):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0246:         return []    # 说明：返回当前函数计算得到的结果。
0247:     target_set = set(map(str, target_vars))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0248:     full_times: set[datetime] = set()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0249:     var_times: Dict[datetime, set[str]] = {}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0250:     for year in range(int(start_year), int(end_year)):    # 说明：遍历集合或数据流，逐项完成处理。
0251:         year_dir = os.path.join(root, f"{year:04d}")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0252:         if not os.path.isdir(year_dir):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0253:             continue    # 说明：保留该行以完成当前代码块的语法结构。
0254:         try:    # 说明：进入异常保护块，保证超算任务失败时可诊断。
0255:             day_entries = sorted((e for e in os.scandir(year_dir) if e.is_dir()), key=lambda e: e.name)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0256:         except OSError:    # 说明：捕获异常并提供安全回退或清晰报错。
0257:             continue    # 说明：保留该行以完成当前代码块的语法结构。
0258:         for day_entry in day_entries:    # 说明：遍历集合或数据流，逐项完成处理。
0259:             try:    # 说明：进入异常保护块，保证超算任务失败时可诊断。
0260:                 day = datetime.strptime(day_entry.name, "%Y-%m-%d")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0261:             except ValueError:    # 说明：捕获异常并提供安全回退或清晰报错。
0262:                 continue    # 说明：保留该行以完成当前代码块的语法结构。
0263:             try:    # 说明：进入异常保护块，保证超算任务失败时可诊断。
0264:                 file_entries = sorted((e for e in os.scandir(day_entry.path) if e.is_file() and e.name.endswith(".npy")), key=lambda e: e.name)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0265:             except OSError:    # 说明：捕获异常并提供安全回退或清晰报错。
0266:                 continue    # 说明：保留该行以完成当前代码块的语法结构。
0267:             for entry in file_entries:    # 说明：遍历集合或数据流，逐项完成处理。
0268:                 stamp_match = re.match(r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})(?P<rest>.*)\.npy$", entry.name)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0269:                 if not stamp_match:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0270:                     continue    # 说明：保留该行以完成当前代码块的语法结构。
0271:                 hour = int(stamp_match.group("h"))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0272:                 minute = int(stamp_match.group("m"))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0273:                 second = int(stamp_match.group("s"))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0274:                 if dt_data > 1 and hour % int(dt_data) != 0:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0275:                     continue    # 说明：保留该行以完成当前代码块的语法结构。
0276:                 t = day.replace(hour=hour, minute=minute, second=second)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0277:                 rest = stamp_match.group("rest")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0278:                 if rest in {"", "-era5", "-state", "-all"}:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0279:                     full_times.add(t)    # 说明：调用函数或方法，执行具体工程动作。
0280:                     continue    # 说明：保留该行以完成当前代码块的语法结构。
0281:                 var = _era5_variable_name_from_file(entry.path)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0282:                 if var in target_set:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0283:                     var_times.setdefault(t, set()).add(var)    # 说明：调用函数或方法，执行具体工程动作。
0284:                     continue    # 说明：保留该行以完成当前代码块的语法结构。
0285:                 for candidate in target_set:    # 说明：遍历集合或数据流，逐项完成处理。
0286:                     if rest.endswith("-" + candidate) or rest.endswith("_" + candidate.replace("-", "_")):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0287:                         var_times.setdefault(t, set()).add(candidate)    # 说明：调用函数或方法，执行具体工程动作。
0288:                         break    # 说明：保留该行以完成当前代码块的语法结构。
0289:     complete = set(full_times)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0290:     complete.update(t for t, vars_present in var_times.items() if target_set.issubset(vars_present))    # 说明：调用函数或方法，执行具体工程动作。
0291:     return sorted(complete)    # 说明：返回当前函数计算得到的结果。
0292:     # 说明：空行，用于分隔逻辑块，提高可读性。
0293:     # 说明：空行，用于分隔逻辑块，提高可读性。
0294: def _safe_np_load(path: str | os.PathLike[str]) -> Any:    # 说明：定义函数，复用项目中的关键流程。
0295:     """Load NPY/NPZ safely; plain NPY arrays use mmap to lower host-memory pressure."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0296:     path = str(path)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0297:     mmap_mode = "r" if path.endswith(".npy") else None    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0298:     data = np.load(path, allow_pickle=True, mmap_mode=mmap_mode)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0299:     if isinstance(data, np.lib.npyio.NpzFile):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0300:         with data as npz:    # 说明：使用上下文管理器，安全管理文件、AMP 或 DDP 同步。
0301:             return {k: npz[k] for k in npz.files}    # 说明：返回当前函数计算得到的结果。
0302:     if getattr(data, "shape", None) == () and data.dtype == object:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0303:         obj = data.item()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0304:         if isinstance(obj, Mapping):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0305:             return dict(obj)    # 说明：返回当前函数计算得到的结果。
0306:     return data    # 说明：返回当前函数计算得到的结果。
0307:     # 说明：空行，用于分隔逻辑块，提高可读性。
0308:     # 说明：空行，用于分隔逻辑块，提高可读性。
0309: def _as_channel_first(arr: np.ndarray, grid_shape: Tuple[int, int]) -> np.ndarray:    # 说明：定义函数，复用项目中的关键流程。
0310:     """Convert [H,W], [C,H,W], or [H,W,C] arrays to [C,H,W]."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0311:     arr = np.asarray(arr)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0312:     arr = np.squeeze(arr)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0313:     if arr.ndim == 2:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0314:         return arr[None, :, :]    # 说明：返回当前函数计算得到的结果。
0315:     if arr.ndim == 3:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0316:         h, w = grid_shape    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0317:         if arr.shape[-2:] == (h, w):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0318:             return arr    # 说明：返回当前函数计算得到的结果。
0319:         if arr.shape[:2] == (h, w):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0320:             return np.moveaxis(arr, -1, 0)    # 说明：返回当前函数计算得到的结果。
0321:     if arr.ndim == 4 and arr.shape[0] == 2 and arr.shape[1] == len(PRESSURE_LEVELS):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0322:         return arr.reshape(2 * len(PRESSURE_LEVELS), arr.shape[-2], arr.shape[-1])    # 说明：返回当前函数计算得到的结果。
0323:     raise ValueError(f"Cannot interpret array shape {arr.shape} as channel-first grid {grid_shape}")    # 说明：调用函数或方法，执行具体工程动作。
0324:     # 说明：空行，用于分隔逻辑块，提高可读性。
0325:     # 说明：空行，用于分隔逻辑块，提高可读性。
0326: def _to_float_tensor(x: np.ndarray) -> torch.Tensor:    # 说明：定义函数，复用项目中的关键流程。
0327:     return torch.from_numpy(np.asarray(x, dtype=np.float32))    # 说明：返回当前函数计算得到的结果。
0328:     # 说明：空行，用于分隔逻辑块，提高可读性。
0329:     # 说明：空行，用于分隔逻辑块，提高可读性。
0330: def _empty_obs() -> Dict[str, torch.Tensor]:    # 说明：定义函数，复用项目中的关键流程。
0331:     f = torch.empty(0, dtype=torch.float32)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0332:     l = torch.empty(0, dtype=torch.long)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0333:     return {    # 说明：返回当前函数计算得到的结果。
0334:         "measurement": f,    # 说明：保留该行以完成当前代码块的语法结构。
0335:         "lat": f,    # 说明：保留该行以完成当前代码块的语法结构。
0336:         "lon": f,    # 说明：保留该行以完成当前代码块的语法结构。
0337:         "relative_time": f,    # 说明：保留该行以完成当前代码块的语法结构。
0338:         "channel": l,    # 说明：保留该行以完成当前代码块的语法结构。
0339:         "platform": l,    # 说明：保留该行以完成当前代码块的语法结构。
0340:         "scan_angle": f,    # 说明：保留该行以完成当前代码块的语法结构。
0341:         "sat_zenith_angle": f,    # 说明：保留该行以完成当前代码块的语法结构。
0342:         "solar_zenith_angle": f,    # 说明：保留该行以完成当前代码块的语法结构。
0343:         "pressure": f,    # 说明：保留该行以完成当前代码块的语法结构。
0344:         "height": f,    # 说明：保留该行以完成当前代码块的语法结构。
0345:         "variable_type": l,    # 说明：保留该行以完成当前代码块的语法结构。
0346:         "report_type": l,    # 说明：保留该行以完成当前代码块的语法结构。
0347:         "station_type": l,    # 说明：保留该行以完成当前代码块的语法结构。
0348:         "quality_flag": f,    # 说明：保留该行以完成当前代码块的语法结构。
0349:         "mask": f,    # 说明：保留该行以完成当前代码块的语法结构。
0350:     }    # 说明：保留该行以完成当前代码块的语法结构。
0351:     # 说明：空行，用于分隔逻辑块，提高可读性。
0352:     # 说明：空行，用于分隔逻辑块，提高可读性。
0353: @dataclass(frozen=True)    # 说明：装饰器，修改函数/方法行为或启用框架入口。
0354: class SensorFiles:    # 说明：定义核心类，封装模型、数据或训练职责。
0355:     measurement_suffixes: Tuple[str, ...]    # 说明：保留该行以完成当前代码块的语法结构。
0356:     aux_suffixes: Tuple[str, ...]    # 说明：保留该行以完成当前代码块的语法结构。
0357:     mask_suffixes: Tuple[str, ...]    # 说明：保留该行以完成当前代码块的语法结构。
0358:     # 说明：空行，用于分隔逻辑块，提高可读性。
0359:     # 说明：空行，用于分隔逻辑块，提高可读性。
0360: SAT_FILES = SensorFiles(    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0361:     measurement_suffixes=(    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0362:         "brightness_temperature_value.npy",    # 说明：保留该行以完成当前代码块的语法结构。
0363:         "tmbrs_value.npy",    # 说明：保留该行以完成当前代码块的语法结构。
0364:         "obs_value.npy",    # 说明：保留该行以完成当前代码块的语法结构。
0365:         "measurement.npy",    # 说明：保留该行以完成当前代码块的语法结构。
0366:         "value.npy",    # 说明：保留该行以完成当前代码块的语法结构。
0367:     ),    # 说明：保留该行以完成当前代码块的语法结构。
0368:     aux_suffixes=("auxiliary_value.npy", "metadata_value.npy", "aux_value.npy"),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0369:     mask_suffixes=("mask.npy", "obs_mask.npy"),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0370: )    # 说明：调用函数或方法，执行具体工程动作。
0371: CONV_FILES = SensorFiles(    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0372:     measurement_suffixes=("obs_value.npy", "observation_value.npy", "measurement.npy", "value.npy"),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0373:     aux_suffixes=("auxiliary_value.npy", "metadata_value.npy", "pressure_value.npy", "height_value.npy"),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0374:     mask_suffixes=("mask.npy", "obs_mask.npy"),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0375: )    # 说明：调用函数或方法，执行具体工程动作。
0376:     # 说明：空行，用于分隔逻辑块，提高可读性。
0377:     # 说明：空行，用于分隔逻辑块，提高可读性。
0378: class HealDARetrievalDataset(Dataset):    # 说明：定义核心类，封装模型、数据或训练职责。
0379:     """Variable-length point-cloud observation dataset for retrieval.    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0380:     # 说明：空行，用于分隔逻辑块，提高可读性。
0381:     Parameters mirror the Hydra datamodule config.  The loader never reads ERA5    # 说明：保留该行以完成当前代码块的语法结构。
0382:     targets as model input; ERA5 is used only for ``target``.    # 说明：保留该行以完成当前代码块的语法结构。
0383:     """    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0384:     # 说明：空行，用于分隔逻辑块，提高可读性。
0385:     def __init__(    # 说明：定义函数，复用项目中的关键流程。
0386:         self,    # 说明：保留该行以完成当前代码块的语法结构。
0387:         obs_dir: str,    # 说明：保留该行以完成当前代码块的语法结构。
0388:         era5_dir: str,    # 说明：保留该行以完成当前代码块的语法结构。
0389:         scale_dir: str,    # 说明：保留该行以完成当前代码块的语法结构。
0390:         mode: str,    # 说明：保留该行以完成当前代码块的语法结构。
0391:         sensors: Sequence[str],    # 说明：保留该行以完成当前代码块的语法结构。
0392:         target_variables: Sequence[str] = TARGET_VARS,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0393:         pressure_levels: Sequence[int] = PRESSURE_LEVELS,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0394:         era5_all_vars: Sequence[str] = XICHEN_ERA5_ALL_VARS,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0395:         grid_shape: Sequence[int] = (181, 360),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0396:         obs_window: Mapping[str, int] | None = None,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0397:         no_lookahead: bool = False,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0398:         no_lookahead_window: Mapping[str, int] | None = None,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0399:         dt_data: int = 6,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0400:         dt_obs: int = 3,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0401:         start_year: int = 2016,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0402:         end_year: int = 2022,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0403:         debug: bool = False,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0404:         max_debug_samples: int = 8,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0405:         max_points_per_sensor: int = 250_000,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0406:         strict_time_index: bool = False,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0407:         target_cache_size: int = 16,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0408:         normalize_target: bool = True,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0409:         normalize_obs: bool = True,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0410:         require_obs_stats: bool = False,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0411:         qc: Optional[Mapping[str, Sequence[float]]] = None,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0412:         obs_default_normalization: Optional[Mapping[str, Sequence[float]]] = None,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0413:     ) -> None:    # 说明：保留该行以完成当前代码块的语法结构。
0414:         super().__init__()    # 说明：调用函数或方法，执行具体工程动作。
0415:         self.obs_dir = os.path.abspath(os.path.expanduser(obs_dir))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0416:         self.era5_dir = os.path.abspath(os.path.expanduser(era5_dir))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0417:         self.scale_dir = os.path.abspath(os.path.expanduser(scale_dir))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0418:         self.mode = mode    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0419:         self.sensors = [canonical_sensor(s) for s in sensors]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0420:         if any(s in {"satwnd", "ascat"} for s in self.sensors):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0421:             raise ValueError("Retrieval task must not read satwnd/ascat unless explicitly reconfigured.")    # 说明：调用函数或方法，执行具体工程动作。
0422:         self.target_variables = list(target_variables)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0423:         self.pressure_levels = list(pressure_levels)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0424:         self.era5_all_vars = list(era5_all_vars)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0425:         self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0426:         self.dt_data = int(dt_data)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0427:         self.dt_obs = int(dt_obs)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0428:         self.start_year = int(start_year)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0429:         self.end_year = int(end_year)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0430:         self.debug = bool(debug)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0431:         self.max_debug_samples = int(max_debug_samples)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0432:         self.max_points_per_sensor = int(max_points_per_sensor)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0433:         self.strict_time_index = bool(strict_time_index)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0434:         self.target_cache_size = max(int(target_cache_size), 0)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0435:         self._target_cache: "OrderedDict[datetime, torch.Tensor]" = OrderedDict()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0436:         self.normalize_target = bool(normalize_target)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0437:         self.normalize_obs = bool(normalize_obs)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0438:         self.require_obs_stats = bool(require_obs_stats)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0439:         self.qc = dict(qc or {})    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0440:         self.obs_default_normalization = dict(obs_default_normalization or {})    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0441:     # 说明：空行，用于分隔逻辑块，提高可读性。
0442:         window = no_lookahead_window if no_lookahead else obs_window    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0443:         window = window or {"start_hours": -21, "end_hours": 3}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0444:         self.window_start = int(window.get("start_hours", -21))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0445:         self.window_end = int(window.get("end_hours", 3))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0446:     # 说明：空行，用于分隔逻辑块，提高可读性。
0447:         if not os.path.isdir(self.era5_dir):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0448:             raise FileNotFoundError(    # 说明：调用函数或方法，执行具体工程动作。
0449:                 f"ERA5 directory does not exist: {self.era5_dir}. Expected nested folders like "    # 说明：保留该行以完成当前代码块的语法结构。
0450:                 "YYYY/YYYY-MM-DD/HH:MM:SS-t-1000.npy. Override paths.era5_dir if needed."    # 说明：保留该行以完成当前代码块的语法结构。
0451:             )    # 说明：调用函数或方法，执行具体工程动作。
0452:         self.sensor_dirs = {sensor: self._find_sensor_dir(sensor) for sensor in self.sensors}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0453:         self.sensor_schema = {sensor: self._load_sensor_schema(sensor) for sensor in self.sensors}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0454:         self.sensor_stats = {sensor: self._load_obs_stats(sensor) for sensor in self.sensors}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0455:         self.target_mean, self.target_std = self._load_target_stats() if self.normalize_target else (None, None)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0456:         self.target_times = self._build_target_times()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0457:         if self.debug:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0458:             self.target_times = self.target_times[: self.max_debug_samples]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0459:         if not self.target_times:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0460:             warnings.warn(    # 说明：调用函数或方法，执行具体工程动作。
0461:                 f"No complete ERA5 target files found for {mode} in {self.era5_dir}. "    # 说明：保留该行以完成当前代码块的语法结构。
0462:                 "Expected either YYYY/YYYY-MM-DD/HH:MM:SS.npy or per-variable files such as "    # 说明：保留该行以完成当前代码块的语法结构。
0463:                 "YYYY/YYYY-MM-DD/HH:MM:SS-t-1000.npy for all 26 T/Q targets. "    # 说明：保留该行以完成当前代码块的语法结构。
0464:                 "Check start/end years, dt_data, and path layout.",    # 说明：保留该行以完成当前代码块的语法结构。
0465:                 RuntimeWarning,    # 说明：保留该行以完成当前代码块的语法结构。
0466:             )    # 说明：调用函数或方法，执行具体工程动作。
0467:     # 说明：空行，用于分隔逻辑块，提高可读性。
0468:     def __len__(self) -> int:    # 说明：定义函数，复用项目中的关键流程。
0469:         return len(self.target_times)    # 说明：返回当前函数计算得到的结果。
0470:     # 说明：空行，用于分隔逻辑块，提高可读性。
0471:     def _find_sensor_dir(self, sensor: str) -> str:    # 说明：定义函数，复用项目中的关键流程。
0472:         for cand in SENSOR_DIR_CANDIDATES[sensor]:    # 说明：遍历集合或数据流，逐项完成处理。
0473:             p = os.path.join(self.obs_dir, cand)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0474:             if os.path.isdir(p):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0475:                 return p    # 说明：返回当前函数计算得到的结果。
0476:         # Return the first candidate for clear downstream error messages.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0477:         return os.path.join(self.obs_dir, SENSOR_DIR_CANDIDATES[sensor][0])    # 说明：返回当前函数计算得到的结果。
0478:     # 说明：空行，用于分隔逻辑块，提高可读性。
0479:     def _load_sensor_schema(self, sensor: str) -> Mapping[str, Any]:    # 说明：定义函数，复用项目中的关键流程。
0480:         root = self.sensor_dirs[sensor]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0481:         patterns = [    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0482:             os.path.join(root, "*_schema.json"),    # 说明：调用函数或方法，执行具体工程动作。
0483:             os.path.join(root, f"{sensor}_1.0deg_schema.json"),    # 说明：调用函数或方法，执行具体工程动作。
0484:             os.path.join(root, "schema.json"),    # 说明：调用函数或方法，执行具体工程动作。
0485:         ]    # 说明：保留该行以完成当前代码块的语法结构。
0486:         for pattern in patterns:    # 说明：遍历集合或数据流，逐项完成处理。
0487:             for path in glob(pattern):    # 说明：遍历集合或数据流，逐项完成处理。
0488:                 try:    # 说明：进入异常保护块，保证超算任务失败时可诊断。
0489:                     with open(path, "r", encoding="utf-8") as f:    # 说明：使用上下文管理器，安全管理文件、AMP 或 DDP 同步。
0490:                         return json.load(f)    # 说明：返回当前函数计算得到的结果。
0491:                 except Exception as exc:  # pragma: no cover - diagnostics only    # 说明：捕获异常并提供安全回退或清晰报错。
0492:                     warnings.warn(f"Failed to read schema {path}: {exc}")    # 说明：调用函数或方法，执行具体工程动作。
0493:         return {}    # 说明：返回当前函数计算得到的结果。
0494:     # 说明：空行，用于分隔逻辑块，提高可读性。
0495:     def _load_obs_stats(self, sensor: str) -> Optional[Dict[str, np.ndarray]]:    # 说明：定义函数，复用项目中的关键流程。
0496:         candidates = [    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0497:             os.path.join(self.scale_dir, "retrieval_obs_stats", f"{sensor}.npz"),    # 说明：调用函数或方法，执行具体工程动作。
0498:             os.path.join(self.sensor_dirs[sensor], "retrieval_obs_stats.npz"),    # 说明：调用函数或方法，执行具体工程动作。
0499:             os.path.join(self.sensor_dirs[sensor], "sensor_stats.npz"),    # 说明：调用函数或方法，执行具体工程动作。
0500:         ]    # 说明：保留该行以完成当前代码块的语法结构。
0501:         for path in candidates:    # 说明：遍历集合或数据流，逐项完成处理。
0502:             if os.path.exists(path):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0503:                 with np.load(path) as npz:    # 说明：使用上下文管理器，安全管理文件、AMP 或 DDP 同步。
0504:                     return {k: np.asarray(npz[k], dtype=np.float32) for k in npz.files}    # 说明：返回当前函数计算得到的结果。
0505:         if self.require_obs_stats:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0506:             raise FileNotFoundError(    # 说明：调用函数或方法，执行具体工程动作。
0507:                 f"Observation normalization stats for sensor {sensor!r} were not found. "    # 说明：保留该行以完成当前代码块的语法结构。
0508:                 f"Run: python tools/generate_retrieval_mean_std.py --obs_dir {self.obs_dir} "    # 说明：保留该行以完成当前代码块的语法结构。
0509:                 f"--era5_dir {self.era5_dir} --scale_dir {self.scale_dir} --include_obs_stats"    # 说明：保留该行以完成当前代码块的语法结构。
0510:             )    # 说明：调用函数或方法，执行具体工程动作。
0511:         return None    # 说明：返回当前函数计算得到的结果。
0512:     # 说明：空行，用于分隔逻辑块，提高可读性。
0513:     def _load_target_stats(self) -> Tuple[np.ndarray, np.ndarray]:    # 说明：定义函数，复用项目中的关键流程。
0514:         mean_path = os.path.join(self.scale_dir, "normalize_mean.npz")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0515:         std_path = os.path.join(self.scale_dir, "normalize_std.npz")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0516:         if not os.path.exists(mean_path) or not os.path.exists(std_path):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0517:             raise FileNotFoundError(    # 说明：调用函数或方法，执行具体工程动作。
0518:                 "ERA5 target mean/std files are missing. Expected both files:\n"    # 说明：保留该行以完成当前代码块的语法结构。
0519:                 f"  {mean_path}\n  {std_path}\n"    # 说明：保留该行以完成当前代码块的语法结构。
0520:                 "Generate them with:\n"    # 说明：保留该行以完成当前代码块的语法结构。
0521:                 f"  python tools/generate_retrieval_mean_std.py --era5_dir {self.era5_dir} "    # 说明：保留该行以完成当前代码块的语法结构。
0522:                 f"--scale_dir {self.scale_dir} --target_vars {' '.join(self.target_variables)}"    # 说明：调用函数或方法，执行具体工程动作。
0523:             )    # 说明：调用函数或方法，执行具体工程动作。
0524:         mean_npz = dict(np.load(mean_path))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0525:         std_npz = dict(np.load(std_path))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0526:         missing = [v for v in self.target_variables if v not in mean_npz or v not in std_npz]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0527:         if missing:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0528:             raise KeyError(    # 说明：调用函数或方法，执行具体工程动作。
0529:                 f"Missing target variable stats in {self.scale_dir}: {missing}. "    # 说明：保留该行以完成当前代码块的语法结构。
0530:                 "Regenerate normalize_mean.npz / normalize_std.npz for the T/Q-13 target list."    # 说明：保留该行以完成当前代码块的语法结构。
0531:             )    # 说明：调用函数或方法，执行具体工程动作。
0532:         mean = np.concatenate([np.asarray(mean_npz[v]).reshape(1) for v in self.target_variables]).astype(np.float32)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0533:         std = np.concatenate([np.asarray(std_npz[v]).reshape(1) for v in self.target_variables]).astype(np.float32)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0534:         std = np.where(std == 0, 1.0, std)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0535:         return mean, std    # 说明：返回当前函数计算得到的结果。
0536:     # 说明：空行，用于分隔逻辑块，提高可读性。
0537:     def _build_target_times(self) -> List[datetime]:    # 说明：定义函数，复用项目中的关键流程。
0538:         times: List[datetime] = []    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0539:         if self.strict_time_index:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0540:             return collect_era5_target_times(    # 说明：返回当前函数计算得到的结果。
0541:                 self.era5_dir, self.target_variables, self.start_year, self.end_year, self.dt_data    # 说明：保留该行以完成当前代码块的语法结构。
0542:             )    # 说明：调用函数或方法，执行具体工程动作。
0543:     # 说明：空行，用于分隔逻辑块，提高可读性。
0544:         start = datetime(self.start_year, 1, 1)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0545:         end = datetime(self.end_year, 1, 1)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0546:         t = start    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0547:         while t < end:    # 说明：循环处理缓存或时间序列，直到满足终止条件。
0548:             if era5_time_has_targets(self.era5_dir, t, self.target_variables):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0549:                 times.append(t)    # 说明：调用函数或方法，执行具体工程动作。
0550:             elif not os.path.exists(self.era5_dir):    # 说明：执行备用条件分支，覆盖另一类配置或状态。
0551:                 break    # 说明：保留该行以完成当前代码块的语法结构。
0552:             t += timedelta(hours=self.dt_data)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0553:             if self.debug and len(times) >= self.max_debug_samples:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0554:                 break    # 说明：保留该行以完成当前代码块的语法结构。
0555:     # 说明：空行，用于分隔逻辑块，提高可读性。
0556:         # If the data are real but not aligned to the configured dt_data grid    # 说明：中文/配置注释，说明相邻代码或参数用途。
0557:         # (for example files named 22:00:00-t-1000.npy), fall back to a scan of    # 说明：中文/配置注释，说明相邻代码或参数用途。
0558:         # the actual nested ERA5 tree instead of reporting an empty dataset.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0559:         if not times and os.path.isdir(self.era5_dir):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0560:             times = collect_era5_target_times(    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0561:                 self.era5_dir, self.target_variables, self.start_year, self.end_year, self.dt_data    # 说明：保留该行以完成当前代码块的语法结构。
0562:             )    # 说明：调用函数或方法，执行具体工程动作。
0563:         if not times and os.path.isdir(self.era5_dir) and self.dt_data != 1:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0564:             scanned = collect_era5_target_times(    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0565:                 self.era5_dir, self.target_variables, self.start_year, self.end_year, dt_data=1    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0566:             )    # 说明：调用函数或方法，执行具体工程动作。
0567:             if scanned:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0568:                 warnings.warn(    # 说明：调用函数或方法，执行具体工程动作。
0569:                     f"No complete ERA5 targets were aligned to dt_data={self.dt_data} hours. "    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0570:                     "Falling back to all complete target times found in the nested per-variable ERA5 layout. "    # 说明：保留该行以完成当前代码块的语法结构。
0571:                     "Set datamodule.data.strict_time_index=true or datamodule.data.dt_data=1 to make this explicit.",    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0572:                     RuntimeWarning,    # 说明：保留该行以完成当前代码块的语法结构。
0573:                 )    # 说明：调用函数或方法，执行具体工程动作。
0574:                 times = scanned    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0575:         return times    # 说明：返回当前函数计算得到的结果。
0576:     # 说明：空行，用于分隔逻辑块，提高可读性。
0577:     def _find_time_file(self, root: str, t: datetime, suffixes: Iterable[str]) -> Optional[str]:    # 说明：定义函数，复用项目中的关键流程。
0578:         for suffix in suffixes:    # 说明：遍历集合或数据流，逐项完成处理。
0579:             path = datetime_path(root, t, suffix)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0580:             if os.path.exists(path):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0581:                 return path    # 说明：返回当前函数计算得到的结果。
0582:         # fallback for slightly different naming conventions    # 说明：中文/配置注释，说明相邻代码或参数用途。
0583:         base = os.path.join(root, f"{t.year:04d}", f"{t.year:04d}-{t.month:02d}-{t.day:02d}")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0584:         if os.path.isdir(base):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0585:             stamp = f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}"    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0586:             for suffix in suffixes:    # 说明：遍历集合或数据流，逐项完成处理。
0587:                 matches = sorted(glob(os.path.join(base, f"{stamp}*{suffix}")))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0588:                 if matches:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0589:                     return matches[0]    # 说明：返回当前函数计算得到的结果。
0590:         return None    # 说明：返回当前函数计算得到的结果。
0591:     # 说明：空行，用于分隔逻辑块，提高可读性。
0592:     def _load_target_variable_file(self, path: str, var: str) -> np.ndarray:    # 说明：定义函数，复用项目中的关键流程。
0593:         data = _safe_np_load(path)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0594:         if isinstance(data, Mapping):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0595:             if var in data:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0596:                 arr = np.asarray(data[var]).squeeze()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0597:             elif len(data) == 1:    # 说明：执行备用条件分支，覆盖另一类配置或状态。
0598:                 arr = np.asarray(next(iter(data.values()))).squeeze()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0599:             else:    # 说明：执行默认分支，保证逻辑闭环。
0600:                 raise KeyError(f"ERA5 variable file {path} does not contain {var!r}; keys={sorted(data.keys())}")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0601:         else:    # 说明：执行默认分支，保证逻辑闭环。
0602:             arr = np.asarray(data).squeeze()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0603:         if arr.ndim == 3:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0604:             cf = _as_channel_first(arr, self.grid_shape)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0605:             if cf.shape[0] != 1:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0606:                 raise ValueError(f"ERA5 variable file {path} should contain one channel, got {cf.shape}")    # 说明：调用函数或方法，执行具体工程动作。
0607:             arr = cf[0]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0608:         if arr.shape != self.grid_shape:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0609:             raise ValueError(f"ERA5 variable file {path} has shape {arr.shape}, expected {self.grid_shape}")    # 说明：调用函数或方法，执行具体工程动作。
0610:         return arr.astype(np.float32)    # 说明：返回当前函数计算得到的结果。
0611:     # 说明：空行，用于分隔逻辑块，提高可读性。
0612:     def _load_target(self, t: datetime) -> torch.Tensor:    # 说明：定义函数，复用项目中的关键流程。
0613:         """Load and optionally normalize one ERA5 T/Q target, with a small per-worker LRU cache."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0614:         if self.target_cache_size > 0 and t in self._target_cache:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0615:             cached = self._target_cache.pop(t)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0616:             self._target_cache[t] = cached    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0617:             return cached.clone()    # 说明：返回当前函数计算得到的结果。
0618:         path = find_era5_fullstate_file(self.era5_dir, t)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0619:         if path is not None:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0620:             data = _safe_np_load(path)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0621:             if isinstance(data, Mapping):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0622:                 missing = [v for v in self.target_variables if v not in data]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0623:                 if missing:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0624:                     raise KeyError(f"ERA5 file {path} does not contain target variables {missing}")    # 说明：调用函数或方法，执行具体工程动作。
0625:                 arr = np.stack([np.asarray(data[v]).squeeze() for v in self.target_variables], axis=0)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0626:             else:    # 说明：执行默认分支，保证逻辑闭环。
0627:                 arr = _as_channel_first(np.asarray(data), self.grid_shape)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0628:                 if arr.shape[0] == len(self.target_variables):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0629:                     pass    # 说明：保留该行以完成当前代码块的语法结构。
0630:                 elif set(self.target_variables).issubset(self.era5_all_vars) and arr.shape[0] >= len(self.era5_all_vars):    # 说明：执行备用条件分支，覆盖另一类配置或状态。
0631:                     idx = [self.era5_all_vars.index(v) for v in self.target_variables]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0632:                     arr = arr[idx]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0633:                 else:    # 说明：执行默认分支，保证逻辑闭环。
0634:                     raise ValueError(    # 说明：调用函数或方法，执行具体工程动作。
0635:                         f"ERA5 target {path} has {arr.shape[0]} channels. Provide era5_all_vars in the "    # 说明：保留该行以完成当前代码块的语法结构。
0636:                         "Hydra config or pre-extract the 26 T/Q channels."    # 说明：保留该行以完成当前代码块的语法结构。
0637:                     )    # 说明：调用函数或方法，执行具体工程动作。
0638:         else:    # 说明：执行默认分支，保证逻辑闭环。
0639:             paths = {var: find_era5_variable_file(self.era5_dir, t, var) for var in self.target_variables}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0640:             missing = [var for var, var_path in paths.items() if var_path is None]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0641:             if missing:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0642:                 day = _era5_day_dir(self.era5_dir, t)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0643:                 stamp = _era5_stamp(t)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0644:                 raise FileNotFoundError(    # 说明：调用函数或方法，执行具体工程动作。
0645:                     "ERA5 target files were not found for "    # 说明：保留该行以完成当前代码块的语法结构。
0646:                     f"{t.isoformat()} under {day}. Expected either {stamp}.npy or per-variable files "    # 说明：调用函数或方法，执行具体工程动作。
0647:                     f"like {stamp}-t-1000.npy. Missing target variables: {missing[:8]}"    # 说明：保留该行以完成当前代码块的语法结构。
0648:                     + (" ..." if len(missing) > 8 else "")    # 说明：调用函数或方法，执行具体工程动作。
0649:                 )    # 说明：调用函数或方法，执行具体工程动作。
0650:             arr = np.stack(    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0651:                 [self._load_target_variable_file(paths[var], var) for var in self.target_variables], axis=0    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0652:             )    # 说明：调用函数或方法，执行具体工程动作。
0653:         arr = arr.astype(np.float32)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0654:         if self.normalize_target and self.target_mean is not None and self.target_std is not None:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0655:             arr = (arr - self.target_mean[:, None, None]) / self.target_std[:, None, None]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0656:         tensor = _to_float_tensor(arr)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0657:         if self.target_cache_size > 0:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0658:             self._target_cache[t] = tensor    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0659:             while len(self._target_cache) > self.target_cache_size:    # 说明：循环处理缓存或时间序列，直到满足终止条件。
0660:                 self._target_cache.popitem(last=False)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0661:         return tensor.clone()    # 说明：返回当前函数计算得到的结果。
0662:     # 说明：空行，用于分隔逻辑块，提高可读性。
0663:     def _field_index(self, sensor: str, field_names: Sequence[str]) -> Optional[int]:    # 说明：定义函数，复用项目中的关键流程。
0664:         schema = self.sensor_schema.get(sensor, {})    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0665:         candidates: List[str] = []    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0666:         for block in ("auxiliary_value", "metadata_value", "obs_value"):    # 说明：遍历集合或数据流，逐项完成处理。
0667:             fields = schema.get(block, {}).get("fields_in_order") if isinstance(schema.get(block), Mapping) else None    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0668:             if fields:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0669:                 candidates = list(fields)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0670:                 break    # 说明：保留该行以完成当前代码块的语法结构。
0671:         if not candidates and isinstance(schema.get("fields_in_order"), Sequence):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0672:             candidates = list(schema["fields_in_order"])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0673:         lower = {str(v).lower(): i for i, v in enumerate(candidates)}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0674:         for name in field_names:    # 说明：遍历集合或数据流，逐项完成处理。
0675:             if name.lower() in lower:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0676:                 return lower[name.lower()]    # 说明：返回当前函数计算得到的结果。
0677:         return None    # 说明：返回当前函数计算得到的结果。
0678:     # 说明：空行，用于分隔逻辑块，提高可读性。
0679:     @staticmethod    # 说明：装饰器，修改函数/方法行为或启用框架入口。
0680:     @lru_cache(maxsize=8)    # 说明：装饰器，修改函数/方法行为或启用框架入口。
0681:     def _lat_lon_grid(grid_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:    # 说明：定义函数，复用项目中的关键流程。
0682:         h, w = grid_shape    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0683:         lat = np.linspace(90.0, -90.0, h, dtype=np.float32)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0684:         lon = np.linspace(0.0, 360.0, w, endpoint=False, dtype=np.float32)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0685:         lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0686:         return lat2d, lon2d    # 说明：返回当前函数计算得到的结果。
0687:     # 说明：空行，用于分隔逻辑块，提高可读性。
0688:     def _normalize_measurements(self, sensor: str, measurement: np.ndarray, channel: np.ndarray) -> np.ndarray:    # 说明：定义函数，复用项目中的关键流程。
0689:         if not self.normalize_obs:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0690:             return measurement.astype(np.float32)    # 说明：返回当前函数计算得到的结果。
0691:         stats = self.sensor_stats.get(sensor)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0692:         if stats is not None and "mean" in stats and "std" in stats:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0693:             mean = stats["mean"]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0694:             std = np.where(stats["std"] == 0, 1.0, stats["std"])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0695:             ch = np.clip(channel.astype(int), 0, len(mean) - 1)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0696:             return ((measurement - mean[ch]) / std[ch]).astype(np.float32)    # 说明：返回当前函数计算得到的结果。
0697:         # Configurable safe defaults; these are not used for field discovery.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0698:         key = "satellite" if sensor in SATELLITE_SENSORS else "conventional"    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0699:         default = self.obs_default_normalization.get(key, None)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0700:         if default is not None and len(default) == 2:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0701:             mean, std = float(default[0]), max(float(default[1]), 1e-6)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0702:             return ((measurement - mean) / std).astype(np.float32)    # 说明：返回当前函数计算得到的结果。
0703:         return measurement.astype(np.float32)    # 说明：返回当前函数计算得到的结果。
0704:     # 说明：空行，用于分隔逻辑块，提高可读性。
0705:     def _load_aux_array(self, sensor: str, t: datetime, files: SensorFiles) -> Optional[np.ndarray]:    # 说明：定义函数，复用项目中的关键流程。
0706:         root = self.sensor_dirs[sensor]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0707:         aux_path = self._find_time_file(root, t, files.aux_suffixes)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0708:         if aux_path is None:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0709:             return None    # 说明：返回当前函数计算得到的结果。
0710:         aux = _safe_np_load(aux_path)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0711:         if isinstance(aux, Mapping):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0712:             # Keep mapping support by stacking numeric 2-D/3-D fields in sorted key order.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0713:             fields = []    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0714:             for key in sorted(aux):    # 说明：遍历集合或数据流，逐项完成处理。
0715:                 val = np.asarray(aux[key])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0716:                 if val.ndim >= 2:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0717:                     fields.append(np.squeeze(val))    # 说明：调用函数或方法，执行具体工程动作。
0718:             if not fields:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0719:                 return None    # 说明：返回当前函数计算得到的结果。
0720:             aux = np.stack(fields, axis=0)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0721:         try:    # 说明：进入异常保护块，保证超算任务失败时可诊断。
0722:             return _as_channel_first(np.asarray(aux), self.grid_shape)    # 说明：返回当前函数计算得到的结果。
0723:         except Exception:    # 说明：捕获异常并提供安全回退或清晰报错。
0724:             return None    # 说明：返回当前函数计算得到的结果。
0725:     # 说明：空行，用于分隔逻辑块，提高可读性。
0726:     def _extract_aux_field(    # 说明：定义函数，复用项目中的关键流程。
0727:         self,    # 说明：保留该行以完成当前代码块的语法结构。
0728:         sensor: str,    # 说明：保留该行以完成当前代码块的语法结构。
0729:         aux: Optional[np.ndarray],    # 说明：保留该行以完成当前代码块的语法结构。
0730:         names: Sequence[str],    # 说明：保留该行以完成当前代码块的语法结构。
0731:         flat_idx: np.ndarray,    # 说明：保留该行以完成当前代码块的语法结构。
0732:         default: float = np.nan,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0733:     ) -> np.ndarray:    # 说明：保留该行以完成当前代码块的语法结构。
0734:         if aux is None:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0735:             return np.full(len(flat_idx), default, dtype=np.float32)    # 说明：返回当前函数计算得到的结果。
0736:         idx = self._field_index(sensor, names)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0737:         if idx is None or idx >= aux.shape[0]:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0738:             return np.full(len(flat_idx), default, dtype=np.float32)    # 说明：返回当前函数计算得到的结果。
0739:         return aux[idx].reshape(-1)[flat_idx].astype(np.float32)    # 说明：返回当前函数计算得到的结果。
0740:     # 说明：空行，用于分隔逻辑块，提高可读性。
0741:     def _load_sensor_at_time(self, sensor: str, obs_time: datetime, target_time: datetime) -> Dict[str, torch.Tensor]:    # 说明：定义函数，复用项目中的关键流程。
0742:         root = self.sensor_dirs[sensor]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0743:         files = SAT_FILES if sensor in SATELLITE_SENSORS else CONV_FILES    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0744:         measurement_path = self._find_time_file(root, obs_time, files.measurement_suffixes)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0745:         if measurement_path is None or not os.path.exists(measurement_path):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0746:             return _empty_obs()    # 说明：返回当前函数计算得到的结果。
0747:     # 说明：空行，用于分隔逻辑块，提高可读性。
0748:         raw = _safe_np_load(measurement_path)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0749:         if isinstance(raw, Mapping):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0750:             # Point-cloud npz/dict path.  We only consume fields actually present.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0751:             return self._point_mapping_to_obs(sensor, raw, obs_time, target_time)    # 说明：返回当前函数计算得到的结果。
0752:     # 说明：空行，用于分隔逻辑块，提高可读性。
0753:         arr = _as_channel_first(np.asarray(raw), self.grid_shape)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0754:         c, h, w = arr.shape    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0755:         if (h, w) != self.grid_shape:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0756:             raise ValueError(f"{measurement_path} has grid {(h, w)}, expected {self.grid_shape}")    # 说明：调用函数或方法，执行具体工程动作。
0757:     # 说明：空行，用于分隔逻辑块，提高可读性。
0758:         mask_path = self._find_time_file(root, obs_time, files.mask_suffixes)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0759:         if mask_path and os.path.exists(mask_path):    # 说明：执行条件分支，处理不同运行环境或配置情况。
0760:             mask_arr = np.asarray(_safe_np_load(mask_path)).squeeze()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0761:             if mask_arr.ndim == 2:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0762:                 valid_mask = np.broadcast_to(mask_arr[None, :, :] > 0, (c, h, w))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0763:             else:    # 说明：执行默认分支，保证逻辑闭环。
0764:                 valid_mask = _as_channel_first(mask_arr, self.grid_shape) > 0    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0765:                 if valid_mask.shape[0] == 1:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0766:                     valid_mask = np.broadcast_to(valid_mask, (c, h, w))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0767:         else:    # 说明：执行默认分支，保证逻辑闭环。
0768:             valid_mask = np.isfinite(arr)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0769:     # 说明：空行，用于分隔逻辑块，提高可读性。
0770:         valid = np.isfinite(arr) & valid_mask    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0771:         if sensor in SATELLITE_SENSORS:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0772:             rng = self.qc.get("infrared_bt_range" if sensor == "hrs4" else "microwave_bt_range", (0.0, 400.0))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0773:             valid &= (arr >= float(rng[0])) & (arr <= float(rng[1]))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0774:         ch_idx, ij = np.nonzero(valid.reshape(c, -1))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0775:         if len(ch_idx) == 0:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0776:             return _empty_obs()    # 说明：返回当前函数计算得到的结果。
0777:     # 说明：空行，用于分隔逻辑块，提高可读性。
0778:         if self.max_points_per_sensor > 0 and len(ch_idx) > self.max_points_per_sensor:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0779:             # Deterministic thinning preserves global coverage and avoids loader OOM.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0780:             keep = np.linspace(0, len(ch_idx) - 1, self.max_points_per_sensor, dtype=np.int64)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0781:             ch_idx = ch_idx[keep]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0782:             ij = ij[keep]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0783:     # 说明：空行，用于分隔逻辑块，提高可读性。
0784:         measurement = arr.reshape(c, -1)[ch_idx, ij].astype(np.float32)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0785:         lat_grid, lon_grid = self._lat_lon_grid(self.grid_shape)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0786:         lat = lat_grid.reshape(-1)[ij].astype(np.float32)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0787:         lon = lon_grid.reshape(-1)[ij].astype(np.float32)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0788:         rel_time = np.full(len(ch_idx), (obs_time - target_time).total_seconds() / 3600.0, dtype=np.float32)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0789:         aux = self._load_aux_array(sensor, obs_time, files)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0790:     # 说明：空行，用于分隔逻辑块，提高可读性。
0791:         pressure = self._extract_aux_field(sensor, aux, ("pressure", "pres", "prs", "p"), ij)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0792:         height = self._extract_aux_field(sensor, aux, ("height", "hgt", "elev", "elevation", "hmsl", "hols"), ij)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0793:         if sensor in CONVENTIONAL_SENSORS:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0794:             p_rng = self.qc.get("pressure_range_hpa", (0.5, 1100.0))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0795:             h_rng = self.qc.get("height_range_m", (0.0, 60000.0))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0796:             p_ok = ~np.isfinite(pressure) | ((pressure >= float(p_rng[0])) & (pressure <= float(p_rng[1])))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0797:             h_ok = ~np.isfinite(height) | ((height >= float(h_rng[0])) & (height <= float(h_rng[1])))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0798:             keep = p_ok & h_ok    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0799:             measurement, lat, lon, rel_time, ch_idx, ij, pressure, height = (    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0800:                 measurement[keep], lat[keep], lon[keep], rel_time[keep], ch_idx[keep], ij[keep], pressure[keep], height[keep]    # 说明：保留该行以完成当前代码块的语法结构。
0801:             )    # 说明：调用函数或方法，执行具体工程动作。
0802:             if len(measurement) == 0:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0803:                 return _empty_obs()    # 说明：返回当前函数计算得到的结果。
0804:     # 说明：空行，用于分隔逻辑块，提高可读性。
0805:         platform = self._extract_aux_field(sensor, aux, ("platform", "platform_id", "sat_id", "satellite_id", "said", "siid"), ij, default=0.0)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0806:         scan = self._extract_aux_field(sensor, aux, ("scan_angle", "scanline", "fov", "fovn"), ij)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0807:         satza = self._extract_aux_field(sensor, aux, ("satellite_zenith_angle", "satellite_za", "saza"), ij)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0808:         solza = self._extract_aux_field(sensor, aux, ("solar_zenith_angle", "solza", "soza"), ij)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0809:         report_type = self._extract_aux_field(sensor, aux, ("report_type", "report", "type"), ij, default=0.0)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0810:         station_type = self._extract_aux_field(sensor, aux, ("station_type", "station", "stype"), ij, default=0.0)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0811:         quality = self._extract_aux_field(sensor, aux, ("quality_flag", "quality", "qc", "lsql", "scan_quality_flags"), ij, default=0.0)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0812:         measurement = self._normalize_measurements(sensor, measurement, ch_idx)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0813:     # 说明：空行，用于分隔逻辑块，提高可读性。
0814:         obs = {    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0815:             "measurement": _to_float_tensor(measurement),    # 说明：调用函数或方法，执行具体工程动作。
0816:             "lat": _to_float_tensor(lat),    # 说明：调用函数或方法，执行具体工程动作。
0817:             "lon": _to_float_tensor(lon),    # 说明：调用函数或方法，执行具体工程动作。
0818:             "relative_time": _to_float_tensor(rel_time),    # 说明：调用函数或方法，执行具体工程动作。
0819:             "channel": torch.from_numpy(ch_idx.astype(np.int64)),    # 说明：调用函数或方法，执行具体工程动作。
0820:             "platform": torch.from_numpy(np.nan_to_num(platform, nan=0.0).astype(np.int64)),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0821:             "scan_angle": _to_float_tensor(scan),    # 说明：调用函数或方法，执行具体工程动作。
0822:             "sat_zenith_angle": _to_float_tensor(satza),    # 说明：调用函数或方法，执行具体工程动作。
0823:             "solar_zenith_angle": _to_float_tensor(solza),    # 说明：调用函数或方法，执行具体工程动作。
0824:             "pressure": _to_float_tensor(pressure),    # 说明：调用函数或方法，执行具体工程动作。
0825:             "height": _to_float_tensor(height),    # 说明：调用函数或方法，执行具体工程动作。
0826:             "variable_type": torch.from_numpy(ch_idx.astype(np.int64)),    # 说明：调用函数或方法，执行具体工程动作。
0827:             "report_type": torch.from_numpy(np.nan_to_num(report_type, nan=0.0).astype(np.int64)),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0828:             "station_type": torch.from_numpy(np.nan_to_num(station_type, nan=0.0).astype(np.int64)),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0829:             "quality_flag": _to_float_tensor(quality),    # 说明：调用函数或方法，执行具体工程动作。
0830:             "mask": torch.ones(len(measurement), dtype=torch.float32),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0831:         }    # 说明：保留该行以完成当前代码块的语法结构。
0832:         return obs    # 说明：返回当前函数计算得到的结果。
0833:     # 说明：空行，用于分隔逻辑块，提高可读性。
0834:     def _point_mapping_to_obs(    # 说明：定义函数，复用项目中的关键流程。
0835:         self,    # 说明：保留该行以完成当前代码块的语法结构。
0836:         sensor: str,    # 说明：保留该行以完成当前代码块的语法结构。
0837:         data: Mapping[str, np.ndarray],    # 说明：保留该行以完成当前代码块的语法结构。
0838:         obs_time: datetime,    # 说明：保留该行以完成当前代码块的语法结构。
0839:         target_time: datetime,    # 说明：保留该行以完成当前代码块的语法结构。
0840:     ) -> Dict[str, torch.Tensor]:    # 说明：保留该行以完成当前代码块的语法结构。
0841:         lower = {str(k).lower(): k for k in data}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0842:     # 说明：空行，用于分隔逻辑块，提高可读性。
0843:         def get_any(names: Sequence[str], default: float = np.nan) -> np.ndarray:    # 说明：定义函数，复用项目中的关键流程。
0844:             for name in names:    # 说明：遍历集合或数据流，逐项完成处理。
0845:                 if name.lower() in lower:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0846:                     return np.asarray(data[lower[name.lower()]]).reshape(-1)    # 说明：返回当前函数计算得到的结果。
0847:             n = len(next(iter(data.values())).reshape(-1))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0848:             return np.full(n, default, dtype=np.float32)    # 说明：返回当前函数计算得到的结果。
0849:     # 说明：空行，用于分隔逻辑块，提高可读性。
0850:         measurement = get_any(("observation", "measurement", "obs", "value", "brightness_temperature"))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0851:         n = len(measurement)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0852:         lat = get_any(("lat", "latitude", "obs_latitude"))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0853:         lon = get_any(("lon", "longitude", "obs_longitude"))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0854:         rel_time = get_any(("relative_time", "dt", "time_offset"), default=(obs_time - target_time).total_seconds() / 3600.0)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0855:         channel = get_any(("channel", "channel_index", "sensor_index", "variable_type"), default=0.0).astype(np.int64)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0856:         valid = np.isfinite(measurement) & np.isfinite(lat) & np.isfinite(lon)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0857:         valid &= (lat >= -90.0) & (lat <= 90.0) & (lon >= -360.0) & (lon <= 720.0)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0858:         if sensor in SATELLITE_SENSORS:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0859:             rng = self.qc.get("infrared_bt_range" if sensor == "hrs4" else "microwave_bt_range", (0.0, 400.0))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0860:             valid &= (measurement >= float(rng[0])) & (measurement <= float(rng[1]))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0861:         idx = np.nonzero(valid)[0]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0862:         if self.max_points_per_sensor > 0 and len(idx) > self.max_points_per_sensor:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0863:             idx = idx[np.linspace(0, len(idx) - 1, self.max_points_per_sensor, dtype=np.int64)]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0864:         if len(idx) == 0:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0865:             return _empty_obs()    # 说明：返回当前函数计算得到的结果。
0866:         measurement = self._normalize_measurements(sensor, measurement[idx].astype(np.float32), channel[idx])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0867:         return {    # 说明：返回当前函数计算得到的结果。
0868:             "measurement": _to_float_tensor(measurement),    # 说明：调用函数或方法，执行具体工程动作。
0869:             "lat": _to_float_tensor(lat[idx].astype(np.float32)),    # 说明：调用函数或方法，执行具体工程动作。
0870:             "lon": _to_float_tensor(lon[idx].astype(np.float32)),    # 说明：调用函数或方法，执行具体工程动作。
0871:             "relative_time": _to_float_tensor(rel_time[idx].astype(np.float32)),    # 说明：调用函数或方法，执行具体工程动作。
0872:             "channel": torch.from_numpy(channel[idx].astype(np.int64)),    # 说明：调用函数或方法，执行具体工程动作。
0873:             "platform": torch.from_numpy(get_any(("platform", "satellite", "sat_id", "said"), 0.0)[idx].astype(np.int64)),    # 说明：调用函数或方法，执行具体工程动作。
0874:             "scan_angle": _to_float_tensor(get_any(("scan_angle", "fov", "fovn"))[idx].astype(np.float32)),    # 说明：调用函数或方法，执行具体工程动作。
0875:             "sat_zenith_angle": _to_float_tensor(get_any(("satellite_zenith_angle", "satellite_za", "saza"))[idx].astype(np.float32)),    # 说明：调用函数或方法，执行具体工程动作。
0876:             "solar_zenith_angle": _to_float_tensor(get_any(("solar_zenith_angle", "solza", "soza"))[idx].astype(np.float32)),    # 说明：调用函数或方法，执行具体工程动作。
0877:             "pressure": _to_float_tensor(get_any(("pressure", "pres"))[idx].astype(np.float32)),    # 说明：调用函数或方法，执行具体工程动作。
0878:             "height": _to_float_tensor(get_any(("height", "elev", "hmsl", "hols"))[idx].astype(np.float32)),    # 说明：调用函数或方法，执行具体工程动作。
0879:             "variable_type": torch.from_numpy(get_any(("variable_type", "variable", "channel"), 0.0)[idx].astype(np.int64)),    # 说明：调用函数或方法，执行具体工程动作。
0880:             "report_type": torch.from_numpy(get_any(("report_type", "type"), 0.0)[idx].astype(np.int64)),    # 说明：调用函数或方法，执行具体工程动作。
0881:             "station_type": torch.from_numpy(get_any(("station_type", "station"), 0.0)[idx].astype(np.int64)),    # 说明：调用函数或方法，执行具体工程动作。
0882:             "quality_flag": _to_float_tensor(get_any(("quality_flag", "quality", "qc"), 0.0)[idx].astype(np.float32)),    # 说明：调用函数或方法，执行具体工程动作。
0883:             "mask": torch.ones(len(idx), dtype=torch.float32),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0884:         }    # 说明：保留该行以完成当前代码块的语法结构。
0885:     # 说明：空行，用于分隔逻辑块，提高可读性。
0886:     def _load_sensor_window(self, sensor: str, target_time: datetime) -> Dict[str, torch.Tensor]:    # 说明：定义函数，复用项目中的关键流程。
0887:         parts: List[Dict[str, torch.Tensor]] = []    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0888:         for hour in range(self.window_start, self.window_end + 1, self.dt_obs):    # 说明：遍历集合或数据流，逐项完成处理。
0889:             obs_time = target_time + timedelta(hours=hour)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0890:             part = self._load_sensor_at_time(sensor, obs_time, target_time)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0891:             if part["measurement"].numel() > 0:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0892:                 parts.append(part)    # 说明：调用函数或方法，执行具体工程动作。
0893:         if not parts:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0894:             return _empty_obs()    # 说明：返回当前函数计算得到的结果。
0895:         out: Dict[str, torch.Tensor] = {}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0896:         for key in parts[0]:    # 说明：遍历集合或数据流，逐项完成处理。
0897:             out[key] = torch.cat([p[key] for p in parts], dim=0)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0898:         return out    # 说明：返回当前函数计算得到的结果。
0899:     # 说明：空行，用于分隔逻辑块，提高可读性。
0900:     def __getitem__(self, index: int) -> Dict[str, Any]:    # 说明：定义函数，复用项目中的关键流程。
0901:         target_time = self.target_times[index]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0902:         target = self._load_target(target_time)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0903:         obs = {sensor: self._load_sensor_window(sensor, target_time) for sensor in self.sensors}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0904:         return {    # 说明：返回当前函数计算得到的结果。
0905:             "target": target,    # 说明：保留该行以完成当前代码块的语法结构。
0906:             "target_time": target_time.isoformat(),    # 说明：调用函数或方法，执行具体工程动作。
0907:             "target_time_epoch": int(target_time.replace(tzinfo=None).timestamp()),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0908:             "observations": obs,    # 说明：保留该行以完成当前代码块的语法结构。
0909:             "target_variables": self.target_variables,    # 说明：保留该行以完成当前代码块的语法结构。
0910:             "pressure_levels": self.pressure_levels,    # 说明：保留该行以完成当前代码块的语法结构。
0911:         }    # 说明：保留该行以完成当前代码块的语法结构。
0912:     # 说明：空行，用于分隔逻辑块，提高可读性。
0913:     # 说明：空行，用于分隔逻辑块，提高可读性。
0914: def collate_retrieval_batch(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:    # 说明：定义函数，复用项目中的关键流程。
0915:     """Collate retrieval samples without padding observation point clouds."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0916:     target = torch.stack([item["target"] for item in batch], dim=0)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0917:     sensors = list(batch[0]["observations"].keys())    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0918:     observations: Dict[str, List[Dict[str, torch.Tensor]]] = {sensor: [] for sensor in sensors}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0919:     for item in batch:    # 说明：遍历集合或数据流，逐项完成处理。
0920:         for sensor in sensors:    # 说明：遍历集合或数据流，逐项完成处理。
0921:             observations[sensor].append(item["observations"][sensor])    # 说明：调用函数或方法，执行具体工程动作。
0922:     return {    # 说明：返回当前函数计算得到的结果。
0923:         "target": target,    # 说明：保留该行以完成当前代码块的语法结构。
0924:         "observations": observations,    # 说明：保留该行以完成当前代码块的语法结构。
0925:         "target_time": [item["target_time"] for item in batch],    # 说明：保留该行以完成当前代码块的语法结构。
0926:         "target_time_epoch": torch.tensor([item["target_time_epoch"] for item in batch], dtype=torch.long),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0927:         "target_variables": batch[0]["target_variables"],    # 说明：保留该行以完成当前代码块的语法结构。
0928:         "pressure_levels": batch[0]["pressure_levels"],    # 说明：保留该行以完成当前代码块的语法结构。
0929:     }    # 说明：保留该行以完成当前代码块的语法结构。
```

## src/models/retrieval/healda_hpx_vit.py

```text
0001: # -*- coding: utf-8 -*-    # 说明：中文/配置注释，说明相邻代码或参数用途。
0002: """ViT backbone for HealDA-style retrieval.    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0003:     # 说明：空行，用于分隔逻辑块，提高可读性。
0004: The class name keeps the HPX terminology used by HealDA.  When HPX dependencies    # 说明：保留该行以完成当前代码块的语法结构。
0005: are unavailable, the model operates on the 181x360 lat-lon fallback grid while    # 说明：保留该行以完成当前代码块的语法结构。
0006: retaining patch encode / Transformer / patch decode structure.    # 说明：保留该行以完成当前代码块的语法结构。
0007: """    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0008:     # 说明：空行，用于分隔逻辑块，提高可读性。
0009: from __future__ import annotations    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0010:     # 说明：空行，用于分隔逻辑块，提高可读性。
0011: import math    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0012: from typing import Sequence    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0013:     # 说明：空行，用于分隔逻辑块，提高可读性。
0014: import torch    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0015: import torch.utils.checkpoint as checkpoint    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0016: from torch import nn    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0017: import torch.nn.functional as F    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0018:     # 说明：空行，用于分隔逻辑块，提高可读性。
0019:     # 说明：空行，用于分隔逻辑块，提高可读性。
0020: class DropPath(nn.Module):    # 说明：定义核心类，封装模型、数据或训练职责。
0021:     def __init__(self, drop_prob: float = 0.0) -> None:    # 说明：定义函数，复用项目中的关键流程。
0022:         super().__init__()    # 说明：调用函数或方法，执行具体工程动作。
0023:         self.drop_prob = float(drop_prob)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0024:     # 说明：空行，用于分隔逻辑块，提高可读性。
0025:     def forward(self, x: torch.Tensor) -> torch.Tensor:    # 说明：定义函数，复用项目中的关键流程。
0026:         if self.drop_prob == 0.0 or not self.training:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0027:             return x    # 说明：返回当前函数计算得到的结果。
0028:         keep = 1.0 - self.drop_prob    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0029:         shape = (x.shape[0],) + (1,) * (x.ndim - 1)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0030:         random = keep + torch.rand(shape, dtype=x.dtype, device=x.device)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0031:         random.floor_()    # 说明：调用函数或方法，执行具体工程动作。
0032:         return x.div(keep) * random    # 说明：返回当前函数计算得到的结果。
0033:     # 说明：空行，用于分隔逻辑块，提高可读性。
0034:     # 说明：空行，用于分隔逻辑块，提高可读性。
0035: class TransformerBlock(nn.Module):    # 说明：定义核心类，封装模型、数据或训练职责。
0036:     """Pre-norm Transformer block with query/key RMS-normalized attention input."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0037:     # 说明：空行，用于分隔逻辑块，提高可读性。
0038:     def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, drop_path: float = 0.0) -> None:    # 说明：定义函数，复用项目中的关键流程。
0039:         super().__init__()    # 说明：调用函数或方法，执行具体工程动作。
0040:         self.norm1 = nn.LayerNorm(dim)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0041:         self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0042:         self.drop_path = DropPath(drop_path)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0043:         self.norm2 = nn.LayerNorm(dim)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0044:         hidden = int(dim * mlp_ratio)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0045:         self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0046:     # 说明：空行，用于分隔逻辑块，提高可读性。
0047:     def forward(self, x: torch.Tensor) -> torch.Tensor:    # 说明：定义函数，复用项目中的关键流程。
0048:         y = self.norm1(x)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0049:         # RMS-normalize Q/K/V input for bf16 stability; this is a lightweight approximation    # 说明：中文/配置注释，说明相邻代码或参数用途。
0050:         # of HealDA's q/k RMS normalization.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0051:         y = y / y.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0052:         attn_out, _ = self.attn(y, y, y, need_weights=False)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0053:         x = x + self.drop_path(attn_out)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0054:         x = x + self.drop_path(self.mlp(self.norm2(x)))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0055:         return x    # 说明：返回当前函数计算得到的结果。
0056:     # 说明：空行，用于分隔逻辑块，提高可读性。
0057:     # 说明：空行，用于分隔逻辑块，提高可读性。
0058: class LatLonViTBackbone(nn.Module):    # 说明：定义核心类，封装模型、数据或训练职责。
0059:     """Patch encode -> global Transformer -> patch decode backbone."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0060:     # 说明：空行，用于分隔逻辑块，提高可读性。
0061:     def __init__(    # 说明：定义函数，复用项目中的关键流程。
0062:         self,    # 说明：保留该行以完成当前代码块的语法结构。
0063:         in_channels: int,    # 说明：保留该行以完成当前代码块的语法结构。
0064:         out_channels: int = 26,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0065:         img_size: Sequence[int] = (181, 360),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0066:         patch_size: Sequence[int] = (6, 6),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0067:         dim: int = 512,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0068:         depth: int = 12,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0069:         heads: int = 8,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0070:         mlp_ratio: float = 4.0,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0071:         dropout: float = 0.05,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0072:         drop_path: float = 0.1,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0073:         use_checkpoint: bool = False,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0074:     ) -> None:    # 说明：保留该行以完成当前代码块的语法结构。
0075:         super().__init__()    # 说明：调用函数或方法，执行具体工程动作。
0076:         self.img_size = (int(img_size[0]), int(img_size[1]))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0077:         self.patch_size = (int(patch_size[0]), int(patch_size[1]))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0078:         self.dim = int(dim)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0079:         self.use_checkpoint = bool(use_checkpoint)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0080:         self.patch_encode = nn.Conv2d(in_channels, dim, kernel_size=self.patch_size, stride=self.patch_size)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0081:         h, w = self.img_size    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0082:         ph, pw = self.patch_size    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0083:         self.pad_h = (math.ceil(h / ph) * ph) - h    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0084:         self.pad_w = (math.ceil(w / pw) * pw) - w    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0085:         gh = (h + self.pad_h) // ph    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0086:         gw = (w + self.pad_w) // pw    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0087:         self.grid_tokens = (gh, gw)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0088:         self.pos_embed = nn.Parameter(torch.zeros(1, gh * gw, dim))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0089:         dpr = torch.linspace(0, drop_path, depth).tolist() if depth > 0 else []    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0090:         self.blocks = nn.ModuleList([    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0091:             TransformerBlock(dim=dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout, drop_path=dpr[i])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0092:             for i in range(depth)    # 说明：遍历集合或数据流，逐项完成处理。
0093:         ])    # 说明：调用函数或方法，执行具体工程动作。
0094:         self.norm = nn.LayerNorm(dim)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0095:         self.patch_decode = nn.ConvTranspose2d(dim, dim, kernel_size=self.patch_size, stride=self.patch_size)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0096:         self.out_proj = nn.Sequential(nn.GroupNorm(8 if dim % 8 == 0 else 1, dim), nn.SiLU(), nn.Conv2d(dim, out_channels, kernel_size=1))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0097:         nn.init.trunc_normal_(self.pos_embed, std=0.02)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0098:     # 说明：空行，用于分隔逻辑块，提高可读性。
0099:     def forward(self, x: torch.Tensor) -> torch.Tensor:    # 说明：定义函数，复用项目中的关键流程。
0100:         b, _, h, w = x.shape    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0101:         if self.pad_h or self.pad_w:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0102:             x = F.pad(x, (0, self.pad_w, 0, self.pad_h))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0103:         x = self.patch_encode(x)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0104:         gh, gw = x.shape[-2:]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0105:         tokens = x.flatten(2).transpose(1, 2)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0106:         if tokens.shape[1] != self.pos_embed.shape[1]:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0107:             pos = F.interpolate(    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0108:                 self.pos_embed.transpose(1, 2).view(1, self.dim, *self.grid_tokens),    # 说明：调用函数或方法，执行具体工程动作。
0109:                 size=(gh, gw), mode="bilinear", align_corners=False,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0110:             ).flatten(2).transpose(1, 2)    # 说明：调用函数或方法，执行具体工程动作。
0111:         else:    # 说明：执行默认分支，保证逻辑闭环。
0112:             pos = self.pos_embed    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0113:         tokens = tokens + pos    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0114:         for block in self.blocks:    # 说明：遍历集合或数据流，逐项完成处理。
0115:             if self.use_checkpoint and self.training:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0116:                 tokens = checkpoint.checkpoint(block, tokens, use_reentrant=False)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0117:             else:    # 说明：执行默认分支，保证逻辑闭环。
0118:                 tokens = block(tokens)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0119:         tokens = self.norm(tokens)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0120:         x = tokens.transpose(1, 2).view(b, self.dim, gh, gw)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0121:         x = self.patch_decode(x)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0122:         x = x[..., :h, :w]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0123:         return self.out_proj(x)    # 说明：返回当前函数计算得到的结果。
0124:     # 说明：空行，用于分隔逻辑块，提高可读性。
0125:     # 说明：空行，用于分隔逻辑块，提高可读性。
0126: class HPXViTBackbone(LatLonViTBackbone):    # 说明：定义核心类，封装模型、数据或训练职责。
0127:     """Compatibility alias for the HealDA HPX ViT backbone."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0128:     # 说明：空行，用于分隔逻辑块，提高可读性。
0129:     pass    # 说明：保留该行以完成当前代码块的语法结构。
```

## src/models/retrieval/healda_xichen_retrieval.py

```text
0001: # -*- coding: utf-8 -*-    # 说明：中文/配置注释，说明相邻代码或参数用途。
0002: """Top-level HealDA-style multi-source T/Q profile retrieval model."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0003:     # 说明：空行，用于分隔逻辑块，提高可读性。
0004: from __future__ import annotations    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0005:     # 说明：空行，用于分隔逻辑块，提高可读性。
0006: from typing import Any, Dict, Mapping, Sequence    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0007:     # 说明：空行，用于分隔逻辑块，提高可读性。
0008: import torch    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0009: from torch import nn    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0010:     # 说明：空行，用于分隔逻辑块，提高可读性。
0011: from .healda_hpx_vit import HPXViTBackbone, LatLonViTBackbone    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0012: from .healda_obs_encoder import HealDAObservationEncoder    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0013: from .retrieval_decoder import ProfileRetrievalDecoder    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0014:     # 说明：空行，用于分隔逻辑块，提高可读性。
0015: MODEL_SIZES: Dict[str, Dict[str, int]] = {    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0016:     "tiny": {"dim": 256, "depth": 6, "heads": 4, "obs_token_dim": 32, "sensor_embed_dim": 128},    # 说明：保留该行以完成当前代码块的语法结构。
0017:     "base": {"dim": 512, "depth": 12, "heads": 8, "obs_token_dim": 32, "sensor_embed_dim": 256},    # 说明：保留该行以完成当前代码块的语法结构。
0018:     "full_healda_like": {"dim": 1024, "depth": 24, "heads": 16, "obs_token_dim": 32, "sensor_embed_dim": 512},    # 说明：保留该行以完成当前代码块的语法结构。
0019: }    # 说明：保留该行以完成当前代码块的语法结构。
0020:     # 说明：空行，用于分隔逻辑块，提高可读性。
0021:     # 说明：空行，用于分隔逻辑块，提高可读性。
0022: class HealDAXiChenRetrieval(nn.Module):    # 说明：定义核心类，封装模型、数据或训练职责。
0023:     """Observation-only, background-free retrieval model.    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0024:     # 说明：空行，用于分隔逻辑块，提高可读性。
0025:     Input batch format is produced by ``collate_retrieval_batch``.  Forward output    # 说明：保留该行以完成当前代码块的语法结构。
0026:     is ``[B, 26, 181, 360]`` where channels are ``t-50..t-1000`` followed by    # 说明：保留该行以完成当前代码块的语法结构。
0027:     ``q-50..q-1000``.    # 说明：保留该行以完成当前代码块的语法结构。
0028:     """    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0029:     # 说明：空行，用于分隔逻辑块，提高可读性。
0030:     def __init__(    # 说明：定义函数，复用项目中的关键流程。
0031:         self,    # 说明：保留该行以完成当前代码块的语法结构。
0032:         sensors: Sequence[str] = ("atms", "amsua", "mhs", "hrs4", "gdas_prebufr"),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0033:         target_vars: Sequence[str] | None = None,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0034:         pressure_levels: Sequence[int] | None = None,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0035:         output_channels: int = 26,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0036:         output_grid: Sequence[int] = (181, 360),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0037:         grid_backend: str = "hpx",    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0038:         fallback_grid_backend: str = "latlon",    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0039:         hpx_nside: int = 64,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0040:         model_size: str = "base",    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0041:         dim: int | None = None,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0042:         depth: int | None = None,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0043:         heads: int | None = None,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0044:         obs_token_dim: int | None = None,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0045:         sensor_embed_dim: int | None = None,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0046:         patch_size: Sequence[int] = (6, 6),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0047:         mlp_ratio: float = 4.0,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0048:         dropout: float = 0.05,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0049:         drop_path: float = 0.1,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0050:         concat_observability_mask: bool = True,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0051:         use_gradient_checkpointing: bool = False,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0052:         **kwargs: Any,    # 说明：保留该行以完成当前代码块的语法结构。
0053:     ) -> None:    # 说明：保留该行以完成当前代码块的语法结构。
0054:         super().__init__()    # 说明：调用函数或方法，执行具体工程动作。
0055:         if model_size not in MODEL_SIZES:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0056:             raise ValueError(f"Unknown model_size {model_size!r}; choose one of {sorted(MODEL_SIZES)}")    # 说明：调用函数或方法，执行具体工程动作。
0057:         preset = MODEL_SIZES[model_size]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0058:         self.sensors = list(sensors)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0059:         self.pressure_levels = list(pressure_levels or [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0060:         self.target_vars = list(target_vars or [*(f"t-{p}" for p in self.pressure_levels), *(f"q-{p}" for p in self.pressure_levels)])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0061:         self.output_channels = int(output_channels)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0062:         self.output_grid = (int(output_grid[0]), int(output_grid[1]))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0063:         self.grid_backend = grid_backend    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0064:         self.fallback_grid_backend = fallback_grid_backend    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0065:         self.hpx_nside = int(hpx_nside)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0066:         self.model_size = model_size    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0067:         self.concat_observability_mask = bool(concat_observability_mask)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0068:     # 说明：空行，用于分隔逻辑块，提高可读性。
0069:         self.dim = int(dim or preset["dim"])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0070:         self.depth = int(depth or preset["depth"])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0071:         self.heads = int(heads or preset["heads"])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0072:         self.obs_token_dim = int(obs_token_dim or preset["obs_token_dim"])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0073:         self.sensor_embed_dim = int(sensor_embed_dim or preset["sensor_embed_dim"])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0074:     # 说明：空行，用于分隔逻辑块，提高可读性。
0075:         active_grid_backend = grid_backend    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0076:         if grid_backend == "hpx":    # 说明：执行条件分支，处理不同运行环境或配置情况。
0077:             # The current XiChen-compatible implementation keeps the HealDA HPX API but    # 说明：中文/配置注释，说明相邻代码或参数用途。
0078:             # trains on the public02 [181, 360] lat-lon labels.  Native HPX scatter/regrid    # 说明：中文/配置注释，说明相邻代码或参数用途。
0079:             # remains available through tools/regrid_hpx_latlon.py, but the training path    # 说明：中文/配置注释，说明相邻代码或参数用途。
0080:             # deliberately falls back to lat-lon unless a future native HPX module replaces    # 说明：中文/配置注释，说明相邻代码或参数用途。
0081:             # HPXAggregation with an earth2grid-backed implementation.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0082:             if fallback_grid_backend != "latlon":    # 说明：执行条件分支，处理不同运行环境或配置情况。
0083:                 raise ImportError("grid_backend=hpx requested, but this package currently requires fallback_grid_backend=latlon for training")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0084:             active_grid_backend = "latlon"    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0085:         self.active_grid_backend = active_grid_backend    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0086:     # 说明：空行，用于分隔逻辑块，提高可读性。
0087:         self.obs_encoder = HealDAObservationEncoder(    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0088:             sensors=self.sensors,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0089:             grid_shape=self.output_grid,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0090:             token_dim=self.obs_token_dim,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0091:             sensor_embed_dim=self.sensor_embed_dim,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0092:             grid_backend=self.active_grid_backend,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0093:             hpx_nside=self.hpx_nside,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0094:         )    # 说明：调用函数或方法，执行具体工程动作。
0095:         in_channels = self.sensor_embed_dim + (len(self.sensors) if self.concat_observability_mask else 0)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0096:         backbone_cls = HPXViTBackbone if self.active_grid_backend == "hpx" else LatLonViTBackbone    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0097:         self.backbone = backbone_cls(    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0098:             in_channels=in_channels,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0099:             out_channels=self.output_channels,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0100:             img_size=self.output_grid,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0101:             patch_size=patch_size,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0102:             dim=self.dim,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0103:             depth=self.depth,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0104:             heads=self.heads,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0105:             mlp_ratio=mlp_ratio,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0106:             dropout=dropout,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0107:             drop_path=drop_path,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0108:             use_checkpoint=use_gradient_checkpointing,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0109:         )    # 说明：调用函数或方法，执行具体工程动作。
0110:         self.decoder = ProfileRetrievalDecoder(self.output_channels, output_channels=self.output_channels, pressure_levels=self.pressure_levels)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0111:     # 说明：空行，用于分隔逻辑块，提高可读性。
0112:     def forward(self, batch: Mapping[str, Any] | None = None, *, observations: Mapping[str, Any] | None = None, as_profile: bool = False) -> torch.Tensor:    # 说明：定义函数，复用项目中的关键流程。
0113:         if batch is not None:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0114:             observations = batch["observations"]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0115:         if observations is None:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0116:             raise ValueError("HealDAXiChenRetrieval.forward requires a batch or observations mapping")    # 说明：调用函数或方法，执行具体工程动作。
0117:         obs_features, obs_masks = self.obs_encoder(observations)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0118:         if self.concat_observability_mask:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0119:             x = torch.cat([obs_features, obs_masks.to(dtype=obs_features.dtype)], dim=1)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0120:         else:    # 说明：执行默认分支，保证逻辑闭环。
0121:             x = obs_features    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0122:         y = self.backbone(x)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0123:         return self.decoder(y, as_profile=as_profile)    # 说明：返回当前函数计算得到的结果。
0124:     # 说明：空行，用于分隔逻辑块，提高可读性。
0125:     def estimate_vram_gb(self, batch_size: int = 1) -> float:    # 说明：定义函数，复用项目中的关键流程。
0126:         """Very coarse activation-memory estimate for run planning."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0127:         h, w = self.output_grid    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0128:         ph, pw = self.backbone.patch_size    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0129:         tokens = ((h + self.backbone.pad_h) // ph) * ((w + self.backbone.pad_w) // pw)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0130:         attention = batch_size * self.depth * self.heads * tokens * tokens * 2 / 1024**3    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0131:         activations = batch_size * self.depth * tokens * self.dim * 8 / 1024**3    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0132:         params = sum(p.numel() for p in self.parameters()) * 4 / 1024**3    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0133:         return float(params + activations + attention)    # 说明：返回当前函数计算得到的结果。
```

## tools/smoke_retrieval_model.py

```text
0001: #!/usr/bin/env python    # 说明：脚本解释器声明，确保可直接执行。
0002: # -*- coding: utf-8 -*-    # 说明：中文/配置注释，说明相邻代码或参数用途。
0003: """Run a synthetic forward/loss/backward smoke test for the retrieval model."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0004:     # 说明：空行，用于分隔逻辑块，提高可读性。
0005: from __future__ import annotations    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0006:     # 说明：空行，用于分隔逻辑块，提高可读性。
0007: import argparse    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0008: import sys    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0009: from pathlib import Path    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0010: from typing import Dict, List    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0011:     # 说明：空行，用于分隔逻辑块，提高可读性。
0012: PROJECT_ROOT = Path(__file__).resolve().parents[1]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0013: if str(PROJECT_ROOT) not in sys.path:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0014:     sys.path.insert(0, str(PROJECT_ROOT))    # 说明：调用函数或方法，执行具体工程动作。
0015:     # 说明：空行，用于分隔逻辑块，提高可读性。
0016: import torch    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0017:     # 说明：空行，用于分隔逻辑块，提高可读性。
0018: from src.losses.retrieval_tq_loss import RetrievalTQLoss    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0019: from src.models.retrieval.healda_xichen_retrieval import HealDAXiChenRetrieval    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0020: from src.utils.device import autocast, configure_accelerator_performance    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0021:     # 说明：空行，用于分隔逻辑块，提高可读性。
0022:     # 说明：空行，用于分隔逻辑块，提高可读性。
0023: def make_obs(n: int, device_cpu: torch.device) -> Dict[str, torch.Tensor]:    # 说明：定义函数，复用项目中的关键流程。
0024:     """Create one synthetic point-cloud observation dictionary on CPU."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0025:     lat = torch.linspace(-80.0, 80.0, n, device=device_cpu)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0026:     lon = torch.linspace(0.0, 359.0, n, device=device_cpu)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0027:     ch = torch.arange(n, device=device_cpu) % 16    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0028:     return {    # 说明：返回当前函数计算得到的结果。
0029:         "measurement": torch.randn(n, device=device_cpu) * 0.1,    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0030:         "lat": lat,    # 说明：保留该行以完成当前代码块的语法结构。
0031:         "lon": lon,    # 说明：保留该行以完成当前代码块的语法结构。
0032:         "relative_time": torch.zeros(n, device=device_cpu),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0033:         "channel": ch.long(),    # 说明：调用函数或方法，执行具体工程动作。
0034:         "platform": torch.zeros(n, dtype=torch.long, device=device_cpu),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0035:         "scan_angle": torch.zeros(n, device=device_cpu),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0036:         "sat_zenith_angle": torch.zeros(n, device=device_cpu),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0037:         "solar_zenith_angle": torch.zeros(n, device=device_cpu),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0038:         "pressure": torch.full((n,), 500.0, device=device_cpu),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0039:         "height": torch.full((n,), 5000.0, device=device_cpu),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0040:         "variable_type": ch.long(),    # 说明：调用函数或方法，执行具体工程动作。
0041:         "report_type": torch.zeros(n, dtype=torch.long, device=device_cpu),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0042:         "station_type": torch.zeros(n, dtype=torch.long, device=device_cpu),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0043:         "quality_flag": torch.zeros(n, device=device_cpu),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0044:         "mask": torch.ones(n, device=device_cpu),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0045:     }    # 说明：保留该行以完成当前代码块的语法结构。
0046:     # 说明：空行，用于分隔逻辑块，提高可读性。
0047:     # 说明：空行，用于分隔逻辑块，提高可读性。
0048: def main() -> None:    # 说明：定义函数，复用项目中的关键流程。
0049:     parser = argparse.ArgumentParser()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0050:     parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0051:     parser.add_argument("--model_size", choices=["tiny", "base", "full_healda_like"], default="tiny")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0052:     parser.add_argument("--batch_size", type=int, default=1)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0053:     parser.add_argument("--points", type=int, default=256)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0054:     parser.add_argument("--grid", nargs=2, type=int, default=[181, 360])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0055:     parser.add_argument("--bf16", action="store_true")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0056:     parser.add_argument("--fast_cpu", action="store_true", help="use a tiny custom model for CPU-only CI smoke tests")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0057:     args = parser.parse_args()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0058:     # 说明：空行，用于分隔逻辑块，提高可读性。
0059:     device_type = "cuda" if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0060:     device = torch.device(args.device if device_type == "cuda" else "cpu")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0061:     configure_accelerator_performance(device_type=device_type)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0062:     # 说明：空行，用于分隔逻辑块，提高可读性。
0063:     sensors = ["atms", "amsua", "mhs", "hrs4", "gdas_prebufr"]    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0064:     observations: Dict[str, List[Dict[str, torch.Tensor]]] = {    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0065:         sensor: [make_obs(args.points, torch.device("cpu")) for _ in range(args.batch_size)] for sensor in sensors    # 说明：调用函数或方法，执行具体工程动作。
0066:     }    # 说明：保留该行以完成当前代码块的语法结构。
0067:     grid = (int(args.grid[0]), int(args.grid[1]))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0068:     batch = {"observations": observations, "target": torch.randn(args.batch_size, 26, *grid, device=device)}    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0069:     model_kwargs = dict(model_size=args.model_size, sensors=sensors, output_grid=grid)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0070:     if args.fast_cpu:    # 说明：执行条件分支，处理不同运行环境或配置情况。
0071:         model_kwargs.update(dict(dim=32, depth=1, heads=4, obs_token_dim=8, sensor_embed_dim=32, patch_size=(6, 6), dropout=0.0, drop_path=0.0))    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0072:     model = HealDAXiChenRetrieval(**model_kwargs).to(device)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0073:     loss_fn = RetrievalTQLoss()    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0074:     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0075:     dtype = torch.bfloat16 if args.bf16 and device_type == "cuda" else torch.float32    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0076:     # 说明：空行，用于分隔逻辑块，提高可读性。
0077:     model.train()    # 说明：调用函数或方法，执行具体工程动作。
0078:     optimizer.zero_grad(set_to_none=True)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0079:     with autocast(device_type, dtype=dtype):    # 说明：使用上下文管理器，安全管理文件、AMP 或 DDP 同步。
0080:         pred = model(batch)    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0081:         losses = loss_fn(pred, batch["target"])    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0082:     losses["total_loss"].backward()    # 说明：调用函数或方法，执行具体工程动作。
0083:     optimizer.step()    # 说明：调用函数或方法，执行具体工程动作。
0084:     print(f"forward_shape={tuple(pred.shape)}")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0085:     print(f"loss={float(losses['total_loss'].detach()):.6f}")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0086:     print("smoke_status=ok")    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0087:     # 说明：空行，用于分隔逻辑块，提高可读性。
0088:     # 说明：空行，用于分隔逻辑块，提高可读性。
0089: if __name__ == "__main__":    # 说明：执行条件分支，处理不同运行环境或配置情况。
0090:     main()    # 说明：调用函数或方法，执行具体工程动作。
```

## configs/datamodule/retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml

```text
0001: # HealDA-style retrieval datamodule: observations -> ERA5 T/Q profiles.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0002: _target_: src.datamodules.retrieval.healda_datamodule.HealDARetrievalDataModule    # 说明：保留该行以完成当前代码块的语法结构。
0003:     # 说明：空行，用于分隔逻辑块，提高可读性。
0004: obs_dir: ${paths.obs_dir}    # 说明：保留该行以完成当前代码块的语法结构。
0005: era5_dir: ${paths.era5_dir}    # 说明：保留该行以完成当前代码块的语法结构。
0006: scale_dir: ${paths.scale_dir}    # 说明：保留该行以完成当前代码块的语法结构。
0007:     # 说明：空行，用于分隔逻辑块，提高可读性。
0008: sensors:    # 说明：保留该行以完成当前代码块的语法结构。
0009:   satellite:    # 说明：保留该行以完成当前代码块的语法结构。
0010:     - atms    # 说明：保留该行以完成当前代码块的语法结构。
0011:     - amsua    # 说明：保留该行以完成当前代码块的语法结构。
0012:     - mhs    # 说明：保留该行以完成当前代码块的语法结构。
0013:     - hrs4    # 说明：保留该行以完成当前代码块的语法结构。
0014:   conventional:    # 说明：保留该行以完成当前代码块的语法结构。
0015:     - gdas_prebufr    # 说明：保留该行以完成当前代码块的语法结构。
0016:     # 说明：空行，用于分隔逻辑块，提高可读性。
0017: target:    # 说明：保留该行以完成当前代码块的语法结构。
0018:   variables:    # 说明：保留该行以完成当前代码块的语法结构。
0019:     - t    # 说明：保留该行以完成当前代码块的语法结构。
0020:     - q    # 说明：保留该行以完成当前代码块的语法结构。
0021:   pressure_levels: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]    # 说明：保留该行以完成当前代码块的语法结构。
0022:   output_channels: 26    # 说明：保留该行以完成当前代码块的语法结构。
0023:     # 说明：空行，用于分隔逻辑块，提高可读性。
0024: # Data and indexing behavior.  ERA5 is supervision only and is never exposed as an input.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0025: data:    # 说明：保留该行以完成当前代码块的语法结构。
0026:   obs_dir: ${paths.obs_dir}    # 说明：保留该行以完成当前代码块的语法结构。
0027:   era5_dir: ${paths.era5_dir}    # 说明：保留该行以完成当前代码块的语法结构。
0028:   grid_shape: [181, 360]    # 说明：保留该行以完成当前代码块的语法结构。
0029:   obs_window:    # 说明：保留该行以完成当前代码块的语法结构。
0030:     start_hours: -21    # 说明：保留该行以完成当前代码块的语法结构。
0031:     end_hours: 3    # 说明：保留该行以完成当前代码块的语法结构。
0032:   no_lookahead: false    # 说明：保留该行以完成当前代码块的语法结构。
0033:   no_lookahead_window:    # 说明：保留该行以完成当前代码块的语法结构。
0034:     start_hours: -24    # 说明：保留该行以完成当前代码块的语法结构。
0035:     end_hours: 0    # 说明：保留该行以完成当前代码块的语法结构。
0036:   # public02 ERA5 can contain non-00/06/12/18 files such as 22:00:00-t-1000.npy.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0037:   # Use dt_data=1 + strict_time_index=true for exact nested-file scanning on the supercomputer.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0038:   dt_data: 1    # 说明：保留该行以完成当前代码块的语法结构。
0039:   dt_obs: 3    # 说明：保留该行以完成当前代码块的语法结构。
0040:   max_points_per_sensor: 250000    # 说明：保留该行以完成当前代码块的语法结构。
0041:   strict_time_index: true    # 说明：保留该行以完成当前代码块的语法结构。
0042:   target_cache_size: 16    # 说明：保留该行以完成当前代码块的语法结构。
0043:   normalize_target: true    # 说明：保留该行以完成当前代码块的语法结构。
0044:   normalize_obs: true    # 说明：保留该行以完成当前代码块的语法结构。
0045:   require_obs_stats: false    # 说明：保留该行以完成当前代码块的语法结构。
0046:   # XiChen 13-level ERA5 state order used only to select T/Q label channels when files contain full states.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0047:   era5_all_vars: [    # 说明：保留该行以完成当前代码块的语法结构。
0048:     "t2m", "u10", "v10", "msl",    # 说明：保留该行以完成当前代码块的语法结构。
0049:     "z-50", "z-100", "z-150", "z-200", "z-250", "z-300", "z-400", "z-500", "z-600", "z-700", "z-850", "z-925", "z-1000",    # 说明：保留该行以完成当前代码块的语法结构。
0050:     "u-50", "u-100", "u-150", "u-200", "u-250", "u-300", "u-400", "u-500", "u-600", "u-700", "u-850", "u-925", "u-1000",    # 说明：保留该行以完成当前代码块的语法结构。
0051:     "v-50", "v-100", "v-150", "v-200", "v-250", "v-300", "v-400", "v-500", "v-600", "v-700", "v-850", "v-925", "v-1000",    # 说明：保留该行以完成当前代码块的语法结构。
0052:     "t-50", "t-100", "t-150", "t-200", "t-250", "t-300", "t-400", "t-500", "t-600", "t-700", "t-850", "t-925", "t-1000",    # 说明：保留该行以完成当前代码块的语法结构。
0053:     "q-50", "q-100", "q-150", "q-200", "q-250", "q-300", "q-400", "q-500", "q-600", "q-700", "q-850", "q-925", "q-1000",    # 说明：保留该行以完成当前代码块的语法结构。
0054:   ]    # 说明：保留该行以完成当前代码块的语法结构。
0055:     # 说明：空行，用于分隔逻辑块，提高可读性。
0056: qc:    # 说明：保留该行以完成当前代码块的语法结构。
0057:   microwave_bt_range: [0, 400]    # 说明：保留该行以完成当前代码块的语法结构。
0058:   infrared_bt_range: [0, 400]    # 说明：保留该行以完成当前代码块的语法结构。
0059:   humidity_range: [0, 1]    # 说明：保留该行以完成当前代码块的语法结构。
0060:   wind_range: [-150, 150]    # 说明：保留该行以完成当前代码块的语法结构。
0061:   temperature_range: [150, 350]    # 说明：保留该行以完成当前代码块的语法结构。
0062:   pressure_range_hpa: [0.5, 1100]    # 说明：保留该行以完成当前代码块的语法结构。
0063:   height_range_m: [0, 60000]    # 说明：保留该行以完成当前代码块的语法结构。
0064:     # 说明：空行，用于分隔逻辑块，提高可读性。
0065: # Used only when retrieval_obs_stats/<sensor>.npz is missing. Override after data inspection if needed.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0066: obs_default_normalization:    # 说明：保留该行以完成当前代码块的语法结构。
0067:   satellite: [250.0, 50.0]    # 说明：保留该行以完成当前代码块的语法结构。
0068:   conventional: [0.0, 1.0]    # 说明：保留该行以完成当前代码块的语法结构。
0069:     # 说明：空行，用于分隔逻辑块，提高可读性。
0070: start_train_year: 2016    # 说明：保留该行以完成当前代码块的语法结构。
0071: start_val_year: 2022    # 说明：保留该行以完成当前代码块的语法结构。
0072: start_test_year: 2023    # 说明：保留该行以完成当前代码块的语法结构。
0073: end_year: 2024    # 说明：保留该行以完成当前代码块的语法结构。
0074:     # 说明：空行，用于分隔逻辑块，提高可读性。
0075: seed: 1024    # 说明：保留该行以完成当前代码块的语法结构。
0076: batch_size: 1    # 说明：保留该行以完成当前代码块的语法结构。
0077: num_workers: 8    # 说明：保留该行以完成当前代码块的语法结构。
0078: shuffle: true    # 说明：保留该行以完成当前代码块的语法结构。
0079: pin_memory: true    # 说明：保留该行以完成当前代码块的语法结构。
0080: prefetch_factor: 4    # 说明：保留该行以完成当前代码块的语法结构。
0081:     # 说明：空行，用于分隔逻辑块，提高可读性。
0082: debug: false    # 说明：保留该行以完成当前代码块的语法结构。
0083: max_debug_samples: 8    # 说明：保留该行以完成当前代码块的语法结构。
```

## configs/model/retrieval/healda_xichen_tq13.yaml

```text
0001: # HealDA-style model that outputs [B,26,181,360].    # 说明：中文/配置注释，说明相邻代码或参数用途。
0002: # Kept at the model-config root so the user-facing override model.model_size=tiny works.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0003: model_size: base    # 说明：保留该行以完成当前代码块的语法结构。
0004:     # 说明：空行，用于分隔逻辑块，提高可读性。
0005: net:    # 说明：保留该行以完成当前代码块的语法结构。
0006:   _target_: src.models.retrieval.healda_xichen_retrieval.HealDAXiChenRetrieval    # 说明：保留该行以完成当前代码块的语法结构。
0007:   sensors:    # 说明：保留该行以完成当前代码块的语法结构。
0008:     - atms    # 说明：保留该行以完成当前代码块的语法结构。
0009:     - amsua    # 说明：保留该行以完成当前代码块的语法结构。
0010:     - mhs    # 说明：保留该行以完成当前代码块的语法结构。
0011:     - hrs4    # 说明：保留该行以完成当前代码块的语法结构。
0012:     - gdas_prebufr    # 说明：保留该行以完成当前代码块的语法结构。
0013:   target_vars: [    # 说明：保留该行以完成当前代码块的语法结构。
0014:     "t-50", "t-100", "t-150", "t-200", "t-250", "t-300", "t-400", "t-500", "t-600", "t-700", "t-850", "t-925", "t-1000",    # 说明：保留该行以完成当前代码块的语法结构。
0015:     "q-50", "q-100", "q-150", "q-200", "q-250", "q-300", "q-400", "q-500", "q-600", "q-700", "q-850", "q-925", "q-1000",    # 说明：保留该行以完成当前代码块的语法结构。
0016:   ]    # 说明：保留该行以完成当前代码块的语法结构。
0017:   pressure_levels: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]    # 说明：保留该行以完成当前代码块的语法结构。
0018:   output_channels: 26    # 说明：保留该行以完成当前代码块的语法结构。
0019:   output_grid: [181, 360]    # 说明：保留该行以完成当前代码块的语法结构。
0020:   grid_backend: hpx    # 说明：保留该行以完成当前代码块的语法结构。
0021:   fallback_grid_backend: latlon    # 说明：保留该行以完成当前代码块的语法结构。
0022:   hpx_nside: 64    # 说明：保留该行以完成当前代码块的语法结构。
0023:   model_size: ${model.model_size}    # 说明：保留该行以完成当前代码块的语法结构。
0024:   patch_size: [6, 6]    # 说明：保留该行以完成当前代码块的语法结构。
0025:   mlp_ratio: 4.0    # 说明：保留该行以完成当前代码块的语法结构。
0026:   dropout: 0.05    # 说明：保留该行以完成当前代码块的语法结构。
0027:   drop_path: 0.1    # 说明：保留该行以完成当前代码块的语法结构。
0028:   concat_observability_mask: true    # 说明：保留该行以完成当前代码块的语法结构。
0029:   use_gradient_checkpointing: ${oc.select:training.use_gradient_checkpointing,false}    # 说明：保留该行以完成当前代码块的语法结构。
0030:     # 说明：空行，用于分隔逻辑块，提高可读性。
0031: model_size_options:    # 说明：保留该行以完成当前代码块的语法结构。
0032:   tiny:    # 说明：保留该行以完成当前代码块的语法结构。
0033:     dim: 256    # 说明：保留该行以完成当前代码块的语法结构。
0034:     depth: 6    # 说明：保留该行以完成当前代码块的语法结构。
0035:     heads: 4    # 说明：保留该行以完成当前代码块的语法结构。
0036:     obs_token_dim: 32    # 说明：保留该行以完成当前代码块的语法结构。
0037:     sensor_embed_dim: 128    # 说明：保留该行以完成当前代码块的语法结构。
0038:   base:    # 说明：保留该行以完成当前代码块的语法结构。
0039:     dim: 512    # 说明：保留该行以完成当前代码块的语法结构。
0040:     depth: 12    # 说明：保留该行以完成当前代码块的语法结构。
0041:     heads: 8    # 说明：保留该行以完成当前代码块的语法结构。
0042:     obs_token_dim: 32    # 说明：保留该行以完成当前代码块的语法结构。
0043:     sensor_embed_dim: 256    # 说明：保留该行以完成当前代码块的语法结构。
0044:   full_healda_like:    # 说明：保留该行以完成当前代码块的语法结构。
0045:     dim: 1024    # 说明：保留该行以完成当前代码块的语法结构。
0046:     depth: 24    # 说明：保留该行以完成当前代码块的语法结构。
0047:     heads: 16    # 说明：保留该行以完成当前代码块的语法结构。
0048:     obs_token_dim: 32    # 说明：保留该行以完成当前代码块的语法结构。
0049:     sensor_embed_dim: 512    # 说明：保留该行以完成当前代码块的语法结构。
```

## configs/training/retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml

```text
0001: epochs: 50    # 说明：保留该行以完成当前代码块的语法结构。
0002: seed: 1024    # 说明：保留该行以完成当前代码块的语法结构。
0003: resume_ckpt: false    # 说明：保留该行以完成当前代码块的语法结构。
0004: pretrain_ckpt: null    # 说明：保留该行以完成当前代码块的语法结构。
0005: resume_pretrain: false    # 说明：保留该行以完成当前代码块的语法结构。
0006:     # 说明：空行，用于分隔逻辑块，提高可读性。
0007: lr: 1e-4    # 说明：保留该行以完成当前代码块的语法结构。
0008: betas: [0.9, 0.95]    # 说明：保留该行以完成当前代码块的语法结构。
0009: weight_decay: 0.05    # 说明：保留该行以完成当前代码块的语法结构。
0010: max_grad_norm: 1.0    # 说明：保留该行以完成当前代码块的语法结构。
0011: gradient_accumulation_steps: 2    # 说明：保留该行以完成当前代码块的语法结构。
0012:     # 说明：空行，用于分隔逻辑块，提高可读性。
0013: warmup_epochs: 5    # 说明：保留该行以完成当前代码块的语法结构。
0014: scheduler_type: cosine_warmup    # 说明：保留该行以完成当前代码块的语法结构。
0015: find_unused_parameters: false    # 说明：保留该行以完成当前代码块的语法结构。
0016: broadcast_buffers: false    # 说明：保留该行以完成当前代码块的语法结构。
0017: gradient_as_bucket_view: true    # 说明：保留该行以完成当前代码块的语法结构。
0018: static_graph: false    # 说明：保留该行以完成当前代码块的语法结构。
0019: bucket_cap_mb: 64    # 说明：保留该行以完成当前代码块的语法结构。
0020:     # 说明：空行，用于分隔逻辑块，提高可读性。
0021: use_amp: true    # 说明：保留该行以完成当前代码块的语法结构。
0022: precision:    # 说明：保留该行以完成当前代码块的语法结构。
0023:   type: bf16    # 说明：保留该行以完成当前代码块的语法结构。
0024: device: cuda    # 说明：保留该行以完成当前代码块的语法结构。
0025: cudnn_benchmark: true    # 说明：保留该行以完成当前代码块的语法结构。
0026: deterministic: false    # 说明：保留该行以完成当前代码块的语法结构。
0027: float32_matmul_precision: high    # 说明：保留该行以完成当前代码块的语法结构。
0028: fused_adamw: false    # 说明：保留该行以完成当前代码块的语法结构。
0029: use_gradient_checkpointing: false    # 说明：保留该行以完成当前代码块的语法结构。
0030: compile:    # 说明：保留该行以完成当前代码块的语法结构。
0031:   enabled: false    # 说明：保留该行以完成当前代码块的语法结构。
0032:   target: backbone    # 说明：保留该行以完成当前代码块的语法结构。
0033:   mode: max-autotune-no-cudagraphs    # 说明：保留该行以完成当前代码块的语法结构。
0034:     # 说明：空行，用于分隔逻辑块，提高可读性。
0035: log_dir: ${paths.output_dir}    # 说明：保留该行以完成当前代码块的语法结构。
0036: profile: false    # 说明：保留该行以完成当前代码块的语法结构。
0037: log_every_n_steps: 20    # 说明：保留该行以完成当前代码块的语法结构。
0038:     # 说明：空行，用于分隔逻辑块，提高可读性。
0039: # Debug overrides can be supplied from CLI:    # 说明：中文/配置注释，说明相邻代码或参数用途。
0040: # debug=true model.net.model_size=tiny datamodule.batch_size=1 datamodule.num_workers=0    # 说明：中文/配置注释，说明相邻代码或参数用途。
0041: limit_train_batches: null    # 说明：保留该行以完成当前代码块的语法结构。
0042: limit_val_batches: null    # 说明：保留该行以完成当前代码块的语法结构。
```

## scripts/retrieval/train_healda_xichen_tq13.slurm

```text
0001: #!/bin/bash    # 说明：脚本解释器声明，确保可直接执行。
0002: #SBATCH --job-name=xichen_retrieval_tq13    # 说明：中文/配置注释，说明相邻代码或参数用途。
0003: #SBATCH --nodes=1    # 说明：中文/配置注释，说明相邻代码或参数用途。
0004: #SBATCH --partition=zs    # 说明：中文/配置注释，说明相邻代码或参数用途。
0005: #SBATCH --ntasks-per-node=2    # 说明：中文/配置注释，说明相邻代码或参数用途。
0006: #SBATCH --gres=gpu:2    # 说明：中文/配置注释，说明相邻代码或参数用途。
0007: #SBATCH --nodelist=gnode02    # 说明：中文/配置注释，说明相邻代码或参数用途。
0008: #SBATCH --cpus-per-task=8    # 说明：中文/配置注释，说明相邻代码或参数用途。
0009: #SBATCH --mem=500G    # 说明：中文/配置注释，说明相邻代码或参数用途。
0010: #SBATCH --output=slurmlogs/%x_%j.out    # 说明：中文/配置注释，说明相邻代码或参数用途。
0011: #SBATCH --error=slurmlogs/%x_%j.err    # 说明：中文/配置注释，说明相邻代码或参数用途。
0012:     # 说明：空行，用于分隔逻辑块，提高可读性。
0013: source ~/.bashrc    # 说明：保留该行以完成当前代码块的语法结构。
0014: conda activate xichen_v1    # 说明：保留该行以完成当前代码块的语法结构。
0015:     # 说明：空行，用于分隔逻辑块，提高可读性。
0016: export NCCL_DEBUG=INFO    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0017: export NCCL_IB_DISABLE=1    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0018: export NCCL_P2P_DISABLE=1    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0019: export NCCL_NSOCKS_PERTHREAD=2    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0020: export NCCL_SOCKET_NTHREADS=4    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0021: export PYTHONFAULTHANDLER=1    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0022: export HYDRA_FULL_ERROR=1    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0023:     # 说明：空行，用于分隔逻辑块，提高可读性。
0024: torchrun \    # 说明：保留该行以完成当前代码块的语法结构。
0025:   --nproc_per_node=2 \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0026:   --master_port=29501 \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0027:   main.py \    # 说明：保留该行以完成当前代码块的语法结构。
0028:   --config-name=train.yaml \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0029:   datamodule=retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0030:   model=retrieval/healda_xichen_tq13.yaml \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0031:   pipeline=retrieval/trainer.yaml \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0032:   training=retrieval/healda_atms_amsua_mhs_hrs4_gdas_tq13.yaml \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0033:   loss_fn=retrieval_tq_huber.yaml \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0034:   paths=retrieval_public02.yaml \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0035:   paths.obs_dir=/public02/data/Observation/observation_npy/ \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0036:   paths.era5_dir=/public02/data/era5_np181x360_level13 \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0037:   paths.scale_dir=/public02/data/era5_np181x360_level13/normalized_mean_std \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0038:   training.device=cuda \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0039:   training.precision.type=bf16 \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0040:   training.gradient_accumulation_steps=2 \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0041:   datamodule.data.strict_time_index=true \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0042:   datamodule.data.dt_data=1 \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0043:   datamodule.num_workers=8 \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0044:   datamodule.prefetch_factor=4 \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0045:   model.model_size=base \    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0046:   task_name=healda_xichen_retrieval_atms_amsua_mhs_hrs4_gdas_tq13    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
```

## setup.py

```text
0001: #!/usr/bin/env python    # 说明：脚本解释器声明，确保可直接执行。
0002: # -*- coding: utf-8 -*-    # 说明：中文/配置注释，说明相邻代码或参数用途。
0003: """Package metadata for XiChen/HealDA retrieval."""    # 说明：模块或函数文档字符串，说明该代码块的工程作用。
0004:     # 说明：空行，用于分隔逻辑块，提高可读性。
0005: from setuptools import find_packages, setup    # 说明：导入运行所需模块，支撑训练、数据读取或分布式功能。
0006:     # 说明：空行，用于分隔逻辑块，提高可读性。
0007: setup(    # 说明：调用函数或方法，执行具体工程动作。
0008:     name="xichen-healda-retrieval",    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0009:     version="1.1.0",    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0010:     description="XiChen HealDA-style multi-source observation to ERA5 T/Q13 retrieval",    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0011:     author="XiChen/HealDA retrieval engineering",    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0012:     packages=find_packages(),    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0013:     install_requires=[    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0014:         "torch>=2.1",    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0015:         "hydra-core>=1.3",    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0016:         "omegaconf>=2.3",    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0017:         "numpy>=1.23",    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0018:         "tqdm>=4.64",    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0019:         "tensorboard>=2.12",    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0020:         "pyyaml>=6.0",    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0021:     ],    # 说明：保留该行以完成当前代码块的语法结构。
0022: )    # 说明：调用函数或方法，执行具体工程动作。
```

## requirements-h100.txt

```text
0001: # H100/CUDA environment dependencies for the retrieval training task.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0002: # Install a CUDA-matched PyTorch wheel from the official PyTorch index before using this file when needed.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0003: torch>=2.1    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0004: hydra-core>=1.3    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0005: omegaconf>=2.3    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0006: numpy>=1.23    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0007: tqdm>=4.64    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0008: tensorboard>=2.12    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0009: pyyaml>=6.0    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
0010: # Optional native HPX regridding; training falls back to lat-lon when unavailable.    # 说明：中文/配置注释，说明相邻代码或参数用途。
0011: earth2grid; platform_system == "Linux"    # 说明：设置变量、配置项或对象属性，驱动后续训练流程。
```