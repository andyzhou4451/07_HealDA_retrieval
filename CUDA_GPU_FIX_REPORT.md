# CUDA/GPU cleanup audit report

## Scope

Based on the previous H100-optimized project, I re-checked the project for legacy NPU defaults, CUDA/H100 launch robustness, inference defaults, and obvious runtime bugs.

## Main fixes

1. Changed all training YAML defaults from `device: "npu"` to `device: "cuda"`.
2. Added `gpu` as a supported alias for `cuda` in `src/utils/device.py`.
3. Changed automatic accelerator selection to prefer CUDA/H100 first, with NPU only as a legacy compatibility fallback.
4. Removed unused `torch_npu` imports from inference paths that should run on CUDA-only environments.
5. Converted shell launch scripts from Ascend/HCCL environment variables to CUDA/NCCL variables:
   - `PYTORCH_NPU_ALLOC_CONF` -> `PYTORCH_CUDA_ALLOC_CONF`
   - `ASCEND_RT_VISIBLE_DEVICES` -> `CUDA_VISIBLE_DEVICES`
   - HCCL variables -> safe NCCL defaults
6. Normalized fixed GPU id lists in scripts to `0,1` for 2-GPU H100 jobs and `0` for single-GPU inference jobs.
7. Kept the retrieval Slurm template using `conda activate xichen_v1`, `torchrun --nproc_per_node=2`, and `training.device=cuda`.
8. Fixed a retrieval gradient-accumulation edge case: final partial accumulation now scales loss by the actual remaining micro-batch count instead of always dividing by `gradient_accumulation_steps`.
9. Fixed UTC timestamp generation in the retrieval dataset to avoid timezone-dependent epoch values.
10. Fixed an inference obsoperator bug where a missing ERA5 file could leave `era5` undefined.
11. Added missing dependencies used by project code to `setup.py` and `requirements-h100.txt`, including `click`, `dask`, `scipy`, `python-dateutil`, `einops`, `timm`, `torchvision`, `matplotlib`, `xarray`, and `pyrootutils`.
12. Removed generated `__pycache__` files from the deliverable project.

## Validation performed in this container

```bash
python -m compileall -q main.py src tools inference plots
python tools/smoke_retrieval_model.py --device cpu --fast_cpu --grid 12 24 --points 8
python - <<'PY'
from src.utils.device import normalize_device_type
assert normalize_device_type('gpu') == 'cuda'
assert normalize_device_type('cuda') == 'cuda'
assert normalize_device_type('cpu') == 'cpu'
print('alias_check=ok')
PY
```

Observed smoke output:

```text
forward_shape=(1, 26, 12, 24)
smoke_status=ok
alias_check=ok
```

## Runtime notes

- Real CUDA/H100 training still needs to be validated on the cluster because this container has no H100, no `/public02` mount, and no project Hydra environment.
- For retrieval, keep using `training.device=cuda` or `training.device=gpu`; both now resolve to CUDA.
- `NCCL_IB_DISABLE=1` remains a safe default. If your cluster has working IB/RoCE and the admin recommends it, set `NCCL_IB_DISABLE=0` and configure `NCCL_SOCKET_IFNAME` to the correct interface.
