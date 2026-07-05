# TH-HPC4 One-Node Four-GPU Submission

This repository includes `scripts/th_hpc4_gpu4.sh` for the TH-HPC4 GPU system.
The script follows the site manual flow: submit with `yhbatch`, load the PyTorch
module, and run the training command through `yhrun`.

Default resources:

- Partition: `gpu1`
- Nodes: `1`
- GPUs per node: `4`
- CPUs per GPU: `8`
- PyTorch module: `pytorch/1.11.0-cu11.3-py3.9`

Submit:

```bash
yhbatch scripts/th_hpc4_gpu4.sh
```

The default training command is:

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 main.py --config-name=train_h100_80gb_single_gpu.yaml hardware.single_gpu=false training.single_gpu=false training.device=cuda task_name=healda_xichen_tq13_hpc4_4gpu
```

Override the command without editing the script:

```bash
HPC4_TRAIN_CMD='python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 main.py --config-name=train_h100_80gb_single_gpu.yaml hardware.single_gpu=false training.single_gpu=false training.device=cuda epochs=20 task_name=healda_xichen_tq13_hpc4_4gpu_debug' yhbatch scripts/th_hpc4_gpu4.sh
```

Use `yhbatch -p gpu5 scripts/th_hpc4_gpu4.sh` if the operator assigns a
different 8-GPU partition. Keep `--gpus-per-node=4` for a one-node four-card job.
