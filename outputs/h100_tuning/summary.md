# H100 single GPU tuning summary

dry_run: True
num_trials: 2
recommended: `{'batch_size': 2, 'effective_batch_size': 2, 'patch_size': [8, 8], 'num_workers': 4, 'prefetch_factor': 2, 'pin_memory': True, 'persistent_workers': True, 'precision': 'bf16', 'compile': False, 'channels_last': True, 'oom': False, 'nan_or_inf': False, 'returncode': 0, 'samples_per_second': 2.9079700005769253, 'max_memory_reserved_gb': 40.0625, 'max_memory_allocated_gb': 34.053125, 'memory_utilization': 0.50078125, 'step_time_mean': 0.6877650043168296, 'step_time_p50': 0.6877650043168296, 'step_time_p90': 0.7565415047485127}`

## Top stable trials
- bs=2, patch=[8, 8], precision=bf16, workers=4, sps=2.9080, reserved=40.06GB
- bs=1, patch=[8, 8], precision=bf16, workers=4, sps=2.4400, reserved=33.06GB
