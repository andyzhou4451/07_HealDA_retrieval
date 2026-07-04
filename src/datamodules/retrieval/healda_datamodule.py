# -*- coding: utf-8 -*-
"""DataModule wrapper for HealDA-style T/Q retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .healda_dataset import HealDARetrievalDataset, TARGET_VARS, PRESSURE_LEVELS, XICHEN_ERA5_ALL_VARS, collate_retrieval_batch


class HealDARetrievalDataModule:
    """Pure PyTorch datamodule used by ``src.pipeline.retrieval.trainer``."""

    def __init__(
        self,
        obs_dir: str,
        era5_dir: str,
        scale_dir: str,
        sensors: dict | list,
        target: dict | None = None,
        data: dict | None = None,
        qc: dict | None = None,
        obs_default_normalization: dict | None = None,
        start_train_year: int = 2016,
        start_val_year: int = 2022,
        start_test_year: int = 2023,
        end_year: int = 2024,
        batch_size: int = 1,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        seed: int = 1024,
        debug: bool = False,
        max_debug_samples: int = 8,
        distributed: bool = False,
        num_replicas: int = 1,
        rank: int = 0,
        **kwargs,
    ) -> None:
        self.obs_dir = obs_dir
        self.era5_dir = era5_dir
        self.scale_dir = scale_dir
        self.sensors_cfg = sensors
        self.target_cfg = target or {}
        self.data_cfg = data or {}
        self.qc = qc or {}
        self.obs_default_normalization = obs_default_normalization or {}
        self.start_train_year = int(start_train_year)
        self.start_val_year = int(start_val_year)
        self.start_test_year = int(start_test_year)
        self.end_year = int(end_year)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.shuffle = bool(shuffle)
        self.pin_memory = bool(pin_memory)
        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = bool(persistent_workers)
        self.seed = int(seed)
        self.debug = bool(debug)
        self.max_debug_samples = int(max_debug_samples)
        self.distributed = bool(distributed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.extra_kwargs = dict(kwargs)

        self.train_data = None
        self.val_data = None
        self.test_data = None

    @staticmethod
    def _as_list(value) -> list[str]:
        """Convert a Hydra/OmegaConf scalar or list-like value into a plain list.

        OmegaConf ``DictConfig``/``ListConfig`` objects are not plain ``dict``/``list``
        instances.  Using ``list(DictConfig)`` on the configured ``sensors`` block
        returns the group keys (``satellite`` and ``conventional``), which are not
        real sensors.  This helper keeps the configured leaf sensor names only.
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, Mapping):
            out: list[str] = []
            for nested in value.values():
                out.extend(HealDARetrievalDataModule._as_list(nested))
            return out
        if isinstance(value, Sequence):
            return [str(v) for v in value]
        return [str(value)]

    @property
    def sensors(self) -> list[str]:
        """Return flat sensor aliases, never Hydra group names.

        Accepts either

        ``sensors: [atms, amsua, mhs, hrs4, gdas_prebufr]``

        or the recommended grouped form

        ``sensors: {satellite: [...], conventional: [...]}``.
        """
        if isinstance(self.sensors_cfg, Mapping):
            flat = []
            for group in ("satellite", "conventional"):
                flat.extend(self._as_list(self.sensors_cfg.get(group, [])))
            if not flat:
                # Fallback for custom mappings whose values are already sensor lists.
                flat = self._as_list(self.sensors_cfg)
        else:
            flat = self._as_list(self.sensors_cfg)

        # Preserve order while avoiding accidental duplicates from aliases.
        seen: set[str] = set()
        deduped: list[str] = []
        for sensor in flat:
            sensor = str(sensor).strip()
            if sensor and sensor not in seen:
                seen.add(sensor)
                deduped.append(sensor)
        return deduped

    def _worker_init_fn(self, worker_id: int) -> None:
        worker_seed = int(self.seed) + int(self.rank) * 100000 + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def _get_sampler(self, dataset, shuffle: bool):
        if self.distributed and self.num_replicas > 1:
            return DistributedSampler(dataset, num_replicas=self.num_replicas, rank=self.rank, shuffle=shuffle, seed=self.seed)
        return None

    def _make_dataset(self, mode: str, start_year: int, end_year: int) -> HealDARetrievalDataset:
        target_variables = self.target_cfg.get("target_vars", self.target_cfg.get("variables", TARGET_VARS))
        # Convert target.variables=[t,q] into explicit t-50... q-1000 names.
        if target_variables and set(target_variables).issubset({"t", "q"}):
            levels = self.target_cfg.get("pressure_levels", PRESSURE_LEVELS)
            target_variables = [*(f"t-{p}" for p in levels), *(f"q-{p}" for p in levels)]
        return HealDARetrievalDataset(
            obs_dir=self.obs_dir,
            era5_dir=self.era5_dir,
            scale_dir=self.scale_dir,
            mode=mode,
            sensors=self.sensors,
            target_variables=target_variables,
            pressure_levels=self.target_cfg.get("pressure_levels", PRESSURE_LEVELS),
            era5_all_vars=self.data_cfg.get("era5_all_vars", XICHEN_ERA5_ALL_VARS),
            grid_shape=self.data_cfg.get("grid_shape", [181, 360]),
            obs_window=self.data_cfg.get("obs_window", {"start_hours": -21, "end_hours": 3}),
            no_lookahead=self.data_cfg.get("no_lookahead", False),
            no_lookahead_window=self.data_cfg.get("no_lookahead_window", {"start_hours": -24, "end_hours": 0}),
            dt_data=self.data_cfg.get("dt_data", 6),
            dt_obs=self.data_cfg.get("dt_obs", 3),
            start_year=start_year,
            end_year=end_year,
            debug=self.debug,
            max_debug_samples=self.max_debug_samples,
            max_points_per_sensor=self.data_cfg.get("max_points_per_sensor", 250_000),
            strict_time_index=self.data_cfg.get("strict_time_index", False),
            target_cache_size=self.data_cfg.get("target_cache_size", 16),
            normalize_target=self.data_cfg.get("normalize_target", True),
            normalize_obs=self.data_cfg.get("normalize_obs", True),
            require_obs_stats=self.data_cfg.get("require_obs_stats", False),
            require_sensor_dirs=self.data_cfg.get("require_sensor_dirs", True),
            filter_empty_observation_windows=self.data_cfg.get("filter_empty_observation_windows", True),
            min_required_sensors=self.data_cfg.get("min_required_sensors", 1),
            require_complete_obs_window=self.data_cfg.get("require_complete_obs_window", False),
            qc=self.qc,
            obs_default_normalization=self.obs_default_normalization,
        )

    def setup(self) -> None:
        if self.train_data is None:
            self.train_data = self._make_dataset("train", self.start_train_year, self.start_val_year)
        if self.val_data is None:
            self.val_data = self._make_dataset("val", self.start_val_year, self.start_test_year)
        if self.test_data is None:
            self.test_data = self._make_dataset("test", self.start_test_year, self.end_year)

    def _loader(self, dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        sampler = self._get_sampler(dataset, shuffle=shuffle)
        kwargs = dict(
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None and shuffle),
            num_workers=self.num_workers,
            collate_fn=collate_retrieval_batch,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            worker_init_fn=self._worker_init_fn,
            persistent_workers=(self.num_workers > 0 and self.persistent_workers),
        )
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        self.setup()
        return self._loader(self.train_data, shuffle=self.shuffle, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        self.setup()
        return self._loader(self.val_data, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        self.setup()
        return self._loader(self.test_data, shuffle=False, drop_last=False)
