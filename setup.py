#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Package metadata for XiChen/HealDA retrieval."""

from setuptools import find_packages, setup

setup(
    name="xichen-healda-retrieval",
    version="1.1.0",
    description="XiChen HealDA-style multi-source observation to ERA5 T/Q13 retrieval",
    author="XiChen/HealDA retrieval engineering",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "numpy>=1.23",
        "tqdm>=4.64",
        "tensorboard>=2.12",
        "pyyaml>=6.0",
        "python-dateutil>=2.8",
        "click>=8.0",
        "dask>=2023.1",
        "scipy>=1.9",
        "matplotlib>=3.6",
        "einops>=0.6",
        "timm>=0.9",
        "torchvision>=0.16",
        "xarray>=2023.1",
        "pyrootutils>=1.0",
    ],
)
