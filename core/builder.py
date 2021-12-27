"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-03-31 17:48:41
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-09-26 00:51:50
"""

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from pyutils.config import configs
from pyutils.datasets import get_dataset
from pyutils.loss import AdaptiveLossSoft, KLLossMixed
from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from pyutils.optimizer.sam import SAM
from pyutils.typing import DataLoader, Optimizer, Scheduler
from torch.types import Device

from core.models import *
from pyutils.optimizer.radam import RAdam

__all__ = [
    "make_dataloader",
    "make_model",
    "make_weight_optimizer",
    "make_arch_optimizer",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]


def make_dataloader(name: str = None) -> Tuple[DataLoader, DataLoader]:
    name = (name or configs.dataset.name).lower()
    train_dataset, test_dataset = get_dataset(
        name,
        configs.dataset.img_height,
        configs.dataset.img_width,
        dataset_dir=configs.dataset.root,
        transform=configs.dataset.transform,
    )
    validation_dataset = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=configs.run.batch_size,
        shuffle=int(configs.dataset.shuffle),
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
    )

    validation_loader = (
        torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=configs.run.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=configs.dataset.num_workers,
        )
        if validation_dataset is not None
        else None
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=configs.run.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
    )

    return train_loader, validation_loader, test_loader


def make_model(device: Device, random_state: int = None, model_cfg = None) -> nn.Module:
    model_name = model_cfg.name
    if "resnet" in model_name.lower():
        model = eval(model_name)(
            in_channels=configs.dataset.in_channels,
            num_classes=configs.dataset.num_classes,
            in_bit=model_cfg.input_bit,
            w_bit=model_cfg.weight_bit,
            device=device,
        ).to(device)
        model.reset_parameters()
    else:
        model = None
        raise NotImplementedError(f"Not supported model name: {model_name}")

    return model


def make_optimizer(params, name: str = None, configs=None) -> Optimizer:
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=configs.lr,
            momentum=configs.momentum,
            weight_decay=configs.weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            betas=getattr(configs, "betas", (0.9, 0.999)),
        )
    elif name == "radam":
        optimizer = RAdam(params, lr=configs.lr, weight_decay=configs.weight_decay
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif name == "sam_sgd":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.5),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            momenum=0.9,
        )
    elif name == "sam_adam":
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.001),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    else:
        raise NotImplementedError(name)

    return optimizer


def make_scheduler(optimizer: Optimizer, name: str = None) -> Scheduler:
    name = (name or configs.scheduler.name).lower()
    if name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(configs.run.n_epochs), eta_min=float(configs.scheduler.lr_min)
        )
    elif name == "cosine_warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=configs.run.n_epochs,
            max_lr=configs.optimizer.lr,
            min_lr=configs.scheduler.lr_min,
            warmup_steps=int(configs.scheduler.warmup_steps),
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=configs.scheduler.lr_gamma)
    else:
        raise NotImplementedError(name)

    return scheduler


def make_criterion(name: str = None) -> nn.Module:
    name = (name or configs.criterion.name).lower()
    if name == "nll":
        criterion = nn.NLLLoss()
    elif name == "mse":
        criterion = nn.MSELoss()
    elif name == "mae":
        criterion = nn.L1Loss()
    elif name == "ce":
        criterion = nn.CrossEntropyLoss()
    elif name == "adaptive":
        criterion = AdaptiveLossSoft(alpha_min=-1.0, alpha_max=1.0)
    elif name == "mixed_kl":
        criterion = KLLossMixed(
            T=getattr(configs.criterion, "T", 3),
            alpha=getattr(configs.criterion, "alpha", 0.9),
        )
    else:
        raise NotImplementedError(name)
    return criterion
