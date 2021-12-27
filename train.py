"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-05-18 01:49:14
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-07-07 22:59:26
"""
#!/usr/bin/env python
# coding=UTF-8
import argparse
import os
from typing import Iterable

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler

from core import builder


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    device: torch.device,
    teacher: nn.Module = None,
    soft_criterion: Criterion = None,
) -> None:
    model.train()
    step = epoch * len(train_loader)
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(data)

        def _get_loss(output, target):
            if teacher:
                with torch.no_grad():
                    teacher_score = teacher(data).detach()
                loss = soft_criterion(output, teacher_score, target)
            else:
                loss = criterion(output, target)
            return loss

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        loss = _get_loss(output, target)
        class_loss = loss
        if configs.criterion.ortho_loss_weight > 0:
            ortho_loss = model.get_ortho_loss()
            loss = loss + configs.criterion.ortho_loss_weight * ortho_loss
        else:
            ortho_loss = torch.zeros(1)
        loss.backward()

        optimizer.step()

        step += 1

        if batch_idx % int(configs.run.log_interval) == 0:
            log = "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4f} Class Loss: {:.4f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.data.item(),
                class_loss.data.item(),
            )
            if configs.criterion.ortho_loss_weight > 0:
                log += " Ortho Loss: {:.4f}".format(ortho_loss.item())
            lg.info(log)

            mlflow.log_metrics({"train_loss": loss.item()}, step=step)

    scheduler.step()
    accuracy = 100.0 * correct.float() / len(train_loader.dataset)
    lg.info(f"Train Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f})%")
    mlflow.log_metrics({"train_acc": accuracy.item(), "lr": get_learning_rate(optimizer)}, step=epoch)


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
) -> None:
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in validation_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)

            val_loss += criterion(output, target).data.item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100.0 * correct.float() / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    lg.info(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            val_loss, correct, len(validation_loader.dataset), accuracy
        )
    )
    mlflow.log_metrics({"val_acc": accuracy.data.item(), "val_loss": val_loss}, step=epoch)


def test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
) -> None:
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)

            val_loss += criterion(output, target).data.item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(test_loader)
    loss_vector.append(val_loss)

    accuracy = 100.0 * correct.float() / len(test_loader.dataset)
    accuracy_vector.append(accuracy)

    lg.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            val_loss, correct, len(test_loader.dataset), accuracy
        )
    )
    mlflow.log_metrics({"test_acc": accuracy.data.item(), "test_loss": val_loss}, step=epoch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic) == True:
        set_torch_deterministic()

    model = builder.make_model(
        device,
        int(configs.run.random_state) if int(configs.run.deterministic) else None,
        model_cfg=configs.model,
    )

    train_loader, validation_loader, test_loader = builder.make_dataloader()
    optimizer = builder.make_optimizer(
        [p for p in model.parameters() if p.requires_grad], configs.optimizer.name, configs.optimizer
    )
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion().to(device)
    saver = BestKModelSaver(k=int(configs.checkpoint.save_best_model_k))

    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}_{configs.dataset.img_height}x{configs.dataset.img_width}_ortho-{configs.criterion.ortho_loss_weight}_ib-{configs.model.input_bit}_wb-{configs.model.weight_bit}_qb-{configs.mlg.basis_bit}_qu-{configs.mlg.coeff_in_bit}_qv-{configs.mlg.coeff_out_bit}_proj-{configs.mlg.projection_alg if configs.mlg.projection_alg is not None else 0}_kd-{int(configs.mlg.kd)}"

    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}"
    if len(configs.checkpoint.model_comment) > 0:
        checkpoint += "_" + configs.checkpoint.model_comment
    checkpoint += ".pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    mlflow.set_experiment(configs.run.experiment)
    experiment = mlflow.get_experiment_by_name(configs.run.experiment)

    # run_id_prefix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mlflow.start_run(run_name=model_name)
    mlflow.log_params(
        {
            "exp_name": configs.run.experiment,
            "exp_id": experiment.experiment_id,
            "run_id": mlflow.active_run().info.run_id,
            "inbit": configs.model.input_bit,
            "wbit": configs.model.weight_bit,
            "init_lr": configs.optimizer.lr,
            "checkpoint": checkpoint,
            "restore_checkpoint": configs.checkpoint.restore_checkpoint,
            "pid": os.getpid(),
        }
    )

    lossv, accv = [0], [0]
    epoch = 0
    try:
        lg.info(
            f"Experiment {configs.run.experiment} ({experiment.experiment_id}) starts. Run ID: ({mlflow.active_run().info.run_id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
        )
        lg.info(configs)
        if int(configs.checkpoint.resume):
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )

            lg.info("Validate resumed model...")
            test(
                model,
                test_loader,
                0,
                criterion,
                [],
                [],
                device=device,
            )

        ## set MLG for model
        base_in = int(configs.mlg.base_in)
        base_out = int(configs.mlg.base_out)
        quant_ratio_b = int(configs.quantize.quant_ratio_basis)
        quant_ratio_u = int(configs.quantize.quant_ratio_coeff_in)
        quant_ratio_v = int(configs.quantize.quant_ratio_coeff_out)
        quant_ratio_in = int(configs.quantize.quant_ratio_in)
        quant_ratio_b=quant_ratio_b if quant_ratio_b > 1e-8 else None
        quant_ratio_u=quant_ratio_u if quant_ratio_u > 1e-8 else None
        quant_ratio_v=quant_ratio_v if quant_ratio_v > 1e-8 else None
        quant_ratio_in=quant_ratio_in if quant_ratio_in > 1e-8 else None

        if base_in > 0 or base_out > 0:
            model.enable_dynamic_weight(
                base_in=base_in,
                base_out=base_out,
                last_layer=False,
            )
        model.assign_separate_weight_bit(
            int(configs.mlg.basis_bit),
            int(configs.mlg.coeff_in_bit),
            int(configs.mlg.coeff_out_bit),
            quant_ratio_b=quant_ratio_b,
            quant_ratio_u=quant_ratio_u,
            quant_ratio_v=quant_ratio_v,
        )
        model.set_quant_ratio(
            quant_ratio_b=quant_ratio_b,
            quant_ratio_u=quant_ratio_u,
            quant_ratio_v=quant_ratio_v,
            quant_ratio_in=quant_ratio_in,
        )

        n_lowrank_params = model.get_total_num_params(fullrank=False)
        n_fullrank_params = model.get_total_num_params(fullrank=True)
        compress_ratio = n_lowrank_params / n_fullrank_params
        lowrank_mem = model.get_total_param_size(fullrank=False, fullprec=False)
        fullrank_mem = model.get_total_param_size(fullrank=True, fullprec=True)
        mem_ratio = lowrank_mem / fullrank_mem
        lg.info(
            f"Parameter count {model.get_num_params()}, compression ratio: {n_lowrank_params} / {n_fullrank_params} = {compress_ratio}\n"
        )
        lg.info(
            f"Parameter size: {model.get_param_size()}, memory compression ratio: {lowrank_mem} / {fullrank_mem} = {mem_ratio}"
        )

        ## set teacher model in knowledge distillation
        if configs.mlg.kd and configs.teacher.name and configs.teacher.checkpoint:
            lg.info(f"Build teacher model {configs.teacher.name}")
            teacher = builder.make_model(
                device,
                int(configs.run.random_state) if int(configs.run.deterministic) else None,
                model_cfg=configs.teacher,
            )
            load_model(teacher, path=configs.teacher.checkpoint)
            teacher_criterion = builder.make_criterion(name="ce").to(device)
            soft_criterion = builder.make_criterion(name=configs.soft_criterion.name).to(device)
            teacher.assign_separate_weight_bit(32, 32, 32)
            teacher.eval()
            lg.info(f"Validate teacher model {configs.teacher.name}")
            test(teacher, test_loader, -1, teacher_criterion, [], [], device)
            model.approximate_target_model(teacher, alg=configs.mlg.projection_alg)
        else:
            teacher = None
            soft_criterion = None

        for epoch in range(1, int(configs.run.n_epochs) + 1):
            train(
                model,
                train_loader,
                optimizer,
                scheduler,
                epoch,
                criterion,
                device,
                teacher=teacher,
                soft_criterion=soft_criterion,
            )
            if validation_loader is not None:
                lg.info(f"Validating model...")
                validate(
                    model,
                    validation_loader,
                    epoch,
                    criterion,
                    lossv,
                    accv,
                    device=device,
                )
                lg.info(f"Testing model...")
                test(
                    model,
                    test_loader,
                    epoch,
                    criterion,
                    [],
                    [],
                    device=device,
                )
            else:
                lg.info(f"Testing model...")
                test(
                    model,
                    test_loader,
                    epoch,
                    criterion,
                    lossv,
                    accv,
                    device=device,
                )
            saver.save_model(model, accv[-1], epoch=epoch, path=checkpoint, save_model=False, print_msg=True)
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
