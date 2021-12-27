'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-12-27 16:09:50
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-12-27 16:14:10
'''

import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from torchpack.utils.config import configs

dataset = "cifar10"
model = "resnet18"
exp = "pretrain"
root = f"log/{dataset}/{model}/{exp}"
script = "train.py"
config_file = f"configs/{dataset}/{model}/train/{exp}.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]
    base_in, base_out, qb, qu, qv, id = args
    with open(
        os.path.join(root, f"bi-{base_in}_bo-{base_out}_qb-{qb}_qu-{qu}_qv-{qv}_run-{id}.log"), "w"
    ) as wfid:
        exp = [
            f"--mlg.base_in={base_in}",
            f"--mlg.base_out={base_out}",
            f"--mlg.basis_bit={qb}",
            f"--mlg.coeff_in_bit={qu}",
            f"--mlg.coeff_out_bit={qv}",
            f"--run.random_state={41+id}",
        ]
        logger.info(f"running command {' '.join(pres + exp)}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [(0, 0, 32, 32, 32, 1)]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
