import json
import os
import shutil
import random

import torch
import numpy as np


class Params:
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


class AverageMeter:
    """Computes and stores the average and current value, usualy"""

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.random.seed(seed)


def save_dict_to_json(dic, json_path):
    dic = {k: float(v) for k, v in dic.items()}
    with open(json_path, "w") as f:
        json.dump(dic, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    filepath = os.path.join(checkpoint, "last.pth.tar")
    if not os.path.exists(checkpoint):
        print(f"Checkpoint Directory does not exist! Making directory {checkpoint}")
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exits")

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "best.pth.tar"))


def load_checkpoint(checkpoint, model: torch.nn.Module, optimizer: torch.optim = None):
    if not os.path.exists(checkpoint):
        raise (f"File doesn't exist{checkpoint}")

    if not torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])

    return checkpoint
