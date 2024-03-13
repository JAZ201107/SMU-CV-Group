import argparse
import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from engine import train_and_evaluate
from dataloader.dataloader import fetch_dataloader
from model.vit import vit
from model.metrics import metrics
from utils.misc import Params, seed_everything
from utils.log import set_logger


def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="data/64x64_SIGNS",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--model_dir",
        default="experiments/base_model",
        help="Directory containing params.json",
    )
    parser.add_argument(
        "--restore_file",
        default=None,
        help="Optional, name of the file in --model_dir containing weights to reload before training",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    json_path = os.path.join(args.model_dir, "params.json")

    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"

    # Load params
    params = Params(json_path)
    params.cuda = torch.cuda.is_available()  # if cuda is avaible

    seed_everything(42)
    set_logger(os.path.join(args.model_dir, "train.log"))

    logging.info("Loading the datasets...")

    dataloader = fetch_dataloader(["train", "val"], args.data_dir, params)
    train_dl = dataloader["train"]
    val_dl = dataloader["val"]
    logging.info("- done. ")

    # Initialize model
    model = vit(params).cuda() if params.cuda else vit(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    loss_fn = nn.BCELoss()
    metrics = metrics

    logging.info(f"Starting training for {params.num_epochs} epochs(s)")
    train_and_evaluate(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics=metrics,
        params=params,
        model_dir=args.model_dir,
        restore_file=args.restore_file,
    )
