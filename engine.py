import os
from tqdm import tqdm
import logging


import torch
import torch.nn as nn
import numpy as np

import config
from utils.misc import save_checkpoint, load_checkpoint, save_dict_to_json, AverageMeter


def evaluate(model: nn.Module, loss_fn, dataloader, metrics, params):
    model.eval()
    summ = []

    loss_avg = AverageMeter()

    with tqdm(total=len(dataloader)) as t:
        for data_batch, labels_batch in dataloader:
            if params.cuda:
                data_batch, labels_batch = data_batch.cuda(
                    non_blocking=True,
                ), labels_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.view(-1, 1).float()
            output_batch = model(data_batch)
            loss = loss_fn(
                output_batch,
            )

            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            summary_batch = {
                metric: metrics[metrics](labels_batch, output_batch)
                for metric in metrics
            }

            loss_avg.update(loss.item())

            summary_batch["loss"] = loss.item()
            summ.append(summary_batch)

            del data_batch
            del labels_batch
            del output_batch

            t.set_postfix(
                loss="{:05.3f}".format(loss_avg()),
                roc_auc="{:05.3f}".format(summary_batch["roc_auc"]),
            )
            t.update()

    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " : ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items()
    )
    logging.info("- Eval metrics: " + metrics_string)


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim,
    loss_fn,
    dataloader,
    metrics,
    params,
    scheduler=None,
):

    model.train()
    summ = []
    loss_avg = AverageMeter()

    with tqdm(total=len(dataloader)) as t:
        for i, (data_batch, labels_batch) in enumerate(dataloader):
            if params.cuda:
                data_batch, labels_batch = data_batch.cuda(
                    non_blocking=True
                ), labels_batch.cuda(non_blocking=True)

            labels_batch = labels_batch.view(-1, 1).float()
            output_batch = model(data_batch)

            loss = loss_fn(output_batch, labels_batch)

            optimizer.zero_grad()
            loss.backward()

            if scheduler:
                scheduler.step()
            optimizer.step()

            if i % config.SAVE_SUMMARY_STEPS == 0:
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

            summary_batch = {
                metric: metrics[metric](labels_batch, output_batch)
                for metric in metrics
            }

            summary_batch["loss"] = loss.item()
            summ.append(summary_batch)

            loss_avg.update(loss.item())
            del data_batch
            del labels_batch
            del output_batch

            t.set_postfix(
                loss="{:05.3f}".format(loss_avg()),
                roc_auc="{:05.3f}".format(summary_batch["roc_auc"]),
            )
            t.update()

    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items()
    )
    logging.info("- Train metrics: " + metrics_string)

    return metrics_mean


def train_and_evaluate(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    metrics,
    params,
    scheduler=None,
    model_dir=None,
    restore_file: bool = None,
):
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file, config.MODEL_END)
        logging.info(f"Restoring parameters from {restore_path}")
        load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        logging.info(f"Epoch {epoch + 1} / {params.num_epochs}")

        train_one_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            dataloader=train_dataloader,
            metrics=metrics,
            params=params,
            scheduler=scheduler,
        )

        val_metrics = evaluate(
            model=model,
            loss_fn=loss_fn,
            dataloader=val_dataloader,
            metrics=metrics,
            params=params,
        )

        val_acc = val_metrics["accuracy"]
        is_best = val_acc >= best_val_acc

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optim_dict": optimizer.state_dict(),
            },
            is_best=is_best,
            checkpoint=model_dir,
        )

        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            save_dict_to_json(val_metrics, best_json_path)
