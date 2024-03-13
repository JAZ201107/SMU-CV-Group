import torch
from sklearn.metrics import roc_auc_score, accuracy_score

import config


def get_roc_auc_score(labels, outputs):
    predict = (outputs >= config.THRESHOLD).astype(int)

    return roc_auc_score(labels, predict)


def accuracy(labels, outputs):
    predict = (outputs >= config.THRESHOLD).astype(int)
    return accuracy_score(labels, predict)


metrics = {
    "accuracy": accuracy,
    "roc_auc": get_roc_auc_score,
}
