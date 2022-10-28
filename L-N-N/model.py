import torch


def lnn(X, w, b):
    return torch.matmul(X, w) + b
