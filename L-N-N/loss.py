import torch


def loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
