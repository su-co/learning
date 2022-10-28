import torch


def sgd(params, lr, batch_size):
    with torch.no_grad():  # 被with torch.no_grad()包住的代码，不用跟踪反向梯度计算
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
