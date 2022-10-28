import random
import torch


def get_batch(batch_size, features, labels):
    num = len(features)  # 获取样本数量
    indices = list(range(num))  # 产生下标list
    random.shuffle(indices)  # 打乱indices，并不产生新的list
    for i in range(0, num, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num)])
        yield features[batch_indices], labels[batch_indices]
