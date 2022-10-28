import torch

'''
根据y=wx+b+varepsilon生成特征X和标签y
num代表数量
'''


def generate_data(w, b, num):
    X = torch.normal(0, 1, (num, 2))
    y = torch.matmul(X, w) + b
    y = y + torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)
