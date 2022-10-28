"""试图为电压和电流的关系建立一个模型。你能使用自动微分来学习模型的参数吗?"""
import torch

from generate_data import generate_data
from model import lnn
from get_batch import get_batch
from loss import loss
from sgd import sgd

if __name__ == '__main__':
    # 生成1k个样本数据
    true_w = torch.tensor([20.])
    true_b = 0.
    features, labels = generate_data(true_w, true_b, 1000)

    # 初始化模型参数
    # w = torch.tensor([0., 0.], requires_grad=True)  # 如果w是0
    w = torch.normal(0, 0.01, size=(1, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 训练
    lr = 0.03
    epochs = 10
    batch_size = 10
    net = lnn
    for epoch in range(epochs):
        for X, y in get_batch(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()  # 求l关于w，b的梯度方向
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    # 显示求得的w，b
    print(w)
    print(b)
