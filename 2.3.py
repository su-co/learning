import torch

x = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# print(x)
# print(len(x))  # len()函数只会描述第一个轴的长度
# print(x.shape)  # shape将会描述每一个轴的长度
# print(x.T)    # 矩阵转置

# 降维求和
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# print(A, A.shape)
# A_sum_axis0 = A.sum(axis=0)  # 沿着轴0（第一个轴）
# print(A_sum_axis0, A_sum_axis0.shape)
# print(A.mean(), A.sum() / A.numel())
# print(A.mean(axis=0))  # 指定轴

# 非降维求和
# print(A.sum(), A.sum(axis=1, keepdims=True))  # 保持维度不变求和

# 计算向量-向量的点积
# torch.dot(x, y)

# 计算矩阵-向量
# torch.mv(A, x)

# 计算矩阵-矩阵
# torch.mm(A, B)

# 范数用于形容向量的大小
# 二范数通常省略下标2，分量平方和开根
u = torch.tensor([3.0, -4.0])
# print(torch.norm(u))
# 一范数是元素的绝对值之和
# print(torch.abs(u).sum())

# 矩阵范数
a = torch.ones(4, 9)
# print(a)
# print(torch.norm(a))
