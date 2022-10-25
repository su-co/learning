import torch

# 1.
# x = torch.arange(12).reshape(3, 4)
# print(x == x.T.T)

# 4.
# x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
# print(len(x))

# 6. A.sum(axis=1)变成1x5的向量，经过广播后，变成5x5的向量，和A的形状不同
# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# print(A)
# print(A.sum(axis=1))
# print(A/A.sum(axis=1))

# 7.
# x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
# print(x)
# print(x.sum(axis=2))


