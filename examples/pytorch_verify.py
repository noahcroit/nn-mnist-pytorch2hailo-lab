import torch

# generate randomized tensor
rand = torch.rand(3, 4)
print(rand)

# basic tensor multiplication
data1 = [[1, 2], [3, 4]]
data2 = [[1], [2]]
data3 = [1, 2]
t1 = torch.tensor(data1)
t2 = torch.tensor(data2)
t3 = torch.tensor(data3)
r1 = torch.matmul(t1, t2)
r2 = torch.matmul(t1, t3)
print("matrix A =", t1)
print("vector x =", t2)
print("transpose of x =", t3)
print("A * x =")
print(r1)
print("xT * A =")
print(r2)



