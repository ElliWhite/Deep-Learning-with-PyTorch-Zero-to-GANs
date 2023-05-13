import torch

# Number
t1 = torch.tensor(4.)
print(t1.dtype)

# Vector
t2 = torch.tensor([1., 2, 3, 4])


# Matrix
t3 = torch.tensor([[5., 6], 
                   [7, 8], 
                   [9, 10]])

# 3-dimensional array
t4 = torch.tensor([
    [[11, 12, 13], 
     [13, 14, 15]], 
    [[15, 16, 17], 
     [17, 18, 19.]]])

print(t1)
print(t1.shape)

print(t2)
print(t2.shape)

print(t3)
print(t3.shape)

print(t4)
print(t4.shape)

# Create tensors.
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)

# Arithmetic operations 
y = w * x + b
print(y)

# Compute derivatives
y.backward()

# Display gradients
print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad)

import numpy as np

x = np.array([[1, 2], [3, 4]])

y = torch.from_numpy(x)

print(x.dtype)
print(y.dtype)

# Convert a torch tensor to a numpy array
z = y.numpy()

print(z.dtype)