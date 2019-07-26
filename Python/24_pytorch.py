import torch


# Creating tensors in PyTorch -----------------------------------------------------------------------------------------

# Create random tensor of size 3 by 3
your_first_tensor = torch.rand(3, 3)

# Calculate the shape of the tensor
tensor_size = your_first_tensor.shape

# Create a matrix of ones with shape 3 by 3
tensor_of_ones = torch.ones(3, 3)

# Create an identity matrix with shape 3 by 3
identity_tensor = torch.eye(3)

# Element-wise multiply tensor_of_ones with identity_tensor
matrices_multiplied = torch.matmul(tensor_of_ones, identity_tensor)

# Element-wise multiply tensor_of_ones with identity_tensor
element_multiplication = tensor_of_ones * identity_tensor


# Forward propagation -------------------------------------------------------------------------------------------------

x = torch.rand(1000, 1000)
y = torch.rand(1000, 1000)
z = torch.rand(1000, 1000)

q = x * y
f = torch.matmul(z, q)

mean_f = torch.mean(f)


# Backpropagation by auto-differentiation -----------------------------------------------------------------------------

