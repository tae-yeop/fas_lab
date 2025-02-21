import torch

# Dummy Data for Training (Replace with real dataset)
x_train = torch.rand(100, 1, 30, 128, 128)
y_train = torch.randint(0, 2, (100,))
x_val = torch.rand(20, 1, 30, 128, 128)
y_val = torch.randint(0, 2, (20,))