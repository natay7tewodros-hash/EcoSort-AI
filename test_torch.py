import torch

print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())

x = torch.rand(2, 3)
print("A small random tensor:")
print(x)