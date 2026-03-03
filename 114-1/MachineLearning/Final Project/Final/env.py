import torch
print('Torch CUDA:', torch.version.cuda)
print(f"Cuda OK: {torch.cuda.is_available()}")
