import torch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Get the CUDA version used by PyTorch
print("PyTorch CUDA version:", torch.version.cuda)

# Get the GPU device name (if available)
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))