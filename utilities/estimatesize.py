import torch.nn as nn

def count_parameters_and_estimate_size(model: nn.Module):
    """
    Counts the number of parameters in a PyTorch model and estimates the file size of the saved .pt file.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        tuple: A tuple containing the total number of parameters and the estimated file size in megabytes (MB).
    """
    total_params = sum(p.numel() for p in model.parameters())
    # Assuming each parameter is stored as a 32-bit float (4 bytes)
    estimated_size_in_bytes = total_params * 4
    estimated_size_in_mb = estimated_size_in_bytes / (1024 ** 2)
    return total_params, estimated_size_in_mb

def size_report(model):
    total_params, estimated_size = count_parameters_and_estimate_size(model)
    print(f"Total Parameters: {total_params}")
    print(f"Estimated .pt File Size: {estimated_size:.2f} MB")

