import torch

def test_cuda():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Capability: {torch.cuda.get_device_capability(0)}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0)} bytes")
        print(f"CUDA Memory Cached: {torch.cuda.memory_reserved(0)} bytes")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    test_cuda()