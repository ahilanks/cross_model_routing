import torch
import time

print("===== CUDA Diagnostic Test =====")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
    print("Memory Allocated:", round(torch.cuda.memory_allocated(0)/1024**2, 1), "MB")
    print("Memory Cached:", round(torch.cuda.memory_reserved(0)/1024**2, 1), "MB")

    print("Attempting to initialize CUDA context...")
    # Create a small tensor on the device
    torch.ones(1, device=device) 
    print("CUDA context successfully initialized.")

    # Create some random tensors on GPU
    print("\nRunning a simple GPU computation...")
    a = torch.randn((10000, 10000), device=device)
    b = torch.randn((10000, 10000), device=device)
    torch.cuda.synchronize()
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()
    print("Matrix multiplication successful!")
    print(f"Time taken: {end - start:.4f} seconds")

    # Test gradient computation
    print("\nTesting autograd on GPU...")
    x = torch.randn((1024, 1024), device=device, requires_grad=True)
    y = torch.sum(x ** 2)
    y.backward()
    print("Autograd test passed!")

    # Clear cache
    torch.cuda.empty_cache()
    print("\nAll CUDA tests passed ✅")
else:
    print("❌ CUDA not available! Check your driver or runtime setup.")
