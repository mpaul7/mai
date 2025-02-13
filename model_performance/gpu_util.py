import GPUtil

gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU id: {gpu.id}")
    print(f"  Load: {gpu.load*100:.2f}%")
    print(f"  Free Memory: {gpu.memoryFree} MB")
    print(f"  Used Memory: {gpu.memoryUsed} MB")
    print(f"  Total Memory: {gpu.memoryTotal} MB")
    print("-" * 30)
