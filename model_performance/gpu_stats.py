import time
import matplotlib.pyplot as plt
import pynvml
import psutil  # For CPU and system memory stats

# Initialize NVML and get a handle for GPU 0
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Update index if using multiple GPUs

# Lists to store recorded data
timestamps = []
# GPU stats (in GB and %)
memory_total_list = []
memory_used_list = []
memory_free_list = []
gpu_util_list = []
# CPU and system memory stats
cpu_util_list = []
sys_mem_total_list = []
sys_mem_used_list = []
sys_mem_free_list = []

start_time = time.time()
duration = 300   # Total duration in seconds (5 minutes)
interval = 10    # Sampling interval in seconds

while time.time() - start_time < duration:
    current_time = time.time() - start_time  # Elapsed time
    
    # GPU stats using NVML (convert bytes to GB)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    
    timestamps.append(current_time)
    memory_total_list.append(mem_info.total / 1024**3)
    memory_used_list.append(mem_info.used / 1024**3)
    memory_free_list.append(mem_info.free / 1024**3)
    gpu_util_list.append(gpu_util.gpu)
    
    # CPU and system RAM stats using psutil
    cpu_util = psutil.cpu_percent(interval=None)  # CPU utilization in %
    sys_mem = psutil.virtual_memory()
    cpu_util_list.append(cpu_util)
    sys_mem_total_list.append(sys_mem.total / 1024**3)
    sys_mem_used_list.append(sys_mem.used / 1024**3)
    sys_mem_free_list.append(sys_mem.available / 1024**3)
    
    # Print current stats to the console
    print(f"[{current_time:.1f} sec] GPU Memory Total: {mem_info.total / 1024**3:.2f} GB")
    print(f"[{current_time:.1f} sec] GPU Memory Used: {mem_info.used / 1024**3:.2f} GB")
    print(f"[{current_time:.1f} sec] GPU Memory Free: {mem_info.free / 1024**3:.2f} GB")
    print(f"[{current_time:.1f} sec] GPU Utilization: {gpu_util.gpu}%")
    print(f"[{current_time:.1f} sec] CPU Utilization: {cpu_util}%")
    print(f"[{current_time:.1f} sec] System RAM Total: {sys_mem.total / 1024**3:.2f} GB")
    print(f"[{current_time:.1f} sec] System RAM Used: {sys_mem.used / 1024**3:.2f} GB")
    print(f"[{current_time:.1f} sec] System RAM Free: {sys_mem.available / 1024**3:.2f} GB")
    print("-" * 50)
    
    time.sleep(interval)

pynvml.nvmlShutdown()

# Plotting all the recorded statistics
plt.figure(figsize=(12, 12))

# Subplot 1: GPU Memory Usage (GB)
plt.subplot(4, 1, 1)
plt.plot(timestamps, memory_total_list, label='Total GPU Memory (GB)', linestyle='--', color='blue')
plt.plot(timestamps, memory_used_list, label='Used GPU Memory (GB)', color='red')
plt.plot(timestamps, memory_free_list, label='Free GPU Memory (GB)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Memory (GB)')
plt.title('GPU Memory Usage Over Time')
plt.legend()
plt.grid(True)

# Subplot 2: GPU Utilization (%)
plt.subplot(4, 1, 2)
plt.plot(timestamps, gpu_util_list, label='GPU Utilization (%)', color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Utilization (%)')
plt.title('GPU Utilization Over Time')
plt.legend()
plt.grid(True)

# Subplot 3: CPU Utilization (%)
plt.subplot(4, 1, 3)
plt.plot(timestamps, cpu_util_list, label='CPU Utilization (%)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Utilization (%)')
plt.title('CPU Utilization Over Time')
plt.legend()
plt.grid(True)

# Subplot 4: System RAM Memory Usage (GB)
plt.subplot(4, 1, 4)
plt.plot(timestamps, sys_mem_total_list, label='Total RAM (GB)', linestyle='--', color='blue')
plt.plot(timestamps, sys_mem_used_list, label='Used RAM (GB)', color='red')
plt.plot(timestamps, sys_mem_free_list, label='Free RAM (GB)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Memory (GB)')
plt.title('System RAM Memory Usage Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('gpu_stats.png')
plt.show()
