import time
import matplotlib.pyplot as plt
import pynvml
import psutil  # For CPU and system memory stats
from tqdm import tqdm  # Progress bar library

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
# For average CPU utilization (if needed)
cpu_util_list = []
# For per-core CPU utilizations (each element is a list of per-core percentages)
cpu_core_utils = []
# System RAM stats (in GB)
sys_mem_total_list = []
sys_mem_used_list = []
sys_mem_free_list = []

start_time = time.time()
duration =  300 # Total duration in seconds (5 minutes)
interval = 5    # Sampling interval in seconds
num_iterations = duration // interval  # Number of iterations

# Use tqdm to show progress bar over the iterations
for _ in tqdm(range(num_iterations), desc="Progress:", unit="iter"):
    current_time = time.time() - start_time  # Elapsed time
    
    # GPU stats using NVML (convert bytes to GB)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    
    timestamps.append(current_time)
    memory_total_list.append(mem_info.total / 1024**3)
    memory_used_list.append(mem_info.used / 1024**3)
    memory_free_list.append(mem_info.free / 1024**3)
    gpu_util_list.append(gpu_util.gpu)
    
    # CPU utilization for all cores using psutil (list of per-core percentages)
    core_utils = psutil.cpu_percent(interval=None, percpu=True)
    cpu_core_utils.append(core_utils)
    # Also compute the average CPU utilization (optional)
    cpu_util_list.append(sum(core_utils) / len(core_utils))
    
    # System RAM stats using psutil (convert bytes to GB)
    sys_mem = psutil.virtual_memory()
    sys_mem_total_list.append(sys_mem.total / 1024**3)
    sys_mem_used_list.append(sys_mem.used / 1024**3)
    sys_mem_free_list.append(sys_mem.available / 1024**3)
    
    # Print current stats to the console
    # print(f"[{current_time:.1f} sec] GPU Memory Total: {mem_info.total / 1024**3:.2f} GB")
    # print(f"[{current_time:.1f} sec] GPU Memory Used: {mem_info.used / 1024**3:.2f} GB")
    # print(f"[{current_time:.1f} sec] GPU Memory Free: {mem_info.free / 1024**3:.2f} GB")
    # print(f"[{current_time:.1f} sec] GPU Utilization: {gpu_util.gpu}%")
    # print(f"[{current_time:.1f} sec] CPU Utilization (avg): {cpu_util_list[-1]:.2f}%")
    # print(f"[{current_time:.1f} sec] CPU Core Utilizations: {core_utils}")
    # print(f"[{current_time:.1f} sec] System RAM Total: {sys_mem.total / 1024**3:.2f} GB")
    # print(f"[{current_time:.1f} sec] System RAM Used: {sys_mem.used / 1024**3:.2f} GB")
    # print(f"[{current_time:.1f} sec] System RAM Free: {sys_mem.available / 1024**3:.2f} GB")
    # print("-" * 60)
    
    time.sleep(interval)

pynvml.nvmlShutdown()

# Plotting all the recorded statistics
plt.figure(figsize=(14, 16))

# Subplot 1: GPU Memory Usage (GB)
plt.subplot(5, 1, 1)
plt.plot(timestamps, memory_total_list, label='Total GPU Memory (GB)', linestyle='--', color='blue')
plt.plot(timestamps, memory_used_list, label='Used GPU Memory (GB)', color='red')
plt.plot(timestamps, memory_free_list, label='Free GPU Memory (GB)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Memory (GB)')
plt.title('GPU Memory Usage Over Time')
plt.legend()
plt.grid(True)

# Subplot 2: GPU Utilization (%)
plt.subplot(5, 1, 2)
plt.plot(timestamps, gpu_util_list, label='GPU Utilization (%)', color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Utilization (%)')
plt.title('GPU Utilization Over Time')
plt.legend()
plt.grid(True)

# Subplot 3: Average CPU Utilization (%)
plt.subplot(5, 1, 3)
plt.plot(timestamps, cpu_util_list, label='Average CPU Utilization (%)', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Utilization (%)')
plt.title('Average CPU Utilization Over Time')
plt.legend()
plt.grid(True)

# Subplot 4: CPU Core Utilization (%)
plt.subplot(5, 1, 4)
num_cores = len(cpu_core_utils[0]) if cpu_core_utils else 0
for core_index in range(num_cores):
    # Extract the usage for this core over all timestamps
    core_usage = [core_stats[core_index] for core_stats in cpu_core_utils]
    plt.plot(timestamps, core_usage, label=f'Core {core_index}')
plt.xlabel('Time (s)')
plt.ylabel('Utilization (%)')
plt.title('CPU Core Utilizations Over Time')
plt.legend(ncol=4, fontsize='small')  # Adjust legend for many lines
plt.grid(True)

# Subplot 5: System RAM Memory Usage (GB)
plt.subplot(5, 1, 5)
plt.plot(timestamps, sys_mem_total_list, label='Total RAM (GB)', linestyle='--', color='blue')
plt.plot(timestamps, sys_mem_used_list, label='Used RAM (GB)', color='red')
plt.plot(timestamps, sys_mem_free_list, label='Free RAM (GB)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Memory (GB)')
plt.title('System RAM Memory Usage Over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
from datetime import datetime
output_file = datetime.now().strftime('gpu_stats_%Y%m%d_%H%M%S.png')
plt.savefig(output_file)



