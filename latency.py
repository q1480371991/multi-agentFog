import numpy as np
import os
import matplotlib.pyplot as plt
# 对比不同算法的延迟数据并可视化
# Base path and models to evaluate
# base_path = "/home/natnael/Desktop/mult-agentFog/output/"
base_path = "./output_lxl/"
models = ["MATD3", "MAPPO", "MASAC", "MAIDDPG"]

# Containers for valid data
valid_models = []
# 初始化各模型的延迟数据字典（键为模型名，值为延迟列表）
episode_latencies = {"MATD3": [], "MAPPO": [], "MASAC": [], "MAIDDPG": []}

# Function to safely load .npy latency metrics 加载.npy格式延迟数据的函数
def load_latency(directory):
    try:
        latencies = np.load(os.path.join(directory, "evaluate/evaluation_latencies.npy"))
    except FileNotFoundError as e:
        print(f"⚠️  Missing file: {e.filename}. Skipping {directory}.")
        return None
    return latencies

# Iterate over models and load latency data 遍历模型并加载延迟数据
for model in models:
    metrics_dir = os.path.join(base_path, model)
    latencies = load_latency(metrics_dir)
    
    if latencies is None:
        continue  # Skip if file is missing
    
    valid_models.append(model)
    episode_latencies[model] = latencies# 存储延迟数据

# Plot only latency
if valid_models:
    x = np.arange(len(next(iter(episode_latencies.values()))))  # Episodes
    plt.figure(figsize=(10, 6))

    for model in valid_models:
        plt.plot(x, episode_latencies[model], label=f"{model} Latency", marker='o')

    plt.title('Episode-wise Comparison of Latency')
    plt.xlabel('Episode')
    plt.ylabel('Latency (s)')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(base_path, "origin/latency_comparison_plot.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Latency-only plot saved to: {save_path}")
else:
    print("❌ No valid latency data found.")
