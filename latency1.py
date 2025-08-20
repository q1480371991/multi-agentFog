import numpy as np
import os
import matplotlib.pyplot as plt
# 对比不同算法的延迟数据并可视化（支持平滑处理）。
# Base path and models to evaluate
# base_path = "/home/natnael/Desktop/mult-agentFog/output/"
base_path = "./output_lxl/"
models = ["MATD3", "MAPPO", "MASAC", "MAIDDPG"]

# Containers for valid data
valid_models = []
episode_latencies = {"MATD3": [], "MAPPO": [], "MASAC": [], "MAIDDPG": []}

# Function to safely load .npy latency metrics 加载.npy格式延迟数据的函数
def load_latency(directory):
    try:
        latencies = np.load(os.path.join(directory, "evaluate/evaluation_latencies.npy"))
    except FileNotFoundError as e:
        print(f"⚠️  Missing file: {e.filename}. Skipping {directory}.")
        return None
    return latencies

# Simple moving average smoothing  简单移动平均平滑函数（减少数据噪声）
def smooth(data, window_size=5):
    # 使用卷积计算移动平均，mode='valid'确保输出长度为原始长度减去窗口大小加1
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Iterate over models and load latency data
for model in models:
    metrics_dir = os.path.join(base_path, model)
    latencies = load_latency(metrics_dir)
    
    if latencies is None:
        continue  # Skip if file is missing
    
    valid_models.append(model)
    episode_latencies[model] = latencies# 存储延迟数据

# Plot only latency
if valid_models:
    plt.figure(figsize=(7, 4))

    for model in valid_models:
        raw_data = episode_latencies[model]# 获取原始延迟数据
        smoothed = smooth(raw_data, window_size=5) # 对原始数据进行平滑处理
        x = np.arange(len(smoothed))# 生成平滑后数据对应的x轴（回合数）
        # 绘制平滑后的延迟曲线，线条更粗（linewidth=2）
        plt.plot(x, smoothed, label=f"{model}", linewidth=2)

    plt.title('Episode-wise Comparison of Latency')
    plt.xlabel('Episode') # x轴标签：回合数
    plt.ylabel('Latency (s)')# y轴标签：延迟（单位：秒）
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(base_path, "origin/latency_comparison_plot_smoothed.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Smoothed latency-only plot saved to: {save_path}")
else:
    print("❌ No valid latency data found.")
