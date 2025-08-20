import numpy as np
import os
import matplotlib.pyplot as plt
# 对比不同算法的奖励、延迟、能量消耗数据并可视化。
# Base path and models to evaluate
# base_path = "/home/natnael/Desktop/mult-agentFog/output/"
base_path = "./output_lxl/"
models = ["MATD3", "MAPPO", "MASAC", "MAIDDPG"]  # Ensure these exist and have .npy files

# Containers for valid data
valid_models = []
episode_rewards = {"MATD3": [], "MAPPO": [], "MASAC": [],  "MAIDDPG": []}
episode_latencies = {"MATD3": [], "MAPPO": [], "MASAC": [],  "MAIDDPG": []}
episode_energies = {"MATD3": [], "MAPPO": [], "MASAC": [],  "MAIDDPG": []}


# Function to safely load .npy metrics for each model 加载.npy格式评估指标的函数
def load_metrics(directory):
    try:
        rewards = np.load(os.path.join(directory, "evaluate/evaluation_rewards.npy"))
        latencies = np.load(os.path.join(directory, "evaluate/evaluation_latencies.npy"))
        energies = np.load(os.path.join(directory, "evaluate/evaluation_energy.npy"))
    except FileNotFoundError as e:
        print(f"⚠️  Missing file: {e.filename}. Skipping {directory}.")
        return None, None, None
    return rewards,latencies,energies

# Iterate over models and load data
for model in models:
    metrics_dir = os.path.join(base_path, model)
    rewards,latencies, energies= load_metrics(metrics_dir)# 加载三种数据
    
    if rewards is None:
        continue  # Skip if any file is missing
    if latencies is None:
        continue  # Skip if any file is missing
    if energies is None:
        continue  # Skip if any file is missing
    
    valid_models.append(model)
    episode_rewards[model] = rewards# 存储奖励数据
    episode_latencies[model] = latencies# 存储延迟数据
    episode_energies[model] = energies# 存储能耗数据

# Plot episode-based comparison
if valid_models:
    # 奖励
    x = np.arange(len(next(iter(episode_rewards.values()))))  # X-axis: episode indices
    plt.figure(figsize=(8, 5))

    for model in valid_models:
        # 绘制每个模型的奖励曲线，带标记点（marker='o'）以区分
        plt.plot(x, episode_rewards[model], label=f"{model} Reward", marker='o')

    plt.title('Episode-wise Comparison of Rewards')
    plt.xlabel('Episode') # x轴标签：回合数
    plt.ylabel('Reward')# y轴标签：奖励值
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(base_path, "origin/comparison/reward_comparison_plot.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Reward-only plot saved to: {save_path}")

    # 延迟
    x = np.arange(len(next(iter(episode_latencies.values()))))  # X-axis: episode indices
    plt.figure(figsize=(8, 5))

    for model in valid_models:
        plt.plot(x, episode_latencies[model], label=f"{model} Latency", marker='o')

    plt.title('Episode-wise Comparison of Latency')
    plt.xlabel('Episode')
    plt.ylabel('Latency')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(base_path, "origin/comparison/latencies_comparison_plot.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"✅ latencies-only plot saved to: {save_path}")

    # 能耗
    x = np.arange(len(next(iter(episode_energies.values()))))  # X-axis: episode indices
    plt.figure(figsize=(8, 5))

    for model in valid_models:
        plt.plot(x, episode_energies[model], label=f"{model} Energy", marker='o')

    plt.title('Episode-wise Comparison of Energy')
    plt.xlabel('Episode')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(base_path, "origin/comparison/energies_comparison_plot.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Energy-only plot saved to: {save_path}")
else:
    print("❌ No valid reward data found.")

    print("❌ No valid evaluation data found. Please check your model folders.")
