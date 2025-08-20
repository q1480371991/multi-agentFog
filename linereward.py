import numpy as np
import os
import matplotlib.pyplot as plt
# 对比不同算法（MATD3、MAPPO、MASAC、MAIDDPG）的能量消耗数据并可视化（支持平滑处理）

#Moving average smoothing function  移动平均平滑函数
# 用于对原始奖励数据进行平滑处理，减少噪声干扰，更清晰展示趋势
def moving_average(data, window_size=10):
    # 使用卷积计算移动平均值：窗口内数据的平均
    # mode='valid'表示输出仅包含窗口完全覆盖的部分
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# --- Base path and models ---
# base_path = "/home/natnael/Desktop/mult-agentFog/output/"
base_path = "./output_lxl/"
models = ["MATD3", "MAPPO", "MASAC", "MAIDDPG"]
window_size = 10  # You can adjust this  移动平均窗口大小（可调整）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Load and smooth rewards 加载并平滑奖励数据
episode_rewards = {} # 存储各模型的平滑后奖励数据（键：模型名，值：平滑奖励列表）
valid_models = []

for model in models:
    reward_path = os.path.join(base_path, model, "evaluate/evaluation_rewards.npy")
    if os.path.exists(reward_path):
        # 加载原始奖励数据
        rewards = np.load(reward_path)
        # 检查数据量是否满足移动平均窗口要求
        if len(rewards) >= window_size:
            # 对奖励数据进行平滑处理并存储
            episode_rewards[model] = moving_average(rewards, window_size)
            valid_models.append(model)
        else:
            print(f"⚠️ Not enough data for moving average in: {model}")
    else:
        print(f"⚠️ Missing: {reward_path}")

# Plot smoothed reward curves 绘制平滑后的奖励曲线
if valid_models:
    # x轴数据：平滑后的数据长度（因窗口滑动，长度比原始数据短）
    x = np.arange(len(next(iter(episode_rewards.values()))))  # Adjusted X-axis after smoothing
    plt.figure(figsize=(12, 6))
    # 为每个有效模型绘制平滑奖励曲线
    for idx, model in enumerate(valid_models):
        plt.plot(
            x,
            episode_rewards[model],
            label=f"{model} (smoothed)", # 图例：模型名+平滑标记
            color=colors[idx % len(colors)],
            linewidth=2
        )

    plt.title("Smoothed Episode-wise Reward Comparison", fontsize=16)
    plt.xlabel("Episode", fontsize=14)# x轴标签：回合数
    plt.ylabel("Reward", fontsize=14)# y轴标签：奖励值
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(base_path, "origin/smoothed_reward_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✅ Smoothed reward graph saved at: {save_path}")
else:
    print("❌ No valid reward data found.")
