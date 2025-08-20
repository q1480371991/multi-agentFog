import numpy as np
import os
import matplotlib.pyplot as plt
# 同时对比不同算法 奖励、延迟、能量 三种指标，计算各指标的平均值并绘制柱状图
# Base path and models to evaluate
# base_path = "/home/natnael/Desktop/mult-agentFog/output/"
base_path = "./output_lxl/"
models = ["MATD3", "MAPPO", "MASAC", "MAIDDPG"]  # Ensure these exist and have .npy files

# Containers for valid data 存储有效数据的容器
valid_models = []
avg_rewards = []# 存储各模型的平均奖励
avg_latencies = [] # 存储各模型的平均延迟
avg_energies = [] # 存储各模型的平均能量消耗

# Function to safely load .npy metrics 加载.npy格式评估指标的函数
def load_metrics(directory):
    try:
        # 加载奖励、延迟、能量三种评估数据
        rewards = np.load(os.path.join(directory, "evaluate/evaluation_rewards.npy"))
        latencies = np.load(os.path.join(directory, "evaluate/evaluation_latencies.npy"))
        energies = np.load(os.path.join(directory, "evaluate/evaluation_energy.npy"))
    except FileNotFoundError as e:
        print(f"⚠️  Missing file: {e.filename}. Skipping {directory}.")
        return None, None, None
    return rewards, latencies, energies

# Iterate over models
for model in models:
    metrics_dir = os.path.join(base_path, model)
    rewards, latencies, energies = load_metrics(metrics_dir)# 加载三种数据
    
    if rewards is None:
        continue  # Skip if any file is missing若数据加载失败，跳过当前模型
    
    valid_models.append(model)
    # 计算并存储各指标的平均值（将每回合数据聚合为单值）
    avg_rewards.append(np.mean(rewards))
    avg_latencies.append(np.mean(latencies))
    avg_energies.append(np.mean(energies))

# Plot only if valid data exists 绘制多指标对比柱状图（若有有效数据）
if valid_models:
    x = np.arange(len(valid_models))# x轴为模型索引
    width = 0.15  # Thinner bars

    # Adjusted the position of bars for minimal space between models
    fig, ax = plt.subplots(figsize=(8, 5))  # Smaller plot size
    # 绘制三种指标的柱状图，通过x轴偏移区分（左：奖励，中：延迟，右：能量）
    rect1 = ax.bar(x - width, avg_rewards, width, label="Avg Reward", color='skyblue')
    rect2 = ax.bar(x, avg_latencies, width, label="Avg Latency", color='lightcoral')
    rect3 = ax.bar(x + width, avg_energies, width, label="Avg Energy", color='lightgreen')

    ax.set_ylabel("Metric Value")# y轴标签：指标值
    ax.set_title("Comparison of Multi-Agent RL Algorithms")
    ax.set_xticks(x)# 设置x轴刻度位置
    ax.set_xticklabels(valid_models)# x轴刻度标签为模型名称
    ax.legend()


    # 定义柱状图数值标注函数（在每个柱子顶部显示数值）
    def annotate_bars(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom')

    annotate_bars(rect1)
    annotate_bars(rect2)
    annotate_bars(rect3)

    plt.tight_layout()
    save_path = os.path.join(base_path, "origin/comparison/comparison_plot.png")
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Comparison plot saved to: {save_path}")
else:
    print("❌ No valid evaluation data found. Please check your model folders.")
