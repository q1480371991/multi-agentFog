import numpy as np
import os
import matplotlib.pyplot as plt
# 对比不同算法（MATD3、MAPPO、MASAC、MAIDDPG）的能量消耗数据并可视化。

# --- 移动平均平滑函数 ---
# 用于对数据进行平滑处理，减少噪声干扰，使曲线更易观察趋势
# window_size: 滑动窗口大小，越大平滑效果越强
def moving_average(data, window_size=10):
    # 使用卷积    # np.convolve计算数据与窗口的卷积，mode='valid'表示只返回有效卷积结果（去除边界效应）
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# --- 可选：截断异常值 ---
# 去除数据中的极端值，避免异常点对可视化的影响
# low_percentile: 下限分位数，低于此值的视为异常
# high_percentile: 上限分位数，高于此值的视为异常
def clip_outliers(data, low_percentile=5, high_percentile=95):
    lower = np.percentile(data, low_percentile)# 计算下限阈值
    upper = np.percentile(data, high_percentile)# 计算上限阈值
    return np.clip(data, lower, upper)# 将数据截断在[lower, upper]范围内
if __name__ == "__main__":
    # --- Base path and models ---
    # base_path = "/home/natnael/Desktop/mult-agentFog/output/"
    base_path = "./output_lxl/"
    models = ["MATD3", "MAPPO", "MASAC", "MAIDDPG"]
    window_size = 10 # 移动平均窗口大小
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']# 每条曲线的颜色（确保与模型顺序对应）

    # Load, clip, and smooth energy   加载、截断和平滑能量消耗数据
    episode_energies = {} # 存储各模型的能量消耗数据（平滑后）
    valid_models = []# 存储有有效数据的模型名称

    for model in models:
        energy_path=base_path+model+"/evaluate/evaluation_energy.npy"
        # energy_path = os.path.join(base_path, model, "weight/evaluation_energy.npy")

        if os.path.exists(energy_path):
            total_energy = np.load(energy_path) # 加载能量消耗数据（npy格式，二进制数组）

            # Optional: clip extreme values (e.g., energy spikes)  可选：截断极端值（如能量消耗的尖峰）
            clipped_energy = clip_outliers(total_energy)

            # Ensure we have enough data for smoothing  确保数据量足够进行平滑处理（至少大于等于窗口大小）
            if len(clipped_energy) >= window_size:
                # 对截断后的数据进行移动平均平滑
                smoothed_energy = moving_average(clipped_energy, window_size)
                episode_energies[model] = smoothed_energy# 存储平滑后的数据
                valid_models.append(model)# 记录有效模型
            else:
                print(f"⚠️ Not enough data for smoothing in: {model}")
        else:
            print(f"⚠️ Missing: {energy_path}")

    # --- Plot smoothed energy curves  绘制平滑后的能量消耗曲线---
    if valid_models:
        x = np.arange(len(next(iter(episode_energies.values()))))  # Uniform x-axis
        plt.figure(figsize=(12, 6))

        for idx, model in enumerate(valid_models):
            # 绘制每个模型的能量消耗曲线
            plt.plot(
                x,
                episode_energies[model],
                label=f"{model}",
                color=colors[idx % len(colors)],
                linewidth=2
            )

        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Energy Consumption", fontsize=14)
        plt.title("Smoothed Episode-wise Energy Comparison", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        save_path = os.path.join(base_path, "origin/smoothed_energy_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

        print(f"✅ Smoothed energy graph saved at: {save_path}")
    else:
        print("❌ No valid energy data found.")
