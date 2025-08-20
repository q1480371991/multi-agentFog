import numpy as np
import os
import matplotlib.pyplot as plt
# 可视化 MATD3 算法的原始延迟数据。
# Path to MATD3 evaluation output
matd3_path = "./output_lxl/"
latency_file = os.path.join(matd3_path, "MATD3/evaluate/evaluation_latencies.npy")

# Load latency
try:
    # 从npy文件中加载延迟数据（每个元素对应一回合的总延迟）
    latencies = np.load(latency_file)
    # 生成x轴数据（回合数，与延迟数据长度一致）
    x = np.arange(len(latencies))
except FileNotFoundError:
    print(f"❌ Could not find {latency_file}")
    exit()

# Plot
plt.figure(figsize=(6, 4))
# 绘制MATD3算法的延迟曲线，红色线条，标签为"MATD3 Latency"
plt.plot(x, latencies, label="MATD3 Latency", color='red')
plt.title("MATD3 Episode-wise Latency")
plt.xlabel("Episode")# x轴标签：回合数
plt.ylabel("Latency (s)")# y轴标签：延迟（单位：秒）
plt.grid(True)
plt.legend()

# Save and show
save_path = os.path.join(matd3_path, "origin/matd3_latency_plot.png")
plt.tight_layout()
plt.savefig(save_path)
plt.show()

print(f"✅ Raw MATD3 latency plot saved to: {save_path}")
