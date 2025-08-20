import numpy as np
import os
import matplotlib.pyplot as plt
# 可视化 MATD3 算法的原始奖励数据。

# Path to MATD3 evaluation output
# matd3_path = "/home/natnael/Desktop/mult-agentFog/output/MATD3"
matd3_path = "./output_lxl/"
reward_file = os.path.join(matd3_path, "MATD3/evaluate/evaluation_rewards.npy")

# Load reward
try:
    # 从npy文件中加载奖励数据（每个元素对应一回合的奖励值）
    rewards = np.load(reward_file)
    # 生成x轴数据（回合数，与奖励数据长度一致）
    x = np.arange(len(rewards))
except FileNotFoundError:
    print(f"❌ Could not find {reward_file}")
    exit()

# Plot
plt.figure(figsize=(6, 4))
# 绘制MATD3算法的奖励曲线，蓝色线条，标签为"MATD3 Reward"
plt.plot(x, rewards, label="MATD3 Reward", color='blue')
#plt.title("MATD3 Episode-wise Reward")
plt.xlabel("Episode")# x轴标签：回合数
plt.ylabel("Reward")# y轴标签：奖励值
plt.grid(True)
plt.legend()

# Save and show
save_path = os.path.join(matd3_path, "origin/matd3_reward_plot.png")
plt.tight_layout()
plt.savefig(save_path)
plt.show()

print(f"✅ Raw MATD3 reward plot saved to: {save_path}")
