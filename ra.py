import numpy as np
import os
import matplotlib.pyplot as plt

# Path to MATD3 evaluation output
matd3_path = "/home/natnael/Desktop/mult-agentFog/output/MATD3"
reward_file = os.path.join(matd3_path, "evaluation_rewards.npy")

# Load reward
try:
    rewards = np.load(reward_file)
    x = np.arange(len(rewards))
except FileNotFoundError:
    print(f"❌ Could not find {reward_file}")
    exit()

# Plot
plt.figure(figsize=(6, 4))
plt.plot(x, rewards, label="MATD3 Reward", color='blue')
#plt.title("MATD3 Episode-wise Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()

# Save and show
save_path = os.path.join(matd3_path, "matd3_reward_plot.png")
plt.tight_layout()
plt.savefig(save_path)
plt.show()

print(f"✅ Raw MATD3 reward plot saved to: {save_path}")
