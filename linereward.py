import numpy as np
import os
import matplotlib.pyplot as plt

# --- Moving average smoothing function ---
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# --- Base path and models ---
base_path = "/home/natnael/Desktop/mult-agentFog/output/"
models = ["MATD3", "MAPPO", "MASAC", "MAIDDPG"]
window_size = 10  # You can adjust this
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# --- Load and smooth rewards ---
episode_rewards = {}
valid_models = []

for model in models:
    reward_path = os.path.join(base_path, model, "evaluation_rewards.npy")
    if os.path.exists(reward_path):
        rewards = np.load(reward_path)
        if len(rewards) >= window_size:
            episode_rewards[model] = moving_average(rewards, window_size)
            valid_models.append(model)
        else:
            print(f"⚠️ Not enough data for moving average in: {model}")
    else:
        print(f"⚠️ Missing: {reward_path}")

# --- Plot smoothed reward curves ---
if valid_models:
    x = np.arange(len(next(iter(episode_rewards.values()))))  # Adjusted X-axis after smoothing
    plt.figure(figsize=(12, 6))

    for idx, model in enumerate(valid_models):
        plt.plot(
            x,
            episode_rewards[model],
            label=f"{model} (smoothed)",
            color=colors[idx % len(colors)],
            linewidth=2
        )

    plt.title("Smoothed Episode-wise Reward Comparison", fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(base_path, "smoothed_reward_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✅ Smoothed reward graph saved at: {save_path}")
else:
    print("❌ No valid reward data found.")
