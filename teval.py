import numpy as np
import os
import matplotlib.pyplot as plt

# Base path and models to evaluate
base_path = "/home/natnael/Desktop/mult-agentFog/output/"
models = ["MATD3", "MAPPO", "MASAC", "MAIDDPG"]  # Ensure these exist and have .npy files

# Containers for valid data
valid_models = []
episode_rewards = {"MATD3": [], "MAPPO": [], "MASAC": [],  "MAIDDPG": []}
#episode_latencies = {"MATD3": [], "MAPPO": [], "MASAC": [],  "MAIDDPG": []}
#episode_energies = {"MATD3": [], "MAPPO": [], "MASAC": [],  "MAIDDPG": []}


# Function to safely load .npy metrics for each model
def load_metrics(directory):
    try:
        rewards = np.load(os.path.join(directory, "evaluation_rewards.npy"))
        #latencies = np.load(os.path.join(directory, "evaluation_latencies.npy"))
        #energies = np.load(os.path.join(directory, "evaluation_energy.npy"))
    except FileNotFoundError as e:
        print(f"⚠️  Missing file: {e.filename}. Skipping {directory}.")
        return None, None, None
    return rewards

# Iterate over models and load data
for model in models:
    metrics_dir = os.path.join(base_path, model)
    rewards = load_metrics(metrics_dir)
    
    if rewards is None:
        continue  # Skip if any file is missing
    
    valid_models.append(model)
    episode_rewards[model] = rewards
    #episode_latencies[model] = latencies
    #episode_energies[model] = energies

# Plot episode-based comparison
# Plot only rewards
if valid_models:
    x = np.arange(len(next(iter(episode_rewards.values()))))  # X-axis: episode indices
    plt.figure(figsize=(8, 5))

    for model in valid_models:
        plt.plot(x, episode_rewards[model], label=f"{model} Reward", marker='o')

    #plt.title('Episode-wise Comparison of Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(base_path, "reward_comparison_plot.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Reward-only plot saved to: {save_path}")
else:
    print("❌ No valid reward data found.")

    print("❌ No valid evaluation data found. Please check your model folders.")
