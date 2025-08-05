import numpy as np
import os
import matplotlib.pyplot as plt

# Base path and models to evaluate
base_path = "/home/natnael/Desktop/mult-agentFog/output/"
models = ["MATD3", "MAPPO", "MASAC", "MAIDDPG"]  # Ensure these exist and have .npy files

# Containers for valid data
valid_models = []
avg_rewards = []
avg_latencies = []
avg_energies = []

# Function to safely load .npy metrics
def load_metrics(directory):
    try:
        rewards = np.load(os.path.join(directory, "evaluation_rewards.npy"))
        latencies = np.load(os.path.join(directory, "evaluation_latencies.npy"))
        energies = np.load(os.path.join(directory, "evaluation_energy.npy"))
    except FileNotFoundError as e:
        print(f"⚠️  Missing file: {e.filename}. Skipping {directory}.")
        return None, None, None
    return rewards, latencies, energies

# Iterate over models
for model in models:
    metrics_dir = os.path.join(base_path, model)
    rewards, latencies, energies = load_metrics(metrics_dir)
    
    if rewards is None:
        continue  # Skip if any file is missing
    
    valid_models.append(model)
    avg_rewards.append(np.mean(rewards))
    avg_latencies.append(np.mean(latencies))
    avg_energies.append(np.mean(energies))

# Plot only if valid data exists
if valid_models:
    x = np.arange(len(valid_models))
    width = 0.15  # Thinner bars

    # Adjusted the position of bars for minimal space between models
    fig, ax = plt.subplots(figsize=(8, 5))  # Smaller plot size
    rect1 = ax.bar(x - width, avg_rewards, width, label="Avg Reward", color='skyblue')
    rect2 = ax.bar(x, avg_latencies, width, label="Avg Latency", color='lightcoral')
    rect3 = ax.bar(x + width, avg_energies, width, label="Avg Energy", color='lightgreen')

    ax.set_ylabel("Metric Value")
    #ax.set_title("Comparison of Multi-Agent RL Algorithms")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_models)
    ax.legend()

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
    save_path = os.path.join(base_path, "comparison_plot.png")
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Comparison plot saved to: {save_path}")
else:
    print("❌ No valid evaluation data found. Please check your model folders.")
