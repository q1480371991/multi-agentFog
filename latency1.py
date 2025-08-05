import numpy as np
import os
import matplotlib.pyplot as plt

# Base path and models to evaluate
base_path = "/home/natnael/Desktop/mult-agentFog/output/"
models = ["MATD3", "MAPPO", "MASAC", "MAIDDPG"]

# Containers for valid data
valid_models = []
episode_latencies = {"MATD3": [], "MAPPO": [], "MASAC": [], "MAIDDPG": []}

# Function to safely load .npy latency metrics
def load_latency(directory):
    try:
        latencies = np.load(os.path.join(directory, "evaluation_latencies.npy"))
    except FileNotFoundError as e:
        print(f"⚠️  Missing file: {e.filename}. Skipping {directory}.")
        return None
    return latencies

# Simple moving average smoothing
def smooth(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Iterate over models and load latency data
for model in models:
    metrics_dir = os.path.join(base_path, model)
    latencies = load_latency(metrics_dir)
    
    if latencies is None:
        continue  # Skip if file is missing
    
    valid_models.append(model)
    episode_latencies[model] = latencies

# Plot only latency
if valid_models:
    plt.figure(figsize=(7, 4))

    for model in valid_models:
        raw_data = episode_latencies[model]
        smoothed = smooth(raw_data, window_size=5)
        x = np.arange(len(smoothed))
        plt.plot(x, smoothed, label=f"{model}", linewidth=2)

   #plt.title('Episode-wise Comparison of Latency')
    plt.xlabel('Episode')
    plt.ylabel('Latency (s)')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(base_path, "latency_comparison_plot_smoothed.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Smoothed latency-only plot saved to: {save_path}")
else:
    print("❌ No valid latency data found.")
