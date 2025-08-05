import numpy as np
import os
import matplotlib.pyplot as plt

# --- Moving average smoothing function ---
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# --- Optional: Clip outliers ---
def clip_outliers(data, low_percentile=5, high_percentile=95):
    lower = np.percentile(data, low_percentile)
    upper = np.percentile(data, high_percentile)
    return np.clip(data, lower, upper)

# --- Base path and models ---
base_path = "/home/natnael/Desktop/mult-agentFog/output/"
models = ["MATD3", "MAPPO", "MASAC", "MAIDDPG"]
window_size = 10
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# --- Load, clip, and smooth energy ---
episode_energies = {}
valid_models = []

for model in models:
    energy_path = os.path.join(base_path, model, "evaluation_energy.npy")
    
    if os.path.exists(energy_path):
        total_energy = np.load(energy_path)

        # Optional: clip extreme values (e.g., energy spikes)
        clipped_energy = clip_outliers(total_energy)

        # Ensure we have enough data for smoothing
        if len(clipped_energy) >= window_size:
            smoothed_energy = moving_average(clipped_energy, window_size)
            episode_energies[model] = smoothed_energy
            valid_models.append(model)
        else:
            print(f"⚠️ Not enough data for smoothing in: {model}")
    else:
        print(f"⚠️ Missing: {energy_path}")

# --- Plot smoothed energy curves ---
if valid_models:
    x = np.arange(len(next(iter(episode_energies.values()))))  # Uniform x-axis
    plt.figure(figsize=(12, 6))

    for idx, model in enumerate(valid_models):
        plt.plot(
            x,
            episode_energies[model],
            label=f"{model}",
            color=colors[idx % len(colors)],
            linewidth=2
        )

    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Energy Consumption", fontsize=14)
    #plt.title("Smoothed Episode-wise Energy Comparison", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(base_path, "smoothed_energy_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"✅ Smoothed energy graph saved at: {save_path}")
else:
    print("❌ No valid energy data found.")
