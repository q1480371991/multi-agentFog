import numpy as np
import os
import matplotlib.pyplot as plt

# Path to MATD3 evaluation output
matd3_path = "/home/natnael/Desktop/mult-agentFog/output/MATD3"
latency_file = os.path.join(matd3_path, "evaluation_latencies.npy")

# Load latency
try:
    latencies = np.load(latency_file)
    x = np.arange(len(latencies))
except FileNotFoundError:
    print(f"❌ Could not find {latency_file}")
    exit()

# Plot
plt.figure(figsize=(6, 4))
plt.plot(x, latencies, label="MATD3 Latency", color='red')
#plt.title("MATD3 Episode-wise Latency")
plt.xlabel("Episode")
plt.ylabel("Latency (s)")
plt.grid(True)
plt.legend()

# Save and show
save_path = os.path.join(matd3_path, "matd3_latency_plot.png")
plt.tight_layout()
plt.savefig(save_path)
plt.show()

print(f"✅ Raw MATD3 latency plot saved to: {save_path}")
