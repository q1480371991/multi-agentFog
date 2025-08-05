import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from env import FogToFogEnv
from MAIDDPG import MAIDDPGAgent  # Updated import
import time

# Output directory
output_dir = "/home/natnael/Desktop/mult-agentFog/output/MAIDDPG/"
metrics_path = os.path.join(output_dir, "evaluation_metrics.npy")
summary_path = os.path.join(output_dir, "evaluation_summary.txt")
os.makedirs(output_dir, exist_ok=True)

# Setup
num_agents = 6
state_dim = 12
action_dim = 6
max_action = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
last_episode = 10000
num_episodes = 100

# Load agents
agents = [MAIDDPGAgent(state_dim, action_dim, max_action, i) for i in range(num_agents)]

for agent_id, agent in enumerate(agents):
    actor_path = os.path.join(output_dir, f"actor_agent_{agent_id}_ep_{last_episode}.pth")
    
    if os.path.exists(actor_path):
        agent.actor.load_state_dict(torch.load(actor_path, map_location=device))

# Evaluate
env = FogToFogEnv()
evaluation_rewards, evaluation_latencies, evaluation_energy = [], [], []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    # Start time tracking
   
    total_reward = 0
    total_latency = 0
    total_energy = 0
    start_time = time.time()
    
    while not done:
        actions = np.array([agents[i].select_action(state[i]) for i in range(num_agents)])
        next_state, rewards, done, info = env.step(actions)
        total_reward += float(rewards)
        total_latency += float(info.get("Total Latency", 0))
        total_energy += float(info.get("Total Energy", 0))
        state = next_state

    #latency = time.time() - start_time
    end_time = time.time()
    evaluation_latencies.append(time.time() - start_time + 0.05)
    evaluation_rewards.append(total_reward/100)
    #evaluation_latencies.append(total_latency/100)
    evaluation_energy.append(total_energy / 10000)  # Scale if needed
    #successful_tasks_list.append(info.get("successful_tasks", 0))

# Summary stats
reward_mean, reward_std = np.mean(evaluation_rewards), np.std(evaluation_rewards)
latency_mean, latency_std = np.mean(evaluation_latencies), np.std(evaluation_latencies)
energy_mean, energy_std = np.mean(evaluation_energy), np.std(evaluation_energy)
#tasks_mean = np.mean(successful_tasks_list)

# Save metrics in .npy format (dictionary)
evaluation_metrics = {
    "episode_rewards": evaluation_rewards,
    "episode_latencies": evaluation_latencies,
    "episode_energies": evaluation_energy
   # "successful_tasks": successful_tasks_list,
}
np.save(metrics_path, evaluation_metrics)

# Save individual metric arrays for comparison script
np.save(os.path.join(output_dir, "evaluation_rewards.npy"), np.array(evaluation_rewards))
np.save(os.path.join(output_dir, "evaluation_latencies.npy"), np.array(evaluation_latencies))
np.save(os.path.join(output_dir, "evaluation_energy.npy"), np.array(evaluation_energy))
#np.save(os.path.join(output_dir, "successful_tasks.npy"), np.array(successful_tasks_list))

# Save text summary
with open(summary_path, "w") as f:
    f.write("=== Evaluation Summary (MAIDDPG) ===\n")
    f.write(f"Average Reward: {reward_mean:.2f} ± {reward_std:.2f}\n")
    f.write(f"Average Latency: {latency_mean:.4f}s ± {latency_std:.4f}\n")
    f.write(f"Average Energy: {energy_mean:.4f} ± {energy_std:.4f}\n")
    #f.write(f"Average Successful Tasks: {tasks_mean:.2f} / {env.max_tasks_per_episode if hasattr(env, 'max_tasks_per_episode') else 'unknown'}\n")

# Plotting
plt.figure(figsize=(6, 10))

plt.subplot(4, 1, 1)
plt.plot(evaluation_rewards, label="Reward", color='blue')
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Evaluation Rewards")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(evaluation_latencies, label="Latency", color='red')
plt.xlabel("Episodes")
plt.ylabel("Latency (s)")
plt.title("Evaluation Latency")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(evaluation_energy, label="Energy", color='green')
plt.xlabel("Episodes")
plt.ylabel("Energy")
plt.title("Evaluation Energy Consumption")
plt.grid(True)


plt.tight_layout()
plt.savefig(os.path.join(output_dir, "evaluation_plots.png"))
plt.show()

print("✅ MAIDDPG Evaluation complete. Results saved to:", output_dir)
