import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from fogenv import FogToFogEnv
from MAPPO import MAPPOAgent  # Assuming this is your PPO agent implementation
import time

# Output directory for saving evaluation results
# output_dir = "/home/natnael/Desktop/mult-agentFog/output/MAPPO/"
output_dir = "./output_lxl/MAPPO/"
os.makedirs(output_dir, exist_ok=True)

# Load trained agents
num_agents = 6# 智能体数量（与训练时保持一致）
state_dim = 12  # 状态维度（示例：6个雾节点 × 2个状态特征（CPU+带宽））
action_dim = 6  # 动作维度（对应6个雾节点的任务分配策略）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the episode to load the models from
last_episode = 10000  # You can change this to the episode number you want to evaluate
max_action = 1  # Ensure this is correctly set based on your training setup
# 初始化多智能体（与训练时的智能体结构一致）
agents = [MAPPOAgent(state_dim, action_dim, i) for i in range(num_agents)]

# 为每个智能体加载训练好的模型参数
for agent_id, agent in enumerate(agents):
    actor_path = os.path.join(output_dir, f"weight/actor_agent_{agent_id}_ep_{last_episode}.pth")
    critic_path = os.path.join(output_dir, f"weight/critic_agent_{agent_id}_ep_{last_episode}.pth")
    
    # 加载策略网络参数（评估主要依赖策略网络生成动作）
    if os.path.exists(actor_path):
        agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
    else:
        print(f"Actor model file not found: {actor_path}")
    # 加载价值网络参数（评估中可能用于辅助分析，非必需）
    if os.path.exists(critic_path):
        agent.critic.load_state_dict(torch.load(critic_path, map_location=device))
    else:
        print(f"Critic model file not found: {critic_path}")

# 初始化雾计算环境（与训练时的环境配置一致，确保评估场景一致）
env = FogToFogEnv()
num_episodes = 100
success_count = 0
execution_latencies = []
evaluation_rewards = []
evaluation_energy = []

# Start evaluation
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    # Start time tracking
    start_time = time.time()
    total_reward = 0
    total_latency = 0
    total_energy = 0
    
    while not done:
        actions = np.array([agents[i].select_action(state[i])[0] for i in range(num_agents)])  # Extract only action
        next_state, rewards, done, info = env.step(actions)
        
        total_reward += float(rewards)
        total_latency += float(info.get("Total Latency", 0))
        total_energy += float(info.get("Total Energy", 0))
        state = next_state
    
    # End time tracking
    end_time = time.time()
    execution_latencies.append(end_time - start_time)
    evaluation_rewards.append(rewards)
    evaluation_energy.append(total_energy / 10000)  # Normalize energy
    
    if info.get("Successful Tasks", 0) > 0:
        success_count += 1

# Compute success rate and average execution latency
success_rate = success_count / num_episodes
avg_latency = np.mean(execution_latencies)
avg_reward = np.mean(evaluation_rewards)
avg_energy = np.mean(evaluation_energy)

# Save evaluation results
np.save(f"{output_dir}evaluate/evaluation_rewards.npy", evaluation_rewards)
np.save(f"{output_dir}evaluate/execution_latencies.npy", execution_latencies)
np.save(f"{output_dir}evaluate/evaluation_energy.npy", evaluation_energy)

with open(f"{output_dir}evaluate/evaluation_metrics.txt", "w") as f:
    f.write(f"Success Rate: {success_rate}\n")
    f.write(f"Average Execution Latency: {avg_latency}\n")
    f.write(f"Average Evaluation Reward: {avg_reward}\n")
    f.write(f"Average Evaluation Energy: {avg_energy}\n")

# Create a single figure for all plots
plt.figure(figsize=(15, 10))

# Plot Evaluation Rewards
plt.subplot(3, 1, 1)
plt.plot(evaluation_rewards, label="Evaluation Reward", color='b')
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()
plt.title("Evaluation Rewards")

# Plot Execution Latencies
plt.subplot(3, 1, 2)
plt.plot(execution_latencies, label="Execution Latency", color='r')
plt.xlabel("Episodes")
plt.ylabel("Latency (s)")
plt.legend()
plt.title("Execution Latency")

# Plot Evaluation Energy Consumption
plt.subplot(3, 1, 3)
plt.plot(evaluation_energy, label="Evaluation Energy", color='g')
plt.xlabel("Episodes")
plt.ylabel("Energy Consumption")
plt.legend()
plt.title("Evaluation Energy Consumption")

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "evaluate/evaluation_plots.png"))  # Save the full plot
plt.close()

print("Evaluation complete. Results saved to", output_dir)
