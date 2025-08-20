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
num_episodes = 100 # 评估的总回合数
success_count = 0# 记录成功完成任务的回合数
execution_latencies = [] # 存储每回合的执行延迟（评估耗时）
evaluation_rewards = []# 存储每回合的总奖励（归一化后）
evaluation_energy = []# 存储每回合的能量消耗（归一化后）

# Start evaluation
for episode in range(num_episodes):
    state = env.reset() # 重置环境，获取初始状态（每个智能体对应一个子状态）
    done = False
    
    # 记录当前回合的开始时间（用于计算执行延迟）
    start_time = time.time()
    total_reward = 0 # 累积当前回合的总奖励
    total_latency = 0 # 累积当前回合的环境反馈延迟
    total_energy = 0# 累积当前回合的能量消耗

    # 单回合内的交互循环（直到任务完成）
    while not done:
        # 每个智能体根据当前状态选择动作
        actions = np.array([agents[i].select_action(state[i])[0] for i in range(num_agents)])
        # 环境执行动作，返回下一状态、奖励、回合结束标志和额外信息（延迟、能量等）
        next_state, rewards, done, info = env.step(actions)
        # 累积奖励、环境反馈的延迟和能量消耗
        total_reward += float(rewards)
        total_latency += float(info.get("Total Latency", 0))
        total_energy += float(info.get("Total Energy", 0))
        state = next_state # 更新状态为下一状态，准备下一次决策

    # 计算当前回合的执行时间（从开始到结束的耗时）
    end_time = time.time()
    execution_latencies.append(end_time - start_time)
    # 存储归一化后的奖励（除以100，与训练时的处理一致）
    evaluation_rewards.append(rewards)
    # 存储归一化后的能量消耗（除以10000，缩小数值范围）
    evaluation_energy.append(total_energy / 10000)  # Normalize energy
    # 检查当前回合是否有成功完成的任务（从info中获取标志）
    if info.get("Successful Tasks", 0) > 0:
        success_count += 1

# Compute success rate and average execution latency  计算评估指标的统计结果
success_rate = success_count / num_episodes# 成功率 = 成功回合数 / 总回合数
avg_latency = np.mean(execution_latencies)# 平均执行延迟
avg_reward = np.mean(evaluation_rewards)# 平均奖励
avg_energy = np.mean(evaluation_energy) # 平均能量消耗

# Save evaluation results 保存各指标的数组
np.save(f"{output_dir}evaluate/evaluation_rewards.npy", evaluation_rewards)
np.save(f"{output_dir}evaluate/evaluation_latencies.npy", execution_latencies)
np.save(f"{output_dir}evaluate/evaluation_energy.npy", evaluation_energy)


# 保存文本格式的评估摘要
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
