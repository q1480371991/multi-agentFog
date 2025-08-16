import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from fogenv import FogToFogEnv
from MASAC import SACAgent  # Change to MASAC agent import
import time

# Output directory for saving evaluation results
# output_dir = "/home/natnael/Desktop/mult-agentFog/output/MASAC/"
output_dir = "./output_lxl/MASAC/"
os.makedirs(output_dir, exist_ok=True)

# Load trained agents 加载训练好的智能体参数
num_agents = 6# 智能体数量（对应6个雾节点）
state_dim = 12  # 状态维度（示例：6个雾节点 × 每个节点的CPU/带宽等特征）
action_dim = 6  # 动作维度（每个动作对应6个雾节点的任务分配策略）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 指定要加载的模型对应的训练回合数
last_episode = 10000  # You can change this to the episode number you want to evaluate
max_action = 1  # Ensure this is correctly set based on your training setup

# 初始化6个SAC智能体（每个智能体对应一个雾节点）
agents = [SACAgent(state_dim, action_dim, max_action, i) for i in range(num_agents)]

# 为每个智能体加载训练好的模型参数
for agent_id, agent in enumerate(agents):
    actor_path = os.path.join(output_dir, f"weight/actor_agent_{agent_id}_ep_{last_episode}.pth")
    critic1_path = os.path.join(output_dir, f"weight/critic1_agent_{agent_id}_ep_{last_episode}.pth")
    critic2_path = os.path.join(output_dir, f"weight/critic2_agent_{agent_id}_ep_{last_episode}.pth")
    
    # 加载策略网络参数（若文件存在）
    if os.path.exists(actor_path):
        agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
    else:
        print(f"Actor model file not found: {actor_path}")
    # 加载第一个价值网络参数（若文件存在）
    if os.path.exists(critic1_path):
        agent.critic1.load_state_dict(torch.load(critic1_path, map_location=device))
    else:
        print(f"Critic1 model file not found: {critic1_path}")
    # 加载第二个价值网络参数（若文件存在）
    if os.path.exists(critic2_path):
        agent.critic2.load_state_dict(torch.load(critic2_path, map_location=device))
    else:
        print(f"Critic2 model file not found: {critic2_path}")

# 初始化雾计算环境
env = FogToFogEnv()
num_episodes = 100# 评估的总回合数
success_count = 0# 成功处理任务的回合计数
execution_latencies = []# 记录每个回合的执行延迟（评估脚本运行耗时）
evaluation_rewards = []   # 记录每个回合的奖励
evaluation_energy = []# 记录每个回合的能量消耗（归一化后）

# 开始评估循环
for episode in range(num_episodes):
    # 重置环境，获取初始状态（每个智能体对应一个子状态）
    state = env.reset()
    done = False # 回合结束标志（任务处理完成时为True）
    
    # 记录回合开始时间（用于计算执行延迟）
    start_time = time.time()
    total_reward = 0# 本回合的总奖励
    total_latency = 0# 本回合的总延迟（环境内部计算的任务延迟）
    total_energy = 0# 本回合的总能量消耗
    # 回合内循环（直到任务处理完成）
    while not done:
        # 每个智能体根据当前状态选择动作（仅提取动作值，忽略日志概率）
        actions = np.array([agents[i].select_action(state[i])[0] for i in range(num_agents)])  # Extract only action
        # 执行动作，获取下一状态、奖励、回合结束标志和额外信息（延迟、能量等）
        next_state, rewards, done, info = env.step(actions)

        # 累积本回合的奖励、延迟和能量消耗
        total_reward += float(rewards)
        total_latency += float(info.get("Total Latency", 0))# 从info字典中获取总延迟
        total_energy += float(info.get("Total Energy", 0))# 从info字典中获取总能量
        state = next_state

    # 记录回合结束时间，计算并保存本回合的执行延迟（脚本运行耗时）
    end_time = time.time()
    execution_latencies.append(end_time - start_time)
    # 记录本回合的最终奖励
    evaluation_rewards.append(total_reward/100)
    # 记录归一化后的能量消耗（除以10000缩小数值范围）
    evaluation_energy.append(total_energy/10000)
    # 判断本回合是否有成功处理的任务（通过info字典的"Successful Tasks"键）
    if info.get("Successful Tasks", 0) > 0:
        success_count += 1

# Compute success rate and average execution latency  计算评估指标
success_rate = success_count / num_episodes # 成功率 = 成功回合数 / 总回合数
avg_latency = np.mean(execution_latencies)# 平均执行延迟
avg_reward = np.mean(evaluation_rewards)# 平均奖励
avg_energy = np.mean(evaluation_energy)# 平均能量消耗

# 保存评估结果（以numpy数组形式，便于后续分析）
np.save(f"{output_dir}evaluate/evaluation_rewards.npy", evaluation_rewards)
np.save(f"{output_dir}evaluate/execution_latencies.npy", execution_latencies)
np.save(f"{output_dir}evaluate/evaluation_energy.npy", evaluation_energy)
# 将评估指标保存到文本文件（便于直接查看）
with open(f"{output_dir}evaluate/evaluation_metrics.txt", "w") as f:
    f.write(f"Success Rate: {success_rate}\n")
    f.write(f"Average Execution Latency: {avg_latency}\n")
    f.write(f"Average Evaluation Reward: {avg_reward}\n")
    f.write(f"Average Evaluation Energy: {avg_energy}\n")

# 创建一个包含所有指标的图表（3个子图，分别展示奖励、延迟、能量）
plt.figure(figsize=(15, 10))

# 绘制评估奖励曲线
plt.subplot(3, 1, 1)
plt.plot(evaluation_rewards, label="Evaluation Reward", color='b')
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()
plt.title("Evaluation Rewards")

# 绘制执行延迟曲线
plt.subplot(3, 1, 2)
plt.plot(execution_latencies, label="Execution Latency", color='r')
plt.xlabel("Episodes")
plt.ylabel("Latency (s)")
plt.legend()
plt.title("Execution Latency")

# 绘制能量消耗曲线
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

