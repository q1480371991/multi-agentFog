import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from fogenv import FogToFogEnv
from networks import Actor, Critic
from replay_buffer import ReplayBuffer
from MATD3 import MATD3Agent
import time

# Output directory
# output_dir = "/home/natnael/Desktop/mult-agentFog/output/MATD/"
output_dir = "./output_lxl/MATD3/"
metrics_path = os.path.join(output_dir, "evaluate/evaluation_metrics.npy")
summary_path = os.path.join(output_dir, "evaluate/evaluation_summary.txt")
os.makedirs(output_dir, exist_ok=True)

# Setup
num_agents = 6# 智能体数量（与训练时一致，6个雾节点）
state_dim = 12# 状态维度（6个节点 × 2个特征：CPU和带宽）
action_dim = 6# 动作维度（6个节点的资源分配决策）
max_action = 1# 动作最大值（与训练时一致，用于动作缩放）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
last_episode = 10000
num_episodes = 100# 评估的回合数（共测试100次）

# Load agents
agents = [MATD3Agent(state_dim, action_dim, max_action, i) for i in range(num_agents)]

for agent_id, agent in enumerate(agents):
    actor_path = os.path.join(output_dir, f"weight/actor_agent_{agent_id}_ep_{last_episode}.pth")
    critic1_path = os.path.join(output_dir, f"weight/critic1_agent_{agent_id}_ep_{last_episode}.pth")
    critic2_path = os.path.join(output_dir, f"weight/critic2_agent_{agent_id}_ep_{last_episode}.pth")
    # 加载模型参数（如果文件存在）
    if os.path.exists(actor_path):
        agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
    if os.path.exists(critic1_path):
        agent.critic1.load_state_dict(torch.load(critic1_path, map_location=device))
    if os.path.exists(critic2_path):
        agent.critic2.load_state_dict(torch.load(critic2_path, map_location=device))

# 初始化评估环境
env = FogToFogEnv()
# 初始化存储评估指标的列表
evaluation_rewards, evaluation_latencies, evaluation_energy = [], [], []
success_count = 0# 记录成功完成任务的回合数
# 开始评估循环（共num_episodes回合）
for episode in range(num_episodes):
    state = env.reset()# 重置环境，获取初始状态
    done = False# 回合结束标志
   # Start time tracking
    start_time = time.time() # 记录当前回合开始时间（用于计算总延迟）
    total_reward = 0 # 累计当前回合的奖励
    total_latency = 0# 累计当前回合的延迟
    total_energy = 0# 累计当前回合的能量消耗
    # 单回合内的交互循环（直到任务完成或超时）
    while not done:
        # 每个智能体根据当前状态选择动作（无探索噪声，纯策略执行）
        actions = np.array([agents[i].select_action(state[i]) for i in range(num_agents)])
        # 执行动作，获取环境反馈
        next_state, rewards, done, info = env.step(actions)
        # 累计奖励和能量消耗
        total_reward += float(rewards)
        total_energy += float(info.get("Total Energy", 0)) # 从info中提取总能量
        state = next_state# 更新状态
    # 计算当前回合的总延迟（从开始到结束的时间差）
    evaluation_latencies.append(time.time() - start_time)
    # 存储标准化后的奖励（除以100，与训练时处理一致）
    evaluation_rewards.append(total_reward/100)
    # 存储标准化后的能量消耗（除以10000，与训练时处理一致）
    evaluation_energy.append(total_energy / 10000)
    # 检查当前回合是否成功（从info中获取成功标志）
    if info.get("success", False):
        success_count += 1

# 计算评估指标的统计量
success_rate = success_count / num_episodes # 成功率 = 成功回合数 / 总回合数
# 奖励的均值和标准差
reward_mean, reward_std = np.mean(evaluation_rewards), np.std(evaluation_rewards)
# 延迟的均值和标准差
latency_mean, latency_std = np.mean(evaluation_latencies), np.std(evaluation_latencies)
# 能量消耗的均值和标准差
energy_mean, energy_std = np.mean(evaluation_energy), np.std(evaluation_energy)

# Save metrics in .npy format (dictionary) 将所有评估指标存储为字典，便于后续分析
evaluation_metrics = {
    "episode_rewards": evaluation_rewards,
    "episode_latencies": evaluation_latencies,
    "episode_energies": evaluation_energy
}
# 保存指标到.npy文件（二进制格式，便于快速加载）
np.save(os.path.join(output_dir, "evaluate/evaluation_metrics.npy"), evaluation_metrics)

# Save individual metric arrays for comparison script
# 单独保存每个指标的数组（用于跨算法对比脚本，如teval.py、toteval.py）
np.save(os.path.join(output_dir, "evaluate/evaluation_rewards.npy"), np.array(evaluation_rewards))
np.save(os.path.join(output_dir, "evaluate/evaluation_latencies.npy"), np.array(evaluation_latencies))
np.save(os.path.join(output_dir, "evaluate/evaluation_energy.npy"), np.array(evaluation_energy))

# Save text summary将评估摘要写入文本文件（便于人工阅读）
with open(summary_path, "w") as f:
    f.write("=== Evaluation Summary ===\n")
    f.write(f"Success Rate: {success_rate:.2f}\n")
    f.write(f"Average Reward: {reward_mean:.2f} ± {reward_std:.2f}\n")
    f.write(f"Average Latency: {latency_mean:.2f} ± {latency_std:.2f}\n")
    f.write(f"Average Energy: {energy_mean:.2f} ± {energy_std:.2f}\n")

# Plotting 绘制评估指标图表（3个子图，分别展示奖励、延迟、能量）
plt.figure(figsize=(10, 15))
# 第一个子图：奖励曲线
plt.subplot(3, 1, 1)
plt.plot(evaluation_rewards, label="Reward", color='blue')
plt.xlabel("Episodes")
plt.ylabel("Average Reward")
#plt.title("Evaluation Rewards")
plt.grid(True)
# 第二个子图：延迟曲线
plt.subplot(3, 1, 2)
plt.plot(evaluation_latencies, label="Latency", color='red')
plt.xlabel("Episodes")
plt.ylabel("Average Latency (s)")
#plt.title("Evaluation Latency")
plt.grid(True)
# 第三个子图：能量消耗曲线
plt.subplot(3, 1, 3)
plt.plot(evaluation_energy, label="Energy", color='green')
plt.xlabel("Episodes")
plt.ylabel("Average Energy Consumption")
#plt.title("Evaluation Energy Consumption")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "evaluate/evaluation_plots.png"))
plt.show()

print("✅ Evaluation complete. All results saved to:", output_dir)
