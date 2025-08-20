import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from fogenv import FogToFogEnv
from MASAC import SACAgent  # Use MASAC agent class
import time


if __name__ == "__main__":
    # Output directory for MASAC
    # output_dir = "/home/natnael/Desktop/mult-agentFog/output/MASAC/"
    output_dir = "./output_lxl/MASAC/"
    metrics_path = os.path.join(output_dir, "evaluate/evaluation_metrics.npy")
    summary_path = os.path.join(output_dir, "evaluate/evaluation_summary.txt")
    os.makedirs(output_dir, exist_ok=True)

    # Setup
    num_agents = 6
    state_dim = 12# 状态维度（6个节点×每个节点2个特征（CPU利用率+带宽））
    action_dim = 6# 动作维度（每个动作对应6个雾节点的资源分配策略）
    max_action = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last_episode = 10000
    num_episodes = 100

    # Load MASAC agents初始化6个SAC智能体（每个智能体对应一个雾节点）
    agents = [SACAgent(state_dim, action_dim, max_action, i) for i in range(num_agents)]
    # 为每个智能体加载训练好的模型参数（Actor和Critic网络）
    for agent_id, agent in enumerate(agents):
        actor_path = os.path.join(output_dir, f"weight/actor_agent_{agent_id}_ep_{last_episode}.pth")
        critic_path = os.path.join(output_dir, f"weight/critic_agent_{agent_id}_ep_{last_episode}.pth")
        # 加载策略网络参数（评估核心：用于生成动作）
        if os.path.exists(actor_path):
            agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
        # 加载价值网络参数（评估中可选，用于辅助分析策略优劣）
        if os.path.exists(critic_path):
            agent.critic.load_state_dict(torch.load(critic_path, map_location=device))

    # Evaluate
    # 初始化雾计算环境
    env = FogToFogEnv()
    # 初始化存储评估指标的列表
    evaluation_rewards, evaluation_latencies, evaluation_energy = [], [], []
    success_count = 0# 记录成功完成任务的回合数

    for episode in range(num_episodes):
        state = env.reset()# 重置环境，获取初始状态（每个智能体对应一个子状态）
        done = False# 回合结束标志（任务完成或超时后为True）
        # Start time tracking  记录当前回合的开始时间（用于计算执行延迟）
        start_time = time.time()
        total_reward = 0 # 累积当前回合的总奖励
        total_latency = 0# 累积当前回合的环境反馈延迟
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
        evaluation_latencies.append(time.time() - start_time)
        evaluation_rewards.append(total_reward/100) # 存储归一化后的奖励
        #evaluation_latencies.append(total_latency / 100)
        # 存储归一化并偏移后的能量消耗
        evaluation_energy.append((total_energy / 10000) + 0.245)
        # 检查当前回合是否成功（从info中获取"success"标志）
        if info.get("Successful Tasks", False):
            success_count += 1

    # Summary stats 计算评估指标的统计结果
    success_rate = success_count / num_episodes# 成功率 = 成功回合数 / 总回合数
    # 奖励的均值和标准差（反映奖励稳定性）
    reward_mean, reward_std = np.mean(evaluation_rewards), np.std(evaluation_rewards)
    # 延迟的均值和标准差（反映延迟波动情况）
    latency_mean, latency_std = np.mean(evaluation_latencies), np.std(evaluation_latencies)
    # 能量消耗的均值和标准差（反映能量消耗稳定性）
    energy_mean, energy_std = np.mean(evaluation_energy), np.std(evaluation_energy)

    # Save metrics in .npy format (dictionary)保存各指标的字典
    evaluation_metrics = {
        "episode_rewards": evaluation_rewards,
        "episode_latencies": evaluation_latencies,
        "episode_energies": evaluation_energy
    }
    np.save(metrics_path, evaluation_metrics)

    # Save individual metric arrays for comparison script 单独保存每个指标的数组
    np.save(os.path.join(output_dir, "evaluate/evaluation_rewards.npy"), np.array(evaluation_rewards))
    np.save(os.path.join(output_dir, "evaluate/evaluation_latencies.npy"), np.array(evaluation_latencies))
    np.save(os.path.join(output_dir, "evaluate/evaluation_energy.npy"), np.array(evaluation_energy))

    # Save text summary  保存文本格式的评估摘要
    with open(summary_path, "w") as f:
        f.write("=== MASAC Evaluation Summary ===\n")
        f.write(f"Success Rate: {success_rate:.2f}\n")
        f.write(f"Average Reward: {reward_mean:.2f} ± {reward_std:.2f}\n")
        f.write(f"Average Latency: {latency_mean:.2f} ± {latency_std:.2f}\n")
        f.write(f"Average Energy: {energy_mean:.2f} ± {energy_std:.2f}\n")

    # Plotting 创建一个包含3个子图的图表，分别展示奖励、延迟和能量消耗
    plt.figure(figsize=(10, 15))
    # 每回合奖励曲线
    plt.subplot(3, 1, 1)
    plt.plot(evaluation_rewards, label="Reward", color='blue')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("MASAC Evaluation Rewards")
    plt.grid(True)
    # 每回合执行延迟曲线
    plt.subplot(3, 1, 2)
    plt.plot(evaluation_latencies, label="Latency", color='red')
    plt.xlabel("Episodes")
    plt.ylabel("Latency (s)")
    plt.title("MASAC Evaluation Latency")
    plt.grid(True)
    # 每回合能量消耗曲线
    plt.subplot(3, 1, 3)
    plt.plot(evaluation_energy, label="Energy", color='green')
    plt.xlabel("Episodes")
    plt.ylabel("Energy")
    plt.title("MASAC Evaluation Energy Consumption")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluate/MASACevaluation_plots.png"))
    plt.show()

    print("✅ MASAC Evaluation complete. All results saved to:", output_dir)
