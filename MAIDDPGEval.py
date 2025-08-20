import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from fogenv import FogToFogEnv
from MAIDDPG import MAIDDPGAgent  # Updated import
import time
if __name__ == "__main__":
    # Output directory
    output_dir = "./output_lxl/MAIDDPG/"
    metrics_path = os.path.join(output_dir, "evaluate/evaluation_metrics.npy")
    summary_path = os.path.join(output_dir, "evaluate/evaluation_summary.txt")
    os.makedirs(output_dir, exist_ok=True)

    # Setup
    num_agents = 6
    state_dim = 12  # 状态维度（例如：6个雾节点×2个特征）
    action_dim = 6  # 动作维度（例如：6个雾节点的资源分配）
    max_action = 1  # 动作最大值（用于动作缩放，需与训练时一致）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last_episode = 10000
    num_episodes = 100  # 评估的总回合数

    # 加载训练好的智能体模型
    agents = [MAIDDPGAgent(state_dim, action_dim, max_action, i) for i in range(num_agents)]
    # 为每个智能体加载训练好的Actor模型（策略网络，用于决策）
    for agent_id, agent in enumerate(agents):
        actor_path = os.path.join(output_dir, f"weight/actor_agent_{agent_id}_ep_{last_episode}.pth")

        if os.path.exists(actor_path):
            agent.actor.load_state_dict(torch.load(actor_path, map_location=device))

    # 开始评估
    env = FogToFogEnv()
    # 记录评估指标的列表  每回合奖励、每回合耗时（或延迟）、每回合能量消耗、成功任务数
    evaluation_rewards, evaluation_latencies, evaluation_energy, successful_tasks_list = [], [], [], []
    # 循环评估指定回合数
    for episode in range(num_episodes):
        # 重置环境，获取初始状态
        state = env.reset()
        done = False  # 回合结束标志

        # 初始化当前回合的统计变量
        total_reward = 0  # 总奖励
        total_latency = 0  # 总延迟（环境反馈的延迟）
        total_energy = 0  # 总能量消耗
        start_time = time.time()  # 记录回合开始时间（用于计算实际耗时）
        # 单回合内的交互循环（直到回合结束）
        while not done:
            # 所有智能体根据当前状态选择动作（无探索，纯利用训练好的策略）
            actions = np.array([agents[i].select_action(state[i]) for i in range(num_agents)])
            # 环境执行动作，返回下一状态、奖励、是否结束、额外信息
            next_state, rewards, done, info = env.step(actions)

            # 累积当前回合的指标
            total_reward += float(rewards)  # 累加奖励
            total_latency += float(info.get("Total Latency", 0))  # 累加环境反馈的延迟
            total_energy += float(info.get("Total Energy", 0))  # 累加环境反馈的能量消耗
            state = next_state  # 更新状态，准备下一时刻的决策

        # 计算当前回合的评估指标并记录
        end_time = time.time()
        evaluation_latencies.append(time.time() - start_time + 0.05)  # 实际耗时（加0.05可能是校准值）
        evaluation_rewards.append(total_reward / 100)  # 奖励归一化
        # evaluation_latencies.append(total_latency/100)
        evaluation_energy.append(total_energy / 10000)  # 能量归一化
        successful_tasks_list.append(info.get("Successful Tasks", 0))
        print(
            f"评估回合 {episode + 1}/{num_episodes} 完成，奖励: {evaluation_rewards[-1]:.2f}，延迟: {evaluation_latencies[-1]:.4f}s")

    # 评估结果统计与保存
    # 计算评估指标的均值和标准差
    reward_mean, reward_std = np.mean(evaluation_rewards), np.std(evaluation_rewards)
    latency_mean, latency_std = np.mean(evaluation_latencies), np.std(evaluation_latencies)
    energy_mean, energy_std = np.mean(evaluation_energy), np.std(evaluation_energy)
    tasks_mean = np.mean(successful_tasks_list)

    # 1. 保存所有评估指标（.npy格式，方便后续加载分析）
    evaluation_metrics = {
        "episode_rewards": evaluation_rewards,
        "episode_latencies": evaluation_latencies,
        "episode_energies": evaluation_energy,
        "successful_tasks": successful_tasks_list
    }
    np.save(metrics_path, evaluation_metrics)

    # 2. 保存单独的指标数组（方便与其他算法对比）
    np.save(os.path.join(output_dir + "evaluate/", "evaluation_rewards.npy"), np.array(evaluation_rewards))
    np.save(os.path.join(output_dir + "evaluate/", "evaluation_latencies.npy"), np.array(evaluation_latencies))
    np.save(os.path.join(output_dir + "evaluate/", "evaluation_energy.npy"), np.array(evaluation_energy))
    np.save(os.path.join(output_dir + "evaluate/", "successful_tasks.npy"), np.array(successful_tasks_list))

    # 3. 保存文本格式的评估摘要（方便快速查看关键结果）
    with open(summary_path, "w") as f:
        f.write("=== Evaluation Summary (MAIDDPG) ===\n")
        f.write(f"Average Reward: {reward_mean:.2f} ± {reward_std:.2f}\n")
        f.write(f"Average Latency: {latency_mean:.4f}s ± {latency_std:.4f}\n")
        f.write(f"Average Energy: {energy_mean:.4f} ± {energy_std:.4f}\n")
        # f.write(f"Average Successful Tasks: {tasks_mean:.2f} / {env.max_tasks_per_episode if hasattr(env, 'max_tasks_per_episode') else 'unknown'}\n")

    # 绘制评估指标图表
    plt.figure(figsize=(6, 10))
    # 子图1：每回合奖励
    plt.subplot(4, 1, 1)
    plt.plot(evaluation_rewards, label="Reward", color='blue')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Evaluation Rewards")
    plt.grid(True)
    # 子图2：每回合延迟
    plt.subplot(4, 1, 2)
    plt.plot(evaluation_latencies, label="Latency", color='red')
    plt.xlabel("Episodes")
    plt.ylabel("Latency (s)")
    plt.title("Evaluation Latency")
    plt.grid(True)
    # 子图3：每回合能量消耗
    plt.subplot(4, 1, 3)
    plt.plot(evaluation_energy, label="Energy", color='green')
    plt.xlabel("Episodes")
    plt.ylabel("Energy")
    plt.title("Evaluation Energy Consumption")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir + "evaluate/", "evaluation_plots.png"))
    plt.show()

    print("✅ MAIDDPG Evaluation complete. Results saved to:", output_dir)

