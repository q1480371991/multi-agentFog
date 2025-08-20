import os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import optim
import numpy as np
from networks import Actor, Critic
from replay_buffer import ReplayBuffer
from fogenv import FogToFogEnv
import matplotlib.pyplot as plt
from multiprocessing import Queue, Process

class MAIDDPGAgent:
    """多智能体深度确定性策略梯度（MAIDDPG）智能体类"""
    def __init__(self, state_dim, action_dim, max_action, agent_id, buffer_size=int(1e6)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化策略网络（Actor）和价值网络（Critic）
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)# 输出具体动作
        self.critic = Critic(state_dim, action_dim).to(self.device) # 评估动作价值

        # 初始化目标网络（用于稳定训练）
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        # 目标网络初始参数与主网络一致
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 定义优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)# 策略网络优化器
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)# 价值网络优化器
        # 超参数
        self.gamma = 0.99 # 折扣因子，权衡当前和未来奖励
        self.tau = 0.005 # 目标网络软更新系数
        self.agent_id = agent_id# 智能体编号（多智能体区分）
        # 初始化经验回放缓冲区
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=buffer_size)
        self.batch_size = 64# 每次更新的样本数量

    def select_action(self, state):
        """根据当前状态选择动作（评估模式，无梯度计算）"""
        self.actor.eval() # 切换到评估模式（关闭dropout等）
        # 状态转为张量并添加批次维度
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():# 禁用梯度计算
            action = self.actor(state)# 策略网络输出动作
        self.actor.train()# 切回训练模式
        return action.cpu().numpy()[0] # 转为numpy数组返回

    def store_experience(self, state, action, reward, next_state, done):
        """将经验（s,a,r,s',done）存入回放缓冲区"""
        self.replay_buffer.add(state, action, next_state, reward, done)

    def update(self):
        """从回放缓冲区采样并更新网络参数"""
        # 缓冲区样本不足时不更新
        if self.replay_buffer.size() < self.batch_size:
            return
        # 从缓冲区采样批次数据
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(self.batch_size)
        # 转为张量并移动到计算设备
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)# 增加维度匹配
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)

        # 计算目标Q值（使用目标网络）
        with torch.no_grad(): # 无梯度计算
            next_actions = self.actor_target(next_states)# 目标策略网络生成下一状态动作
            target_q = self.critic_target(next_states, next_actions) # 目标价值网络评估下一状态价值
            # 目标Q值 = 即时奖励 + 折扣因子 * 下一状态价值（若未结束）
            target_q_value = rewards + (1 - dones) * self.gamma * target_q
        # 计算当前Q值（主网络）并计算价值网络损失
        current_q = self.critic(states, actions) # 主价值网络评估当前动作价值
        critic_loss = F.mse_loss(current_q, target_q_value) # 均方误差损失

        # 更新价值网络
        self.critic_optimizer.zero_grad()# 清空梯度
        critic_loss.backward() # 反向传播计算梯度
        self.critic_optimizer.step() # 优化器更新参数

        # 计算策略网络损失（最大化价值网络评估的当前动作价值）
        actor_loss = -self.critic(states, self.actor(states)).mean() # 负号表示最大化
        # 更新策略网络
        self.actor_optimizer.zero_grad()# 清空梯度
        actor_loss.backward()# 反向传播计算梯度
        self.actor_optimizer.step()# 优化器更新参数
        # 软更新目标网络（缓慢跟踪主网络参数，提高稳定性）
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, source, target):
        """软更新目标网络参数：target = tau*source + (1-tau)*target"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

def run_agent(agent_id, state_q, action_q, reward_q, next_state_q, done_q, agent_args):
    """多进程中运行单个智能体的函数（并行处理）"""
    # 初始化智能体
    agent = MAIDDPGAgent(*agent_args, agent_id)
    while True:
        state = state_q.get()# 从队列获取状态
        if state is None:# 收到终止信号时退出
            break
        action = agent.select_action(state) # 选择动作
        action_q.put((agent_id, action)) # 将动作放入队列
        # 从队列获取奖励、下一状态和结束标志
        reward, next_state, done = reward_q.get(), next_state_q.get(), done_q.get()
        # 存储经验并更新网络
        agent.store_experience(state, action, reward, next_state, done)
        agent.update()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) # 设置多进程启动方式
    # output_dir = "/home/natnael/Desktop/mult-agentFog/output/MAIDDPG/"
    output_dir = "./output_lxl/MAIDDPG/"
    os.makedirs(output_dir, exist_ok=True)

    # 环境和智能体配置
    num_agents = 6
    env = FogToFogEnv()
    state_dim = 12# 状态维度（例如：6个雾节点×2个特征）
    action_dim = 6 # 动作维度（例如：6个雾节点的资源分配）
    # 初始化进程间通信队列
    state_queues = [Queue() for _ in range(num_agents)] # 状态队列
    action_queues = [Queue() for _ in range(num_agents)]# 动作队列
    reward_queues = [Queue() for _ in range(num_agents)] # 奖励队列
    next_state_queues = [Queue() for _ in range(num_agents)] # 下一状态队列
    done_queues = [Queue() for _ in range(num_agents)]# 结束标志队列
    # 初始化智能体列表
    agents = [MAIDDPGAgent(state_dim, action_dim, 1, i) for i in range(num_agents)]
    # 启动多进程（每个智能体一个进程）
    processes = [Process(target=run_agent, args=(i, state_queues[i], action_queues[i], reward_queues[i], next_state_queues[i], done_queues[i], (state_dim, action_dim, 1))) for i in range(num_agents)]

    for p in processes:
        p.start()
    # 训练参数
    num_episodes = 10000# 总训练回合数
    # 记录训练指标
    episode_rewards, episode_latencies, episode_energies, episode_successful_Tasks = [], [], [], []
    # 开始训练循环
    for episode in range(num_episodes):
        state = env.reset()# 重置环境，获取初始状态
        done = False# 回合结束标志
        # 初始化当前回合指标
        episode_reward, episode_latency, episode_energy, episode_successful, time_steps = 0, 0, 0, 0, 0
        # 单回合内交互循环
        while not done:
            # 所有智能体选择动作
            actions = [agent.select_action(state) for agent in agents]
            # 环境执行动作，获取反馈
            next_state, reward, done, info = env.step(actions)
            # 处理环境信息（归一化指标）
            info["Total Latency"] /= 100# 延迟归一化
            info["Total Energy"] /= 10000# 能量归一化
            info["Successful Tasks"]
            # 计算集体奖励（多智能体共享）
            collective_reward = sum(reward) / num_agents if isinstance(reward, (list, np.ndarray)) else reward
            individual_rewards = [collective_reward] * num_agents# 每个智能体获得相同奖励
            # 向各智能体的队列中放入经验数据
            for i in range(num_agents):
                reward_queues[i].put(reward[i] if isinstance(reward, (list, np.ndarray)) else reward)
                next_state_queues[i].put(next_state)
                done_queues[i].put(done)
            # 更新状态和回合指标
            state = next_state
            #episode_reward += reward / num_agents if isinstance(reward, (list, np.ndarray)) else reward
            episode_reward += reward.item()# 累积奖励
            episode_latency += info.get("Total Latency", 0)# 累积延迟
            episode_energy += info.get("Total Energy", 0)  # 累积能量消耗
            episode_successful += info.get("Successful Tasks", 0) # 累积成功任务数
            time_steps += 1# 步数计数
        # 记录回合指标（平均每步奖励）
        episode_rewards.append(episode_reward / time_steps)
        episode_latencies.append(episode_latency)
        episode_energies.append(episode_energy)
        episode_successful_Tasks.append(episode_successful)

        print(f"Episode {episode+1}, Total Reward: {episode_reward}, Total Latency: {episode_latency}, Total Energy: {episode_energy}, successful_Tasks: {episode_successful}")
        # 只保存最后一轮的模型
        # if (episode + 1) % 100 == 0 or episode + 1 == num_episodes:
        if  episode + 1 == num_episodes:
            for agent_id, agent in enumerate(agents):
                torch.save(agent.actor.state_dict(), os.path.join(output_dir, f"weight/actor_agent_{agent_id}_ep_{episode+1}.pth"))
                torch.save(agent.critic.state_dict(), os.path.join(output_dir, f"weight/critic_agent_{agent_id}_ep_{episode+1}.pth"))
            print(f"Models saved at episode {episode+1}")
    # 保存训练指标为numpy文件
    np.save(os.path.join(output_dir, "train/training_metrics.npy"), {
        "episode_rewards": episode_rewards,
        "episode_latencies": episode_latencies,
        "episode_energies": episode_energies,
        "episode_successful": episode_successful_Tasks
    })
    print(f"Training metrics saved to {output_dir}/train/training_metrics.npy")

    # 绘制训练指标图表
    plt.figure(figsize=(15, 10))

    # 子图1：回合奖励
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, label="Episode Rewards", color='b')
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    plt.title("Episode Rewards")

    # 子图2：回合延迟
    plt.subplot(2, 2, 2)
    plt.plot(episode_latencies, label="Episode Latencies", color='r')
    plt.xlabel("Episodes")
    plt.ylabel("Latencies (s)")
    plt.legend()
    plt.title("Episode Latencies")

    # 子图3：回合能量消耗
    plt.subplot(2, 2, 3)
    plt.plot(episode_energies, label="Episode Energies", color='g')
    plt.xlabel("Episodes")
    plt.ylabel("Energy Consumption")
    plt.legend()
    plt.title("Episode Energy Consumption")

    # 子图4：成功任务数
    plt.subplot(2, 2, 4)
    plt.plot(episode_successful_Tasks, label="Successful Tasks", color='m')
    plt.xlabel("Episodes")
    plt.ylabel("Successful Tasks")
    plt.legend()
    plt.title("Successful Tasks per Episode")
    plt.show()
    plt.close()

    print("Training and evaluation complete. Results saved to", output_dir)
        






        
