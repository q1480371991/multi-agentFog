import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import multiprocessing
from multiprocessing import Process, Queue
import torch.nn.functional as F
from fogenv import FogToFogEnv
from networks import Actor, Critic# Actor和Critic网络定义
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Actor和Critic网络的备用定义（若networks.py中未定义时使用）
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, max_action):
#         super(Actor, self).__init__()
# 三层全连接网络：输入状态 -> 256维 -> 256维 -> 动作输出
#         self.net = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, action_dim),
#             nn.Sigmoid()  # Output in range (0,1)# 输出范围(0,1)，与max_action配合控制动作幅度
#         )
#         self.max_action = max_action# 动作最大值（用于缩放输出）
    
#     def forward(self, state):
#         return self.net(state) * self.max_action# 缩放动作到指定范围

# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()
#         # 双Q网络结构（Q1和Q2），用于减轻过估计问题
#         self.q1 = nn.Sequential(
#             nn.Linear(state_dim + action_dim, 256),# 输入为状态+动作拼接
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)# 输出Q值
#         )
#         self.q2 = nn.Sequential(
#             nn.Linear(state_dim + action_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
    
#     def forward(self, state, action):
#         sa = torch.cat([state, action], dim=-1) # 拼接状态和动作
#         return self.q1(sa), self.q2(sa) # 返回两个Q值
class GaussianNoise:
    """高斯噪声生成器，用于探索策略空间"""
    def __init__(self, action_dim, mu=0, sigma=0.2, sigma_decay=0.99, sigma_min=0.01):
        self.action_dim = action_dim  # 动作维度
        self.mu = mu  # 噪声均值
        self.sigma = sigma   # 初始标准差（控制探索强度）
        self.sigma_decay = sigma_decay # 噪声衰减因子（随训练减少探索）
        self.sigma_min = sigma_min  # 最小标准差（保证一定探索）

    def sample(self):
        """
        Samples noise based on the Gaussian distribution.
        The noise is added to the action chosen by the policy.
        """
        """生成高斯噪声，并衰减标准差"""
        noise = np.random.normal(self.mu, self.sigma, self.action_dim)
        # Decay the noise sigma after every call to encourage exploration reduction  衰减噪声强度，平衡探索与利用
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)
        return noise

    def reset(self):
        """Resets the noise parameters, useful for resetting the environment or training phases."""
        """重置噪声参数"""
        self.sigma = self.initial_sigma

# ReplayBuffer的备用定义（若replay_buffer.py中未定义时使用）
# class ReplayBuffer:
#     def __init__(self, state_dim, action_dim, max_size=int(1e6)):
#         self.state_dim = state_dim# 状态维度
#         self.action_dim = action_dim# 动作维度
#         self.buffer = [] # 存储经验的列表
#         self.max_size = max_size# 缓冲区最大容量
#         self.pointer = 0# 循环指针（用于覆盖旧经验）
    
    def add(self, state, action, next_state, reward, done):
        """添加经验到缓冲区"""
        if len(self.buffer) < self.max_size:
            self.buffer.append((state, action, next_state, reward, done))
        else:
            self.buffer[self.pointer] = (state, action, next_state, reward, done)
        self.pointer = (self.pointer + 1) % self.max_size# 循环更新指针
    
    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards), np.array(dones)
    
    def size(self):
        """返回当前缓冲区大小"""
        return len(self.buffer)

class MATD3Agent:
    """多智能体TD3（MATD3）智能体类"""
    def __init__(self, state_dim, action_dim, max_action, agent_id, buffer_size=int(1e6)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化策略网络（Actor）和价值网络（Critic1、Critic2）
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)

        # 初始化目标网络（用于稳定训练）
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim).to(self.device)

        # 复制初始参数到目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)# 策略优化器
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=1e-3# 价值网络优化器（合并参数）
        )
        # 算法超参数
        self.gamma = 0.99# 折扣因子（未来奖励权重）
        self.tau = 0.005# 软更新系数（目标网络更新幅度）
        self.agent_id = agent_id# 智能体ID（用于区分多智能体）

        # 初始化经验回放缓冲区
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=buffer_size)
        self.batch_size = 64# 每次更新采样的批量大小
        
    def select_action(self, state):
        """根据当前状态选择动作（用于评估时无噪声输出）"""
        self.actor.eval()# 切换到评估模式（关闭dropout等）
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)# 转换为张量并增加批次维度
        with torch.no_grad():
            action = self.actor(state)
        self.actor.train()# 切换回训练模式
        return action.cpu().numpy()[0]# 转换为numpy数组并返回

    def store_experience(self, state, action, reward, next_state, done):
        # Store experience in replay buffer 将经验存储到回放缓冲区
        self.replay_buffer.add(state, action, next_state, reward, done)

    def update(self):
        """更新网络参数（核心训练逻辑）"""
        if self.replay_buffer.size() < self.batch_size:
            # 缓冲区数据不足时不更新
            return (0.0, 0.0, 0.0)  # Return valid values instead of None  返回默认损失值

        # Sample experiences   从缓冲区采样经验
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(self.batch_size)
        # 转换为张量并移动到指定设备
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)# 增加维度适配网络输出
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)

        # Compute target Q-values  计算目标Q值（使用目标网络）
        with torch.no_grad():# 目标网络不计算梯度
            next_actions = self.actor_target(next_states) # 目标策略的下一动作
            next_q1 = self.critic1_target(next_states, next_actions) # 目标Q1值
            next_q2 = self.critic2_target(next_states, next_actions)# 目标Q2值
            # 取两个Q值的最小值（TD3的关键改进，减轻过估计）
            target_q_value = rewards + (1 - dones) * self.gamma * torch.min(next_q1, next_q2)

        # Compute critic losses 计算当前Q值和价值网络损失
        q1 = self.critic1(states, actions) # 当前Q1值
        q2 = self.critic2(states, actions)# 当前Q2值
        critic1_loss = F.mse_loss(q1, target_q_value)# Q1损失（均方误差）
        critic2_loss = F.mse_loss(q2, target_q_value)# Q2损失（均方误差）

        # Update critics  更新价值网络
        self.critic_optimizer.zero_grad()# 清空梯度
        (critic1_loss + critic2_loss).backward()# 反向传播计算梯度
        self.critic_optimizer.step()# 更新参数
        # 初始化策略损失
        actor_loss = 0.0  # Default to 0 if no actor update

        # Delayed policy update TD3的延迟策略更新（每2步更新一次策略，提高稳定性）
        if self.agent_id % 2 == 0:
            # 策略损失：最大化价值网络评估（使用Q1网络）
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            # 更新策略网络
            self.actor_optimizer.zero_grad()# 清空梯度
            actor_loss.backward()# 反向传播
            self.actor_optimizer.step() # 更新参数

            # Soft update targets  软更新目标网络（缓慢跟踪主网络）
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)

        return critic1_loss.item(), critic2_loss.item(), actor_loss.item()  # Always return values   返回损失值

    def soft_update(self, source, target):
        """软更新目标网络参数（目标 = τ*源 + (1-τ)*目标）"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

# Function to run the agent in parallel
def run_agent(agent_id, state_queue, action_queue, reward_queue, next_state_queue, done_queue, agents, action_dim):
    """多进程训练的子进程函数（每个智能体一个进程）"""
    agent = agents[agent_id]
    while True:
        state = state_queue.get()# 从队列获取状态
        if state is None:  # Sentinel value to stop the process  收到终止信号时退出
            break

        # Select action   选择动作并发送到主进程
        action = agent.select_action(state)
        action_queue.put((agent_id, action, state))

        # Get reward and next state   从队列获取奖励、下一状态和终止标志
        reward = reward_queue.get()
        next_state = next_state_queue.get()
        done = done_queue.get()

        # Store experience and update  存储经验并更新网络
        agent.store_experience(state, action, reward, next_state, done)
        if len(agent.replay_buffer) >= agent.batch_size:
            agent.update()

# Main function for parallel training
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    # output_dir = "/home/natnael/Desktop/mult-agentFog/output/MATD/"
    output_dir = "./output_lxl/MATD3/"
    os.makedirs(output_dir, exist_ok=True)
    
    num_agents = 6
    env = FogToFogEnv()
    state_dim = 12  # 6 fog nodes * 2 (CPU + bandwidth) 状态维度（6个节点 × 2个特征：CPU和带宽）
    action_dim = 6  # 6 fog nodes  动作维度（6个节点的任务分配策略）
    experience_queue = multiprocessing.Queue()# 经验队列（预留）
    stop_event = multiprocessing.Event() # 进程终止信号

    num_episodes = 10000
    state_queue = Queue()
    action_queue = Queue()
    reward_queue = Queue()
    next_state_queue = Queue()
    done_queue = Queue()
    info_queue = Queue()
    # 创建6个MATD3智能体
    agents = [MATD3Agent(state_dim=state_dim, action_dim=action_dim, max_action=1, agent_id=i) for i in range(num_agents)]
    # 启动多进程（每个智能体一个进程）
    processes = [Process(target=run_agent, args=(i, state_queue, action_queue, reward_queue, next_state_queue, done_queue, agents, action_dim)) for i in range(num_agents)]
    for p in processes:
        p.start()

    episode_rewards, episode_latencies, episode_energies, episode_successful_Tasks = [], [], [], []
    for episode in range(num_episodes):
        state = env.reset()# 重置环境，获取初始状态（包含所有智能体的观测）
        done = False
        episode_reward, episode_latency, episode_energy, episode_successful, time_steps = 0, 0, 0, 0, 0
        while not done:
            # 所有智能体根据当前状态选择动作（无探索噪声）
            actions = [agent.select_action(state) for agent in agents]
            # 环境执行联合动作，返回下一状态、奖励、终止标志和额外信息
            next_state, reward, done, info = env.step(actions)

            info["Total Latency"] /= 100#延迟指标标准化
            info["Total Energy"] /= 10000#能量指标标准化
            info["Successful Tasks"]#成功任务数记录
            # 计算集体奖励（全局奖励平均分配）
            collective_reward = sum(reward) / num_agents if isinstance(reward, (list, np.ndarray)) else reward
            # 生成每个智能体的个体奖励
            individual_rewards = [collective_reward] * num_agents
            # 向每个智能体的通信队列推送反馈
            for i in range(num_agents):
                reward_queue.put(reward[i] if isinstance(reward, (list, np.ndarray)) else reward)
                next_state_queue.put(next_state)
                done_queue.put(done)

            state = next_state

            episode_reward += reward.item()
            episode_latency += info.get("Total Latency", 0)
            episode_energy += info.get("Total Energy", 0)
            episode_successful += info.get("Successful Tasks", 0)
            time_steps += 1
        # 累积episode内的指标（奖励、延迟、能量、成功任务数）
        episode_rewards.append(episode_reward / time_steps)# 平均每步奖励
        episode_latencies.append(episode_latency)
        episode_energies.append(episode_energy)
        episode_successful_Tasks.append(episode_successful)


        print(f"Episode {episode+1}, Total Reward: {episode_reward}, Total Latency: {episode_latency}, Total Energy: {episode_energy}, Successful Tasks: {episode_successful}")

        # if (episode + 1) % 100 == 0 or episode + 1 == num_episodes:
        # 只保存最后一轮的模型
        if episode + 1 == num_episodes:
            for agent_id, agent in enumerate(agents):

                torch.save(agent.actor.state_dict(), os.path.join(output_dir, f"weight/actor_agent_{agent_id}_ep_{episode+1}.pth"))
                torch.save(agent.critic1.state_dict(), os.path.join(output_dir, f"weight/critic1_agent_{agent_id}_ep_{episode+1}.pth"))
                torch.save(agent.critic2.state_dict(), os.path.join(output_dir, f"weight/critic2_agent_{agent_id}_ep_{episode+1}.pth"))
            print(f"Models saved at episode {episode+1}")

        metrics = {
        "episode_rewards": episode_rewards,
        "episode_latencies": episode_latencies,
        "episode_energies": episode_energies,
        "episode_successful": episode_successful_Tasks
    }
    # 训练结束后保存所有指标
    np.save(os.path.join(output_dir, "train/training_metrics.npy"), metrics)
    print(f"Training metrics saved to {output_dir}train/training_metrics.npy")
    np.save(os.path.join(output_dir, "train/training_metrics.npy"), {"episode_rewards": episode_rewards, "episode_latencies": episode_latencies, "episode_energies": episode_energies, "episode_successful_Tasks": episode_successful_Tasks})
    plt.plot(episode_rewards, label="Rewards")
    plt.plot(episode_latencies, label="Latencies")
    plt.plot(episode_energies, label="Energies")
    plt.plot(episode_successful_Tasks, label="Successful Tasks")
    plt.legend()
    plt.show()
