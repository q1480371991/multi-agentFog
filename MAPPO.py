import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import multiprocessing as mp
from multiprocessing import Process, Queue
import torch.nn.functional as F
from fogenv import FogToFogEnv
import matplotlib.pyplot as plt
"""
Multi-Agent Proximal Policy Optimization (MAPPO) 核心实现
适用场景：雾计算环境中的多智能体资源分配/任务调度优化
功能：通过PPO算法优化多智能体的联合策略，最小化延迟、最大化资源利用率等
"""


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    """策略网络（Actor）：输入状态，输出动作分布的均值和标准差"""
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_log_std = nn.Linear(256, action_dim)# 输出动作标准差的对数

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.sigmoid(self.fc_mean(x))  # 用sigmoid确保动作在[0,1]范围内
        log_std = self.fc_log_std(x).clamp(-20, 2)# 限制标准差范围，保证数值稳定性
        return mean, log_std


    def sample(self, state):
        """从动作分布中采样动作，并计算对数概率"""
        mean, log_std = self.forward(state)
        std = log_std.exp()# 标准差（指数化对数标准差）
        dist = torch.distributions.Normal(mean, std)# 正态分布
        action = dist.rsample() # 重参数化采样（便于梯度回传）
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True) # 动作对数概率总和
        return action, log_prob

class Critic(nn.Module):
    """价值网络（Critic）：输入状态，输出状态价值（V值）"""
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)# 输出状态价值

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
class GaussianNoise:
    """高斯噪声生成器：用于探索阶段增加动作随机性"""
    def __init__(self, action_dim, mu=0, sigma=0.2, sigma_decay=0.98, sigma_min=0.01):
        self.action_dim = action_dim# 动作维度
        self.mu = mu # 噪声均值
        self.sigma = sigma # 初始标准差
        self.sigma_decay = sigma_decay# 标准差衰减系数
        self.sigma_min = sigma_min # 最小标准差

    def reset(self):
        pass  # 高斯噪声无需重置状态

    def sample(self):
        """生成高斯噪声，并衰减标准差（逐渐减少探索）"""
        noise = np.random.normal(self.mu, self.sigma, self.action_dim)
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)  # Decay sigma
        return noise
class MAPPOAgent:
    """MAPPO智能体类：每个智能体包含独立的Actor和Critic，以及经验回放和更新逻辑"""
    def __init__(self, state_dim, action_dim, agent_id):

        self.device = torch.device("cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.gamma = 0.99# 折扣因子（未来奖励的衰减系数）
        self.epsilon = 0.2  # PPO剪辑范围（限制策略更新幅度）
        self.agent_id = agent_id # 智能体ID
        self.replay_buffer = []# 经验回放缓冲区
    
    def select_action(self, state):
        """根据当前状态选择动作（带探索）"""
        self.actor.eval() # 切换到评估模式（关闭dropout等）
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():# 不计算梯度，节省资源
            action, log_prob = self.actor.sample(state)
        self.actor.train() # 切换回训练模式
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]# 转换为numpy数组返回
    
    def store_experience(self, state, action, reward, next_state, done, log_prob):
        """存储经验到回放缓冲区"""
        self.replay_buffer.append((state, action, reward, next_state, done, log_prob))
        # 限制缓冲区大小，防止内存溢出
        if len(self.replay_buffer) > 100000:
            self.replay_buffer.pop(0)
    
    def update(self):
        """更新策略网络和价值网络（PPO核心逻辑）"""
        if len(self.replay_buffer) < 1000: # 缓冲区数据不足时不更新
            return
        # 随机采样批次数据
        batch = random.sample(self.replay_buffer, 64)
        states, actions, rewards, next_states, dones, log_probs_old = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).unsqueeze(1)

        # 计算目标价值（TDtarget）
        with torch.no_grad():
            target_values = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        # 计算优势函数（Advantage）
        values = self.critic(states)
        advantages = target_values - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)# 优势归一化

        # 计算新的动作和对数概率
        new_actions, new_log_probs = self.actor.sample(states)
        ratio = torch.exp(new_log_probs - log_probs_old) # 新旧策略概率比

        # 熵正则化（鼓励探索）
        entropy_loss = - (torch.exp(new_log_probs) * new_log_probs).mean()  # Entropy regularization
        # PPO剪辑损失
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        actor_loss -= 0.01 * entropy_loss  # 加入熵正则化

        # 价值网络损失（MSE）
        critic_loss = F.mse_loss(values, target_values)

        # 更新策略网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新价值网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

# Function to run an agent in parallel
def run_agent(agent_id, state_queue, action_queue, reward_queue, next_state_queue, done_queue, agents, action_dim):
    agent = agents[agent_id]
    noise = GaussianNoise(action_dim, sigma=0.1, sigma_decay=0.98, sigma_min=0.001) # 噪声生成器
    while True:
        state = state_queue.get() # 从队列获取状态
        if state is None:  # 收到终止信号时退出
            break

        # 选择动作（带噪声探索）
        action = agent.select_action(state, noise)
        action_queue.put((agent_id, action, state))

        # 获取环境反馈
        reward = reward_queue.get()
        next_state = next_state_queue.get()
        done = done_queue.get()

        # 存储经验并更新
        agent.store_experience(state, action, reward, next_state)
        if len(agent.replay_buffer) >= agent.batch_size:
            agent.update()    

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)# 设置多进程启动方式
    # output_dir = "/home/natnael/Desktop/mult-agentFog/output/MAPPO/"
    output_dir = "./output_lxl/MAPPO/"
    os.makedirs(output_dir, exist_ok=True)
    
    num_agents = 6# 智能体数量
    env = FogToFogEnv() # 初始化雾计算环境
    state_dim = 12  # 状态维度（6个雾节点 × 2个特征：CPU和带宽）
    action_dim = 6  # 动作维度（6个雾节点的资源分配）
    experience_queue = mp.Queue()
    stop_event = mp.Event()

    num_episodes = 10000# 训练总回合数
    state_queues = [Queue() for _ in range(num_agents)]
    action_queues = [Queue() for _ in range(num_agents)]
    reward_queues = [Queue() for _ in range(num_agents)]
    next_state_queues = [Queue() for _ in range(num_agents)]
    done_queues = [Queue() for _ in range(num_agents)]
    # 创建智能体列表
    agents = [MAPPOAgent(state_dim=12, action_dim=6, agent_id=i) for i in range(num_agents)]
    # 创建并启动多进程（每个智能体一个进程）
    processes = [Process(target=run_agent, args=(i, state_queues, action_queues, reward_queues, next_state_queues, done_queues, agents, action_dim)) for i in range(num_agents)]
    for p in processes:
        p.start()

    # 记录训练指标
    episode_rewards, episode_latencies, episode_energies, episode_successful_Tasks = [], [], [], []
    for episode in range(num_episodes):
        state = env.reset() # 重置环境
        done = False
        episode_reward, episode_latency, episode_energy, episode_successful, time_steps = 0, 0, 0, 0, 0
        while not done:
            # 所有智能体选择动作
            actions, log_probs = zip(*[agent.select_action(state) for agent in agents])
            # 环境执行动作，获取反馈
            next_state, reward, done, info = env.step(actions)
            # 归一化指标（根据环境特性调整）
            info["Total Latency"] /= 100
            info["Total Energy"] /= 10000
            info["Successful Tasks"]
            # 计算集体奖励（平均每个智能体的奖励）
            collective_reward = sum(reward) / num_agents if isinstance(reward, (list, np.ndarray)) else reward
            individual_rewards = [collective_reward] * num_agents
            # 向每个智能体的队列发送反馈
            for i in range(num_agents):
                reward_queues[i].put(reward[i] if isinstance(reward, (list, np.ndarray)) else reward)
                next_state_queues[i].put(next_state)
                done_queues[i].put(done)
            # 更新状态和指标
            state = next_state
            episode_reward += reward.item()
            episode_latency += info.get("Total Latency", 0)
            episode_energy += info.get("Total Energy", 0)
            episode_successful += info.get("Successful Tasks", 0)
            time_steps += 1
        # 记录每回合的平均指标
        episode_rewards.append(episode_reward / time_steps)
        episode_latencies.append(episode_latency)
        episode_energies.append(episode_energy)
        episode_successful_Tasks.append(episode_successful)
        # 打印训练进度
        print(f"Episode {episode+1}, Total Reward: {episode_reward}, Total Latency: {episode_latency}, Total Energy: {episode_energy}, Successful Tasks: {episode_successful}")
        # 定期保存模型
        # if (episode + 1) % 100 == 0 or episode + 1 == num_episodes:
        # 只保存最后一轮的模型
        if episode + 1 == num_episodes:
            for agent_id, agent in enumerate(agents):
                torch.save(agent.actor.state_dict(), os.path.join(output_dir, f"weight/actor_agent_{agent_id}_ep_{episode+1}.pth"))
                torch.save(agent.critic.state_dict(), os.path.join(output_dir, f"weight/critic_agent_{agent_id}_ep_{episode+1}.pth"))
            print(f"Models saved at episode {episode+1}")

    # 保存训练指标
    metrics = {
        "episode_rewards": episode_rewards,
        "episode_latencies": episode_latencies,
        "episode_energies": episode_energies,
        "episode_successful": episode_successful_Tasks
        }
    np.save(os.path.join(output_dir, "train/training_metrics.npy"), metrics)
    print(f"Training metrics saved to {output_dir}train/training_metrics.npy")

    # 绘制训练指标图表
    plt.figure(figsize=(15, 10))

    # 绘制奖励曲线
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, label="Episode Rewards", color='b')
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    plt.title("Episode Rewards")

    # 绘制延迟曲线
    plt.subplot(2, 2, 2)
    plt.subplot(2, 2, 2)
    plt.plot(episode_latencies, label="Episode Latencies", color='r')
    plt.xlabel("Episodes")
    plt.ylabel("Latencies (s)")
    plt.legend()
    plt.title("Episode Latencies")

    # 绘制能量曲线
    plt.subplot(2, 2, 3)
    plt.plot(episode_energies, label="Episode Energies", color='g')
    plt.xlabel("Episodes")
    plt.ylabel("Energy Consumption")
    plt.legend()
    plt.title("Episode Energy Consumption")

    # 绘制成功任务数曲线
    plt.subplot(2, 2, 4)
    plt.plot(episode_successful_Tasks, label="Successful Tasks", color='m')
    plt.xlabel("Episodes")
    plt.ylabel("Successful Tasks")
    plt.legend()
    plt.title("Successful Tasks per Episode")
    plt.show()
    plt.close()

    print("Training and evaluation complete. Results saved to", output_dir)
        






        
