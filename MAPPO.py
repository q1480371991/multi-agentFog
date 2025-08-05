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

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.sigmoid(self.fc_mean(x))  # Ensure actions are between 0 and 1
        log_std = self.fc_log_std(x).clamp(-20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
class GaussianNoise:
    def __init__(self, action_dim, mu=0, sigma=0.2, sigma_decay=0.98, sigma_min=0.01):
        self.action_dim = action_dim
        self.mu = mu
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min

    def reset(self):
        pass  # No state to reset for Gaussian noise

    def sample(self):
        noise = np.random.normal(self.mu, self.sigma, self.action_dim)
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)  # Decay sigma
        return noise
class MAPPOAgent:
    def __init__(self, state_dim, action_dim, agent_id):
        self.device = torch.device("cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.gamma = 0.99
        self.epsilon = 0.2  # PPO clip range
        self.agent_id = agent_id
        self.replay_buffer = []
    
    def select_action(self, state):
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.actor.sample(state)
        self.actor.train()
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]
    
    def store_experience(self, state, action, reward, next_state, done, log_prob):
        self.replay_buffer.append((state, action, reward, next_state, done, log_prob))
        if len(self.replay_buffer) > 100000:
            self.replay_buffer.pop(0)
    
    def update(self):
        if len(self.replay_buffer) < 1000:
            return
        batch = random.sample(self.replay_buffer, 64)
        states, actions, rewards, next_states, dones, log_probs_old = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).unsqueeze(1)
        
        with torch.no_grad():
            target_values = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        
        values = self.critic(states)
        advantages = target_values - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)  # Normalization
        
        new_actions, new_log_probs = self.actor.sample(states)
        ratio = torch.exp(new_log_probs - log_probs_old)
        
        entropy_loss = - (torch.exp(new_log_probs) * new_log_probs).mean()  # Entropy regularization
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        actor_loss -= 0.01 * entropy_loss  # Adding entropy term
        
        critic_loss = F.mse_loss(values, target_values)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

# Function to run an agent in parallel
def run_agent(agent_id, state_queue, action_queue, reward_queue, next_state_queue, done_queue, agents, action_dim):
    agent = agents[agent_id]
    noise = GaussianNoise(action_dim, sigma=0.1, sigma_decay=0.98, sigma_min=0.001)
    while True:
        state = state_queue.get()
        if state is None:  # Sentinel value to stop the process
            break

        # Select action
        action = agent.select_action(state, noise)
        action_queue.put((agent_id, action, state))

        # Get reward and next state
        reward = reward_queue.get()
        next_state = next_state_queue.get()
        done = done_queue.get()

        # Store experience and update
        agent.store_experience(state, action, reward, next_state)
        if len(agent.replay_buffer) >= agent.batch_size:
            agent.update()    

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    output_dir = "/home/natnael/Desktop/mult-agentFog/output/MAPPO/"
    os.makedirs(output_dir, exist_ok=True)
    
    num_agents = 6
    env = FogToFogEnv()
    state_dim = 12  # 6 fog nodes * 2 (CPU + bandwidth)
    action_dim = 6  # 6 fog nodes
    experience_queue = mp.Queue()
    stop_event = mp.Event()

    num_episodes = 10000
    state_queues = [Queue() for _ in range(num_agents)]
    action_queues = [Queue() for _ in range(num_agents)]
    reward_queues = [Queue() for _ in range(num_agents)]
    next_state_queues = [Queue() for _ in range(num_agents)]
    done_queues = [Queue() for _ in range(num_agents)]
    #info_queue = Queue()
    agents = [MAPPOAgent(state_dim=12, action_dim=6, agent_id=i) for i in range(num_agents)]
    processes = [Process(target=run_agent, args=(i, state_queues, action_queues, reward_queues, next_state_queues, done_queues, agents, action_dim)) for i in range(num_agents)]
    for p in processes:
        p.start()
    
   
    episode_rewards, episode_latencies, episode_energies, episode_successful_Tasks = [], [], [], []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward, episode_latency, episode_energy, episode_successful, time_steps = 0, 0, 0, 0, 0
        while not done:
            actions, log_probs = zip(*[agent.select_action(state) for agent in agents])
            next_state, reward, done, info = env.step(actions)
            info["Total Latency"] /= 100
            info["Total Energy"] /= 10000
            info["Successful Tasks"]
            collective_reward = sum(reward) / num_agents if isinstance(reward, (list, np.ndarray)) else reward
            individual_rewards = [collective_reward] * num_agents
            for i in range(num_agents):
                reward_queues[i].put(reward[i] if isinstance(reward, (list, np.ndarray)) else reward)
                next_state_queues[i].put(next_state)
                done_queues[i].put(done)

            state = next_state
            episode_reward += reward.item()
            episode_latency += info.get("Total Latency", 0)
            episode_energy += info.get("Total Energy", 0)
            episode_successful += info.get("Successful Tasks", 0)
            time_steps += 1
        
        episode_rewards.append(episode_reward / time_steps)
        episode_latencies.append(episode_latency)
        episode_energies.append(episode_energy)
        episode_successful_Tasks.append(episode_successful)

        print(f"Episode {episode+1}, Total Reward: {episode_reward}, Total Latency: {episode_latency}, Total Energy: {episode_energy}, Successful Tasks: {episode_successful}")

        if (episode + 1) % 100 == 0 or episode + 1 == num_episodes:
            for agent_id, agent in enumerate(agents):
                torch.save(agent.actor.state_dict(), os.path.join(output_dir, f"actor_agent_{agent_id}_ep_{episode+1}.pth"))
                torch.save(agent.critic.state_dict(), os.path.join(output_dir, f"critic_agent_{agent_id}_ep_{episode+1}.pth"))
            print(f"Models saved at episode {episode+1}")

        # Save training metrics
    metrics = {
        "episode_rewards": episode_rewards,
        "episode_latencies": episode_latencies,
        "episode_energies": episode_energies,
        "episode_successful": episode_successful_Tasks
        }
    np.save(os.path.join(output_dir, "training_metrics.npy"), metrics)
    print(f"Training metrics saved to {output_dir}/training_metrics.npy")

    # Plot the metrics
    plt.figure(figsize=(15, 10))

    # Plot Episode Rewards
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, label="Episode Rewards", color='b')
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    plt.title("Episode Rewards")

    # Plot Episode Latencies
    plt.subplot(2, 2, 2)
    plt.plot(episode_latencies, label="Episode Latencies", color='r')
    plt.xlabel("Episodes")
    plt.ylabel("Latencies (s)")
    plt.legend()
    plt.title("Episode Latencies")

    # Plot Episode Energies
    plt.subplot(2, 2, 3)
    plt.plot(episode_energies, label="Episode Energies", color='g')
    plt.xlabel("Episodes")
    plt.ylabel("Energy Consumption")
    plt.legend()
    plt.title("Episode Energy Consumption")

    # Plot Successful Tasks
    plt.subplot(2, 2, 4)
    plt.plot(episode_successful_Tasks, label="Successful Tasks", color='m')
    plt.xlabel("Episodes")
    plt.ylabel("Successful Tasks")
    plt.legend()
    plt.title("Successful Tasks per Episode")
    plt.show()
    plt.close()

    print("Training and evaluation complete. Results saved to", output_dir)
        






        
