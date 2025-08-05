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
from networks import Actor, Critic
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, max_action):
#         super(Actor, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, action_dim),
#             nn.Sigmoid()  # Output in range (0,1)
#         )
#         self.max_action = max_action
    
#     def forward(self, state):
#         return self.net(state) * self.max_action

# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()
#         self.q1 = nn.Sequential(
#             nn.Linear(state_dim + action_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
#         self.q2 = nn.Sequential(
#             nn.Linear(state_dim + action_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
    
#     def forward(self, state, action):
#         sa = torch.cat([state, action], dim=-1)
#         return self.q1(sa), self.q2(sa)
class GaussianNoise:
    def __init__(self, action_dim, mu=0, sigma=0.2, sigma_decay=0.99, sigma_min=0.01):
        self.action_dim = action_dim  # Dimensionality of the action space
        self.mu = mu  # Mean of the noise
        self.sigma = sigma  # Initial standard deviation of the noise
        self.sigma_decay = sigma_decay  # Decay factor for noise over time
        self.sigma_min = sigma_min  # Minimum value for sigma to prevent noise from becoming too small

    def sample(self):
        """
        Samples noise based on the Gaussian distribution.
        The noise is added to the action chosen by the policy.
        """
        noise = np.random.normal(self.mu, self.sigma, self.action_dim)
        # Decay the noise sigma after every call to encourage exploration reduction
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)
        return noise

    def reset(self):
        """Resets the noise parameters, useful for resetting the environment or training phases."""
        self.sigma = self.initial_sigma
# class ReplayBuffer:
#     def __init__(self, state_dim, action_dim, max_size=int(1e6)):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.buffer = []
#         self.max_size = max_size
#         self.pointer = 0
    
    def add(self, state, action, next_state, reward, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append((state, action, next_state, reward, done))
        else:
            self.buffer[self.pointer] = (state, action, next_state, reward, done)
        self.pointer = (self.pointer + 1) % self.max_size
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards), np.array(dones)
    
    def size(self):
        return len(self.buffer)

class MATD3Agent:
    def __init__(self, state_dim, action_dim, max_action, agent_id, buffer_size=int(1e6)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim).to(self.device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=1e-3
        )
        
        self.gamma = 0.99
        self.tau = 0.005
        self.agent_id = agent_id
        
        # Initialize the ReplayBuffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=buffer_size)
        self.batch_size = 64
        
    def select_action(self, state):
        self.actor.eval()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state)
        self.actor.train()
        return action.cpu().numpy()[0]

    def store_experience(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.replay_buffer.add(state, action, next_state, reward, done)

    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return (0.0, 0.0, 0.0)  # Return valid values instead of None

        # Sample experiences
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        
        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q1 = self.critic1_target(next_states, next_actions)
            next_q2 = self.critic2_target(next_states, next_actions)
            target_q_value = rewards + (1 - dones) * self.gamma * torch.min(next_q1, next_q2)

        # Compute critic losses
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1, target_q_value)
        critic2_loss = F.mse_loss(q2, target_q_value)

        # Update critics
        self.critic_optimizer.zero_grad()
        (critic1_loss + critic2_loss).backward()
        self.critic_optimizer.step()

        actor_loss = 0.0  # Default to 0 if no actor update

        # Delayed policy update
        if self.agent_id % 2 == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update targets
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)

        return critic1_loss.item(), critic2_loss.item(), actor_loss.item()  # Always return values

    def soft_update(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

# Function to run the agent in parallel
def run_agent(agent_id, state_queue, action_queue, reward_queue, next_state_queue, done_queue, agents, action_dim):
    agent = agents[agent_id]
    while True:
        state = state_queue.get()
        if state is None:  # Sentinel value to stop the process
            break

        # Select action
        action = agent.select_action(state)
        action_queue.put((agent_id, action, state))

        # Get reward and next state
        reward = reward_queue.get()
        next_state = next_state_queue.get()
        done = done_queue.get()

        # Store experience and update
        agent.store_experience(state, action, reward, next_state, done)
        if len(agent.replay_buffer) >= agent.batch_size:
            agent.update()

# Main function for parallel training
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    output_dir = "/home/natnael/Desktop/mult-agentFog/output/MATD/"
    os.makedirs(output_dir, exist_ok=True)
    
    num_agents = 6
    env = FogToFogEnv()
    state_dim = 12  # 6 fog nodes * 2 (CPU + bandwidth)
    action_dim = 6  # 6 fog nodes
    experience_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()

    num_episodes = 10000
    state_queue = Queue()
    action_queue = Queue()
    reward_queue = Queue()
    next_state_queue = Queue()
    done_queue = Queue()
    info_queue = Queue()
    agents = [MATD3Agent(state_dim=state_dim, action_dim=action_dim, max_action=1, agent_id=i) for i in range(num_agents)]
    
    processes = [Process(target=run_agent, args=(i, state_queue, action_queue, reward_queue, next_state_queue, done_queue, agents, action_dim)) for i in range(num_agents)]
    for p in processes:
        p.start()

    episode_rewards, episode_latencies, episode_energies, episode_successful_Tasks = [], [], [], []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward, episode_latency, episode_energy, episode_successful, time_steps = 0, 0, 0, 0, 0
        while not done:
            actions = [agent.select_action(state) for agent in agents]
            next_state, reward, done, info = env.step(actions)
            info["Total Latency"] /= 100
            info["Total Energy"] /= 10000
            info["Successful Tasks"]
            collective_reward = sum(reward) / num_agents if isinstance(reward, (list, np.ndarray)) else reward
            individual_rewards = [collective_reward] * num_agents
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
        
        episode_rewards.append(episode_reward / time_steps)
        episode_latencies.append(episode_latency)
        episode_energies.append(episode_energy)
        episode_successful_Tasks.append(episode_successful)


        print(f"Episode {episode+1}, Total Reward: {episode_reward}, Total Latency: {episode_latency}, Total Energy: {episode_energy}, Successful Tasks: {episode_successful}")

        if (episode + 1) % 100 == 0 or episode + 1 == num_episodes:
            for agent_id, agent in enumerate(agents):
                torch.save(agent.actor.state_dict(), os.path.join(output_dir, f"actor_agent_{agent_id}_ep_{episode+1}.pth"))
                torch.save(agent.critic1.state_dict(), os.path.join(output_dir, f"critic1_agent_{agent_id}_ep_{episode+1}.pth"))
                torch.save(agent.critic2.state_dict(), os.path.join(output_dir, f"critic2_agent_{agent_id}_ep_{episode+1}.pth"))
            print(f"Models saved at episode {episode+1}")

        metrics = {
        "episode_rewards": episode_rewards,
        "episode_latencies": episode_latencies,
        "episode_energies": episode_energies,
        "episode_successful": episode_successful_Tasks
    }
    np.save(os.path.join(output_dir, "training_metrics.npy"), metrics)
    print(f"Training metrics saved to {output_dir}/training_metrics.npy")
    np.save(os.path.join(output_dir, "training_metrics.npy"), {"episode_rewards": episode_rewards, "episode_latencies": episode_latencies, "episode_energies": episode_energies, "episode_successful_Tasks": episode_successful_Tasks})
    plt.plot(episode_rewards, label="Rewards")
    plt.plot(episode_latencies, label="Latencies")
    plt.plot(episode_energies, label="Energies")
    plt.plot(episode_successful_Tasks, label="Successful Tasks")
    plt.legend()
    plt.show()
