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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x).clamp(-20, 2)  # Clamping for numerical stability
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        action = torch.tanh(action) * self.max_action
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        # Ensure both state and action are tensors
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        return self.critic1(x)



class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, agent_id, buffer_size=int(1e6)):
        self.device = torch.device("cpu")  # Force CPU usage
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)

        self.target_critic1 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=0.0003
        )

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=buffer_size)
        self.batch_size = 64
        self.gamma = 0.99
        self.alpha = 0.2
        self.agent_id = agent_id

    def select_action(self, state):
        self.actor.eval()
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.actor(state)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        self.actor.train()
        return np.clip(action.cpu().numpy()[0], -1, 1), np.clip(log_prob.cpu().numpy()[0], -1, 1)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done)) 
        # Keep the replay buffer size in check
        if len(self.replay_buffer) > 100000:  # Replay buffer size limit
            self.replay_buffer.pop(0)

    def update(self):
        if len(self.replay_buffer) < 1000:
            return
        batch = random.sample(self.replay_buffer, 64)
        states, actions, rewards, next_states, dones = zip(*batch)

        #states = torch.FloatTensor(states).to(device)
        states = torch.tensor(states, dtype=torch.float32) if not isinstance(states, torch.Tensor) else state
        actions = torch.tensor(actions, dtype=torch.float32) if not isinstance(actions, torch.Tensor) else action

        #states = torch.FloatTensor(np.array(states)).to(device)
        #actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device).unsqueeze(1)

        with torch.no_grad():
            next_actions = self.actor(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = rewards + self.gamma * torch.min(target_q1, target_q2) * (1 - dones)

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic_loss = torch.nn.MSELoss()(current_q1, target_q) + torch.nn.MSELoss()(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic1(states, self.actor(states)).mean() + self.alpha * torch.mean(actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        for local_params, target_params in zip(local_model.parameters(), target_model.parameters()):
            target_params.data.copy_(tau * local_params.data + (1.0 - tau) * target_params.data)

def run_agent(agent_id, experience_queue, stop_event, agent):
    while not stop_event.is_set():
        if not experience_queue.empty():
            state, action, reward, next_state, done = experience_queue.get()
            agent.store_experience(state, action, reward, next_state, done)
            agent.update()
    print(f"Agent {agent_id} stopped.")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    print("Using device: CPU")

    # output_dir = "/home/natnael/Desktop/mult-agentFog/output/MASAC/"
    output_dir = "./output_lxl/MASAC/"
    os.makedirs(output_dir, exist_ok=True)

    num_agents = 6
    env = FogToFogEnv()
    experience_queue = Queue()
    stop_event = multiprocessing.Event()

    agents = [SACAgent(state_dim=12, action_dim=6, max_action=1.0, agent_id=i) for i in range(num_agents)]

    processes = [Process(target=run_agent, args=(i, experience_queue, stop_event, agents[i])) for i in range(num_agents)]
    for p in processes:
        p.start()

    num_episodes = 10000
    episode_rewards = []
    episode_latencies = []
    episode_energies = []
    episode_successful_Tasks =[]

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_latency = 0
        episode_energy = 0
        episode_successful = 0
        time_steps = 0

        while not done:
            actions = [agent.select_action(state)[0] for agent in agents]
            next_state, reward, done, info = env.step(actions)
            info["Total Latency"] /= 100
            info["Total Energy"] /= 10000
            info["Successful Tasks"]
            collective_reward = sum(reward) / num_agents if isinstance(reward, (list, np.ndarray)) else reward
            individual_rewards = [collective_reward] * num_agents
            for i in range(num_agents):
                experience_queue.put((state, actions[i], individual_rewards[i], next_state, done))

            episode_reward += collective_reward
            episode_latency += info["Total Latency"]
            episode_energy += info["Total Energy"]
            episode_successful += info.get("Successful Tasks", 0)
            time_steps += 1
            state = next_state

        episode_rewards.append(episode_reward / time_steps)
        episode_latencies.append(episode_latency)
        episode_energies.append(episode_energy)
        episode_successful_Tasks.append(episode_successful)

        print(f"Episode {episode+1}, Total Reward: {episode_reward}, Total Latency: {episode_latency}, Total Energy: {episode_energy}, Successful Tasks: {episode_successful}")

                # Save models periodically
        # if (episode + 1) % 100 == 0:
        # 只保存最后一轮的模型
        if episode + 1 == num_episodes:
            for agent_id, agent in enumerate(agents):
                torch.save(agent.actor.state_dict(), os.path.join(output_dir, f"weight/actor_agent_{agent_id}_ep_{episode+1}.pth"))
                torch.save(agent.critic1.state_dict(), os.path.join(output_dir, f"weight/critic1_agent_{agent_id}_ep_{episode+1}.pth"))
                torch.save(agent.critic2.state_dict(), os.path.join(output_dir, f"weight/critic2_agent_{agent_id}_ep_{episode+1}.pth"))

            print(f"Models saved at episode {episode+1}")

    stop_event.set()
    for p in processes:
        p.join()

    metrics = {"episode_rewards": episode_rewards, "episode_latencies": episode_latencies, "episode_energies": episode_energies, "episode_successful_Tasks": episode_successful_Tasks}
    np.save(os.path.join(output_dir, "train/training_metrics.npy"), metrics)
    print(f"Training metrics saved to {output_dir}train/training_metrics.npy")

    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.show()

    plt.plot(episode_latencies)
    plt.title("Episode Latencies")
    plt.show()

    plt.plot(episode_energies)
    plt.title("Episode Energies")
    plt.show()

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
