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
    def __init__(self, state_dim, action_dim, max_action, agent_id, buffer_size=int(1e6)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.tau = 0.005
        self.agent_id = agent_id

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
        self.replay_buffer.add(state, action, next_state, reward, done)

    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        states, actions, next_states, rewards, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q_value = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

def run_agent(agent_id, state_q, action_q, reward_q, next_state_q, done_q, agent_args):
    agent = MAIDDPGAgent(*agent_args, agent_id)
    while True:
        state = state_q.get()
        if state is None:
            break
        action = agent.select_action(state)
        action_q.put((agent_id, action))
        reward, next_state, done = reward_q.get(), next_state_q.get(), done_q.get()
        agent.store_experience(state, action, reward, next_state, done)
        agent.update()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    output_dir = "/home/natnael/Desktop/mult-agentFog/output/MAIDDPG/"
    os.makedirs(output_dir, exist_ok=True)

    num_agents = 6
    env = FogToFogEnv()
    state_dim = 12
    action_dim = 6

    state_queues = [Queue() for _ in range(num_agents)]
    action_queues = [Queue() for _ in range(num_agents)]
    reward_queues = [Queue() for _ in range(num_agents)]
    next_state_queues = [Queue() for _ in range(num_agents)]
    done_queues = [Queue() for _ in range(num_agents)]

    agents = [MAIDDPGAgent(state_dim, action_dim, 1, i) for i in range(num_agents)]

    processes = [Process(target=run_agent, args=(i, state_queues[i], action_queues[i], reward_queues[i], next_state_queues[i], done_queues[i], (state_dim, action_dim, 1))) for i in range(num_agents)]

    for p in processes:
        p.start()

    num_episodes = 10000
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
                reward_queues[i].put(reward[i] if isinstance(reward, (list, np.ndarray)) else reward)
                next_state_queues[i].put(next_state)
                done_queues[i].put(done)

            state = next_state
            #episode_reward += reward / num_agents if isinstance(reward, (list, np.ndarray)) else reward
            episode_reward += reward.item()
            episode_latency += info.get("Total Latency", 0)
            episode_energy += info.get("Total Energy", 0)
            episode_successful += info.get("Successful Tasks", 0)
            time_steps += 1

        episode_rewards.append(episode_reward / time_steps)
        episode_latencies.append(episode_latency)
        episode_energies.append(episode_energy)
        episode_successful_Tasks.append(episode_successful)

        print(f"Episode {episode+1}, Total Reward: {episode_reward}, Total Latency: {episode_latency}, Total Energy: {episode_energy}, successful_Tasks: {episode_successful}")
        if (episode + 1) % 100 == 0 or episode + 1 == num_episodes:
            for agent_id, agent in enumerate(agents):
                torch.save(agent.actor.state_dict(), os.path.join(output_dir, f"actor_agent_{agent_id}_ep_{episode+1}.pth"))
                torch.save(agent.critic.state_dict(), os.path.join(output_dir, f"critic_agent_{agent_id}_ep_{episode+1}.pth"))
            print(f"Models saved at episode {episode+1}")

    np.save(os.path.join(output_dir, "training_metrics.npy"), {
        "episode_rewards": episode_rewards,
        "episode_latencies": episode_latencies,
        "episode_energies": episode_energies,
        "episode_successful": episode_successful_Tasks
    })
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
        






        
