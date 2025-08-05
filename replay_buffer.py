import random
import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.buffer = []
        self.idx = 0

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
