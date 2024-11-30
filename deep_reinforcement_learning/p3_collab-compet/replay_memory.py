from collections import deque, namedtuple
import random
import numpy as np
import torch


class ReplayBuffer():
    def __init__(self, buffer_size, batch_size, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experiences = namedtuple('Experience', ('state', "action", "reward", "next_state", "done"))
        self.device = device
    
    def add_transition(self, state, action, reward, next_state, done):
        """
            state: np.ndarray - shape (n_agents, state_dims)
            action: np.ndarray - shape (n_agents, action_dims)
            reward: np.ndarray - shape (n_agents, 1)
            done: np.ndarray - shape (n_agents, 1)
        """
        reward = np.array(reward).reshape(-1, 1)
        done = np.array(done).reshape(-1, 1)
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None], axis=1)).float().to(self.device)  # shape : (n_agents, batch_size, state_dims)
        full_states = torch.from_numpy(np.vstack([e.state.flatten() for e in experiences if e is not None])).float().to(self.device) # shape : (batch_size, n_agents * state_dims)
        actions = torch.from_numpy(np.vstack([e.action.flatten() for e in experiences if e is not None])).float().to(self.device)  # shape: (batch_size, n_agents * action_dims)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None], axis=1)).float().to(self.device) # shape: (n_agents, batch_size, 1)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None], axis=1)).float().to(self.device) # shape : (n_agents, batch_size, state_dims)
        full_next_states = torch.from_numpy(np.vstack([e.next_state.flatten() for e in experiences if e is not None])).float().to(self.device) # shape : (batch_size, n_agents * state_dims)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None], axis=1).astype(np.uint8)).float().to(self.device) # shape: (n_agents, batch_size, 1)

        return (states, full_states, actions, rewards, next_states, full_next_states, dones)
    
    def __len__(self):
        return len(self.memory)