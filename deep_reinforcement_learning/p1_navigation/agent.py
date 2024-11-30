import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from q_network import QNetwork
from config import Config
from replay_buffer import ReplayBuffer


class Agent(nn.Module):
    def __init__(
        self, 
        state_dim=37, 
        action_dim=4, 
        network_architecture=Config.NETWORK_ARCHITECTURE, 
        eps_start = Config.EPSILON_START, 
        eps_decay = Config.EPSILON_DECAY, 
        eps_end = Config.EPSILON_END, 
        gamma = Config.GAMMA, 
        tau = Config.TAU, 
        learn_every = Config.LEARN_EVERY, 
        device=Config.DEVICE, 
        seed=Config.SEED):
        """
            Params
            ======
            state_dim: state dimension
            action_dim: action dimension
            network_architecture: Q network architecture
            eps_start: epsilon start
            eps_decay: epsilon decay
            eps_end: epsilon min
            gamma: discount rate
            tau: update target network
            learn_every: num step for sampling and learning from buffer
        """
        super(Agent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # init epsilon to be eps_start
        self.epsilon = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.gamma = gamma
        self.tau = tau
        self.learn_every = learn_every
        self.seed = np.random.seed(seed)
        self.device = device

        self.primary_network = QNetwork(state_dim, action_dim, network_architecture, seed).to(device)
        self.target_network = QNetwork(state_dim, action_dim, network_architecture, seed).to(device)
        self.optimizer = optim.Adam(self.primary_network.parameters(), lr=Config.LR)

        self.memory = ReplayBuffer(Config.BUFFER_SIZE, Config.BATCH_SIZE, device, seed)

        self.time_step = 0
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_end)

    def set_epsilon_for_inference(self, eps=Config.EPSILON_FOR_INFER):
        self.epsilon = eps

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.primary_network.eval()
        with torch.no_grad():
            q_values = self.primary_network(state)
        self.primary_network.train()
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(self.action_dim))
        else:
            return int(np.argmax(q_values.cpu().squeeze().numpy()))
            # return np.argmax(q_values.cpu().squeeze().numpy())
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        # select best actions of next_states from primary network
        best_acts = self.primary_network(next_states).argmax(dim=-1).unsqueeze(1)
        # gather best value of corresponding actions from target network and calculate true value (estimate)
        q_target_output = self.target_network(next_states).gather(1, best_acts)
        true_value_est = rewards + self.gamma * (q_target_output) * (1-dones)

        q_est = self.primary_network(states).gather(1, actions)
        
        loss = F.mse_loss(true_value_est, q_est)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # soft update the target network with tau
        self.soft_update()

    def soft_update(self):
        for target_param, primary_param in zip(self.target_network.parameters(), self.primary_network.parameters()):
            target_param.data.copy_(self.tau*primary_param.data + (1.0-self.tau)*target_param.data)
    
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add_transition(state, action, reward, next_state, done)

        self.time_step = (self.time_step + 1) % self.learn_every
        if self.time_step == 0 and len(self.memory) >= self.memory.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
    