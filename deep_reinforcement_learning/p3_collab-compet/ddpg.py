from audioop import add
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_dims, net_arch, action_dims, device) -> None:
        super(Actor, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dims, net_arch[0]))
        for i in range(len(net_arch)-1):
            self.layers.append(nn.Linear(net_arch[i], net_arch[i+1]))
        self.layers.append(nn.Linear(net_arch[-1], action_dims))
        self.to(device)

    def forward(self, state):
        output = state
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                output = F.relu(layer(output))
            else:    
                output = torch.tanh(layer(output))
        return output


class Critic(nn.Module):
    def __init__(self, full_state_action_dims, net_arch, device) -> None:
        super(Critic, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(full_state_action_dims, net_arch[0]))
        for i in range(len(net_arch)-1):
            self.layers.append(nn.Linear(net_arch[i], net_arch[i+1]))
        self.layers.append(nn.Linear(net_arch[-1], 1))
        self.to(device)
    
    def forward(self, full_state_action):
        output = full_state_action
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                output = F.relu(layer(output))
            else:    
                output = layer(output)
        return output

class OUNoise:
    def __init__(self, action_dimension, scale=0.3, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
    

class DDPG:
    def __init__(self, config) -> None:
        self.actor = Actor(config.STATE_DIMS, config.ACTOR_NET, config.ACTION_DIMS, config.DEVICE)
        self.actor_target = Actor(config.STATE_DIMS, config.ACTOR_NET, config.ACTION_DIMS, config.DEVICE)
        self.critic = Critic((config.STATE_DIMS + config.ACTION_DIMS)*config.N_AGENTS, config.CRITIC_NET, config.DEVICE)
        self.critic_target = Critic((config.STATE_DIMS + config.ACTION_DIMS)*config.N_AGENTS, config.CRITIC_NET, config.DEVICE)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)
        self.ounoise = OUNoise(config.ACTION_DIMS)
        self.tau = config.TAU
        self.gamma = config.GAMMA
        self.loss = nn.MSELoss()
        self.device = config.DEVICE
        self.soft_update(tau=1)
    
    def soft_update(self, tau=None):
        if tau is None:
            tau = self.tau
        for local_params, target_params in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_params.data.copy_(tau*local_params.data + (1-tau)*target_params.data)
        for local_params, target_params in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_params.data.copy_(tau*local_params.data + (1-tau)*target_params.data)
    
    def select_action(self, state, add_noise=True):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().squeeze().numpy()
        self.actor.train()
        if add_noise:
            action += self.ounoise.noise()
        return np.clip(action, -1, 1)