import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from config import Config as C
from replay_memory import ReplayBuffer
from utils import *

class Actor(nn.Module):
    def __init__(self) -> None:
        super(Actor, self).__init__()
        self.state_dim = C.STATE_DIM
        self.action_dim = C.ACTION_DIM
        self.net_structure = [self.state_dim] + C.ACTOR_NET_STRUCTURE + [self.action_dim]
        self.net = self.create_net()
        self.to(C.DEVICE)

    def create_net(self):
        net = [nn.Linear(self.state_dim, self.net_structure[0])]
        torch.nn.init.kaiming_normal_(net[-1].weight)
        for i in range(len(self.net_structure)-1):
            net.append(nn.ReLU())
            net.append(nn.Linear(self.net_structure[i], self.net_structure[i+1]))
            torch.nn.init.kaiming_normal_(net[-1].weight)
        net.append(nn.Tanh())
        return nn.Sequential(*net)
    
    def forward(self, x):
        out = self.net(x)
        return out


class Critic(nn.Module):
    def __init__(self) -> None:
        super(Critic, self).__init__()
        self.state_dim = C.STATE_DIM
        self.action_dim = C.ACTION_DIM
        self.net_structure = C.CRITIC_NET_STRUCTURE + [1]

        self.w_state = nn.Linear(self.state_dim, self.net_structure[0])
        torch.nn.init.kaiming_normal_(self.w_state.weight)
        self.w_action = nn.Linear(self.action_dim, self.net_structure[0])
        torch.nn.init.kaiming_normal_(self.w_action.weight)
        net = [nn.Linear(self.net_structure[0]*2, self.net_structure[1])]
        torch.nn.init.kaiming_normal_(net[-1].weight)
        for i in range(1, len(self.net_structure)-1):
            net.append(nn.ReLU())
            net.append(nn.Linear(self.net_structure[i], self.net_structure[i+1]))
            torch.nn.init.kaiming_normal_(net[-1].weight)

        self.net = nn.Sequential(*net[:-1])
        self.to(C.DEVICE)

    def forward(self, state, action):
        out_state = self.w_state(state)
        out_action = self.w_action(action)
        out = torch.cat((out_state, out_action), dim=-1)

        out = self.net(out)
        return out

class DDPG:
    def __init__(self) -> None:        
        self.actor = Actor()
        self.actor_target = Actor()
        self.actor_optim = Adam(self.actor.parameters(), lr=C.ACTOR_LR)
        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_optim = Adam(self.critic.parameters(), lr=C.CRITIC_LR)

        self.gamma = C.GAMMA
        self.tau = C.TAU
        self.loss = nn.MSELoss()

        hard_update(self.actor, self.actor_target)
        hard_update(self.critic, self.critic_target)
        self.ounoise = OUNoise(C.ACTION_DIM)
    
    def update_parameters(self, batch):
        states, actions, rewards, next_states, dones = batch

        # Update critic
        next_actions = self.actor_target(next_states)
        next_q = self.critic_target(next_states, next_actions) * (1-dones)
        true_value_est = rewards + (self.gamma*next_q).detach()

        current_q = self.critic(states, actions)
        loss = self.loss(current_q, true_value_est)
        self.critic_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        # Update actor
        policy_grad_loss = -(self.critic(states, self.actor(states))).mean()
        print(policy_grad_loss.shape)
        self.actor_optim.zero_grad()
        policy_grad_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)
    
    def select_action(self, state, add_noise=True):
        state = torch.from_numpy(state).float().unsqueeze(0).to(C.DEVICE)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().squeeze().numpy()
        self.actor.train()
        if add_noise:
            action += self.ounoise.noise()
        action = np.clip(action, -1.0, 1.0)
        return action

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


