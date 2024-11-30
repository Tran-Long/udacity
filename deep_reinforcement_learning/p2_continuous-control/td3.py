import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from replay_memory import ReplayBuffer
from config import Config as C

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_arch, device) -> None:
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_structure = net_arch + [1]

        self.f1 = nn.Linear(self.state_dim + self.action_dim, self.net_structure[0])
        self.f2 = nn.Linear(self.action_dim + self.net_structure[0], self.net_structure[1])
        self.f3 = nn.Linear(self.net_structure[1], 1)

        self.to(device)

    def forward(self, state, action):
        input = torch.cat((state, action), dim=1)
        out = F.relu(self.f1(input))
        out = torch.cat((out, action), dim=-1)
        out = F.relu(self.f2(out))
        out = self.f3(out)
        return out

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_arch, device) -> None:
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_structure = [self.state_dim] + net_arch + [self.action_dim]
        self.net = self.create_net()
        self.to(device)

    def create_net(self):
        net = [nn.Linear(self.state_dim, self.net_structure[0])]
        # torch.nn.init.kaiming_normal_(net[-1].weight)
        for i in range(len(self.net_structure)-1):
            net.append(nn.ReLU())
            net.append(nn.Linear(self.net_structure[i], self.net_structure[i+1]))
            # torch.nn.init.kaiming_normal_(net[-1].weight)
        net.append(nn.Tanh())
        return nn.Sequential(*net)
    
    def forward(self, x):
        out = self.net(x)
        return out

class TD3:
    def __init__(self, state_dim, action_dim, alpha, beta, gamma, tau, update_actor_interval=2, warmup=1000, noise=0.1, device=C.DEVICE) -> None:
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(C.BUFFER_SIZE, C.BATCH_SIZE, device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.update_actor_interval = update_actor_interval
        self.device = device
        self.warmup = warmup

        self.actor = Actor(state_dim, action_dim, C.ACTOR_NET_STRUCTURE, device)
        self.critic1 = Critic(state_dim, action_dim, C.CRITIC_NET_STRUCTURE, device)
        self.critic2 = Critic(state_dim, action_dim, C.CRITIC_NET_STRUCTURE, device)

        self.actor_target = Actor(state_dim, action_dim, C.ACTOR_NET_STRUCTURE, device)
        self.critic1_target = Critic(state_dim, action_dim, C.CRITIC_NET_STRUCTURE, device)
        self.critic2_target = Critic(state_dim, action_dim, C.CRITIC_NET_STRUCTURE, device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=beta)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=beta)

        self.noise = noise
        self.loss = nn.MSELoss()
        self.update_network_parameters(tau=1)
        self.time_step = 0
        self.learn_step = 0

    def select_action(self, state):
        if self.time_step < self.warmup:
            mu = torch.tensor(np.random.normal(scale=self.noise, size=(self.action_dim, )))
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            mu = self.actor(state).to(self.device)
        mu_prime = mu + torch.tensor(np.random.normal(scale=self.noise, size=(self.action_dim, )), dtype=float).to(self.device)
        mu_prime = torch.clamp(mu_prime, -1, 1)
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add_transition(state, action, reward, next_state, done)
        if len(self.memory) >= self.memory.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        next_actions = self.actor_target(next_states) + torch.clamp(torch.from_numpy(np.random.normal(scale=0.2, size=(self.memory.batch_size, self.action_dim))), -0.5, 0.5)
        next_actions = torch.clamp(next_actions, -1.0, 1.0).float()

        target_value1 = self.critic1_target(next_states, next_actions) *  (1-dones)
        target_value2 = self.critic2_target(next_states, next_actions) *  (1-dones)
        current_value1 = self.critic1(states, actions)
        current_value2 = self.critic2(states, actions)
        target_value = rewards + self.gamma*torch.min(target_value1, target_value2)
        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        loss1 = self.loss(target_value, current_value1)
        loss2 = self.loss(target_value, current_value2)
        loss = loss1 + loss2
        loss.backward()
        self.critic1_optim.step()
        self.critic2_optim.step()
        self.learn_step += 1
        if self.learn_step % self.update_actor_interval == 0:
            self.actor_optim.zero_grad()
            actor_loss1 = self.critic1(states, self.actor(states))
            actor_loss1 = -actor_loss1.mean()
            actor_loss1.backward()
            self.actor_optim.step()
            self.update_network_parameters()
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        for primary_param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(tau*primary_param.data + (1-tau)*target_param.data)
        for primary_param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(tau*primary_param.data + (1-tau)*target_param.data)
        for primary_param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau*primary_param.data + (1-tau)*target_param.data)

    def set_inference_status(self):
        self.warmup = 0
        self.noise = 0