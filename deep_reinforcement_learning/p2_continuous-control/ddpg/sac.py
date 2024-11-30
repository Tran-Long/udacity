import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from replay_memory import ReplayBuffer

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_arch, device) -> None:
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_structure = net_arch + [1]

        self.f1 = nn.Linear(self.state_dim + self.action_dim, self.net_structure[0])
        self.f2 = nn.Linear(self.net_structure[0], self.net_structure[1])
        self.f3 = nn.Linear(self.net_structure[1], 1)

        self.to(device)

    def forward(self, state, action):
        input = torch.cat((state, action), dim=1)
        out = F.relu(self.f1(input))
        out = F.relu(self.f2(out))
        out = self.f3(out)
        return out

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, net_arch, device) -> None:
        super(ValueNetwork, self).__init__()
        self.state_dim = state_dim
        self.net_structure = net_arch + [1]

        self.f1 = nn.Linear(self.state_dim, self.net_structure[0])
        self.f2 = nn.Linear(self.net_structure[0], self.net_structure[1])
        self.f3 = nn.Linear(self.net_structure[1], 1)

        self.to(device)
    def forward(self, state):
        out = F.relu(self.f1(state))
        out = F.relu(self.f2(out))
        return self.f3(out)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_arch=[256, 256], device='cpu') -> None:
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_structure = [self.state_dim] + net_arch + [self.action_dim]
        self.reparam_noise = 1e-6

        self.f1 = nn.Linear(state_dim, net_arch[0])
        self.f2 = nn.Linear(net_arch[0], net_arch[1])
        self.mu = nn.Linear(net_arch[1], action_dim)
        self.sigma = nn.Linear(net_arch[1], action_dim)
        
        self.device = device
        self.to(device)

    def forward(self, state):
        out = F.relu(self.f1(state))
        out = F.relu(self.f2(out))
        mu = self.mu(out)
        sigma = self.sigma(out)
        sigma = torch.clamp(sigma, self.reparam_noise, 1)
        return mu, sigma
    
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)
        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        action = torch.tanh(actions) * torch.tensor(1).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1- action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        return action, log_probs
    


class SAC:
    def __init__(self, state_dim, action_dim, alpha=0.0003, beta=0.0003, gamma=0.99, tau=0.005, reward_scale=2, device='cpu') -> None:
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(1000000, 256, device)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim, [256, 256], device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic1 = Critic(state_dim, action_dim, [256, 256], device)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=beta)
        self.critic2 = Critic(state_dim, action_dim, [256, 256], device)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=beta)
        self.value = ValueNetwork(state_dim, [256, 256], device)
        self.value_target = ValueNetwork(state_dim, [256, 256], device)
        self.value_optim = optim.Adam(self.value.parameters(), lr=beta)

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        for primary_param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
            target_param.data.copy_(tau*primary_param.data + (1-tau)*target_param.data)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action, _ = self.actor.sample_normal(state, reparameterize=False)
        return action.cpu().detach().numpy()
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add_transition(state, action, reward, next_state, done)
        if len(self.memory) >= self.memory.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        value = self.value(states)
        target_value = self.value_target(next_states) * (1-dones)

        action, log_probs = self.actor.sample_normal(states, reparameterize=False)
        q1_new_policy = self.critic1(states, action)
        q2_new_policy = self.critic2(states, action)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        self.value_optim.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward()
        self.value_optim.step()

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()

        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        q_hat = self.scale*rewards + self.gamma*target_value
        q1_old_policy = self.critic1(states, actions)
        q2_old_policy = self.critic2(states, actions)
        critic1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic1_optim.step()
        self.critic2_optim.step()

        self.update_network_parameters()
        
