import os
import torch
import torch.nn.functional as F
import numpy as np
from ddpg import DDPG
from replay_memory import ReplayBuffer

class MADDPG:
    def __init__(self, config) -> None:
        self.n_agents = config.N_AGENTS
        self.agents = [DDPG(config) for _ in range(self.n_agents)]
        self.checkpoints_folder = config.CHECKPOINT_FOLDER
        self.memory = ReplayBuffer(config.BUFFER_SIZE, config.BATCH_SIZE, config.DEVICE)
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add_transition(state, action, reward, next_state, done)
        if len(self.memory) >= self.memory.batch_size:
            for agent_idx in range(self.n_agents):
                experiences = self.memory.sample()
                self.learn(agent_idx, experiences)
            for agent in self.agents:
                agent.soft_update()
    
    def act(self, state):
        """
            state = (n_agents, state_dims)
        """
        actions_for_env = []
        for agent_idx, agent in enumerate(self.agents):
            actions_for_env.append(agent.select_action(state[agent_idx]))
        return np.array(actions_for_env)

    def learn(self, agent_idx, experiences):
        learning_agent = self.agents[agent_idx]
        states, full_states, actions, rewards, next_states, full_next_states, dones = experiences
        

        full_next_actions = []
        for idx, agent in enumerate(self.agents):
            action = agent.actor_target(next_states[idx])
            full_next_actions.append(action)
        full_next_actions = torch.concat(full_next_actions, dim=1)

        input_for_t_critics = torch.concat([full_next_states, full_next_actions], dim=1)
        # print(rewards.shape, dones.shape, learning_agent.critic_target(input_for_t_critics).shape)
        target_q = rewards[agent_idx] + learning_agent.gamma * (1-dones[agent_idx]) * learning_agent.critic_target(input_for_t_critics)
        current_q = learning_agent.critic(torch.concat([full_states, actions], dim=1))
        loss = learning_agent.loss(target_q, current_q)
        learning_agent.critic_optim.zero_grad()
        loss.backward()
        learning_agent.critic_optim.step()

        full_actions = []
        for idx, agent in enumerate(self.agents):
            action = agent.actor(states[idx])
            full_actions.append(action)
        full_actions = torch.concat(full_actions, dim=1)
        input_for_ctitic = torch.concat([full_states, full_actions], dim=1)
        actor_loss = -(learning_agent.critic(input_for_ctitic)).mean()
        learning_agent.actor_optim.zero_grad()
        actor_loss.backward()
        learning_agent.actor_optim.step()

    def save_weight(self):
        for idx in range(self.n_agents):
            agent_ckpt_dir = os.path.join(self.checkpoints_folder, "Agent " + str(idx))
            if not os.path.exists(agent_ckpt_dir):
                os.makedirs(agent_ckpt_dir)
            torch.save(self.agents[idx].actor.state_dict(), os.path.join(agent_ckpt_dir, "actor_ckpt.pth"))
            torch.save(self.agents[idx].actor_target.state_dict(), os.path.join(agent_ckpt_dir, "actor_t_ckpt.pth"))
            torch.save(self.agents[idx].critic.state_dict(), os.path.join(agent_ckpt_dir, "critic_ckpt.pth"))
            torch.save(self.agents[idx].critic_target.state_dict(), os.path.join(agent_ckpt_dir, "critic_t_ckpt.pth"))
    
    def load_weight(self):
        for idx in range(self.n_agents):
            agent_ckpt_dir = os.path.join(self.checkpoints_folder, "Agent " + str(idx))
            self.agents[idx].actor.load_state_dict(torch.load(os.path.join(agent_ckpt_dir, "actor_ckpt.pth")))
            self.agents[idx].actor_target.load_state_dict(torch.load(os.path.join(agent_ckpt_dir, "actor_t_ckpt.pth")))
            self.agents[idx].critic.load_state_dict(torch.load(os.path.join(agent_ckpt_dir, "critic_ckpt.pth")))
            self.agents[idx].critic_target.load_state_dict(torch.load(os.path.join(agent_ckpt_dir, "critic_t_ckpt.pth")))