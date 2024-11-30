from ddpg import DDPG
from config import Config as C
from replay_memory import ReplayBuffer

class Agent:
    def __init__(self) -> None:
        self.rl_brain = DDPG()
        self.buffer = ReplayBuffer(C.BUFFER_SIZE, C.BATCH_SIZE, C.DEVICE)
    
    def act(self, state, add_noise=True):
        return self.rl_brain.select_action(state, add_noise).reshape(1, -1)
    
    def step(self, state, action, reward, next_state, done):
        self.buffer.add_transition(state, action, reward, next_state, done)
        if len(self.buffer) >= self.buffer.batch_size:
            experiences = self.buffer.sample()
            self.learn(experiences)
    
    def learn(self, experiences):
        self.rl_brain.update_parameters(experiences)

