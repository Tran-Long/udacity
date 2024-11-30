from unityagents import UnityEnvironment
import numpy as np
from td3 import TD3
from collections import deque
from config import Config as C
import torch

def infer(agent, env, brain_name):
    """
    
    Params
    ======
        agent: the agent for training
        env: the Unity environment
        brain_name: brain_name of the Unity environment
        n_episodes (int): maximum number of training episodes
    """
    env_info = env.reset(train_mode=False)[brain_name]       # reset the environment
    state = env_info.vector_observations[0]                 # get the current state
    score = 0
    while True:
        action = agent.select_action(state)                           # agent choose action
        env_info = env.step(action)[brain_name]             # send the action to the environment
        next_state = env_info.vector_observations[0]        # get the next state
        reward = env_info.rewards[0]                        # get the reward
        done = env_info.local_done[0]
        state = next_state
        score += reward
        if done:
            break

    return score

if __name__ == "__main__":
    agent = TD3(C.STATE_DIM, C.ACTION_DIM, C.ACTOR_LR, C.CRITIC_LR, C.GAMMA, C.TAU)
    agent.actor.load_state_dict(torch.load("checkpoints/actor_checkpoint.pth"))
    agent.set_inference_status()
    env = UnityEnvironment(file_name='./Reacher_Windows_x86_64/Reacher.exe')
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    print(infer(agent, env, brain_name))
