import torch
from unityagents import UnityEnvironment
import numpy as np
from maddpg import MADDPG
import torch
from collections import deque
import matplotlib.pyplot as plt
from config import Config as C

def infer(multi_agent, env, brain_name, n_episodes=5):
    """
    
    Params
    ======
        agent: the agent for training
        env: the Unity environment
        brain_name: brain_name of the Unity environment
        n_episodes (int): maximum number of training episodes
    """
    all_scores = []                                           
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]       # reset the environment
        states = env_info.vector_observations                    # get the current state
        scores = np.zeros(len(env_info.agents))
        while True:
            actions = multi_agent.act(states)                           # agent choose action
            env_info = env.step(actions)[brain_name]             # send the action to the environment
            next_states = env_info.vector_observations           # get the next state
            rewards = env_info.rewards                           # get the reward
            dones = env_info.local_done  
            states = next_states
            scores += np.array(rewards)
            if np.any(dones):                                  # exit loop if episode finished
                break

        all_scores.append(scores)              # save most recent score
        print('\rEpisode {}\t Score: {:.2f} {:.2f}'.format(i_episode, scores[0], scores[1]))
    return all_scores


if __name__ == "__main__":
    env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    my_agent = MADDPG(config=C)
    my_agent.load_weight()
    scores = infer(my_agent, env, brain_name)
