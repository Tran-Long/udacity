from unityagents import UnityEnvironment
import numpy as np
from maddpg import MADDPG
import torch
from collections import deque
import matplotlib.pyplot as plt
from config import Config as C

def train(multi_agent, env, brain_name, n_episodes=C.NUM_EPISODES):
    """
    
    Params
    ======
        agent: the agent for training
        env: the Unity environment
        brain_name: brain_name of the Unity environment
        n_episodes (int): maximum number of training episodes
    """
    all_scores = []                                                 # list containing scores from each episode
    scores_window = deque(maxlen=100)                           # last 100 scores
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
        states = env_info.vector_observations                    # get the current state
        scores = np.zeros(len(env_info.agents))
        while True:
            actions = multi_agent.act(states)                           # agent choose action
            env_info = env.step(actions)[brain_name]             # send the action to the environment
            next_states = env_info.vector_observations           # get the next state
            rewards = env_info.rewards                           # get the reward
            dones = env_info.local_done  
            multi_agent.step(states, actions, rewards, next_states, dones) # agent process the transition
            states = next_states
            scores += np.array(rewards)
            if np.any(dones):                                  # exit loop if episode finished
                break

        scores_window.append(scores)       # save most recent score
        all_scores.append(scores)              # save most recent score
        current_mean = np.mean(scores_window, axis=0)
        print('\rEpisode {}\tAverage Score: {:.2f} {:.2f}'.format(i_episode, current_mean[0], current_mean[1]), end="")
        if i_episode % 500 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f} {:.2f}'.format(i_episode, current_mean[0], current_mean[1]))
        if i_episode % 1000 == 0:
            multi_agent.save_weight()
            print("Save weights, episode: ", i_episode)
        if np.max(np.mean(scores_window, axis=0))>=0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f} {:.2f}'.format(i_episode, current_mean[0], current_mean[1]))
            multi_agent.save_weight()
            break
    return all_scores

def scores_plot(scores, save=False):
    """Plot & save the scores"""
    plt.figure(figsize=(17, 7))
    scores = np.array(scores).transpose()
    for idx in range(len(scores)):
        plt.plot(np.arange(len(scores[idx])), scores[idx])
    plt.legend(["Agent " + str(i) for i in range(len(scores))])
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    if save:
        plt.savefig("Result.png")
        print("Save successfully!")
    plt.show()


if __name__ == "__main__":
    env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    my_agent = MADDPG(config=C)
    scores = train(my_agent, env, brain_name)
    scores_plot(scores, True)
