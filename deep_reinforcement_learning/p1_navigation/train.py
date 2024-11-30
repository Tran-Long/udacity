from unityagents import UnityEnvironment
from collections import deque
import numpy as np
from agent import Agent
import torch
import matplotlib.pyplot as plt
from config import Config

def dqn(agent, env, brain_name, n_episodes=Config.NUM_EPISODE, max_t=Config.MAX_TIMESTEP):
    """Deep Q-Learning.
    
    Params
    ======
        agent: the agent for training
        env: the Unity environment
        brain_name: brain_name of the Unity environment
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    scores = []                                                 # list containing scores from each episode
    scores_window = deque(maxlen=100)                           # last 100 scores
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
        state = env_info.vector_observations[0]                 # get the current state
        score = 0
        for _ in range(max_t):
            action = agent.act(state)                           # agent choose action
            env_info = env.step(action)[brain_name]             # send the action to the environment
            next_state = env_info.vector_observations[0]        # get the next state
            reward = env_info.rewards[0]                        # get the reward
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done) # agent process the transition
            state = next_state
            score += reward
            if done:
                break
        
        agent.update_epsilon()
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 200 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        # save weight because sometime server error :(
        if i_episode % 1000 == 0:
            torch.save(agent.primary_network.state_dict(), Config.CHECKPOINT)
            print("Save weights, episode: ", i_episode)
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.primary_network.state_dict(), Config.CHECKPOINT)
            break
    return scores

def scores_plot(scores, save=False):
    """Plot & save the scores"""
    plt.figure(figsize=(17, 7))
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    if save:
        plt.savefig("Result.png")
        print("Save successfully!")
    plt.show()


if __name__ == "__main__":
    
    env = UnityEnvironment(file_name=Config.UNITY_ENV)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    my_agent = Agent()
    scores = dqn(my_agent, env, brain_name)
    scores_plot(scores, True)