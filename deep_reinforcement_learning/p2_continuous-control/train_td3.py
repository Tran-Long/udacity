from unityagents import UnityEnvironment
import numpy as np
from td3 import TD3
from collections import deque
from config import Config as C
import torch
import matplotlib.pyplot as plt

def train(agent, env, brain_name, n_episodes=C.NUM_EPISODES):
    """
    
    Params
    ======
        agent: the agent for training
        env: the Unity environment
        brain_name: brain_name of the Unity environment
        n_episodes (int): maximum number of training episodes
    """
    scores = []                                                 # list containing scores from each episode
    scores_window = deque(maxlen=100)                           # last 100 scores
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
        state = env_info.vector_observations[0]                 # get the current state
        score = 0
        while True:
            action = agent.select_action(state)                           # agent choose action
            env_info = env.step(action)[brain_name]             # send the action to the environment
            next_state = env_info.vector_observations[0]        # get the next state
            reward = env_info.rewards[0]                        # get the reward
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done) # agent process the transition
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 200 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if i_episode % 1000 == 0:
            torch.save(agent.actor.state_dict(), C.ACTOR_CHECKPOINT)
            torch.save(agent.actor_target.state_dict(), C.ACTOR_T_CHECKPOINT)
            torch.save(agent.critic1.state_dict(), C.CRITIC1_CHECKPOINT)
            torch.save(agent.critic1_target.state_dict(), C.CRITIC1_T_CHECKPOINT)
            torch.save(agent.critic2.state_dict(), C.CRITIC2_CHECKPOINT)
            torch.save(agent.critic2_target.state_dict(), C.CRITIC2_T_CHECKPOINT)
            print("Save weights, episode: ", i_episode)
        if np.mean(scores_window)>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.actor.state_dict(), C.ACTOR_CHECKPOINT)
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
    env = UnityEnvironment(file_name='./Reacher_Windows_x86_64/Reacher.exe')
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    my_agent = TD3(C.STATE_DIM, C.ACTION_DIM, C.ACTOR_LR, C.CRITIC_LR, C.GAMMA, C.TAU, device=C.DEVICE)
    scores = train(my_agent, env, brain_name)
    scores_plot(scores, True)
