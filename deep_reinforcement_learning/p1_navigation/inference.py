from unityagents import UnityEnvironment
from agent import Agent
import torch
from config import Config

env = UnityEnvironment(file_name=Config.UNITY_ENV)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# load weights and set epsilon for inferencing
my_agent = Agent()
my_agent.primary_network.load_state_dict(torch.load(Config.CHECKPOINT, map_location="cpu"))
my_agent.set_epsilon_for_inference(eps=Config.EPSILON_FOR_INFER)

scores = []
for _ in range(Config.NUM_EPiSODE_INFER):
    env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
    state = env_info.vector_observations[0]             # get the current state
    score = 0                                           # initialize the score
    while True:
        action = my_agent.act(state)                    # select an action
        env_info = env.step(action)[brain_name]         # send the action to the environment
        next_state = env_info.vector_observations[0]    # get the next state
        reward = env_info.rewards[0]                    # get the reward
        done = env_info.local_done[0]                   # see if episode has finished
        score += reward                                 # update the score
        state = next_state                              # roll over the state to next time step
        if done:                                        # exit loop if episode finished
            break
    scores.append(score)

print("Scores: {}".format(scores))
print("Average in {} episodes: {}".format(Config.NUM_EPiSODE_INFER, sum(scores)/len(scores)))