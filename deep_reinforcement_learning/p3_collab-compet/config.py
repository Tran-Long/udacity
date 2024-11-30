import torch
class Config:
    SEED = 37
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    N_AGENTS = 2
    STATE_DIMS = 24
    ACTION_DIMS = 2
    ACTOR_NET = [256, 256]
    CRITIC_NET = [256, 256]
    TAU = 0.01
    GAMMA = 0.95
    ACTOR_LR = 1e-4
    CRITIC_LR = 1e-3

    BUFFER_SIZE = 1000000
    BATCH_SIZE = 128
    NUM_EPISODES = 10000

    CHECKPOINT_FOLDER = "checkpoints/"
