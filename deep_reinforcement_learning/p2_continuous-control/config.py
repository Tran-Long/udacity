import torch
class Config:
    SEED = 73
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    STATE_DIM = 33
    ACTION_DIM = 4
    ACTOR_NET_STRUCTURE = [400, 300]
    CRITIC_NET_STRUCTURE = [400, 300]
    ACTOR_LR = 1e-3
    CRITIC_LR = 1e-3
    GAMMA = 0.99
    TAU = 5e-3

    BUFFER_SIZE = 1000000
    BATCH_SIZE = 100

    NUM_EPISODES = 5000

    ACTOR_CHECKPOINT = "./checkpoints/actor_checkpoint.pth"
    ACTOR_T_CHECKPOINT = "./checkpoints/actor_t_checkpoint.pth"
    CRITIC1_CHECKPOINT = "./checkpoints/critic1_checkpoint.pth"
    CRITIC1_T_CHECKPOINT = "./checkpoints/critic1_t_checkpoint.pth"
    CRITIC2_CHECKPOINT = "./checkpoints/critic2_checkpoint.pth"
    CRITIC2_T_CHECKPOINT = "./checkpoints/critic2_t_checkpoint.pth"