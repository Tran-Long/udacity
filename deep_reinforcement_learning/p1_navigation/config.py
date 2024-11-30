import torch

class Config:
    # seed for randomness
    SEED = 3737
    
    # name of the unity environment
    UNITY_ENV = "Banana_Windows_x86_64\Banana.exe"

    # training hyperparams
    BATCH_SIZE = 64                         # batch size for learning
    BUFFER_SIZE = 100000                    # buffer max capacity
    TAU = 1e-3                              # hyperparam update the target network
    NETWORK_ARCHITECTURE = [64, 128, 64]    # hidden layers in the network
    LEARN_EVERY = 4                         # agent will sample and learn after LEARN_EVERY steps
    EPSILON_START = 1.0                     # epsilon start at first episode
    EPSILON_END = 0.05                      # epsilon min
    EPSILON_DECAY = 0.9995                  # epsilon decay after each episode
    EPSILON_FOR_INFER = 0.05                # epsilon for inferencing
    GAMMA = 0.99                            # discount factor
    LR = 0.001                              # learning rate
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    NUM_EPISODE = 5000                      # num of trAaining episodes
    NUM_EPiSODE_INFER = 10                  # num of inferencing episodes
    MAX_TIMESTEP = 300                      # max timestep for each episode

    CHECKPOINT = "checkpoint.pth"           # checkpoint path of trained weight