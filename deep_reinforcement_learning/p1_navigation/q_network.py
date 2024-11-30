import torch
import torch.nn as nn
import torch.nn.functional as F



# class QNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, network_acrchitecture, seed):
#         """
#         Params
#             ======
#             state_dim: state dimension
#             action_dim: action dimension
#             network_architecture: Q network architecture
#         """
#         super(QNetwork, self).__init__()
#         torch.manual_seed(seed)
#         network_acrchitecture = [state_dim] + network_acrchitecture + [action_dim]
#         self.network_achitecture = network_acrchitecture
#         layers = []
#         for i in range(len(network_acrchitecture)-1):
#             layers.append(nn.Linear(network_acrchitecture[i], network_acrchitecture[i+1]))
#             if i != len(network_acrchitecture) - 2:
#                 layers.append(nn.ReLU())
#         self.network = nn.Sequential(*layers)
        
    
#     def forward(self, x):
#         return self.network(x)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, network_acrchitecture, seed):
        """
        Params
            ======
            state_dim: state dimension
            action_dim: action dimension
            network_architecture: Q network architecture
        """
        super(QNetwork, self).__init__()
        torch.manual_seed(seed)
        network_acrchitecture = [state_dim] + network_acrchitecture
        self.network_achitecture = network_acrchitecture
        layers = []
        for i in range(len(network_acrchitecture)-2):
            layers.append(nn.Linear(network_acrchitecture[i], network_acrchitecture[i+1]))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.value_branch = nn.Sequential(
            nn.Linear(network_acrchitecture[-2], network_acrchitecture[-1]), 
            nn.ReLU(),
            nn.Linear(network_acrchitecture[-1], 1)
            )
        self.advantage_branch = nn.Sequential(
            nn.Linear(network_acrchitecture[-2], network_acrchitecture[-1]), 
            nn.ReLU(), 
            nn.Linear(network_acrchitecture[-1], action_dim)
            )
    
    def forward(self, x):
        x = self.backbone(x)
        v = self.value_branch(x)
        a = self.advantage_branch(x)
        return v + (a - a.mean(dim=-1, keepdim=True))
