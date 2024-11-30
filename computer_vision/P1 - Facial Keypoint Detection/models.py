## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        self.conv_seq = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),  # 32*222*222
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32*111*11
            nn.Conv2d(32, 64, kernel_size=5, stride=1),  # 64*107*107
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64*53*53
            nn.Conv2d(64, 128, kernel_size=3, stride=1), # 128*51*51
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 128*25*25
            nn.Conv2d(128, 256, kernel_size=3, stride=1), # 256*23*23
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 256*11*11
            nn.Conv2d(256, 512, kernel_size=3, stride=1), # 512*9*9
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2), # 512*4*4
        )
        self.lin_seq = nn.Sequential(
            nn.Linear(4*4*512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 136)
        )

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv_seq(x)
        x = x.view(x.shape[0], -1)
        x = self.lin_seq(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
