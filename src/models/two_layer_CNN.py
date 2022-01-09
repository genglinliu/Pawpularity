import torch
import torch.nn as nn
import torch.nn.functional as F
from models.multicovariate_conv_layer import Hybrid_Conv2d


class ConvNet_simple(nn.Module):
    """
    Simple two-layer CNN with sequential container
    """
    def __init__(self):
        super(ConvNet_simple, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(387200, 128),
            nn.ReLU(),
            nn.Linear(128, 1) 
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
    

class ConvNet_hybrid(nn.Module):
    """
    Two-layer CNN with one hybrid layer
    """
    def __init__(self): 
        super(ConvNet_hybrid, self).__init__()  
          
        self.hybrid_conv_1 = Hybrid_Conv2d(3, 16, kernel_size=(16, 3, 3, 3), num_cov=12) 
        
        # self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(6195200, 128) # with hybrid, parameter number becomes 6195200
        
    def forward(self, x, cov):
        x = F.relu(self.hybrid_conv_1(x, cov))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(1, -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x