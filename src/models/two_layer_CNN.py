class ConvNet_v1(nn.Module):
    """
    Simple two-layer CNN with sequential container
    """
    def __init__(self):
        super(ConvNet_v1, self).__init__()
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