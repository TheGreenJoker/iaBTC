import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, lookback:int):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(lookback*4, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.fc(x)
