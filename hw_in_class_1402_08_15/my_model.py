import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(32 * 32 * 3, 256)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
