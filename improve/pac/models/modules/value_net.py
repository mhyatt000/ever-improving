import torch
import torch.nn as nn
import torch.nn.functional as F


class QRNet(nn.Module):

    def __init__(self, num_quantiles, num_actions):
        super(QRNet, self).__init__()

        self.num_quantiles = num_quantiles
        self.num_actions = num_actions
        self.fc1 = nn.Linear(384, 64)
        self.fc2 = nn.Linear(2048, 128)
        self.fc3 = nn.Linear(128, num_quantiles)  # * num_actions)

    def forward(self, x):
        bs, seq, *other = x.shape
        x = F.relu(self.fc1(x)).view(bs, seq, -1)
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = x.view(bs, seq, -1)
        return x
