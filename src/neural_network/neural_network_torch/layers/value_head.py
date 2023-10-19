from functools import reduce

import torch.nn.functional as F
from torch import nn

from constants.hyper_parameters import *


class ValueHead(nn.Module):
    def __init__(self, state_shape):
        super(ValueHead, self).__init__()

        self.conv1 = nn.Conv2d(256, 1, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(1)

        self.input_length_linear1 = reduce(lambda a, b: a * b, state_shape)
        self.input_length_linear2 = 256
        self.fc1 = nn.Linear(self.input_length_linear1, self.input_length_linear2)
        self.fc2 = nn.Linear(self.input_length_linear2, 1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = F.relu(x)
        x = x.view(-1, self.input_length_linear1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return torch.tanh(x)