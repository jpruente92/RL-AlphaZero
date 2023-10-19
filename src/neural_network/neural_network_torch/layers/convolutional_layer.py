import torch.nn.functional as F
from torch import nn

from constants.hyper_parameters import *


class ConvolutionalLayer(nn.Module):
    def __init__(self):
        super(ConvolutionalLayer, self).__init__()
        self.conv1 = nn.Conv2d(NO_BOARD_STATES_SAVED * 3, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = F.relu(x)
        return x
