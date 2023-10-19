from functools import reduce

from torch import nn


class PolicyHead(nn.Module):
    def __init__(self, state_shape, nr_actions):
        super(PolicyHead, self).__init__()
        self.conv1 = nn.Conv2d(256, 10, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(10)

        self.input_length_linear = 10 * reduce(lambda a, b: a * b, state_shape)
        self.fc = nn.Linear(self.input_length_linear, nr_actions)

    def forward(self, nn_input):
        x = self.conv1(nn_input)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = x.view(-1, self.input_length_linear)
        x = self.fc(x)
        x = nn.functional.softmax(x, dim=1)
        return x
