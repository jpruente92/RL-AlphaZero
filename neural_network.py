from functools import reduce

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from hyperparameters import *


class Residual_layer(nn.Module):
    def __init__(self):
        super(Residual_layer, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += input
        x = F.relu(x)
        return x


class Convolutional_layer(nn.Module):
    def __init__(self):
        super(Convolutional_layer, self).__init__()
        self.conv1 = nn.Conv2d(NR_BOARD_STATES_SAVED * 3, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = F.relu(x)
        return x


class Policy_head(nn.Module):
    def __init__(self, state_shape, nr_actions):
        super(Policy_head, self).__init__()
        self.conv1 = nn.Conv2d(256, 10, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(10)

        self.input_length_linear = 10 * reduce(lambda a, b: a * b, state_shape)
        self.fc = nn.Linear(self.input_length_linear, nr_actions)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = F.relu(x)
        x = x.view(-1, self.input_length_linear)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x


class Value_head(nn.Module):
    def __init__(self, state_shape):
        super(Value_head, self).__init__()

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


class Neural_network(nn.Module):

    def __init__(self, version, nr_actions, state_shape, name_for_saving):
        super().__init__()
        self.conv = Convolutional_layer()
        # residual blocks
        for block in range(NR_RESIDUAL_LAYERS):
            setattr(self, "res_%i" % block, Residual_layer())
        self.policy_head = Policy_head(state_shape, nr_actions)
        self.value_head = Value_head(state_shape)

        self.to(DEVICE)

        self.name_for_saving = name_for_saving
        if version >= 0 and name_for_saving is not None:
            self.load_model("./neural_networks/{}_version_{}.pth".format(self.name_for_saving, version))

    def forward(self, input):
        x = input
        x = self.conv(x)
        for block in range(NR_RESIDUAL_LAYERS):
            x = getattr(self, "res_%i" % block)(x)
        return self.value_head(x), self.policy_head(x)

    def save_model(self, version):
        torch.save(self.state_dict(), "./neural_networks/{}_version_{}.pth".format(self.name_for_saving, version))

    def load_model(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


class Last_Layer(nn.Module):
    def __init__(self, input_size, nr_actions):
        super().__init__()
        self.policy_head_1 = nn.Linear(input_size, 128)
        self.policy_head_2 = nn.Linear(128, 64)
        self.policy_head_3 = nn.Linear(64, nr_actions)

        self.value_head_1 = nn.Linear(input_size, 128)
        self.value_head_1 = nn.Linear(128, 32)
        self.value_head_1 = nn.Linear(32, 1)

    def forward(self, x):
        policy = self.policy_head_1(x)
        policy = F.relu(policy)
        policy = self.policy_head_2(policy)
        policy = F.relu(policy)
        policy = self.policy_head_3(policy)
        policy = F.softmax(policy)

        value = self.value_head_1(x)
        value = F.relu(value)
        value = self.value_head_2(value)
        value = F.relu(value)
        value = self.value_head_3(value)
        value = F.tanh(value)

        return value, policy
