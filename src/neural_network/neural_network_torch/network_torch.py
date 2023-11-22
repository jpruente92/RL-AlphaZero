from torch import nn, Tensor

from constants.hyper_parameters import *
from neural_network.neural_network_torch.layers.convolutional_layer import ConvolutionalLayer
from neural_network.neural_network_torch.layers.policy_head import PolicyHead
from neural_network.neural_network_torch.layers.residual_layer import ResidualLayer
from neural_network.neural_network_torch.layers.value_head import ValueHead


class NeuralNetworkTorch(nn.Module):

    def __init__(self,
                 no_actions: int,
                 state_shape: tuple
                 ):
        super().__init__()
        self.conv = ConvolutionalLayer()
        # residual blocks
        for block in range(NO_RESIDUAL_LAYERS):
            setattr(self, "res_%i" % block, ResidualLayer())
        self.policy_head = PolicyHead(state_shape, no_actions)
        self.value_head = ValueHead(state_shape)
        self.to(DEVICE)

    def forward(self,
                input_tensor: Tensor
                ):
        x = input_tensor
        x = self.conv(x)
        for block in range(NO_RESIDUAL_LAYERS):
            x = getattr(self, "res_%i" % block)(x)
        return self.value_head(x), self.policy_head(x)
