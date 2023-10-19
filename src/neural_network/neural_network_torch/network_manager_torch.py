from logging import Logger
from typing import List, Literal

import numpy as np
import torch

from constants.hyper_parameters import DEVICE
from neural_network.network_manager import NetworkManagerBase
from neural_network.neural_network_torch.network_torch import NeuralNetworkTorch


class NetworkManagerTorch(NetworkManagerBase):

    def __init__(
            self,
            logger: Logger,
            name_for_saving: str,
            version: int,
            no_actions: int,
            state_shape: tuple
    ):
        super().__init__(
            logger=logger,
            name_for_saving=name_for_saving,
            version=version,
            no_actions=no_actions,
            state_shape=state_shape
        )

    # refactor paths
    def save_model(self, version: int):
        torch.save(self.NEURAL_NETWORK.state_dict(),
                   "./neural_networks/torch/{}_version_{}.pth".format(self.NAME_FOR_SAVING, version))

    def evaluate(
            self,
            states_so_far: List[np.array],
            state_shape: tuple,
            current_player: Literal[-1, 1]
    ) -> (float, np.array):
        nn_input = self._prepare_nn_input(states_so_far, state_shape, current_player)
        nn_input = torch.from_numpy(nn_input).float().to(DEVICE)
        self.NEURAL_NETWORK.eval()
        with torch.no_grad():
            winner, probabilities = self.NEURAL_NETWORK.forward(nn_input).detach().cpu().numpy()
        self.NEURAL_NETWORK.train()
        return winner, probabilities

    def _load_model(self, file_path: str):
        state_dict = torch.load(file_path)
        self.NEURAL_NETWORK.load_state_dict(state_dict)

    def _create_network_from_scratch(self) -> NeuralNetworkTorch:
        return NeuralNetworkTorch(
            no_actions=self.NO_ACTIONS,
            state_shape=self.STATE_SHAPE
        )
