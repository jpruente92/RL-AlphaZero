import os
from logging import Logger
from typing import Literal, List, Tuple

import numpy as np

from constants.constants import NEURAL_NETWORK_DIR_PATH
from constants.hyper_parameters import NO_BOARD_STATES_SAVED
from neural_network.neural_network_torch.network_torch import NeuralNetworkTorch


class NetworkManagerBase:

    def __init__(
            self,
            logger: Logger,
            name_for_saving: str,
            version: int,
            no_actions: int,
            state_shape: tuple
    ):
        self.LOGGER = logger
        self.NAME_FOR_SAVING = name_for_saving
        self.NO_ACTIONS = no_actions
        self.STATE_SHAPE = state_shape
        self.NEURAL_NETWORK = self._create_network_from_scratch()
        self._load_network_data_from_file(name_for_saving, version)

    # region Abstract Methods
    def save_model(self, version: int):
        raise NotImplementedError

    def evaluate(
            self,
            states_so_far: List[np.array],
            state_shape: tuple,
            current_player: Literal[-1, 1]
    ) -> (float, np.array):
        raise NotImplementedError

    def _create_network_from_scratch(self) -> NeuralNetworkTorch:
        raise NotImplementedError

    def _load_network_data_from_file(self, name_for_saving, version) -> None:
        file_path = os.path.join(NEURAL_NETWORK_DIR_PATH, "torch", f"{name_for_saving}_version_{version}.pth")
        if version >= 0 and name_for_saving is not None and os.path.exists(file_path):
            self._load_model(file_path)
        else:
            self.LOGGER.warning(f"Network {name_for_saving} version {version} could not be loaded! "
                                f"Path= {os.path.abspath(file_path)}")

    def _load_model(self, path: str):
        raise NotImplementedError

    # endregion Abstract Methods

    # region Input preparation

    # todo: add unit test
    # noinspection PyArgumentList
    def _prepare_nn_input(
            self,
            states_so_far: List[np.array],
            state_shape: Tuple,
            current_player: Literal[-1, 1]
    ) -> np.array:
        list_of_3_dimensional_states = []
        enough_states_available = len(states_so_far) >= NO_BOARD_STATES_SAVED
        if enough_states_available:
            for state in states_so_far[-NO_BOARD_STATES_SAVED:]:
                list_of_3_dimensional_states.append(self._state_to_3_dimensional_state(state, current_player))
        else:
            for state in states_so_far:
                list_of_3_dimensional_states.append(self._state_to_3_dimensional_state(state, current_player))
            self._fill_with_empty_states(list_of_3_dimensional_states=list_of_3_dimensional_states,
                                         states_so_far=states_so_far,
                                         state_shape=state_shape,
                                         current_player=current_player
                                         )

        nn_input = np.array(list_of_3_dimensional_states)
        nn_input = nn_input.reshape(-1, 3 * NO_BOARD_STATES_SAVED, *state_shape)
        return nn_input

    # todo: add unit test
    def _fill_with_empty_states(
            self,
            list_of_3_dimensional_states: List[List[List[List[int]]]],
            states_so_far: List[np.array],
            state_shape: Tuple,
            current_player: Literal[-1, 1]
    ) -> None:
        for i in range(NO_BOARD_STATES_SAVED - len(states_so_far)):
            empty_state = np.zeros(state_shape)
            list_of_3_dimensional_states.append(self._state_to_3_dimensional_state(empty_state, current_player))

    # todo: add unit test
    def _state_to_3_dimensional_state(
            self,
            state: np.array,
            current_player: Literal[-1, 1]
    ) -> List[List[List[int]]]:
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range(len(state)):
            layer_1.append([])
            layer_2.append([])
            layer_3.append([])
            for j in range(len(state[i])):
                if state[i][j] == 1:
                    layer_1[i].append(1)
                    layer_2[i].append(0)
                elif state[i][j] == -1:
                    layer_1[i].append(0)
                    layer_2[i].append(1)
                else:
                    layer_1[i].append(0)
                    layer_2[i].append(0)
                if current_player == 1:
                    layer_3[i].append(1)
                else:
                    layer_3[i].append(0)
        return [layer_1, layer_2, layer_3]

    # endregion Input preparation
