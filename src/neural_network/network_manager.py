from logging import Logger
from typing import Literal, List, Tuple

import numpy as np
from torch import Tensor

from alpha_zero.replay_buffer_experience import ReplayBufferExperience
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
        self.VERSION = version
        self.NO_ACTIONS = no_actions
        self.STATE_SHAPE = state_shape
        self.NEURAL_NETWORK = self._create_network_from_scratch()
        self._load_network_data_from_file()

    # region Public Abstract Methods
    def clone(self):
        raise NotImplementedError

    def save_model(
            self,
            version: int
    ) -> None:
        raise NotImplementedError

    # todo: clarify which player has to be given together with the states
    def evaluate(
            self,
            states_so_far: List[np.array],
            current_player: Literal[-1, 1]
    ) -> (float, np.array):
        raise NotImplementedError

    def train_batch(
            self,
            learning_rate: float,
            experiences_of_batch: list[ReplayBufferExperience]
    ) -> (float, float):
        raise NotImplementedError

    def compute_loss_of_batch(
            self,
            alpha_zero_agent,
            experiences: list[ReplayBufferExperience],
            number_of_batches_validation: int
    ) -> (float, float):
        raise NotImplementedError

    # endregion Public Abstract Methods

    # region Private Abstract Methods

    def _evaluate_batch(
            self,
            input_tensor: Tensor
    ) -> (Tensor, Tensor):
        raise NotImplementedError

    def _create_network_from_scratch(self) -> NeuralNetworkTorch:
        raise NotImplementedError

    def _load_model(self, path: str):
        raise NotImplementedError

    def _load_network_data_from_file(self) -> None:
        raise NotImplementedError

    # endregion Private Abstract Methods

    # region Public Methods
    # todo: add unit test
    # noinspection PyArgumentList
    def prepare_nn_input(
            self,
            states_so_far: List[np.array],
            current_player: Literal[-1, 1],
            single_evaluation: bool
    ) -> np.array:
        state_shape = states_so_far[0].shape
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
        if not single_evaluation:
            nn_input = np.squeeze(nn_input)
        return nn_input

    def reset_neural_network(self):
        self._load_network_data_from_file()

    # endregion Public Methods

    # region Input preparation

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
