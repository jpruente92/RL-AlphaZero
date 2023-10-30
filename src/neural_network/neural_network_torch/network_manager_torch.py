import os
from logging import Logger
from typing import List, Literal

import numpy as np
import torch
from torch import Tensor

from agents.alpha_zero_agent import AlphaZeroAgent
from alpha_zero.replay_buffer_experience import ReplayBufferExperience
from constants.constants import NEURAL_NETWORK_DIR_PATH
from constants.hyper_parameters import DEVICE, BATCH_SIZE, WEIGHT_POLICY_LOSS, MOMENTUM, WEIGHT_DECAY
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

    # region Public Methods
    
    def clone(self):
        clone = NetworkManagerTorch(
            logger=self.LOGGER,
            name_for_saving=self.NAME_FOR_SAVING,
            version=self.VERSION,
            no_actions=self.NO_ACTIONS,
            state_shape=self.STATE_SHAPE
        )
        return clone
    
    def save_model(
            self,
            version: int
    ) -> None:
        file_path = os.path.join(NEURAL_NETWORK_DIR_PATH, "torch",f"{self.NAME_FOR_SAVING}_version_{version}.pth")
        torch.save(self.NEURAL_NETWORK.state_dict(),
                   file_path)
        self.LOGGER.info(f"Stored neural network to {file_path}")

    def evaluate(
            self,
            states_so_far: List[np.array],
            current_player: Literal[-1, 1]
    ) -> (float, np.array):
        nn_input = self.prepare_nn_input(states_so_far, current_player, single_evaluation=True)
        nn_input = torch.from_numpy(nn_input).float().to(DEVICE)
        self.NEURAL_NETWORK.eval()
        with torch.no_grad():
            winner, probabilities = self.NEURAL_NETWORK.forward(nn_input)
        winner = winner.detach().cpu().numpy()
        probabilities = probabilities.detach().cpu().numpy()
        self.NEURAL_NETWORK.train()
        return winner, probabilities

    def train_batch(
            self,
            learning_rate: float,
            experiences_of_batch: list[ReplayBufferExperience]
    ) -> (float, float):
        input_tensor, outcomes, search_probabilities = self._experiences_to_neural_network_input_and_output(
            experiences_of_batch)
        predicted_outcomes, predicted_move_probabilities = \
            self._evaluate_batch(input_tensor=input_tensor)
        value_loss_tensor = self._compute_values_loss(
            outcomes=outcomes,
            predicted_outcomes=predicted_outcomes
        )
        policy_loss_tensor = self._compute_policy_loss(
            search_probabilities=search_probabilities,
            move_probabilities=predicted_move_probabilities
        )
        loss_tensor = value_loss_tensor + WEIGHT_POLICY_LOSS * policy_loss_tensor

        optimizer = torch.optim.SGD(self.NEURAL_NETWORK.parameters(),
                                    lr=learning_rate,
                                    momentum=MOMENTUM,
                                    weight_decay=WEIGHT_DECAY
                                    )
        optimizer.zero_grad()
        loss_tensor.backward()
        optimizer.step()
        return value_loss_tensor.item(), policy_loss_tensor.item()

    def compute_loss_of_batch(
            self,
            alpha_zero_agent: AlphaZeroAgent,
            experiences: list[ReplayBufferExperience],
            number_of_batches_validation: int
    ) -> (float, float):
        total_value_loss = 0
        total_policy_loss = 0
        for i in range(number_of_batches_validation):
            experiences_of_batch = experiences[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            input_tensor, outcomes, search_probabilities = self._experiences_to_neural_network_input_and_output(
                experiences_of_batch)
            predicted_outcomes, predicted_move_probabilities = \
                alpha_zero_agent.NETWORK_MANAGER._evaluate_batch(input_tensor=input_tensor)

            total_value_loss += self._compute_values_loss(outcomes, predicted_outcomes).item()
            total_policy_loss += self._compute_policy_loss(search_probabilities, predicted_move_probabilities).item()
        return total_policy_loss, total_value_loss

    # endregion Public Methods

    # region Private Methods
    def _load_model(
            self,
            file_path: str
    ) -> None:
        state_dict = torch.load(file_path)
        self.NEURAL_NETWORK.load_state_dict(state_dict)
        
    def _load_network_data_from_file(self) -> None:
        if self.VERSION < 1:
            return

        file_name = f"{self.NAME_FOR_SAVING}_version_{self.VERSION}.pth"
        file_path = os.path.join(NEURAL_NETWORK_DIR_PATH, "torch", file_name)

        if self.NAME_FOR_SAVING is not None and os.path.exists(file_path):
            self.LOGGER.info(f"Loaded network from {file_path}")
            self._load_model(file_path)
        else:
            self.LOGGER.warning(f"Network {self.NAME_FOR_SAVING} self.VERSION {self.VERSION} could not be loaded!"
                                f"Path= {os.path.abspath(file_path)}")

    def _create_network_from_scratch(self) -> NeuralNetworkTorch:
        return NeuralNetworkTorch(
            no_actions=self.NO_ACTIONS,
            state_shape=self.STATE_SHAPE
        )

    def _evaluate_batch(
            self,
            input_tensor: Tensor
    ) -> (Tensor, Tensor):
        predicted_outcomes, predicted_move_probabilities = self.NEURAL_NETWORK.forward(input_tensor)
        predicted_outcomes.squeeze()
        return predicted_outcomes, predicted_move_probabilities

    def _compute_values_loss(
            self,
            outcomes: Tensor,
            predicted_outcomes: Tensor
    ) -> Tensor:
        return ((predicted_outcomes - outcomes) ** 2).mean()

    def _compute_policy_loss(
            self,
            search_probabilities: Tensor,
            move_probabilities: Tensor
    ) -> Tensor:
        return -(search_probabilities * torch.log(move_probabilities)).mean(axis=0).sum()

    def _experiences_to_neural_network_input_and_output(
            self,
            experiences_of_batch: list[ReplayBufferExperience]
    ) -> (Tensor, Tensor, Tensor):
        input_tensor = torch.from_numpy(
            np.array([experience.neural_network_input for experience in experiences_of_batch]))
        input_tensor = input_tensor.squeeze().float().to(DEVICE)
        search_probabilities = torch.from_numpy(
            np.array([experience.search_probabilities for experience in experiences_of_batch]))
        search_probabilities = search_probabilities.squeeze().float().to(DEVICE)
        outcomes = torch.from_numpy(
            np.array([experience.outcome for experience in experiences_of_batch]))
        outcomes = outcomes.squeeze().float().to(DEVICE)
        return input_tensor, outcomes, search_probabilities

    # endregion Private Methods
