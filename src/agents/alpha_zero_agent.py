from logging import Logger
from typing import Literal

from agents.base_agent import BaseAgent
from game_logic.two_player_game import TwoPlayerGame
from monte_carlo_tree_search.mcts import MCTS
from neural_network.network_manager import NetworkManagerBase
from neural_network.neural_network_torch.network_manager_torch import NetworkManagerTorch
from alpha_zero.replay_buffer import ReplayBuffer
from monte_carlo_tree_search.mcts_with_nn import MCTSWithNeuralNetwork


class AlphaZeroAgent(BaseAgent):

    def __init__(
            self,
            logger: Logger,
            network_manager: NetworkManagerBase,
            mcts: MCTS,
            replay_buffer,
            player_number: Literal[-1, 1],
            version=0,
            seconds_per_move=1
    ):
        super().__init__(
            logger=logger,
            name=f"AlphaZero_{version}",
            player_number=player_number,
        )

        self.SECONDS_PER_MOVE = seconds_per_move
        self.NETWORK_MANAGER = network_manager
        self.MCTS = mcts
        self.REPLAY_BUFFER = replay_buffer

        # self.NETWORK_MANAGER = NetworkManagerTorch(
        #     logger=logger,
        #     name_for_saving=name_for_saving,
        #     version=version,
        #     no_actions=game.NO_ACTIONS,
        #     state_shape=game.STATE_SHAPE
        # )
        # self.MCTS = MCTSWithNeuralNetwork(
        #     logger=logger,
        #     seconds_per_move=self.SECONDS_PER_MOVE,
        #     game=game,
        #     player_number=self.player_number,
        #     network_manager=self.NETWORK_MANAGER
        # )
        # self.REPLAY_BUFFER = replay_buffer
        # if self.REPLAY_BUFFER is None:
        #     self.REPLAY_BUFFER = ReplayBuffer(name_for_saving)

        self.VERSION = version
        self.NAME_FOR_SAVING = name_for_saving

    # region Public Methods

    def set_player(self, player_number: Literal[-1, 1]):
        self.player_number = player_number
        self.MCTS.player_number = player_number

    def compute_action(
            self,
            game: TwoPlayerGame

    ) -> int:
        return self.MCTS.step()

    def clone(self):
        clone = AlphaZeroAgent(
            logger=self.LOGGER,
            network_manager=self.NETWORK_MANAGER,
            mcts=self.MCTS,
            replay_buffer=self.REPLAY_BUFFER,
            player_number=self.player_number,
            version=self.VERSION,
            seconds_per_move=self.SECONDS_PER_MOVE,
            name_for_saving=self.NAME_FOR_SAVING
        )
        return clone

    # endregion Public Methods
