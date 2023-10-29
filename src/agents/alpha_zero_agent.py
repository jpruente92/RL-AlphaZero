from logging import Logger
from typing import Literal

from agents.base_agent import BaseAgent
from alpha_zero.replay_buffer import ReplayBuffer
from game_logic.game_state import GameState
from monte_carlo_tree_search.mcts_with_nn import MCTSWithNeuralNetwork
from neural_network.network_manager import NetworkManagerBase


class AlphaZeroAgent(BaseAgent):

    def __init__(
            self,
            logger: Logger,
            network_manager: NetworkManagerBase,
            mcts: MCTSWithNeuralNetwork,
            replay_buffer: ReplayBuffer,
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

        self.VERSION = version

    # region Public Methods

    def set_player(self, player_number: Literal[-1, 1]):
        self.player_number = player_number
        self.MCTS.player_number = player_number

    def compute_action(
            self,
            game_state: GameState

    ) -> int:
        return self.MCTS.step(game_state)

    def clone(self):
        network_manager_clone = self.NETWORK_MANAGER.clone()
        clone = AlphaZeroAgent(
            logger=self.LOGGER,
            network_manager=network_manager_clone,
            mcts=self.MCTS.clone(network_manager_clone),
            replay_buffer=self.REPLAY_BUFFER.clone(),
            player_number=self.player_number,
            version=self.VERSION,
            seconds_per_move=self.SECONDS_PER_MOVE,
        )
        return clone

    def copy(self):
        clone = AlphaZeroAgent(
            logger=self.LOGGER,
            network_manager=self.NETWORK_MANAGER,
            mcts=self.MCTS,
            replay_buffer=self.REPLAY_BUFFER,
            player_number=self.player_number,
            version=self.VERSION,
            seconds_per_move=self.SECONDS_PER_MOVE,
        )
        return clone

    # endregion Public Methods
