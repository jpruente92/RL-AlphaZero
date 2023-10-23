from logging import Logger
from typing import Literal

from agents.base_agent import BaseAgent
from game_logic.game_state import GameState
from game_logic.two_player_game import TwoPlayerGame
from monte_carlo_tree_search.mcts import MCTS


class MCTSAgent(BaseAgent):

    def __init__(self,
                 logger: Logger,
                 game: TwoPlayerGame,
                 player_number: Literal[-1, 1],
                 seconds_per_move: int,
                 name: str = "Monte Carlo Tree Search",
                 ):
        super().__init__(logger=logger, name=name, player_number=player_number)
        self.SECONDS_PER_MOVE = seconds_per_move
        self.MCTS = MCTS(
            logger=logger,
            seconds_per_move=seconds_per_move,
            player_number=player_number,
            game=game
        )

    def set_player(self, player_number: Literal[-1, 1]):
        self.player_number = player_number
        self.MCTS.player_number = player_number

    def compute_action(
            self,
            game_state: GameState
    ) -> int:
        return self.MCTS.step(game_state=game_state)
