from logging import Logger
from typing import Literal

from agents.base_agent import BaseAgent
from game_logic.two_player_game import TwoPlayerGame
from monte_carlo_tree_search.mcts import MCTS


class MCTSAgent(BaseAgent):

    def __init__(self,
                 logger: Logger,
                 player_number: Literal[-1, 1],
                 game: TwoPlayerGame,
                 seconds_per_move: int,
                 name: str = "Monte Carlo Tree Search"
                 ):
        super().__init__(logger=logger, name=name, player_number=player_number, game=game)
        self.SECONDS_PER_MOVE = seconds_per_move
        self.MCTS = MCTS(
            logger=logger,
            seconds_per_move=seconds_per_move,
            game=game,
            player_number=player_number
        )

    def set_player(self, player_number: Literal[-1, 1]):
        self.PLAYER_NUMBER = player_number
        self.MCTS.PLAYER_NUMBER = player_number

    def compute_action(
            self
    ) -> int:
        self.GAME.user_action = None
        return self.MCTS.step()
