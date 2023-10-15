import random
from logging import Logger
from typing import Literal, Optional

from agents.base_agent import BaseAgent
from game_logic.two_player_game import TwoPlayerGame


class RandomizedAgent(BaseAgent):

    def __init__(
            self,
            logger: Logger,
            player_number: Literal[-1, 1],
            game: TwoPlayerGame

    ):
        super().__init__(logger = logger, name="Random", player_number=player_number, game=game)

    def compute_action(
            self,
            training: bool = False
    ) -> int:
        return random.choice(self.GAME.FEASIBLE_ACTIONS)
