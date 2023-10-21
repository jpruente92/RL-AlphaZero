import random
import time
from logging import Logger
from typing import Literal, Optional

from agents.base_agent import BaseAgent
from game_logic.two_player_game import TwoPlayerGame


class RandomizedAgent(BaseAgent):

    def __init__(
            self,
            logger: Logger,
            player_number: Literal[-1, 1],
            game: TwoPlayerGame,
            sleep_time_after_move: int = 0
    ):
        self.SLEEP_TIME_AFTER_MOVE = sleep_time_after_move

        super().__init__(logger=logger, name="Random", player_number=player_number, game=game)

    def compute_action(
            self,
            training: bool = False
    ) -> int:
        time.sleep(self.SLEEP_TIME_AFTER_MOVE)
        return random.choice(self.GAME.FEASIBLE_ACTIONS)
