import random
import time
from logging import Logger
from typing import Literal

from agents.base_agent import BaseAgent
from game_logic.game_state import GameState


class RandomizedAgent(BaseAgent):

    def __init__(
            self,
            logger: Logger,
            player_number: Literal[-1, 1],
            sleep_time_before_move_s: float = 0
    ):
        self.SLEEP_TIME_BEFORE_MOVE_S = sleep_time_before_move_s

        super().__init__(logger=logger, name="Random", player_number=player_number)

    def compute_action(
            self,
            game_state: GameState
    ) -> int:
        time.sleep(self.SLEEP_TIME_BEFORE_MOVE_S)
        return random.choice(game_state.feasible_actions)
